"""Hourly PV energy model (PVWatts-style, built on pvlib)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
import pvlib
from scipy.stats import norm

from .config import SystemConfig
from .weather import Weather

HOURS_PER_YEAR = 8760


def simulate(weather: Weather, system: SystemConfig) -> pd.DataFrame:
    """Run an hourly simulation.

    Returns a frame indexed like the weather with columns (powers in kW,
    irradiance in W/m2): ``poa_global``, ``poa_effective``, ``temp_cell``,
    ``dc_kw`` (after system losses), ``ac_unclipped_kw`` and ``ac_kw``.
    """
    wx, loc = weather.data, weather.location
    solpos = loc.get_solarposition(wx.index)

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=system.tilt_deg,
        surface_azimuth=system.azimuth_deg,
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=wx["dni"],
        ghi=wx["ghi"],
        dhi=wx["dhi"],
        dni_extra=pvlib.irradiance.get_extra_radiation(wx.index),
        albedo=wx["albedo"],
        model=system.transposition_model,
    )
    poa_global = poa["poa_global"].fillna(0.0)

    # Reflection (incidence-angle) losses apply to the beam component only;
    # diffuse IAM is folded into the PVWatts loss budget.
    aoi = pvlib.irradiance.aoi(
        system.tilt_deg, system.azimuth_deg, solpos["apparent_zenith"], solpos["azimuth"]
    )
    iam = pvlib.iam.physical(aoi)
    poa_eff = (poa["poa_direct"].fillna(0.0) * iam + poa["poa_diffuse"].fillna(0.0)).clip(lower=0)

    temp_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][system.racking]
    temp_cell = pvlib.temperature.sapm_cell(
        poa_global, wx["temp_air"], wx["wind_speed"], **temp_params
    )

    pdc_w = pvlib.pvsystem.pvwatts_dc(poa_eff, temp_cell, system.dc_kw * 1000, system.gamma_pdc)
    pdc_w = pdc_w * (1 - system.losses_pct / 100)

    pac0_w = system.ac_kw * 1000
    pdc0_inv_w = pac0_w / system.inverter_eff_nom
    pac_w = pvlib.inverter.pvwatts(pdc_w, pdc0_inv_w, eta_inv_nom=system.inverter_eff_nom)
    # Same efficiency curve without the AC limit -> quantifies clipping loss.
    pac_unclipped_w = pvwatts_inverter_unclipped(pdc_w, pdc0_inv_w, system.inverter_eff_nom)

    return pd.DataFrame(
        {
            "poa_global": poa_global,
            "poa_effective": poa_eff,
            "temp_cell": temp_cell,
            "dc_kw": pdc_w.clip(lower=0).fillna(0.0) / 1000,
            "ac_unclipped_kw": pac_unclipped_w.clip(lower=0).fillna(0.0) / 1000,
            "ac_kw": pac_w.clip(lower=0).fillna(0.0) / 1000,
        },
        index=wx.index,
    )


def pvwatts_inverter_unclipped(
    pdc: pd.Series, pdc0: float, eta_inv_nom: float = 0.96, eta_inv_ref: float = 0.9637
) -> pd.Series:
    """PVWatts v5 inverter efficiency curve (Dobos 2014, eq. 1) with no AC cap.

    Identical to :func:`pvlib.inverter.pvwatts` below the AC rating; above it,
    returns the power the inverter *would* deliver, so ``unclipped - clipped``
    is the energy lost to clipping.
    """
    zeta = pdc / pdc0
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = eta_inv_nom / eta_inv_ref * (-0.0162 * zeta - 0.0059 / zeta + 0.9858)
    # Like pvlib, floor at zero: the curve goes negative at very low loading.
    return (eta * pdc).clip(lower=0).where(pdc > 0, 0.0)


@dataclass(frozen=True)
class EnergyMetrics:
    annual_kwh: float
    specific_yield_kwh_per_kwp: float
    capacity_factor: float
    performance_ratio: float
    poa_insolation_kwh_m2: float
    clipping_loss_kwh: float
    clipping_hours: int
    p90_kwh: float

    @property
    def clipping_loss_pct(self) -> float:
        return 100 * self.clipping_loss_kwh / (self.annual_kwh + self.clipping_loss_kwh)


def energy_metrics(
    sim: pd.DataFrame, system: SystemConfig, yield_uncertainty: float
) -> EnergyMetrics:
    """Headline yield KPIs from an hourly simulation."""
    _check_hourly(sim.index)
    annual = float(sim["ac_kw"].sum())
    poa_kwh_m2 = float(sim["poa_global"].sum() / 1000)
    clipping_kwh = float((sim["ac_unclipped_kw"] - sim["ac_kw"]).clip(lower=0).sum())
    # One-sided 90 % exceedance under a normal approximation.
    p90 = annual * (1 - norm.ppf(0.9) * yield_uncertainty)
    return EnergyMetrics(
        annual_kwh=annual,
        specific_yield_kwh_per_kwp=annual / system.dc_kw,
        capacity_factor=annual / (system.dc_kw * HOURS_PER_YEAR),
        # IEC 61724 PR: final yield (kWh/kWp) / reference yield (kWh/m2 / 1 kW/m2)
        performance_ratio=annual / (poa_kwh_m2 * system.dc_kw),
        poa_insolation_kwh_m2=poa_kwh_m2,
        clipping_loss_kwh=clipping_kwh,
        clipping_hours=int((sim["ac_kw"] >= system.ac_kw * 0.995).sum()),
        p90_kwh=float(p90),
    )


def parameter_sweep(
    weather: Weather, system: SystemConfig, param: str, values: list[float]
) -> pd.DataFrame:
    """Re-run the simulation for each value of one ``SystemConfig`` field."""
    rows = []
    for v in values:
        override: dict[str, Any] = {param: v}
        sim = simulate(weather, replace(system, **override))
        clipped = float((sim["ac_unclipped_kw"] - sim["ac_kw"]).clip(lower=0).sum())
        annual = float(sim["ac_kw"].sum())
        rows.append(
            {
                param: v,
                "annual_kwh": annual,
                "specific_yield": annual / system.dc_kw,
                "clipping_loss_pct": 100 * clipped / (annual + clipped) if annual else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index(param)


def _check_hourly(index: pd.Index) -> None:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        raise ValueError("expected an hourly DatetimeIndex")
    steps_ns = np.diff(index.asi8)
    if not np.all(steps_ns == 3_600 * 10**9):
        raise ValueError("energy sums assume a regular 1-hour timestep")
