from dataclasses import replace

import numpy as np
import pandas as pd
import pvlib
import pytest

from solar_rooftop import energy
from solar_rooftop.config import ProjectConfig
from solar_rooftop.weather import Weather


def test_weather_is_hourly_full_year(weather: Weather) -> None:
    assert len(weather.data) == 8760
    assert weather.data.index.tz is not None
    assert (weather.data.index.minute == 30).all()  # mid-hour labels
    assert weather.data["albedo"].notna().all()


def test_annual_yield_in_plausible_range(sim: pd.DataFrame, cfg: ProjectConfig) -> None:
    m = energy.energy_metrics(sim, cfg.system, cfg.finance.yield_uncertainty)
    # NC low-tilt commercial rooftop: ~1,200-1,400 kWh/kWp, PR ~0.75-0.85
    assert 1200 < m.specific_yield_kwh_per_kwp < 1400
    assert 0.75 < m.performance_ratio < 0.85
    assert 0.13 < m.capacity_factor < 0.17
    assert m.p90_kwh < m.annual_kwh


def test_physical_bounds(sim: pd.DataFrame, cfg: ProjectConfig) -> None:
    assert (sim["ac_kw"] >= 0).all()
    assert sim["ac_kw"].max() <= cfg.system.ac_kw + 1e-6
    assert (sim["ac_kw"] <= sim["dc_kw"] + 1e-6).all()  # inverter can't create energy
    night = sim["poa_global"] == 0
    assert (sim.loc[night, "ac_kw"] == 0).all()


def test_unclipped_curve_matches_pvlib_below_rating() -> None:
    pdc0 = 1000.0
    pdc = pd.Series(np.linspace(0, 1500, 301))
    ours = energy.pvwatts_inverter_unclipped(pdc, pdc0)
    ref = pvlib.inverter.pvwatts(pdc, pdc0)
    below = ref < 0.96 * pdc0 - 1e-9
    np.testing.assert_allclose(ours[below], ref[below], atol=1e-9)
    assert (ours >= ref - 1e-9).all()


def test_clipping_grows_with_dc_ac_ratio(weather: Weather, cfg: ProjectConfig) -> None:
    sweep = energy.parameter_sweep(weather, cfg.system, "dc_ac_ratio", [1.1, 1.3, 1.5])
    clip = sweep["clipping_loss_pct"].to_numpy()
    assert np.all(np.diff(clip) >= 0)
    assert clip[-1] > 0


def test_tilt_optimum_is_reasonable(weather: Weather, cfg: ProjectConfig) -> None:
    # Rule of thumb: optimum fixed tilt sits somewhat below latitude (~36 N here).
    sweep = energy.parameter_sweep(weather, cfg.system, "tilt_deg", [0, 10, 20, 30, 40])
    assert 20 <= sweep["specific_yield"].idxmax() <= 40


def test_yield_scales_linearly_with_size(weather: Weather, cfg: ProjectConfig) -> None:
    small = energy.simulate(weather, replace(cfg.system, dc_kw=150.0))["ac_kw"].sum()
    big = energy.simulate(weather, cfg.system)["ac_kw"].sum()
    assert big / small == pytest.approx(2.0, rel=1e-6)


def test_rejects_non_hourly(sim: pd.DataFrame, cfg: ProjectConfig) -> None:
    with pytest.raises(ValueError, match="1-hour"):
        energy.energy_metrics(sim.iloc[::2], cfg.system, 0.05)
