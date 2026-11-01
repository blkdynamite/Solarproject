"""Project cash-flow model: NPV, IRR, payback and LCOE.

Conventions
-----------
* Annual periods, end-of-year cash flows; year 0 is the construction year.
* The ITC is realised in year 1 (when the owner files taxes).
* Pre-tax scenarios ignore income tax entirely, which suits a tax-exempt owner or
  a quick screen. The after-tax scenario models a taxable commercial owner:
  energy savings raise taxable income, while O&M, inverter replacement and MACRS
  depreciation reduce it. The depreciable basis is reduced by half the ITC
  (IRC §50(c)), and the owner is assumed to have enough tax appetite to use
  losses in the year they occur.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .config import FinanceConfig, SystemConfig


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    itc: float
    after_tax: bool


def default_scenarios(fin: FinanceConfig) -> list[Scenario]:
    return [
        Scenario("pretax_no_itc", "Pre-tax, no incentives", 0.0, after_tax=False),
        Scenario("pretax_itc", f"Pre-tax, {fin.itc:.0%} ITC", fin.itc, after_tax=False),
        Scenario(
            "aftertax_itc_macrs", f"After-tax, {fin.itc:.0%} ITC + MACRS", fin.itc, after_tax=True
        ),
    ]


# ---------------------------------------------------------------------------
# Primitive metrics
# ---------------------------------------------------------------------------
def npv(rate: float, cash_flows: Sequence[float] | np.ndarray) -> float:
    cf = np.asarray(cash_flows, dtype=float)
    return float(np.sum(cf / (1 + rate) ** np.arange(len(cf))))


def irr(cash_flows: Sequence[float] | np.ndarray, lo: float = -0.99, hi: float = 10.0) -> float:
    """Internal rate of return via Brent's method; NaN if there is no sign change.

    Assumes a conventional profile (one outflow followed by inflows), which has a
    unique IRR, so a bracketing root finder is both robust and exact.
    """
    f: Callable[[float], float] = lambda r: npv(r, cash_flows)  # noqa: E731
    f_lo, f_hi = f(lo), f(hi)
    if np.sign(f_lo) == np.sign(f_hi):
        return float("nan")
    return float(brentq(f, lo, hi, xtol=1e-10, maxiter=500))


def payback_years(cash_flows: Sequence[float] | np.ndarray, rate: float = 0.0) -> float | None:
    """Years until cumulative (optionally discounted) cash flow turns positive.

    Interpolates linearly within the crossover year. ``None`` if never repaid.
    """
    cf = np.asarray(cash_flows, dtype=float) / (1 + rate) ** np.arange(len(cash_flows))
    cum = np.cumsum(cf)
    if cum[0] >= 0:
        return 0.0
    idx = np.flatnonzero(cum >= 0)
    if idx.size == 0:
        return None
    y = int(idx[0])
    return float(y - 1 + (-cum[y - 1]) / cf[y])


# ---------------------------------------------------------------------------
# Cash-flow model
# ---------------------------------------------------------------------------
def cash_flow_table(
    year1_kwh: float,
    system: SystemConfig,
    fin: FinanceConfig,
    scenario: Scenario,
    *,
    tariff_override: float | None = None,
    tariff_escalation_override: float | None = None,
) -> pd.DataFrame:
    """Build the annual cash-flow table for one scenario (all values in $ or kWh)."""
    years = np.arange(fin.life_yrs + 1)
    op = years >= 1  # operating years
    t = np.maximum(years - 1, 0)  # escalation exponent: year 1 is the base year

    tariff0 = fin.tariff_per_kwh if tariff_override is None else tariff_override
    esc = (
        fin.tariff_escalation if tariff_escalation_override is None else tariff_escalation_override
    )

    capex = fin.capex_per_w_dc * system.dc_kw * 1000
    energy = np.where(op, year1_kwh * (1 - fin.degradation) ** t, 0.0)
    tariff = np.where(op, tariff0 * (1 + esc) ** t, 0.0)
    savings = energy * tariff
    om = np.where(op, fin.om_per_kw_yr * system.dc_kw * (1 + fin.om_escalation) ** t, 0.0)
    inverter = np.where(
        years == fin.inverter_replace_yr, fin.inverter_replace_per_w_ac * system.ac_kw * 1000, 0.0
    )
    itc = np.where(years == 1, capex * scenario.itc, 0.0)

    depreciation = np.zeros_like(savings)
    tax = np.zeros_like(savings)
    if scenario.after_tax:
        basis = capex * (1 - scenario.itc / 2)
        sched = np.asarray(fin.macrs_schedule)
        n = min(len(sched), fin.life_yrs)
        depreciation[1 : n + 1] = basis * sched[:n]
        taxable = savings - om - inverter - depreciation
        tax = taxable * fin.tax_rate  # negative => tax benefit used elsewhere

    capex_flow = np.where(years == 0, -capex, 0.0)
    net = capex_flow + savings - om - inverter + itc - tax

    return pd.DataFrame(
        {
            "energy_kwh": energy,
            "tariff_per_kwh": tariff,
            "savings": savings,
            "om": -om,
            "inverter_replacement": -inverter,
            "capex": capex_flow,
            "itc": itc,
            "depreciation": depreciation,
            "income_tax": -tax,
            "net_cash_flow": net,
            "cumulative": np.cumsum(net),
        },
        index=pd.Index(years, name="year"),
    )


@dataclass(frozen=True)
class FinanceResult:
    scenario: Scenario
    table: pd.DataFrame
    capex: float
    npv: float
    irr: float
    payback_yrs: float | None
    discounted_payback_yrs: float | None
    lcoe_per_kwh: float

    def summary(self) -> dict[str, object]:
        return {
            "label": self.scenario.label,
            "capex_usd": round(self.capex),
            "npv_usd": round(self.npv),
            "irr_pct": None if np.isnan(self.irr) else round(self.irr * 100, 2),
            "payback_yrs": _round_or_none(self.payback_yrs, 1),
            "discounted_payback_yrs": _round_or_none(self.discounted_payback_yrs, 1),
            "lcoe_per_kwh": round(self.lcoe_per_kwh, 4),
        }


def levelized_cost(
    year1_kwh: float, system: SystemConfig, fin: FinanceConfig, scenario: Scenario
) -> float:
    """Flat (non-escalating) energy price at which NPV = 0, in $/kWh.

    NPV is linear in a flat price ``p``: NPV(p) = NPV(0) + p * dNPV/dp, so the
    break-even price falls out of two evaluations. This definition is
    consistent between pre-tax and after-tax views (taxes on savings, timing of
    the ITC and depreciation are all handled by the same cash-flow model).
    """

    def npv_at(price: float) -> float:
        tbl = cash_flow_table(
            year1_kwh, system, fin, scenario, tariff_override=price, tariff_escalation_override=0.0
        )
        return npv(fin.discount_rate, tbl["net_cash_flow"].to_numpy())

    n0, n1 = npv_at(0.0), npv_at(1.0)
    return -n0 / (n1 - n0)


def evaluate(
    year1_kwh: float, system: SystemConfig, fin: FinanceConfig, scenario: Scenario
) -> FinanceResult:
    table = cash_flow_table(year1_kwh, system, fin, scenario)
    cf = table["net_cash_flow"].to_numpy()
    return FinanceResult(
        scenario=scenario,
        table=table,
        capex=-float(table["capex"].sum()),
        npv=npv(fin.discount_rate, cf),
        irr=irr(cf),
        payback_yrs=payback_years(cf),
        discounted_payback_yrs=payback_years(cf, fin.discount_rate),
        lcoe_per_kwh=levelized_cost(year1_kwh, system, fin, scenario),
    )


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Sensitivity:
    label: str
    low_label: str
    high_label: str
    low: dict[str, float]
    high: dict[str, float]


def default_sensitivities(fin: FinanceConfig) -> list[Sensitivity]:
    """One-at-a-time swings around the base case (``yield_mult`` scales energy)."""
    return [
        Sensitivity("Energy yield", "-10 %", "+10 %", {"yield_mult": 0.9}, {"yield_mult": 1.1}),
        Sensitivity(
            "Installed cost",
            "+15 %",
            "-15 %",
            {"capex_per_w_dc": fin.capex_per_w_dc * 1.15},
            {"capex_per_w_dc": fin.capex_per_w_dc * 0.85},
        ),
        Sensitivity(
            "Retail tariff",
            "-15 %",
            "+15 %",
            {"tariff_per_kwh": fin.tariff_per_kwh * 0.85},
            {"tariff_per_kwh": fin.tariff_per_kwh * 1.15},
        ),
        Sensitivity(
            "Tariff escalation",
            "-1 pt",
            "+1 pt",
            {"tariff_escalation": fin.tariff_escalation - 0.01},
            {"tariff_escalation": fin.tariff_escalation + 0.01},
        ),
        Sensitivity(
            "Discount rate",
            "+2 pt",
            "-2 pt",
            {"discount_rate": fin.discount_rate + 0.02},
            {"discount_rate": fin.discount_rate - 0.02},
        ),
        Sensitivity(
            "O&M cost",
            "+25 %",
            "-25 %",
            {"om_per_kw_yr": fin.om_per_kw_yr * 1.25},
            {"om_per_kw_yr": fin.om_per_kw_yr * 0.75},
        ),
        Sensitivity(
            "Degradation", "0.8 %/yr", "0.3 %/yr", {"degradation": 0.008}, {"degradation": 0.003}
        ),
        Sensitivity("ITC", "0 %", f"{fin.itc * 100:.0f} %", {"itc": 0.0}, {"itc": fin.itc}),
    ]


def sensitivity_table(
    year1_kwh: float,
    system: SystemConfig,
    fin: FinanceConfig,
    scenario: Scenario,
    cases: Sequence[Sensitivity] | None = None,
) -> pd.DataFrame:
    """NPV for each low/high case, sorted by total swing (for a tornado chart)."""
    base_npv = evaluate(year1_kwh, system, fin, scenario).npv
    rows = []
    for case in cases if cases is not None else default_sensitivities(fin):
        row: dict[str, object] = {
            "driver": case.label,
            "low_label": case.low_label,
            "high_label": case.high_label,
        }
        for side, overrides in (("low", case.low), ("high", case.high)):
            ov: dict[str, Any] = dict(overrides)
            mult = ov.pop("yield_mult", 1.0)
            itc = ov.pop("itc", scenario.itc)
            res = evaluate(year1_kwh * mult, system, replace(fin, **ov), replace(scenario, itc=itc))
            row[f"npv_{side}"] = res.npv
        rows.append(row)
    df = pd.DataFrame(rows)
    df["swing"] = (df["npv_high"] - df["npv_low"]).abs()
    df.attrs["base_npv"] = base_npv
    return df.sort_values("swing", ascending=True).reset_index(drop=True)


def _round_or_none(x: float | None, nd: int) -> float | None:
    return None if x is None else round(x, nd)
