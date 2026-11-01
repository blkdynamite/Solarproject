from dataclasses import replace

import numpy as np
import pytest

from solar_rooftop import finance
from solar_rooftop.config import FinanceConfig, ProjectConfig

CFG = ProjectConfig()
SYS, FIN = CFG.system, CFG.finance
YEAR1_KWH = 390_000.0
PRETAX, PRETAX_ITC, AFTERTAX = finance.default_scenarios(FIN)


# --- primitives -------------------------------------------------------------
def test_npv_known_value() -> None:
    assert finance.npv(0.10, [-100, 110]) == pytest.approx(0.0)
    assert finance.npv(0.0, [-100, 50, 60]) == pytest.approx(10.0)


def test_irr_known_value() -> None:
    assert finance.irr([-100, 110]) == pytest.approx(0.10)
    # 3-year annuity at 8 %
    pmt = 100 * 0.08 / (1 - 1.08**-3)
    assert finance.irr([-100, pmt, pmt, pmt]) == pytest.approx(0.08, abs=1e-9)


def test_irr_no_sign_change_is_nan() -> None:
    assert np.isnan(finance.irr([100, 10, 10]))


def test_payback_interpolates() -> None:
    assert finance.payback_years([-100, 40, 40, 40]) == pytest.approx(2.5)
    assert finance.payback_years([-100, 10, 10]) is None


# --- model consistency -----------------------------------------------------
@pytest.mark.parametrize("scenario", [PRETAX, PRETAX_ITC, AFTERTAX])
def test_npv_is_zero_at_irr(scenario: finance.Scenario) -> None:
    res = finance.evaluate(YEAR1_KWH, SYS, FIN, scenario)
    cf = res.table["net_cash_flow"].to_numpy()
    assert finance.npv(res.irr, cf) == pytest.approx(0.0, abs=1e-3)


@pytest.mark.parametrize("scenario", [PRETAX, PRETAX_ITC, AFTERTAX])
def test_lcoe_is_break_even_flat_price(scenario: finance.Scenario) -> None:
    res = finance.evaluate(YEAR1_KWH, SYS, FIN, scenario)
    fin = replace(FIN, tariff_per_kwh=res.lcoe_per_kwh, tariff_escalation=0.0)
    assert finance.evaluate(YEAR1_KWH, SYS, fin, scenario).npv == pytest.approx(0.0, abs=1e-3)


def test_pretax_lcoe_matches_textbook_formula() -> None:
    res = finance.evaluate(YEAR1_KWH, SYS, FIN, PRETAX)
    t = res.table
    disc = (1 + FIN.discount_rate) ** t.index.to_numpy()
    costs = -t["capex"].sum() - ((t["om"] + t["inverter_replacement"]) / disc).sum()
    assert res.lcoe_per_kwh == pytest.approx(costs / (t["energy_kwh"] / disc).sum())


def test_itc_and_depreciation_accounting() -> None:
    t = finance.cash_flow_table(YEAR1_KWH, SYS, FIN, AFTERTAX)
    capex = FIN.capex_per_w_dc * SYS.dc_kw * 1000
    assert t["itc"].sum() == pytest.approx(capex * FIN.itc)
    # Basis reduced by half the ITC; fully depreciated over the MACRS schedule.
    assert t["depreciation"].sum() == pytest.approx(capex * (1 - FIN.itc / 2))
    assert t.loc[0, "net_cash_flow"] == pytest.approx(-capex)


def test_energy_degrades_and_tariff_escalates() -> None:
    t = finance.cash_flow_table(YEAR1_KWH, SYS, FIN, PRETAX)
    assert t.loc[1, "energy_kwh"] == pytest.approx(YEAR1_KWH)
    assert t.loc[2, "energy_kwh"] == pytest.approx(YEAR1_KWH * (1 - FIN.degradation))
    assert t.loc[2, "tariff_per_kwh"] == pytest.approx(
        FIN.tariff_per_kwh * (1 + FIN.tariff_escalation)
    )
    assert t.loc[FIN.inverter_replace_yr, "inverter_replacement"] < 0


def test_incentives_improve_returns() -> None:
    a, b = (finance.evaluate(YEAR1_KWH, SYS, FIN, s) for s in (PRETAX, PRETAX_ITC))
    assert b.npv > a.npv and b.irr > a.irr and b.lcoe_per_kwh < a.lcoe_per_kwh
    assert b.payback_yrs is not None and a.payback_yrs is not None
    assert b.payback_yrs < a.payback_yrs


def test_zero_tax_aftertax_equals_pretax_without_macrs_effect() -> None:
    fin = replace(FIN, tax_rate=0.0)
    a = finance.evaluate(YEAR1_KWH, SYS, fin, PRETAX_ITC)
    b = finance.evaluate(YEAR1_KWH, SYS, fin, AFTERTAX)
    assert a.npv == pytest.approx(b.npv)


def test_sensitivity_table_sorted_and_brackets_base() -> None:
    sens = finance.sensitivity_table(YEAR1_KWH, SYS, FIN, AFTERTAX)
    assert list(sens["swing"]) == sorted(sens["swing"])
    base = sens.attrs["base_npv"]
    yield_row = sens.set_index("driver").loc["Energy yield"]
    assert yield_row["npv_low"] < base < yield_row["npv_high"]


def test_finance_config_validation() -> None:
    with pytest.raises(ValueError):
        FinanceConfig(discount_rate=0.9).validate()
