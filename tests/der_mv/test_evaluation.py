import numpy as np
import pandas as pd
import pytest

from der_mv import evaluation as ev
from der_mv.config import MVConfig, StudyConfig
from der_mv.pipeline import Analysis


def test_holdout_tradeoff_monotone() -> None:
    t = ev.holdout_tradeoff(sigma_raw=1.5, sigma_adjusted=0.4, impact_kw=1.0, mv=MVConfig())
    assert t["ci_halfwidth_pct_raw"].is_monotonic_decreasing
    assert (t["ci_halfwidth_pct_ml_adjusted"] < t["ci_halfwidth_pct_raw"]).all()
    assert t["forgone_mw"].is_monotonic_increasing


def test_dollar_impact_zero_for_exact_estimate() -> None:
    summary = pd.DataFrame(
        {"season_kw": [1.0, 0.5], "true_season_kw": [1.0, 1.0]}, index=["exact", "half"]
    )
    d = ev.dollar_impact(summary, MVConfig(program_devices=1000, capacity_value_per_kw_yr=100))
    assert d.loc["exact", "misstatement_usd"] == pytest.approx(0)
    assert d.loc["half", "misstatement_usd"] == pytest.approx(-50_000)


# --- End-to-end findings on a small simulated fleet ---------------------------
def test_ml_plus_control_is_accurate(analysis: Analysis) -> None:
    s = analysis.summary
    assert abs(s.loc["ml_control", "season_error_pct"]) < 5
    assert abs(s.loc["ml_control", "placebo_mean_kw"]) < 0.05


def test_rct_is_unbiased_within_its_noise(analysis: Analysis) -> None:
    s = analysis.summary.loc["rct"]
    assert s["season_ci_low"] - 0.05 <= s["true_season_kw"] <= s["season_ci_high"] + 0.05


def test_ten_of_ten_underestimates_on_hot_days(analysis: Analysis) -> None:
    s = analysis.summary
    assert s.loc["ten_of_ten", "season_error_pct"] < -30
    assert s.loc["ten_of_ten", "placebo_mean_kw"] < -0.1  # bias shows up with no event too


def test_truth_is_zero_on_placebo_days(analysis: Analysis) -> None:
    r = analysis.results
    assert (r.loc[r["kind"] == "placebo", "truth_kw"] == 0).all()
    assert (r.loc[r["kind"] == "event", "truth_kw"] > 0).all()


def test_ml_model_validates_reasonably(analysis: Analysis) -> None:
    rep = analysis.model_report
    assert rep.hourly_cv_rmse_pct < 25
    assert abs(rep.fleet_bias_pct) < 10
    assert np.isclose(sum(rep.feature_importance.values()), 1.0, atol=0.01)


def test_small_cfg_is_valid(small_cfg: StudyConfig) -> None:
    assert small_cfg.fleet.n_homes == 400
