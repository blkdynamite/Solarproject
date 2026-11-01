import numpy as np
import pandas as pd

from der_mv.config import StudyConfig
from der_mv.pipeline import Analysis
from der_mv.simulate import CONTROL, TREATMENT, simulate_fleet


def test_control_homes_are_never_affected(analysis: Analysis) -> None:
    f = analysis.fleet
    ctl = f.devices.loc[f.devices["grp"] == CONTROL, "device_id"].to_numpy() - 1
    np.testing.assert_array_equal(f.actual_kw[ctl], f.counterfactual_kw[ctl])


def test_holdout_share_is_respected(analysis: Analysis, small_cfg: StudyConfig) -> None:
    share = (analysis.fleet.devices["grp"] == CONTROL).mean()
    assert share == small_cfg.fleet.holdout_share


def test_events_are_the_hottest_weekdays(analysis: Analysis) -> None:
    f = analysis.fleet
    temp = f.weather.set_index("ts")["temp_c"]
    daily_max = temp.resample("D").max()
    event_days = pd.to_datetime(f.events["event_date"])
    assert (event_days.dt.dayofweek < 5).all()
    weekday_max = daily_max[daily_max.index.dayofweek < 5]
    eligible = weekday_max[weekday_max.index >= event_days.min()]
    non_event = eligible[~eligible.index.isin(event_days)]
    assert daily_max.loc[event_days].min() >= non_event.max() - 1e-9


def test_true_impact_positive_during_events(analysis: Analysis) -> None:
    per_event = analysis.fleet.true_impact.groupby("event_id")["impact_kw"].mean()
    assert (per_event > 0.3).all()  # a thermostat program sheds a meaningful share of AC load


def test_telemetry_is_messy_like_real_iot(analysis: Analysis) -> None:
    tel = analysis.fleet.telemetry
    n_dupes = tel.duplicated(["device_id", "ts"]).sum()
    assert n_dupes > 0
    n_expected = len(analysis.fleet.devices) * len(analysis.fleet.hours)
    assert tel.drop_duplicates(["device_id", "ts"]).shape[0] < n_expected  # gaps exist


def test_simulation_is_deterministic(small_cfg: StudyConfig) -> None:
    a = simulate_fleet(small_cfg.site, small_cfg.fleet)
    b = simulate_fleet(small_cfg.site, small_cfg.fleet)
    pd.testing.assert_frame_equal(a.telemetry, b.telemetry)
    assert set(a.devices["grp"]) == {CONTROL, TREATMENT}
