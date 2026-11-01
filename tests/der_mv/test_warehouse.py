import pandas as pd
import pytest

from der_mv import warehouse as wh
from der_mv.config import StudyConfig
from der_mv.pipeline import Analysis


def test_all_sql_data_tests_pass(analysis: Analysis) -> None:
    assert all(t.failing_rows == 0 for t in analysis.tests)
    assert len(analysis.tests) >= 4


def test_deduplication_keeps_one_row_per_device_hour(analysis: Analysis) -> None:
    raw = analysis.fleet.telemetry
    staged = wh.query(analysis.con, "SELECT count(*) AS n FROM stg_telemetry")["n"].iloc[0]
    assert staged == len(raw.drop_duplicates(["device_id", "ts"]))


def test_ten_of_ten_sql_matches_pandas_reference(analysis: Analysis, small_cfg: StudyConfig) -> None:
    """Re-implement the 10-of-10 baseline independently and compare with the SQL model."""
    con, mv = analysis.con, small_cfg.mv
    tel = wh.query(con, "SELECT device_id, ts, day, hod, kw FROM stg_telemetry")
    event_days = set(pd.to_datetime(analysis.fleet.events["event_date"]))
    row = wh.query(
        con,
        "SELECT analysis_id, day FROM int_analysis_days WHERE kind = 'event' ORDER BY day DESC "
        "LIMIT 1",
    ).iloc[0]
    target_day = pd.Timestamp(row["day"])

    for device_id in (1, 2, 3, 50, 123):
        d = tel[tel["device_id"] == device_id]
        counts = d.groupby("day").size()
        days = pd.to_datetime(pd.Series(counts.index))
        ok = (
            (days < target_day)
            & (days.dt.dayofweek < 5)
            & (~days.isin(event_days))
            & (counts.to_numpy() == 24)
        )
        chosen = set(days[ok].sort_values().iloc[-mv.baseline_days :].dt.date)
        ref = (
            d[pd.to_datetime(d["day"]).dt.date.isin(chosen)].groupby("hod")["kw"].mean()
        )
        sql = wh.query(
            con,
            f"SELECT hod, baseline_kw FROM mart_baseline_hourly "
            f"WHERE analysis_id = {row['analysis_id']} AND device_id = {device_id}",
        ).set_index("hod")["baseline_kw"]
        pd.testing.assert_series_equal(
            sql.sort_index(), ref.sort_index(), check_names=False, check_dtype=False, rtol=1e-5
        )


def test_placebo_days_are_non_event_weekdays(analysis: Analysis) -> None:
    days = analysis.analysis_days
    placebo = pd.to_datetime(days.loc[days["kind"] == "placebo", "day"])
    events = pd.to_datetime(days.loc[days["kind"] == "event", "day"])
    assert (placebo.dt.dayofweek < 5).all()
    assert not set(placebo) & set(events)


def test_adjustment_cap_respected(analysis: Analysis, small_cfg: StudyConfig) -> None:
    ed = wh.query(analysis.con, "SELECT * FROM mart_event_device WHERE eligible")
    ratio = ed["adj_baseline_kw"] / ed["baseline_kw"]
    cap = small_cfg.mv.adjustment_cap
    assert ratio.between(1 - cap - 1e-9, 1 + cap + 1e-9).all()


def test_data_test_catches_duplicates(analysis: Analysis) -> None:
    con = analysis.con
    con.execute("CREATE TEMP TABLE _backup AS SELECT * FROM stg_telemetry")
    try:
        con.execute("INSERT INTO stg_telemetry SELECT * FROM stg_telemetry LIMIT 5")
        with pytest.raises(wh.DataQualityError, match="unique"):
            wh.run_tests(con)
    finally:
        con.execute("CREATE OR REPLACE TABLE stg_telemetry AS SELECT * FROM _backup")
    assert all(t.failing_rows == 0 for t in wh.run_tests(con))
