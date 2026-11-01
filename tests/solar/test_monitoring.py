from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from solar_rooftop import monitoring as mon
from solar_rooftop.config import ProjectConfig
from solar_rooftop.monitoring import Status


def _run(sim: pd.DataFrame, cfg: ProjectConfig, faults: list[mon.Fault], seed: int) -> pd.DataFrame:
    meas = mon.synthesize_meter(
        sim["ac_kw"], cfg.system, cfg.monitoring, faults, np.random.default_rng(seed)
    )
    return mon.detect(sim["ac_kw"], meas, cfg.system, cfg.monitoring)


def test_perfect_data_is_all_ok(sim: pd.DataFrame, cfg: ProjectConfig) -> None:
    quiet = replace(cfg, monitoring=replace(cfg.monitoring, hourly_noise=0, daily_noise=0))
    daily = _run(sim, quiet, [], seed=0)
    assert set(daily["status"]) <= {Status.OK.value, Status.LOW_IRRADIANCE.value}
    assert daily["lost_kwh"].sum() == 0


@pytest.mark.parametrize("seed", range(5))
def test_detects_injected_faults(sim: pd.DataFrame, cfg: ProjectConfig, seed: int) -> None:
    faults = mon.default_faults(cfg.site.model_year)
    sc = mon.score(_run(sim, cfg, faults, seed), faults)
    outage, soiling, comms = sc.faults
    assert outage.recall == 1.0 and outage.latency_days == 0
    assert outage.misclassified_days == 0
    assert comms.recall == 1.0
    assert soiling.detected and soiling.latency_days is not None and soiling.latency_days <= 35
    assert soiling.misclassified_days <= 1
    assert sc.false_positive_days <= 3


def test_comms_dropout_not_counted_as_lost_energy(sim: pd.DataFrame, cfg: ProjectConfig) -> None:
    faults = [f for f in mon.default_faults(cfg.site.model_year) if f.kind == "comms_dropout"]
    daily = _run(sim, cfg, faults, seed=1)
    gap = daily.loc[faults[0].start : faults[0].end]
    assert (gap["status"] == Status.NO_DATA.value).all()
    assert (gap["lost_kwh"] == 0).all()
    assert (gap["unmetered_expected_kwh"] > 0).all()


def test_partial_data_day(sim: pd.DataFrame, cfg: ProjectConfig) -> None:
    meas = sim["ac_kw"].copy()
    day = f"{cfg.site.model_year}-06-10"
    meas.loc[f"{day} 11:00" : f"{day} 15:00"] = np.nan
    daily = mon.detect(sim["ac_kw"], meas, cfg.system, cfg.monitoring)
    assert daily.loc[day, "status"] == Status.PARTIAL_DATA.value


def test_two_inverters_down_estimated(sim: pd.DataFrame, cfg: ProjectConfig) -> None:
    y = cfg.site.model_year
    faults = [mon.Fault("inverter_outage", f"{y}-04-10", f"{y}-04-12", "two down", 2)]
    daily = _run(sim, cfg, faults, seed=3)
    hit = daily.loc[f"{y}-04-10" : f"{y}-04-12"]
    flagged = hit[hit["status"] == Status.INVERTER_OUTAGE.value]
    assert len(flagged) >= 2
    assert (flagged["inverters_down_est"] == 2).all()


def test_full_outage_is_major(sim: pd.DataFrame, cfg: ProjectConfig) -> None:
    y = cfg.site.model_year
    faults = [mon.Fault("inverter_outage", f"{y}-04-10", f"{y}-04-11", "site trip", 5)]
    daily = _run(sim, cfg, faults, seed=4)
    assert (daily.loc[f"{y}-04-10" : f"{y}-04-11", "status"] == Status.MAJOR_OUTAGE.value).all()


def test_detector_is_causal(sim: pd.DataFrame, cfg: ProjectConfig) -> None:
    """Results up to a date must not change when later data changes."""
    faults = mon.default_faults(cfg.site.model_year)
    meas = mon.synthesize_meter(
        sim["ac_kw"], cfg.system, cfg.monitoring, faults, np.random.default_rng(0)
    )
    cutoff = f"{cfg.site.model_year}-06-30"
    full = mon.detect(sim["ac_kw"], meas, cfg.system, cfg.monitoring)
    tampered = meas.copy()
    tampered.loc[f"{cfg.site.model_year}-07-01" :] *= 0.5
    part = mon.detect(sim["ac_kw"], tampered, cfg.system, cfg.monitoring)
    pd.testing.assert_series_equal(full.loc[:cutoff, "status"], part.loc[:cutoff, "status"])


def test_group_events_bridges_short_gaps() -> None:
    idx = pd.date_range("2025-01-01", periods=10, freq="D")
    st = [
        "ok",
        "underperformance",
        "underperformance",
        "low_irradiance",
        "underperformance",
        "ok",
        "ok",
        "ok",
        "ok",
        "underperformance",
    ]
    daily = pd.DataFrame({"status": st, "lost_kwh": 1.0, "unmetered_expected_kwh": 0.0}, index=idx)
    events = mon.group_events(daily)
    assert [(e["start"], e["end"], e["flagged_days"]) for e in events] == [
        ("2025-01-02", "2025-01-05", 3),
        ("2025-01-10", "2025-01-10", 1),
    ]
