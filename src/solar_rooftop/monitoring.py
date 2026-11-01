"""Performance monitoring: synthetic meter data, fault detection and scoring.

The detector compares measured daily energy with modelled (expected) energy
and classifies each day. It is deliberately **causal**: every decision uses
only data up to and including that day, so it behaves the same way it would
running live each night on a monitoring platform.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from .config import MonitoringConfig, SystemConfig


class Status(StrEnum):
    OK = "ok"
    LOW_IRRADIANCE = "low_irradiance"  # too little sun for a reliable ratio
    NO_DATA = "no_data"
    PARTIAL_DATA = "partial_data"
    INVERTER_OUTAGE = "inverter_outage"
    MAJOR_OUTAGE = "major_outage"
    UNDERPERFORMANCE = "underperformance"  # gradual: soiling, shading, degradation


ALARMS = frozenset(
    {
        Status.NO_DATA,
        Status.PARTIAL_DATA,
        Status.INVERTER_OUTAGE,
        Status.MAJOR_OUTAGE,
        Status.UNDERPERFORMANCE,
    }
)
#: Alarms that represent lost production (data gaps do not).
PRODUCTION_ALARMS = frozenset(
    {Status.INVERTER_OUTAGE, Status.MAJOR_OUTAGE, Status.UNDERPERFORMANCE}
)

#: Fraction of a day's expected energy that must be covered by meter data.
MIN_DATA_COVERAGE = 0.9


# ---------------------------------------------------------------------------
# Synthetic measured data
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Fault:
    """A ground-truth problem injected into the synthetic meter data.

    ``start`` and ``end`` are inclusive local dates. ``severity`` means
    inverters lost (outage), peak fractional loss (soiling) and is ignored for
    comms dropouts.
    """

    kind: str  # "inverter_outage" | "soiling" | "comms_dropout"
    start: str
    end: str
    description: str
    severity: float = 1.0
    expected_status: Status = Status.INVERTER_OUTAGE

    def mask(self, index: pd.DatetimeIndex) -> np.ndarray:
        days = index.normalize().tz_localize(None)
        return np.asarray((days >= pd.Timestamp(self.start)) & (days <= pd.Timestamp(self.end)))


def default_faults(year: int) -> list[Fault]:
    return [
        Fault(
            "inverter_outage",
            f"{year}-05-12",
            f"{year}-05-20",
            "Inverter 3 tripped offline",
            1,
            Status.INVERTER_OUTAGE,
        ),
        Fault(
            "soiling",
            f"{year}-07-01",
            f"{year}-08-15",
            "Soiling build-up during dry spell (0 -> 8 %)",
            0.08,
            Status.UNDERPERFORMANCE,
        ),
        Fault(
            "comms_dropout",
            f"{year}-10-03",
            f"{year}-10-05",
            "Data-logger comms dropout (plant healthy)",
            0,
            Status.NO_DATA,
        ),
    ]


def synthesize_meter(
    expected_kw: pd.Series,
    system: SystemConfig,
    cfg: MonitoringConfig,
    faults: list[Fault],
    rng: np.random.Generator,
) -> pd.Series:
    """Create a plausible hourly 'meter' series from the model plus injected faults.

    Error model: a day-level multiplicative error (weather/irradiance mismatch
    between model and site, which is what dominates real daily performance
    ratios) and smaller independent hourly noise.
    """
    meas = expected_kw.astype(float).copy()
    days = meas.index.normalize()
    unique_days = days.unique()
    day_err = pd.Series(rng.normal(1.0, cfg.daily_noise, len(unique_days)), index=unique_days)
    meas *= day_err.reindex(days).to_numpy()
    meas *= rng.normal(1.0, cfg.hourly_noise, len(meas))

    for f in faults:
        m = f.mask(meas.index)
        if f.kind == "inverter_outage":
            meas[m] *= 1 - f.severity / system.n_inverters
        elif f.kind == "soiling":
            meas[m] *= 1 - np.linspace(0, f.severity, int(m.sum()))
        elif f.kind == "comms_dropout":
            meas[m] = np.nan
        else:
            raise ValueError(f"Unknown fault kind: {f.kind}")
    return meas.clip(lower=0)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def daily_performance(expected_kw: pd.Series, measured_kw: pd.Series) -> pd.DataFrame:
    """Aggregate hourly series to daily energy and a data-coverage measure."""
    covered = measured_kw.notna()
    d = pd.DataFrame(
        {
            "expected_kwh": expected_kw.resample("D").sum(),
            "expected_covered_kwh": expected_kw.where(covered, 0.0).resample("D").sum(),
            "actual_kwh": measured_kw.resample("D").sum(min_count=1),
        }
    )
    d["data_coverage"] = (d["expected_covered_kwh"] / d["expected_kwh"]).where(
        d["expected_kwh"] > 0, 1.0
    )
    d["perf_index"] = d["actual_kwh"] / d["expected_covered_kwh"]
    return d


def detect(
    expected_kw: pd.Series,
    measured_kw: pd.Series,
    system: SystemConfig,
    cfg: MonitoringConfig,
) -> pd.DataFrame:
    """Classify each day; returns the daily frame with ``status`` and diagnostics.

    Rules (evaluated in order, using only past data):

    1. No / partial meter data -> data alarm, kept apart from production alarms
       so a comms fault is never mistaken for a plant outage.
    2. Very overcast days are skipped: the ratio is noisy when there is little sun.
    3. The performance index (PI = actual / expected) is normalised by a
       trailing median of recent healthy days. This absorbs any constant model
       bias, so thresholds are relative to how *this* plant normally performs.
    4. A sudden step down relative to the last few days, of at least half an
       inverter's share, is an inverter outage (the estimated number of
       inverters down is reported). A drop beyond ``major_outage_threshold``
       is a major outage.
    5. A trailing-median PI more than ``drift_threshold`` below baseline is
       gradual underperformance (soiling, shading, degradation). A step back
       up to baseline (rain or a wash) restarts the trend so the alarm clears.
    """
    d = daily_performance(expected_kw, measured_kw)
    min_expected = cfg.min_expected_kwh_per_kwp * system.dc_kw
    step_threshold = 0.5 / system.n_inverters

    healthy: deque[float] = deque(maxlen=cfg.baseline_window_days)
    recent: deque[float] = deque(maxlen=cfg.drift_window_days)
    min_recent = int(np.ceil(cfg.drift_window_days / 2))

    status, baseline_col, npi_col, trend_col, inv_down = [], [], [], [], []
    for day in d.itertuples():
        baseline = float(np.median(healthy)) if len(healthy) >= cfg.baseline_min_days else 1.0
        npi = day.perf_index / baseline if np.isfinite(day.perf_index) else np.nan
        reference = float(np.median(recent)) if len(recent) >= min_recent else 1.0
        trend = np.nan
        k_down = 0

        if day.data_coverage == 0 or np.isnan(day.actual_kwh):
            s = Status.NO_DATA
        elif day.data_coverage < MIN_DATA_COVERAGE:
            s = Status.PARTIAL_DATA
        elif day.expected_kwh < min_expected:
            s = Status.LOW_IRRADIANCE
        elif npi < 1 - cfg.major_outage_threshold:
            s = Status.MAJOR_OUTAGE
        elif reference - npi >= step_threshold:
            s = Status.INVERTER_OUTAGE
            k_down = max(1, round((reference - npi) * system.n_inverters))
        else:
            # A step *up* back to baseline (rain or a wash) ends any soiling
            # episode: restart the trend so the alarm clears immediately.
            if npi - reference >= cfg.drift_threshold and npi >= 1 - cfg.drift_threshold / 2:
                recent.clear()
            recent.append(npi)
            trend = float(np.median(recent))
            if len(recent) >= min_recent and trend < 1 - cfg.drift_threshold:
                s = Status.UNDERPERFORMANCE
            else:
                s = Status.OK
                healthy.append(day.perf_index)

        status.append(s.value)
        baseline_col.append(baseline)
        npi_col.append(npi)
        trend_col.append(trend)
        inv_down.append(k_down)

    d["baseline_pi"] = baseline_col
    d["normalised_pi"] = npi_col
    d["trend_pi"] = trend_col
    d["inverters_down_est"] = inv_down
    d["status"] = status

    is_loss = d["status"].isin({s.value for s in PRODUCTION_ALARMS})
    target = d["expected_covered_kwh"] * d["baseline_pi"]
    d["lost_kwh"] = np.where(is_loss, (target - d["actual_kwh"]).clip(lower=0), 0.0)
    # Energy that went unmeasured; production during a data gap is unknown.
    d["unmetered_expected_kwh"] = d["expected_kwh"] - d["expected_covered_kwh"]
    return d


def group_events(daily: pd.DataFrame, max_gap_days: int = 2) -> list[dict[str, object]]:
    """Merge consecutive alarm days with the same status into events.

    Gaps of up to ``max_gap_days`` (e.g. a skipped overcast day) are bridged.
    """
    events: list[dict[str, object]] = []
    alarms = daily[daily["status"].isin({s.value for s in ALARMS})]
    for status, grp in alarms.groupby("status", sort=False):
        dates = list(grp.index)
        seg_start = 0
        for i in range(1, len(dates) + 1):
            if i == len(dates) or (dates[i] - dates[i - 1]).days > max_gap_days:
                seg = grp.iloc[seg_start:i]
                events.append(
                    {
                        "status": status,
                        "start": str(seg.index[0].date()),
                        "end": str(seg.index[-1].date()),
                        "flagged_days": len(seg),
                        "lost_kwh": round(float(seg["lost_kwh"].sum()), 1),
                        "unmetered_expected_kwh": round(
                            float(seg["unmetered_expected_kwh"].sum()), 1
                        ),
                    }
                )
                seg_start = i
    return sorted(events, key=lambda e: str(e["start"]))


# ---------------------------------------------------------------------------
# Scoring against ground truth
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FaultScore:
    fault: str
    window: str
    expected_alarm: str
    detected: bool
    first_detection: str | None
    latency_days: int | None
    days_detected: int
    days_evaluable: int
    recall: float | None
    misclassified_days: int


@dataclass(frozen=True)
class DetectionScore:
    faults: list[FaultScore]
    false_positive_days: int
    healthy_days_evaluated: int
    precision: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score(daily: pd.DataFrame, faults: list[Fault]) -> DetectionScore:
    """Day-level precision/recall and per-fault detection latency.

    A day inside a fault window counts as detected when it raises the alarm
    expected for that fault. Overcast days the detector skips are excluded from
    recall (it is not designed to judge them). Any alarm outside every fault
    window is a false positive.
    """
    idx = pd.DatetimeIndex(daily.index)
    status = daily["status"].to_numpy()
    evaluable_day = status != Status.LOW_IRRADIANCE.value
    is_alarm = np.isin(status, [s.value for s in ALARMS])
    in_any = np.zeros(len(daily), dtype=bool)

    per_fault: list[FaultScore] = []
    for f in faults:
        m = f.mask(idx)
        in_any |= m
        n_eval = int((m & evaluable_day).sum())
        hit = m & (status == f.expected_status.value)
        n_hit = int(hit.sum())
        first = idx[hit][0] if n_hit else None
        per_fault.append(
            FaultScore(
                fault=f.description,
                window=f"{f.start} .. {f.end}",
                expected_alarm=f.expected_status.value,
                detected=n_hit > 0,
                first_detection=str(first.date()) if first is not None else None,
                latency_days=(first.tz_localize(None).normalize() - pd.Timestamp(f.start)).days
                if first is not None
                else None,
                days_detected=n_hit,
                days_evaluable=n_eval,
                recall=round(n_hit / n_eval, 3) if n_eval else None,
                misclassified_days=int((m & is_alarm & ~hit).sum()),
            )
        )

    flagged = int(is_alarm.sum())
    true_pos = sum(fs.days_detected for fs in per_fault)
    return DetectionScore(
        faults=per_fault,
        false_positive_days=int((is_alarm & ~in_any).sum()),
        healthy_days_evaluated=int((~in_any & evaluable_day).sum()),
        precision=round(true_pos / flagged, 3) if flagged else None,
    )
