"""M&V estimators of delivered load reduction, with bootstrap uncertainty.

Every method is written in one form: for each analysis day,

    estimate = mean over treated devices of v_T  -  mean over control devices of v_C

where ``v_T`` / ``v_C`` are per-device values over the event window:

==================  ==========================  ==========================
method              v_T (treated)               v_C (control)
==================  ==========================  ==========================
10-of-10            baseline - actual           (not used)
10-of-10 adjusted   adj. baseline - actual      (not used)
ML counterfactual   predicted - actual          (not used)
ML + control group  predicted - actual          predicted - actual
Control group (RCT) -actual                     -actual
==================  ==========================  ==========================

"ML + control group" is a regression-adjusted RCT: the control group's
prediction error on the same day removes any model bias (randomisation makes
the two groups exchangeable), and subtracting predictions removes most of the
home-to-home variation, so it is both unbiased and far more precise than a
plain difference in means. It is the same idea as CUPED in online A/B testing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .simulate import CONTROL, TREATMENT


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    treated_col: str
    control_col: str | None = None


METHODS = [
    Method("ten_of_ten", "10-of-10 baseline", "v_10of10"),
    Method("ten_of_ten_adj", "10-of-10 + day-of adjustment", "v_10of10_adj"),
    Method("ml", "ML counterfactual", "v_ml"),
    Method("ml_control", "ML + control group", "v_ml", "v_ml"),
    Method("rct", "Control group (RCT)", "v_rct", "v_rct"),
]
METHOD_BY_KEY = {m.key: m for m in METHODS}


def device_values(event_device: pd.DataFrame, pred_window_kw: pd.Series) -> pd.DataFrame:
    """Per device x analysis day values for each method (eligible devices only).

    ``pred_window_kw`` is the ML prediction averaged over the event window,
    indexed by (analysis_id, device_id).
    """
    df = event_device[event_device["eligible"]].merge(
        pred_window_kw.rename("pred_kw"),
        left_on=["analysis_id", "device_id"],
        right_index=True,
        how="left",
    )
    df["v_10of10"] = df["baseline_kw"] - df["actual_kw"]
    df["v_10of10_adj"] = df["adj_baseline_kw"] - df["actual_kw"]
    df["v_ml"] = df["pred_kw"] - df["actual_kw"]
    df["v_rct"] = -df["actual_kw"]
    return df


@dataclass(frozen=True)
class Estimates:
    """Point estimates and bootstrap replicates, per method and analysis day."""

    analysis_ids: np.ndarray
    point: pd.DataFrame  # index analysis_id, columns method keys
    replicates: dict[str, np.ndarray]  # method -> (n_boot, n_days)


def estimate(values: pd.DataFrame, n_boot: int, seed: int) -> Estimates:
    """Point estimates plus a Poisson cluster bootstrap over devices.

    Each bootstrap replicate gives every device one Poisson(1) weight, shared
    across all analysis days, so the season-level uncertainty respects the fact
    that the same homes appear in every event. Poisson weights are the standard
    way to bootstrap at scale: no resampling, just weighted sums.
    """
    devices = np.sort(values["device_id"].unique())
    days = np.sort(values["analysis_id"].unique())
    dev_pos = pd.Index(devices).get_indexer(values["device_id"])
    day_pos = pd.Index(days).get_indexer(values["analysis_id"])

    rng = np.random.default_rng(seed)
    weights = rng.poisson(1.0, (n_boot, len(devices))).astype(float)
    ones = np.ones((1, len(devices)))

    def group_means(col: str, grp: str, w: np.ndarray) -> np.ndarray:
        m = (values["grp"] == grp).to_numpy() & values[col].notna().to_numpy()
        mat = np.zeros((len(devices), len(days)))
        cnt = np.zeros((len(devices), len(days)))
        mat[dev_pos[m], day_pos[m]] = values.loc[m, col].to_numpy()
        cnt[dev_pos[m], day_pos[m]] = 1.0
        with np.errstate(invalid="ignore", divide="ignore"):
            return (w @ mat) / (w @ cnt)

    point, reps = {}, {}
    for meth in METHODS:
        p = group_means(meth.treated_col, TREATMENT, ones)[0]
        r = group_means(meth.treated_col, TREATMENT, weights)
        if meth.control_col:
            p = p - group_means(meth.control_col, CONTROL, ones)[0]
            r = r - group_means(meth.control_col, CONTROL, weights)
        point[meth.key], reps[meth.key] = p, r
    return Estimates(days, pd.DataFrame(point, index=pd.Index(days, name="analysis_id")), reps)
