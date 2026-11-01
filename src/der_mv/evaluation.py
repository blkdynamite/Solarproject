"""Score M&V methods against ground truth, placebo days and in dollars."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import MVConfig
from .estimators import METHODS, Estimates


def true_impact_by_day(
    true_impact: pd.DataFrame, values: pd.DataFrame, analysis_days: pd.DataFrame
) -> pd.Series:
    """Average true kW reduction over the *eligible* treated devices, per analysis day.

    Placebo days have a true impact of exactly zero.
    """
    per_device = (
        true_impact.groupby(["event_id", "device_id"])["impact_kw"].mean().rename("impact_kw")
    )
    treated = values[values["grp"] == "treatment"][["analysis_id", "device_id"]]
    joined = treated.merge(
        per_device, left_on=["analysis_id", "device_id"], right_index=True, how="left"
    )
    truth = joined.groupby("analysis_id")["impact_kw"].mean()
    truth = truth.reindex(analysis_days["analysis_id"]).fillna(0.0)  # placebo -> 0
    return truth.rename("truth_kw")


def results_table(
    est: Estimates, truth: pd.Series, analysis_days: pd.DataFrame, confidence: float
) -> pd.DataFrame:
    """Long table: one row per method x analysis day with estimate, CI, truth and error."""
    alpha = (1 - confidence) / 2
    days = analysis_days.set_index("analysis_id")
    rows = []
    for m in METHODS:
        lo, hi = np.nanquantile(est.replicates[m.key], [alpha, 1 - alpha], axis=0)
        for j, aid in enumerate(est.analysis_ids):
            rows.append(
                {
                    "method": m.key,
                    "label": m.label,
                    "analysis_id": int(aid),
                    "kind": days.loc[aid, "kind"],
                    "day": days.loc[aid, "day"],
                    "tmax": float(days.loc[aid, "tmax"]),
                    "estimate_kw": float(est.point.loc[aid, m.key]),
                    "ci_low_kw": float(lo[j]),
                    "ci_high_kw": float(hi[j]),
                    "truth_kw": float(truth.loc[aid]),
                }
            )
    df = pd.DataFrame(rows)
    df["error_kw"] = df["estimate_kw"] - df["truth_kw"]
    df["covered"] = (df["ci_low_kw"] <= df["truth_kw"]) & (df["truth_kw"] <= df["ci_high_kw"])
    return df


def method_summary(results: pd.DataFrame, est: Estimates, confidence: float) -> pd.DataFrame:
    """Accuracy, calibration and season-level estimate for each method."""
    alpha = (1 - confidence) / 2
    ev = results[results["kind"] == "event"]
    pl = results[results["kind"] == "placebo"]
    event_pos = np.isin(est.analysis_ids, ev["analysis_id"].unique())
    true_season = ev.groupby("method")["truth_kw"].mean()

    rows = []
    for m in METHODS:
        e, p = ev[ev["method"] == m.key], pl[pl["method"] == m.key]
        season_reps = est.replicates[m.key][:, event_pos].mean(axis=1)
        lo, hi = np.nanquantile(season_reps, [alpha, 1 - alpha])
        rows.append(
            {
                "method": m.key,
                "label": m.label,
                "season_kw": e["estimate_kw"].mean(),
                "season_ci_low": lo,
                "season_ci_high": hi,
                "true_season_kw": true_season[m.key],
                "season_error_pct": 100 * (e["estimate_kw"].mean() / true_season[m.key] - 1),
                "event_mae_kw": e["error_kw"].abs().mean(),
                "event_error_sd_kw": e["error_kw"].std(),
                "event_worst_error_pct": 100 * (e["error_kw"] / e["truth_kw"]).abs().max(),
                "event_ci_halfwidth_kw": ((e["ci_high_kw"] - e["ci_low_kw"]) / 2).mean(),
                "event_ci_coverage": e["covered"].mean(),
                "placebo_mean_kw": p["estimate_kw"].mean(),
                "placebo_mae_kw": p["error_kw"].abs().mean(),
                "placebo_ci_coverage": p["covered"].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("method")


def bias_temperature_slope(results: pd.DataFrame) -> pd.Series:
    """Event error (kW) per deg C of daily max temperature, by method (OLS slope)."""
    ev = results[results["kind"] == "event"]
    return ev.groupby("method").apply(
        lambda g: float(np.polyfit(g["tmax"], g["error_kw"], 1)[0]), include_groups=False
    )


def holdout_tradeoff(
    sigma_raw: float,
    sigma_adjusted: float,
    impact_kw: float,
    mv: MVConfig,
    shares: np.ndarray | None = None,
) -> pd.DataFrame:
    """Precision of a single event's estimate vs control-group size.

    Half-width of the CI on the treated-minus-control difference:
    z * sigma * sqrt(1/n_T + 1/n_C). The forgone value is the capacity the
    held-out devices could have delivered.
    """
    shares = shares if shares is not None else np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3])
    z = norm.ppf(0.5 + mv.confidence / 2)
    n = mv.program_devices
    n_c, n_t = n * shares, n * (1 - shares)
    se = np.sqrt(1 / n_t + 1 / n_c)
    return pd.DataFrame(
        {
            "holdout_share": shares,
            "control_devices": n_c.round().astype(int),
            "ci_halfwidth_pct_raw": 100 * z * sigma_raw * se / impact_kw,
            "ci_halfwidth_pct_ml_adjusted": 100 * z * sigma_adjusted * se / impact_kw,
            "forgone_mw": n_c * impact_kw / 1000,
            "forgone_value_usd": n_c * impact_kw * mv.capacity_value_per_kw_yr,
        }
    )


def dollar_impact(summary: pd.DataFrame, mv: MVConfig) -> pd.DataFrame:
    """Annual capacity value credited under each method vs the truth, program scale."""
    true_kw = float(summary["true_season_kw"].iloc[0])
    scale = mv.program_devices * mv.capacity_value_per_kw_yr
    out = pd.DataFrame(
        {
            "credited_mw": summary["season_kw"] * mv.program_devices / 1000,
            "credited_value_usd": summary["season_kw"] * scale,
        }
    )
    out["misstatement_usd"] = out["credited_value_usd"] - true_kw * scale
    out.attrs["true_mw"] = true_kw * mv.program_devices / 1000
    out.attrs["true_value_usd"] = true_kw * scale
    return out
