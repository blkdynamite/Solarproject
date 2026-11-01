"""Fleet M&V study: simulate -> SQL layer -> estimate -> evaluate -> report."""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from solar_rooftop.plotting import apply_style

from . import estimators as est_mod
from . import evaluation as ev
from . import ml, plotting, report
from . import warehouse as wh
from .config import StudyConfig
from .simulate import Fleet, simulate_fleet

log = logging.getLogger(__name__)


@dataclass
class Analysis:
    """Everything one simulated program produces, before reporting."""

    fleet: Fleet
    con: duckdb.DuckDBPyConnection
    tests: list[wh.TestResult]
    model_report: ml.ModelReport
    device_values: pd.DataFrame
    analysis_days: pd.DataFrame
    estimates: est_mod.Estimates
    results: pd.DataFrame
    summary: pd.DataFrame


def analyze(cfg: StudyConfig, database: str | Path, raw_dir: Path) -> Analysis:
    """Simulate -> SQL layer -> ML -> estimates -> evaluation for one fleet."""
    f, mv = cfg.fleet, cfg.mv
    window = (f.event_start_hour, f.event_start_hour + f.event_hours)

    log.info("Simulating %s-home fleet (seed %d)", f"{f.n_homes:,}", f.seed)
    fleet = simulate_fleet(cfg.site, f)

    log.info("Building SQL layer and running data tests")
    con = wh.connect(database)
    wh.load_raw(con, fleet, raw_dir)
    wh.build(con, wh.sql_variables(cfg))
    tests = wh.run_tests(con)

    log.info("Training ML counterfactual model")
    model, model_report = ml.train(con, window_hours=window, seed=f.seed)
    pred = ml.predict_analysis_days(con, model)
    in_window = pred["hod"].between(window[0], window[1] - 1)
    pred_window = pred[in_window].groupby(["analysis_id", "device_id"])["pred_kw"].mean()

    log.info("Estimating delivered kW with %d bootstrap replicates", mv.n_bootstrap)
    event_device = wh.query(con, "SELECT * FROM mart_event_device")
    analysis_days = wh.query(con, "SELECT * FROM int_analysis_days ORDER BY day")
    values = est_mod.device_values(event_device, pred_window)
    estimates = est_mod.estimate(values, mv.n_bootstrap, seed=f.seed)

    truth = ev.true_impact_by_day(fleet.true_impact, values, analysis_days)
    results = ev.results_table(estimates, truth, analysis_days, mv.confidence)
    summary = ev.method_summary(results, estimates, mv.confidence)
    return Analysis(
        fleet, con, tests, model_report, values, analysis_days, estimates, results, summary
    )


def validate(cfg: StudyConfig, n_runs: int) -> pd.DataFrame:
    """Repeat the whole study on independent fleets (new homes, assignment and noise).

    Within one fleet every event reuses the same homes, so per-event errors are
    correlated and say little about calibration. Independent fleets do.
    """
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for r in range(n_runs):
            run_cfg = replace(cfg, fleet=replace(cfg.fleet, seed=cfg.fleet.seed + 1000 + r))
            a = analyze(run_cfg, ":memory:", Path(tmp) / f"raw_{r}")
            a.con.close()
            for key, row in a.summary.iterrows():
                rows.append(
                    {
                        "run": r,
                        "method": key,
                        "season_error_pct": row["season_error_pct"],
                        "season_ci_covered": row["season_ci_low"]
                        <= row["true_season_kw"]
                        <= row["season_ci_high"],
                        "season_ci_halfwidth_pct": 100
                        * (row["season_ci_high"] - row["season_ci_low"])
                        / 2
                        / row["true_season_kw"],
                        "placebo_mean_kw": row["placebo_mean_kw"],
                    }
                )
    return pd.DataFrame(rows)


def validation_summary(runs: pd.DataFrame) -> pd.DataFrame:
    g = runs.groupby("method")
    out = pd.DataFrame(
        {
            "mean_error_pct": g["season_error_pct"].mean(),
            "error_sd_pct": g["season_error_pct"].std(),
            "rmse_pct": g["season_error_pct"].apply(lambda e: float(np.sqrt((e**2).mean()))),
            "ci_coverage": g["season_ci_covered"].mean(),
            "ci_halfwidth_pct": g["season_ci_halfwidth_pct"].mean(),
            "placebo_mean_kw": g["placebo_mean_kw"].mean(),
        }
    )
    out.attrs["n_runs"] = int(runs["run"].nunique())
    return out.reindex([m.key for m in est_mod.METHODS])


def run(
    cfg: StudyConfig, out_dir: Path, make_plots: bool = True, validate_runs: int = 0
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mv = cfg.mv
    a = analyze(cfg, out_dir / "warehouse.duckdb", out_dir / "raw")
    con, fleet, values, results, summary = a.con, a.fleet, a.device_values, a.results, a.summary
    analysis_days, tests, model_report = a.analysis_days, a.tests, a.model_report
    dq = wh.data_quality_summary(con)
    heat_slope = ev.bias_temperature_slope(results)
    dollars = ev.dollar_impact(summary, mv)

    # Holdout sizing uses only quantities observable in practice (no ground truth).
    ctl_events = values[(values["grp"] == "control") & (values["kind"] == "event")]
    sigma_raw = float(ctl_events.groupby("analysis_id")["actual_kw"].std().mean())
    sigma_adj = float(ctl_events.groupby("analysis_id")["v_ml"].std().mean())
    impact_est = float(summary.loc["ml_control", "season_kw"])
    tradeoff = ev.holdout_tradeoff(sigma_raw, sigma_adj, impact_est, mv)
    precision_gain = float(
        summary.loc["rct", "event_ci_halfwidth_kw"]
        / summary.loc["ml_control", "event_ci_halfwidth_kw"]
    )

    # Pre-cooling evidence: adjustment factors of dispatched vs control homes on event days.
    adj_factor = (
        values[values["kind"] == "event"].groupby("grp")["adjustment_factor"].median().to_dict()
    )

    results.round(4).to_csv(out_dir / "estimates_by_event.csv", index=False)
    summary.round(4).to_csv(out_dir / "method_summary.csv")
    tradeoff.round(3).to_csv(out_dir / "holdout_tradeoff.csv", index=False)

    hottest = analysis_days[analysis_days["kind"] == "event"].sort_values("tmax").iloc[-1]
    profile = _event_profile(con, fleet, int(hottest["analysis_id"]), mv.adjustment_cap)
    if make_plots:
        log.info("Rendering charts")
        apply_style()
        day_label = pd.Timestamp(hottest["day"]).strftime("%-d %b, %Y")
        plotting.event_day_profile(profile, day_label, cfg, out_dir / "1_event_day_profile.png")
        plotting.event_errors(results, out_dir / "2_method_errors.png")
        plotting.bias_vs_heat(results, out_dir / "3_bias_vs_heat.png")
        plotting.precision(results, precision_gain, out_dir / "4_precision.png")
        plotting.holdout_tradeoff(tradeoff, mv.program_devices, out_dir / "5_holdout_tradeoff.png")

    out: dict[str, Any] = {
        "config": cfg.to_dict(),
        "data_quality": {**dq, "tests": {t.name: t.failing_rows for t in tests}},
        "model": model_report.__dict__,
        "true_season_kw_per_device": round(float(summary["true_season_kw"].iloc[0]), 4),
        "methods": json.loads(summary.round(4).to_json(orient="index")),
        "error_kw_per_degc": heat_slope.round(4).to_dict(),
        "median_adjustment_factor": {k: round(float(v), 3) for k, v in adj_factor.items()},
        "program_value": {
            "devices": mv.program_devices,
            "true_mw": round(dollars.attrs["true_mw"], 2),
            "true_value_usd": round(dollars.attrs["true_value_usd"]),
            "by_method": json.loads(dollars.round(0).to_json(orient="index")),
        },
        "holdout": {
            "sigma_raw_kw": round(sigma_raw, 4),
            "sigma_ml_adjusted_kw": round(sigma_adj, 4),
            "precision_gain": round(precision_gain, 2),
            "tradeoff": json.loads(tradeoff.round(3).to_json(orient="records")),
        },
        "events": analysis_days.assign(day=analysis_days["day"].astype(str)).to_dict("records"),
    }
    if validate_runs:
        log.info("Validating across %d independent fleets", validate_runs)
        runs = validate(cfg, validate_runs)
        runs.round(4).to_csv(out_dir / "validation_runs.csv", index=False)
        vs = validation_summary(runs)
        out["validation"] = {
            "n_runs": vs.attrs["n_runs"],
            "methods": json.loads(vs.round(4).to_json(orient="index")),
        }
    with (out_dir / "summary.json").open("w") as fh:
        json.dump(out, fh, indent=2, default=str)
    (out_dir / "REPORT.md").write_text(report.render(out, cfg))
    con.close()
    return out


def _event_profile(con: Any, fleet: Fleet, analysis_id: int, cap: float) -> pd.DataFrame:
    """Hourly fleet averages on one event day for the eligible devices."""
    prof = wh.query(
        con,
        f"""
        SELECT h.hod,
            avg(h.actual_kw) FILTER (d.grp = 'treatment') AS treated_kw,
            avg(h.actual_kw) FILTER (d.grp = 'control') AS control_kw,
            avg(h.baseline_kw) FILTER (d.grp = 'treatment') AS baseline_kw,
            avg(h.baseline_kw * greatest({1 - cap}, least({1 + cap}, d.adjustment_factor)))
                FILTER (d.grp = 'treatment') AS adj_baseline_kw
        FROM mart_baseline_hourly h
        JOIN mart_event_device d USING (analysis_id, device_id)
        WHERE h.analysis_id = {analysis_id} AND d.eligible
        GROUP BY h.hod ORDER BY h.hod
        """,
    ).set_index("hod")
    ids = wh.query(
        con,
        f"SELECT device_id FROM mart_event_device WHERE analysis_id = {analysis_id} "
        "AND eligible AND grp = 'treatment'",
    )["device_id"].to_numpy()
    day = wh.query(con, f"SELECT day FROM int_analysis_days WHERE analysis_id = {analysis_id}")
    cols = np.flatnonzero(fleet.hours.normalize() == pd.Timestamp(day["day"].iloc[0]))
    cf = fleet.counterfactual_kw[np.ix_(ids - 1, cols)].mean(axis=0)
    prof["counterfactual_kw"] = pd.Series(cf, index=fleet.hours[cols].hour)
    return prof
