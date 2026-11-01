"""End-to-end analysis: energy -> financials -> monitoring -> report artefacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__, energy, finance, plotting
from . import monitoring as mon
from .config import ProjectConfig
from .weather import load_weather

log = logging.getLogger(__name__)

TILT_SWEEP = [float(t) for t in range(0, 46, 5)]
RATIO_SWEEP = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]


def run(cfg: ProjectConfig, out_dir: Path, make_plots: bool = True) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sysc, fin, monc = cfg.system, cfg.finance, cfg.monitoring

    # 1. Energy -----------------------------------------------------------
    log.info("Loading weather and running hourly simulation")
    weather = load_weather(cfg.site)
    sim = energy.simulate(weather, sysc)
    metrics = energy.energy_metrics(sim, sysc, fin.yield_uncertainty)
    tilt = energy.parameter_sweep(weather, sysc, "tilt_deg", sorted({*TILT_SWEEP, sysc.tilt_deg}))
    ratio = energy.parameter_sweep(
        weather, sysc, "dc_ac_ratio", sorted({*RATIO_SWEEP, sysc.dc_ac_ratio})
    )
    log.info("Annual production %.1f MWh", metrics.annual_kwh / 1000)

    # 2. Financials ---------------------------------------------------------
    scenarios = finance.default_scenarios(fin)
    results = [finance.evaluate(metrics.annual_kwh, sysc, fin, s) for s in scenarios]
    base_scenario = scenarios[-1]
    sens = finance.sensitivity_table(metrics.annual_kwh, sysc, fin, base_scenario)
    p90 = finance.evaluate(metrics.p90_kwh, sysc, fin, base_scenario)
    pd.concat({r.scenario.key: r.table for r in results}, names=["scenario"]).round(2).to_csv(
        out_dir / "cash_flows.csv"
    )
    sens.round(0).to_csv(out_dir / "sensitivity.csv", index=False)

    # 3. Monitoring ---------------------------------------------------------
    log.info("Simulating meter data and running fault detection")
    faults = mon.default_faults(cfg.site.model_year)
    measured = mon.synthesize_meter(
        sim["ac_kw"], sysc, monc, faults, np.random.default_rng(monc.seed)
    )
    daily = mon.detect(sim["ac_kw"], measured, sysc, monc)
    events = mon.group_events(daily)
    scores = mon.score(daily, faults)
    daily.round(4).to_csv(out_dir / "daily_performance.csv")
    lost_kwh = float(daily["lost_kwh"].sum())

    # 4. Report -------------------------------------------------------------
    if make_plots:
        log.info("Rendering charts")
        plotting.apply_style()
        plotting.monthly_production(sim, cfg, metrics, out_dir / "1_monthly_production.png")
        plotting.design_tradeoffs(tilt, ratio, cfg, out_dir / "2_design_tradeoffs.png")
        plotting.daily_profile(sim, cfg, out_dir / "3_daily_profile.png")
        plotting.cash_flow(results, out_dir / "4_cash_flow.png")
        plotting.tornado(sens, base_scenario.label, out_dir / "5_npv_sensitivity.png")
        plotting.monitoring(daily, out_dir / "6_performance_monitoring.png")

    best_tilt = float(tilt["specific_yield"].idxmax())
    summary: dict[str, Any] = {
        "version": __version__,
        "config": cfg.to_dict(),
        "energy": {
            "annual_mwh_p50": round(metrics.annual_kwh / 1000, 1),
            "annual_mwh_p90": round(metrics.p90_kwh / 1000, 1),
            "specific_yield_kwh_per_kwp": round(metrics.specific_yield_kwh_per_kwp),
            "capacity_factor_pct": round(metrics.capacity_factor * 100, 1),
            "performance_ratio_pct": round(metrics.performance_ratio * 100, 1),
            "poa_insolation_kwh_m2": round(metrics.poa_insolation_kwh_m2),
            "clipping_loss_pct": round(metrics.clipping_loss_pct, 2),
            "clipping_hours": metrics.clipping_hours,
            "best_tilt_deg": best_tilt,
            "yield_gain_at_best_tilt_pct": round(
                100
                * (tilt["specific_yield"].max() / tilt.loc[sysc.tilt_deg, "specific_yield"] - 1),
                1,
            ),
            "clipping_loss_pct_by_dc_ac_ratio": ratio["clipping_loss_pct"].round(2).to_dict(),
        },
        "finance": {
            **{r.scenario.key: r.summary() for r in results},
            "p90_case_npv_usd": round(p90.npv),
            "sensitivity_npv_usd": {
                row.driver: {row.low_label: round(row.npv_low), row.high_label: round(row.npv_high)}
                for row in sens.itertuples()
            },
        },
        "monitoring": {
            "events": events,
            "detection_score": scores.to_dict(),
            "total_lost_kwh": round(lost_kwh),
            "lost_value_usd": round(lost_kwh * fin.tariff_per_kwh),
        },
    }
    with (out_dir / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return summary


def evaluate_detector(cfg: ProjectConfig, n_seeds: int) -> pd.DataFrame:
    """Monte Carlo check of the detector over many random noise realisations."""
    weather = load_weather(cfg.site)
    expected = energy.simulate(weather, cfg.system)["ac_kw"]
    faults = mon.default_faults(cfg.site.model_year)
    rows = []
    for seed in range(n_seeds):
        measured = mon.synthesize_meter(
            expected, cfg.system, cfg.monitoring, faults, np.random.default_rng(seed)
        )
        daily = mon.detect(expected, measured, cfg.system, cfg.monitoring)
        sc = mon.score(daily, faults)
        row: dict[str, Any] = {"seed": seed, "false_positive_days": sc.false_positive_days}
        for f, fs in zip(faults, sc.faults, strict=True):
            row[f"{f.kind}_recall"] = fs.recall
            row[f"{f.kind}_latency_days"] = fs.latency_days
            row[f"{f.kind}_misclassified"] = fs.misclassified_days
        rows.append(row)
    return pd.DataFrame(rows).set_index("seed")
