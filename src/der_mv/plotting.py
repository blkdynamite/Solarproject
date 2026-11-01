"""Charts for the fleet M&V report (same visual system as the solar charts)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

from solar_rooftop.plotting import AXIS, INK, INK_2, MUTED, SURFACE

from .config import StudyConfig
from .estimators import METHODS

# Categorical slots 1-5 in fixed order (validated palette); one colour per method.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
METHOD_COLOR = {m.key: SLOTS[i] for i, m in enumerate(METHODS)}


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _shade_event(ax: plt.Axes, cfg: StudyConfig) -> None:
    f = cfg.fleet
    s, e = f.event_start_hour, f.event_start_hour + f.event_hours
    ax.axvspan(s, e, color=MUTED, alpha=0.15, lw=0)
    ax.axvspan(s - f.precool_hours, s, color=MUTED, alpha=0.07, lw=0)
    top = ax.get_ylim()[1]
    ax.annotate(
        "Event",
        ((s + e) / 2, top),
        xytext=(0, -12),
        textcoords="offset points",
        ha="center",
        color=INK_2,
        fontsize=8.5,
    )
    ax.annotate(
        "Pre-cool",
        (s - f.precool_hours / 2, top),
        xytext=(0, -24),
        textcoords="offset points",
        ha="center",
        color=INK_2,
        fontsize=8,
    )


def event_day_profile(profile: pd.DataFrame, title_day: str, cfg: StudyConfig, path: Path) -> Path:
    """Fleet-average load on one event day: what happened vs what each method assumes."""
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = profile.index.to_numpy()
    ax.plot(
        x,
        profile["counterfactual_kw"],
        color=INK,
        ls="--",
        lw=1.6,
        label="True counterfactual (unknowable in practice)",
    )
    ax.plot(x, profile["treated_kw"], color=SLOTS[0], label="Dispatched homes (actual)")
    ax.plot(x, profile["control_kw"], color=SLOTS[1], label="Control group (actual)")
    ax.plot(x, profile["baseline_kw"], color=SLOTS[2], label="10-of-10 baseline")
    ax.plot(x, profile["adj_baseline_kw"], color=SLOTS[3], label="10-of-10 + day-of adjustment")
    ax.set_xlim(8, 22)
    ax.set_ylim(0, profile.loc[8:22].to_numpy().max() * 1.15)
    _shade_event(ax, cfg)
    ax.set_xticks(range(8, 23, 2))
    ax.set_xlabel("Hour of day (local standard time)")
    ax.set_ylabel("Average load per home (kW)")
    ax.set_title(f"Hottest event day ({title_day}): the baseline misses the heat")
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, path)


def event_errors(results: pd.DataFrame, path: Path) -> Path:
    """Error of every method on every event day (% of truth) and placebo day (kW)."""
    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(10, 3.8), sharey=True, gridspec_kw={"width_ratios": [1.4, 1]}
    )
    labels = [m.label for m in METHODS]
    y = np.arange(len(METHODS))[::-1]
    for yi, m in zip(y, METHODS, strict=True):
        ev = results[(results["method"] == m.key) & (results["kind"] == "event")]
        pct = (100 * ev["error_kw"] / ev["truth_kw"]).clip(-140, 140)
        a1.scatter(
            pct,
            np.full(len(pct), yi),
            s=26,
            color=METHOD_COLOR[m.key],
            alpha=0.75,
            edgecolor=SURFACE,
            lw=0.8,
            zorder=3,
        )
        a1.plot([pct.mean()] * 2, [yi - 0.3, yi + 0.3], color=INK, lw=2)
        pl = results[(results["method"] == m.key) & (results["kind"] == "placebo")]
        a2.scatter(
            pl["estimate_kw"],
            np.full(len(pl), yi),
            s=26,
            color=METHOD_COLOR[m.key],
            alpha=0.75,
            edgecolor=SURFACE,
            lw=0.8,
            zorder=3,
        )
        a2.plot([pl["estimate_kw"].mean()] * 2, [yi - 0.3, yi + 0.3], color=INK, lw=2)
    for ax in (a1, a2):
        ax.axvline(0, color=AXIS, lw=1)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
    a1.set_yticks(y, labels)
    a1.tick_params(axis="y", length=0, labelcolor=INK)
    a1.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    a1.set_xlabel("Error vs. true delivered kW (each dot = one event; bar = mean)")
    a1.set_title("Event days: accuracy")
    a2.set_xlabel("Estimated kW on days with no event (truth = 0)")
    a2.set_title("Placebo days: bias check")
    return _save(fig, path)


def bias_vs_heat(results: pd.DataFrame, path: Path) -> Path:
    keys = ["ten_of_ten_adj", "ml", "ml_control"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for k in keys:
        ev = results[(results["method"] == k) & (results["kind"] == "event")]
        label = ev["label"].iloc[0]
        ax.scatter(
            ev["tmax"],
            ev["error_kw"],
            s=30,
            color=METHOD_COLOR[k],
            label=label,
            edgecolor=SURFACE,
            lw=0.8,
            zorder=3,
        )
        b, a = np.polyfit(ev["tmax"], ev["error_kw"], 1)
        xs = np.linspace(ev["tmax"].min(), ev["tmax"].max(), 10)
        ax.plot(xs, a + b * xs, color=METHOD_COLOR[k], lw=1.4)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xlabel("Event-day maximum temperature (°C)")
    ax.set_ylabel("Estimate − truth (kW per home)")
    ax.set_title("Baseline errors grow on the hottest days, when capacity is worth most")
    ax.legend(loc="lower left")
    return _save(fig, path)


def precision(results: pd.DataFrame, precision_gain: float, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4))
    ev = results[results["kind"] == "event"]
    days = sorted(ev["day"].unique())
    x = np.arange(len(days))
    for k, off in (("rct", -0.17), ("ml_control", 0.17)):
        d = ev[ev["method"] == k].set_index("day").loc[days]
        ax.errorbar(
            x + off,
            d["estimate_kw"],
            yerr=[d["estimate_kw"] - d["ci_low_kw"], d["ci_high_kw"] - d["estimate_kw"]],
            fmt="o",
            ms=5,
            color=METHOD_COLOR[k],
            elinewidth=2,
            capsize=0,
            label=f"{d['label'].iloc[0]} (90 % CI)",
        )
    truth = ev[ev["method"] == "rct"].set_index("day").loc[days, "truth_kw"]
    ax.scatter(x, truth, marker="_", s=260, color=INK, lw=2, label="True delivered kW", zorder=4)
    ax.set_xticks(x, [pd.Timestamp(d).strftime("%-d %b") for d in days], rotation=45, ha="right")
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("kW reduction per dispatched home")
    ax.set_title(
        f"Same randomised design, {precision_gain:.1f}× tighter: adding ML to the control group"
    )
    ax.legend(loc="lower left", fontsize=8)
    return _save(fig, path)


def holdout_tradeoff(tradeoff: pd.DataFrame, n_devices: int, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = 100 * tradeoff["holdout_share"]
    ax.plot(
        x,
        tradeoff["ci_halfwidth_pct_raw"],
        color=METHOD_COLOR["rct"],
        marker="o",
        ms=4,
        label="Control group (difference in means)",
    )
    ax.plot(
        x,
        tradeoff["ci_halfwidth_pct_ml_adjusted"],
        color=METHOD_COLOR["ml_control"],
        marker="o",
        ms=4,
        label="ML + control group",
    )
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} %"))
    ax.set_xticks(x)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_ylim(bottom=0)
    ax.set_xlabel(f"Share of a {n_devices:,}-device fleet held out as control")
    ax.set_ylabel("± error on one event (90 % CI)")
    ax.set_title("Smaller holdout for the same precision with ML adjustment")
    ax.legend(loc="upper right")
    return _save(fig, path)
