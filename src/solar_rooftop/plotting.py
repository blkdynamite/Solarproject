"""Report charts (static PNGs for the README).

Colours follow a validated, colour-blind-safe categorical order; status colours
(outage / underperformance) are reserved for alarms and always paired with a
text label, so no chart relies on colour alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter

from .config import ProjectConfig
from .energy import EnergyMetrics
from .finance import FinanceResult
from .monitoring import Status

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (fixed order)
CRITICAL, SERIOUS = "#d03b3b", "#ec835a"
SURFACE, INK, INK_2, MUTED, GRID, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
POS, NEG = "#2a78d6", "#e34948"  # diverging poles


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 140,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 9.5,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.edgecolor": AXIS,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "ytick.left": False,
            "lines.linewidth": 2,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
        }
    )


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _kfmt(ax: Axes, axis: str = "y", prefix: str = "") -> None:
    fmt = FuncFormatter(lambda v, _: f"{prefix}{v:,.0f}k" if v >= 0 else f"-{prefix}{-v:,.0f}k")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


# ---------------------------------------------------------------------------
def monthly_production(sim: pd.DataFrame, cfg: ProjectConfig, m: EnergyMetrics, path: Path) -> Path:
    monthly = sim["ac_kw"].resample("MS").sum() / 1000
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(monthly))
    ax.bar(x, monthly.to_numpy(), width=0.7, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    ax.set_xticks(x, monthly.index.strftime("%b"))
    ax.tick_params(axis="x", length=0)
    for i in (int(monthly.argmax()), int(monthly.argmin())):
        ax.annotate(
            f"{monthly.iloc[i]:.0f}",
            (i, monthly.iloc[i]),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=INK_2,
        )
    ax.set_ylabel("AC energy (MWh)")
    site = cfg.site.name.split(" (")[0]
    ax.set_title(
        f"Monthly production: {cfg.system.dc_kw:.0f} kW-DC rooftop, {site}\n"
        f"P50 {m.annual_kwh / 1000:,.0f} MWh/yr · P90 {m.p90_kwh / 1000:,.0f} MWh/yr · "
        f"{m.specific_yield_kwh_per_kwp:,.0f} kWh/kWp"
    )
    return _save(fig, path)


def design_tradeoffs(
    tilt: pd.DataFrame, ratio: pd.DataFrame, cfg: ProjectConfig, path: Path
) -> Path:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.8))

    a1.plot(tilt.index, tilt["specific_yield"], color=SERIES[0], marker="o", ms=4)
    best = float(tilt["specific_yield"].idxmax())
    design = cfg.system.tilt_deg
    gain = tilt["specific_yield"].max() / tilt.loc[design, "specific_yield"] - 1
    a1.axvline(design, color=MUTED, ls="--", lw=1)
    a1.annotate(
        f"Design {design:.0f}°",
        (design, a1.get_ylim()[0]),
        xytext=(4, 6),
        textcoords="offset points",
        color=INK_2,
        fontsize=8.5,
    )
    a1.annotate(
        f"Optimum ≈{best:.0f}° (+{gain:.1%})",
        (best, tilt["specific_yield"].max()),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        color=INK_2,
        fontsize=8.5,
    )
    a1.set_xlabel("Tilt (°)")
    a1.set_ylabel("Specific yield (kWh/kWp/yr)")
    a1.set_title("Tilt vs. yield")

    a2.plot(ratio.index, ratio["clipping_loss_pct"], color=SERIES[0], marker="o", ms=4)
    a2.axvline(cfg.system.dc_ac_ratio, color=MUTED, ls="--", lw=1)
    a2.annotate(
        f"Design {cfg.system.dc_ac_ratio:.2f}",
        (cfg.system.dc_ac_ratio, 0),
        xytext=(4, 6),
        textcoords="offset points",
        color=INK_2,
        fontsize=8.5,
    )
    a2.set_xlabel("DC/AC ratio")
    a2.set_ylabel("Energy lost to clipping (%)")
    a2.set_title("Inverter loading vs. clipping")
    a2.set_ylim(bottom=0)
    return _save(fig, path)


def daily_profile(sim: pd.DataFrame, cfg: ProjectConfig, path: Path) -> Path:
    ac = sim["ac_kw"]
    prof = ac.groupby([ac.index.month, ac.index.hour]).mean()
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    hours = np.arange(24)
    for (mth, name), col in zip([(6, "June"), (3, "March"), (12, "December")], SERIES, strict=True):
        y = prof.loc[mth].reindex(hours, fill_value=0).to_numpy()
        ax.plot(hours, y, color=col, label=name)
        pk = int(y.argmax())
        ax.annotate(
            name,
            (pk, y[pk]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            color=INK_2,
            fontsize=8.5,
        )
    ax.axhline(cfg.system.ac_kw, ls=":", color=MUTED, lw=1.2)
    ax.annotate(
        f"Inverter AC limit ({cfg.system.ac_kw:.0f} kW)",
        (0.5, cfg.system.ac_kw),
        xytext=(0, -12),
        textcoords="offset points",
        color=INK_2,
        fontsize=8.5,
    )
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlim(0, 23)
    ax.set_ylim(0, cfg.system.ac_kw * 1.08)
    ax.set_xlabel("Hour of day (local standard time)")
    ax.set_ylabel("Average AC output (kW)")
    ax.set_title("Average daily production profile")
    ax.legend(loc="center right")
    return _save(fig, path)


def cash_flow(results: list[FinanceResult], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    for res, col in zip(results, SERIES, strict=False):
        cum = res.table["cumulative"] / 1000
        ax.plot(cum.index, cum.to_numpy(), color=col, label=res.scenario.label)
        if res.payback_yrs is not None:
            ax.plot(res.payback_yrs, 0, "o", ms=8, color=col, mec=SURFACE, mew=2, zorder=3)
    ax.axhline(0, color=AXIS, lw=1)
    _kfmt(ax, prefix="$")
    ax.set_xlabel("Project year")
    ax.set_ylabel("Cumulative cash flow (nominal)")
    ax.set_title("Cumulative cash flow by incentive / tax scenario (dots = payback)")
    ax.legend(loc="upper left")
    return _save(fig, path)


def tornado(sens: pd.DataFrame, scenario_label: str, path: Path) -> Path:
    base = float(sens.attrs["base_npv"]) / 1000
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(sens))
    for i, row in sens.iterrows():
        for side in ("low", "high"):
            v = row[f"npv_{side}"] / 1000
            ax.barh(
                i,
                v - base,
                left=base,
                height=0.62,
                color=POS if v >= base else NEG,
                edgecolor=SURFACE,
                linewidth=1,
            )
            if abs(v - base) > 1e-6:
                ax.annotate(
                    row[f"{side}_label"],
                    (v, i),
                    xytext=(4 if v >= base else -4, 0),
                    textcoords="offset points",
                    va="center",
                    ha="left" if v >= base else "right",
                    color=INK_2,
                    fontsize=8,
                )
    ax.axvline(base, color=INK_2, lw=1)
    ax.set_yticks(y, sens["driver"])
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    _kfmt(ax, axis="x", prefix="$")
    ax.set_xlabel("Project NPV")
    ax.set_title(f"NPV sensitivity ({scenario_label}; base ${base:,.0f}k)")
    ax.margins(x=0.12)
    return _save(fig, path)


def monitoring(daily: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    st = daily["status"]
    pi = daily["normalised_pi"]

    ok = st == Status.OK.value
    ax.scatter(daily.index[ok], pi[ok], s=9, color=MUTED, alpha=0.7, lw=0, label="Normal day")
    for status, col, label in [
        (Status.INVERTER_OUTAGE, CRITICAL, "Inverter outage"),
        (Status.MAJOR_OUTAGE, CRITICAL, "Major outage"),
        (Status.UNDERPERFORMANCE, SERIOUS, "Underperformance (soiling?)"),
    ]:
        m = st == status.value
        if m.any():
            ax.scatter(
                daily.index[m],
                pi[m],
                s=22,
                color=col,
                edgecolor=SURFACE,
                lw=0.8,
                label=label,
                zorder=3,
            )

    gaps = daily.index[st.isin([Status.NO_DATA.value, Status.PARTIAL_DATA.value])]
    for dt in gaps:
        ax.axvspan(dt, dt + pd.Timedelta(1, unit="D"), color=MUTED, alpha=0.25, lw=0)
    if len(gaps):
        ax.plot([], [], color=MUTED, alpha=0.4, lw=6, label="No meter data")
        ax.annotate(
            "Comms dropout",
            (gaps[0], 1.07),
            xytext=(-4, 0),
            textcoords="offset points",
            ha="right",
            color=INK_2,
            fontsize=8.5,
        )

    ax.plot(
        daily.index,
        daily["trend_pi"].ffill(limit=3),
        color=SERIES[0],
        lw=1.4,
        label="7-day trailing median",
    )
    ax.axhline(1.0, color=AXIS, lw=1)
    ax.set_ylim(0.65, 1.1)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_ylabel("Actual / expected (vs. baseline)")
    ax.set_title("Daily performance index with automatically flagged issues")
    ax.legend(ncol=3, loc="lower left")
    return _save(fig, path)
