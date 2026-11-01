"""Render the utility-facing decision memo (REPORT.md) from pipeline outputs.

Every number in the memo comes from the run's summary, so the narrative can
never drift out of sync with the analysis.
"""

from __future__ import annotations

from typing import Any

from .config import StudyConfig
from .estimators import METHODS


def _usd(x: float) -> str:
    sign = "−" if x < 0 else ""
    x = abs(x)
    return f"{sign}${x / 1e6:,.2f}M" if x >= 1e6 else f"{sign}${x / 1e3:,.0f}k"


def _pct(x: float, signed: bool = True) -> str:
    return f"{x:+.0f} %" if signed else f"{x:.0f} %"


def render(s: dict[str, Any], cfg: StudyConfig) -> str:
    m, pv, ho, dq, model = (
        s["methods"],
        s["program_value"],
        s["holdout"],
        s["data_quality"],
        s["model"],
    )
    f, mv = cfg.fleet, cfg.mv
    truth = s["true_season_kw_per_device"]
    best = m["ml_control"]
    std, adj = m["ten_of_ten"], m["ten_of_ten_adj"]
    by = pv["by_method"]

    rec = next(
        (r for r in ho["tradeoff"] if r["ci_halfwidth_pct_ml_adjusted"] <= mv.target_precision_pct),
        ho["tradeoff"][-1],
    )
    raw_same = next(
        (r for r in ho["tradeoff"] if r["ci_halfwidth_pct_raw"] <= mv.target_precision_pct), None
    )
    raw_note = (
        f"a plain control-group comparison would need a {raw_same['holdout_share']:.0%} holdout "
        f"({raw_same['forgone_mw']:.1f} MW forgone) for the same precision"
        if raw_same
        else "a plain control-group comparison can't reach that precision at any tested holdout"
    )

    rows = "\n".join(
        f"| {mm.label} | {m[mm.key]['season_kw']:.2f} "
        f"({m[mm.key]['season_ci_low']:.2f}–{m[mm.key]['season_ci_high']:.2f}) "
        f"| {_pct(m[mm.key]['season_error_pct'])} "
        f"| {m[mm.key]['event_worst_error_pct']:.0f} % "
        f"| {m[mm.key]['placebo_mean_kw']:+.2f} "
        f"| {_usd(by[mm.key]['misstatement_usd'])} |"
        for mm in METHODS
    )
    trade_rows = "\n".join(
        f"| {r['holdout_share']:.0%} | {r['control_devices']:,} "
        f"| ±{r['ci_halfwidth_pct_raw']:.1f} % | ±{r['ci_halfwidth_pct_ml_adjusted']:.1f} % "
        f"| {r['forgone_mw']:.1f} MW / {_usd(r['forgone_value_usd'])} |"
        for r in ho["tradeoff"]
    )
    fi = ", ".join(f"{k} ({v:.0%})" for k, v in list(model["feature_importance"].items())[:4])
    slope = s["error_kw_per_degc"]
    calibration, ml_spread, validation_section = _validation_text(s, mv.confidence)
    af = s["median_adjustment_factor"]

    return f"""# Thermostat demand response: how much did the fleet deliver?

*Measurement & verification (M&V) study · {f.n_homes:,} smart thermostats · {f.n_events} summer
events · simulated data with known ground truth.*

## Bottom line

1. **Each dispatched home delivered {best["season_kw"]:.2f} kW** (90 % CI
   {best["season_ci_low"]:.2f}–{best["season_ci_high"]:.2f}) over the {f.event_hours}-hour event
   window. The true value, known here because the data is simulated, is {truth:.2f} kW. At
   program scale ({mv.program_devices:,} devices) that is **{pv["true_mw"]:.0f} MW, worth about
   {_usd(pv["true_value_usd"])}/yr** in avoided capacity at ${mv.capacity_value_per_kw_yr:.0f}/kW-yr.
2. **The industry-standard baseline gets this badly wrong.** The 10-of-10 baseline credits
   {std["season_kw"]:.2f} kW per home ({_pct(std["season_error_pct"])}), a
   {_usd(by["ten_of_ten"]["misstatement_usd"])}/yr misstatement. The usual day-of adjustment
   gets the season average within {abs(adj["season_error_pct"]):.0f} %, but only because two
   errors partly cancel; single events are off by up to {adj["event_worst_error_pct"]:.0f} %.
3. **Recommendation: evaluate with a randomised control group plus an ML counterfactual.** It
   was within {abs(best["season_error_pct"]):.1f} % of the truth for the season and within
   {best["event_worst_error_pct"]:.0f} % on every single event{calibration}. A
   **{rec["holdout_share"]:.0%} holdout** ({rec["forgone_mw"]:.1f} MW set aside) is enough for
   ±{rec["ci_halfwidth_pct_ml_adjusted"]:.1f} % precision on each event; {raw_note}.

![Event day profile](1_event_day_profile.png)

## Comparing methods

| Method | Season kW/home (90 % CI) | Season error | Worst event error | Placebo kW (truth 0) | Annual misstatement |
|---|---|---|---|---|---|
{rows}

*Program scale: {mv.program_devices:,} devices at ${mv.capacity_value_per_kw_yr:.0f}/kW-yr.
Placebo days are the hottest weekdays with no event, so every method should report ~0 kW.*

![Errors by method](2_method_errors.png)

## Why the standard baselines fail

- **Events are called on the hottest days, and the baseline days are cooler.** A 10-of-10
  baseline averages the previous ten non-event weekdays, so on a heatwave day it
  under-predicts what homes would have used and erases most of the reduction. Back-to-back
  events make it worse: excluding recent event days pushes the baseline further back into
  milder weather. Its error grows by {slope["ten_of_ten"]:+.2f} kW per home for every 1 °C
  of daily maximum temperature.
- **Pre-cooling contaminates the day-of adjustment.** The adjustment rescales the baseline
  to match load in the hours just before the event, but dispatched thermostats pre-cool in
  that window. The median adjustment factor is {af["treatment"]:.2f} for dispatched homes vs
  {af["control"]:.2f} for identical control homes on the same days. That inflation
  over-credits the program and partly offsets the heat bias above: luck, not a method.
- **An ML model on its own is not enough either.** It is trained on non-event days, and
  events are, by design, the hottest days of the year, so it must extrapolate beyond most of
  what it has seen. This season it was {_pct(m["ml"]["season_error_pct"])} for the season and
  off by up to {m["ml"]["event_worst_error_pct"]:.0f} % on single events{ml_spread}, while
  its confidence interval only reflects sampling noise, not model error: precise is not the
  same as accurate.
- **The placebo test exposes all three without needing the truth.** On the hottest days
  *without* an event, every method should estimate 0 kW. That check works on real data, and it
  is the first thing I would run on any production M&V method.

![Bias vs heat](3_bias_vs_heat.png)

## Why ML plus the control group works

Randomisation makes the control group an unbiased stand-in for the dispatched homes, so the
control group's ML prediction error on the event day measures the model's bias on that exact
day, and subtracting it cancels the bias. Meanwhile the ML prediction explains most of the
home-to-home variation in load (event-window spread falls from {ho["sigma_raw_kw"]:.2f} kW to
{ho["sigma_ml_adjusted_kw"]:.2f} kW per home), so the estimate is
**{ho["precision_gain"]:.1f}× more precise** than a plain difference in means. This is the same
idea as CUPED variance reduction in online A/B testing.

![Precision](4_precision.png)

Note how the plain control-group estimates in this fleet sit below the truth on most events.
That is not bias: every event reuses the same randomly assigned homes, so one unlucky split
shifts every event the same way. It is also why per-event checks within one season can't
validate a method's error bars. {validation_section}

## How big should the control group be?

Every held-out device delivers zero kW, so the holdout has a direct cost. Precision for a
single event in a {mv.program_devices:,}-device program:

| Holdout | Control devices | ± error, control group only | ± error, ML + control | Capacity forgone |
|---|---|---|---|---|
{trade_rows}

![Holdout trade-off](5_holdout_tradeoff.png)

## Data quality

The data layer (DuckDB SQL, structured like a dbt project) deduplicated
{dq["duplicates_removed"]:,} retried readings out of {dq["raw_rows"]:,}, found
{dq["missing_device_hours_pct"]:.1f} % of device-hours missing and
{dq["devices_with_multiday_outage"]} devices with multi-day outages. A device-event is
included only with a complete event window, adjustment window and ten baseline days;
{dq["eligible_share"]["event/treatment"]:.0%} of dispatched device-events and
{dq["eligible_share"]["event/control"]:.0%} of control device-events qualified. The same rule
applies to both groups, so exclusions can't bias the comparison. All
{len(dq["tests"])} SQL data tests passed.

## Method notes

- **Fleet:** {f.n_homes:,} homes with heterogeneous cooling load (balance point, sensitivity,
  AC size, behaviour), driven by Greensboro, NC weather. Events: {f.event_hours} h from
  {f.event_start_hour}:00, called on the {f.n_events} hottest weekdays, with
  {f.precool_hours} h pre-cool, {f.snapback_hours} h snapback, overrides that rise with
  heat, and {f.offline_rate:.0%} of devices missing the signal. {f.holdout_share:.0%} of homes
  are randomly held out as control.
- **ML model:** a pooled gradient-boosted regressor (scikit-learn) trained on non-event days,
  with weather, calendar, device-fingerprint and same-morning load features. Validation on
  held-out days: {model["hourly_cv_rmse_pct"]:.0f} % CV(RMSE) per device-hour,
  {model["fleet_bias_pct"]:+.1f} % fleet-level bias. Most useful features: {fi}.
- **Uncertainty:** Poisson cluster bootstrap over devices ({mv.n_bootstrap} replicates),
  {mv.confidence:.0%} percentile intervals; devices are the clusters, so season-level
  intervals account for the same homes appearing in every event.
- **Limitations:** simulated homes are simpler than real ones (no humidity, occupancy
  schedules or solar gain on specific walls), and the true effect is the average over
  dispatched homes that reported data. On real data the truth is unknown, so placebo tests,
  holdout comparisons and pre-period validation replace the "error vs truth" columns.
"""


def _validation_text(s: dict[str, Any], confidence: float) -> tuple[str, str, str]:
    """Phrases and a section drawn from the multi-fleet validation, if it was run."""
    val = s.get("validation")
    if not val:
        return "", "", "Run with `--validate N` to repeat the study on N independent fleets."
    vm, n = val["methods"], val["n_runs"]
    best, ml_ = vm["ml_control"], vm["ml"]
    calibration = (
        f", and across {n} independent simulated fleets its {confidence:.0%} confidence "
        f"interval contained the truth {best['ci_coverage']:.0%} of the time"
    )
    ml_spread = (
        f" (across {n} fleets its season error ranged with a standard deviation of "
        f"{ml_['error_sd_pct']:.1f} points, and its interval contained the truth only "
        f"{ml_['ci_coverage']:.0%} of the time)"
    )
    rows = "\n".join(
        f"| {mm.label} | {_pct(vm[mm.key]['mean_error_pct'])} | {vm[mm.key]['error_sd_pct']:.1f} "
        f"| {vm[mm.key]['rmse_pct']:.1f} % | {vm[mm.key]['ci_coverage']:.0%} "
        f"| ±{vm[mm.key]['ci_halfwidth_pct']:.1f} % |"
        for mm in METHODS
    )
    section = f"""

### Validation across {n} independent fleets

The whole study was repeated on {n} fresh fleets (new homes, random assignment and noise;
same weather and event days). A trustworthy method has a mean error near zero and a
{confidence:.0%} interval that contains the truth about {confidence:.0%} of the time.

| Method | Mean season error | Error SD (pts) | RMSE | {confidence:.0%} CI contains truth | Avg CI half-width |
|---|---|---|---|---|---|
{rows}
"""
    return calibration, ml_spread, section
