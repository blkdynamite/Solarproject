# DER Analytics: measuring what distributed energy actually delivers

[![CI](https://github.com/blkdynamite/Solarproject/actions/workflows/ci.yml/badge.svg)](https://github.com/blkdynamite/Solarproject/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13-blue)
![Stack](https://img.shields.io/badge/stack-pandas%20·%20scikit--learn%20·%20DuckDB%20SQL%20·%20pvlib-informational)

Two applied data-science case studies on distributed energy resources (DERs), each built as
tested, reproducible, production-style Python:

| | Question | Techniques |
|---|---|---|
| **[1. Fleet demand-response M&V](#case-study-1-how-many-kw-did-the-thermostat-fleet-deliver)** | How many kW did 2,000 smart thermostats deliver during grid events, how sure are we, and what does a wrong answer cost? | Causal inference (randomised control group), ML counterfactuals (gradient boosting), placebo tests, cluster bootstrap, power analysis, SQL data modelling |
| **[2. Commercial rooftop solar](docs/solar_rooftop.md)** | How much will a 300 kW system produce, does it pay, and can we catch faults automatically? | Physical modelling (pvlib), project finance, anomaly detection with Monte Carlo validation |

---

## Case study 1: How many kW did the thermostat fleet deliver?

Utilities pay demand-response programs for the load they remove from the grid during peak
events. But the reduction itself is never observed: it is the gap between what homes used and
what they *would have* used. That counterfactual is the core measurement and verification (M&V)
problem, and the industry's standard tools for it break down exactly when it matters most.

**Pain points this study tackles:**

1. **Baseline bias.** Events are called on the hottest days, while "X-of-Y" baselines are built
   from earlier, cooler days.
2. **Pre-cooling contaminates the day-of adjustment** that is supposed to fix (1).
3. **No uncertainty.** Programs report one number with no error bars.
4. **Messy IoT data.** Duplicates, gaps, offline devices and customer overrides.
5. **The holdout trade-off.** A control group gives a clean answer, but every held-out device
   delivers nothing.

**Approach.** I simulated a 2,000-home thermostat fleet driven by real Greensboro, NC weather,
with 15 heatwave events, pre-cooling, snapback, overrides, outages and a randomised 20 % holdout.
Simulating means **the true delivered kW is known**, so each method can be scored directly; the
placebo tests and holdout comparisons then carry over to real data, where the truth never is.
Telemetry lands in Parquet, a dbt-style DuckDB SQL layer cleans and models it (with data
tests), and five estimators are compared, from the industry-standard 10-of-10 baseline to a
gradient-boosted counterfactual combined with the randomised control group.

### Results

True delivered reduction: **1.01 kW per home**, or **50 MW (~$5.0M/yr capacity value)** at
a 50,000-device program scale.

| Method | Season estimate (error) | Worst single event | Mean error across 20 fleets | 90 % CI contains truth (20 fleets) | Annual misstatement at 50k devices |
|---|---|---|---|---|---|
| 10-of-10 baseline (industry standard) | −0.01 kW (−101 %) | 136 % | −100 % | 0 % | **−$5.07M** |
| 10-of-10 + day-of adjustment | 0.93 kW (−8.2 %) | 63 % | −7.6 % | 0 % | −$413k |
| ML counterfactual alone | 1.01 kW (+0.2 %) | 42 % | +2.9 % (SD 5.5 pts) | 15 % | +$9k (this season, by luck) |
| Control group (RCT) | 0.96 kW (−5.3 %) | 9 % | +0.1 % (SD 2.9 pts) | 100 % | −$265k |
| **ML + control group** | **1.01 kW (+0.3 %)** | **2 %** | **−0.2 % (SD 0.9 pts)** | **100 %** | **+$14k** |

![Hottest event day](results/fleet_mv/1_event_day_profile.png)

**Findings:**

- **The standard baseline erases the program on peak days.** Events fall on heatwave days
  that are about 6 °C hotter than the baseline days, so the 10-of-10 baseline under-predicts
  what homes would have used by about as much as they actually cut. Its error worsens by
  0.20 kW per home for every 1 °C, so it is worst exactly when capacity is worth most.
- **The common fix works partly by accident.** The day-of adjustment calibrates to the hours
  before the event, but dispatched thermostats pre-cool then (median adjustment factor 1.58 vs
  1.49 for control homes on the same days). That inflation offsets part of the heat bias; it
  is not a method you could defend to a regulator.
- **ML alone is unreliable, and over-confident.** Events are the hottest days, so the model
  must extrapolate. It happened to land close this season, but it missed single events by up to
  42 %, its error varied ±5.5 points across fleets, and its 90 % interval contained the truth
  only 15 % of the time.
- **ML + randomised control group is accurate and precise.** The control group's prediction
  error on the same day cancels the model's bias, and the model removes most home-to-home
  noise, giving **2.2× tighter intervals** than a plain control-group comparison (the CUPED idea
  from A/B testing). For ±2 % per-event precision at 50,000 devices, that means a **1 % holdout
  instead of 10 %: 4.5 MW (~$455k/yr) of capacity no longer sitting idle.**
- **Placebo tests catch every failure without needing the truth.** On the hottest days with no
  event, the 10-of-10 baseline "finds" −0.45 kW and the adjusted baseline +0.11 kW; ML + control
  finds 0.00. This is the check that transfers directly to real program data.

![Method errors](results/fleet_mv/2_method_errors.png)

📄 **[Read the full utility-facing memo](results/fleet_mv/REPORT.md)**: recommendation, method
comparison, why baselines fail, holdout sizing and data quality, all generated from the run so
the numbers can't drift.

### What's under the hood

```
src/der_mv/
├── simulate.py        # fleet physics + behaviour, known ground truth, messy telemetry
├── sql/               # dbt-style SQL models on DuckDB
│   ├── staging/       #   dedupe retried readings, type and range checks
│   ├── intermediate/  #   device-day coverage, event + placebo days
│   ├── marts/         #   10-of-10 baseline in SQL, event-window table, ML features
│   └── tests/         #   data tests: unique keys, group balance, sane adjustments
├── warehouse.py       # Parquet "lake" -> layered SQL build -> data tests
├── ml.py              # pooled gradient-boosted counterfactual, day-level validation
├── estimators.py      # 5 estimators in one form; Poisson cluster bootstrap
├── evaluation.py      # vs truth, placebo days, temperature bias, holdout sizing, $ impact
├── report.py          # renders the decision memo from the run's numbers
└── pipeline.py / cli.py
```

- **Scales by design.** SQL does the heavy lifting (portable to Snowflake and dbt), and the
  bootstrap uses Poisson weights (weighted sums, no resampling), the standard approach for
  millions of devices.
- **No leakage.** The ML model and device fingerprints are trained only on non-event,
  non-placebo days and validated on held-out *days*, not rows.
- **Trust but verify.** The SQL baseline is tested against an independent pandas
  implementation; estimators are tested on synthetic data with known bias; the whole study is
  repeated on 20 independent fleets to check that confidence intervals are calibrated.
- **Next steps** are in the **[roadmap](ROADMAP.md)**: causal-ML targeting of the best
  responders, probabilistic flexibility forecasts, real-data validation, EVs and batteries.

```bash
make install       # pinned, reproducible environment
make fleet-quick   # full study, ~40 s  -> results/fleet_mv/
make fleet         # + validation on 20 independent fleets (~15 min)
```

---

## Case study 2: Commercial rooftop solar

A 300 kW-DC warehouse rooftop system in Greensboro, NC: hourly pvlib yield model with P50/P90,
25-year pre-tax and after-tax (ITC + MACRS) financials with a sensitivity analysis, and a
causal fault detector that finds inverter outages, soiling and data gaps the same day (outages)
or within about three weeks (gradual soiling), validated over 200 Monte Carlo runs.

![Solar monitoring](results/6_performance_monitoring.png)

**[Full write-up →](docs/solar_rooftop.md)**

---

## Engineering

- **Tests:** 82 tests across both case studies, 96 % coverage (`make check` runs ruff, mypy and pytest,
  exactly as CI does on Python 3.11–3.13).
- **Reproducible:** every input in a TOML config (`config/`), seeded randomness, deterministic
  ML training, and a lockfile of tested versions.
- **Typed and linted:** mypy on every function signature; ruff for lint and formatting.
