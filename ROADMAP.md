# Roadmap: DER fleet M&V with AI and statistics

Goal: show how data science answers the question at the heart of every
demand-response (DR) program, **"how many kW did the fleet actually deliver, and
how sure are we?"**, and why the answer matters in dollars.

## Pain points addressed

| # | Pain point | Why it matters |
|---|---|---|
| 1 | **Baseline bias.** Events are called on the hottest days; baselines built from earlier, cooler days misstate the counterfactual. | Programs are over- or under-credited; utilities lose trust in DER capacity. |
| 2 | **Pre-cooling contaminates day-of adjustments.** Thermostats pre-cool before the event, inflating the load that "adjusted" baselines calibrate to. | A common industry fix quietly makes the answer worse. |
| 3 | **No uncertainty.** Results are reported as a single number. | Regulators and planners can't weigh DER against a power plant without error bars. |
| 4 | **Messy IoT data.** Offline devices, telemetry gaps and customer overrides. | Silent data issues bias results; rules must be explicit and tested. |
| 5 | **The holdout trade-off.** A control group gives an unbiased answer, but every held-out device delivers zero kW. | Choosing the control group's size is a cost-vs-precision decision. |

## Phase 1: Quick wins (this iteration)

- [x] Fleet simulator with known ground truth: weather-driven cooling load, events on the hottest weekdays, pre-cool and snapback, heterogeneous response, overrides, outages, telemetry gaps, randomized holdout.
- [x] SQL layer (DuckDB, dbt-style staging -> intermediate -> mart): data-quality tests and the 10-of-10 baseline in SQL.
- [x] Estimators: 10-of-10, 10-of-10 with day-of adjustment, ML counterfactual (gradient boosting), ML corrected by the control group, randomized control trial (RCT).
- [x] Evaluation: bias and error against the truth, placebo tests on non-event days, cluster-bootstrap confidence intervals.
- [x] Holdout-size power analysis and the dollar value of estimation error.
- [x] Utility-facing decision memo, charts, tests, CI.

## Phase 2: Longer targets

- [ ] **Targeting with causal ML:** estimate each home's response (heterogeneous treatment effects, e.g. T-learner or causal forest) and dispatch the best responders first, so the fleet delivers more kW per event with fewer opt-outs.
- [ ] **Flexibility forecasting:** day-ahead forecast of available fleet kW with prediction intervals (quantile gradient boosting plus conformal calibration), scored with pinball loss and interval coverage.
- [ ] **Real-data validation:** NREL End-Use Load Profiles or ecobee Donate Your Data load shapes; placebo tests are the bias check when the truth is unknown.
- [ ] **Analytics engineering:** a full dbt project (dbt-duckdb locally, portable to Snowflake) with schema tests and docs; a small BI dashboard.
- [ ] **More device types:** EV managed charging (charging deferral and rebound) and home batteries (state-of-charge limits, round-trip losses).
- [ ] **Scale:** partitioned Parquet with Polars or Spark to process 1M devices; incremental models.
