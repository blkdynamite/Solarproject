-- One row per analysis day x device: event-window load, baselines and eligibility.
-- The day-of adjustment scales the baseline by actual / baseline over the hours just
-- before the event (capped), the usual fix for "today is hotter than the baseline
-- days". Note those hours include any pre-cooling, which inflates the factor.
WITH h AS (
    SELECT
        *,
        hod >= getvariable('event_start_hour')
            AND hod < getvariable('event_start_hour') + getvariable('event_hours') AS in_window,
        hod >= getvariable('event_start_hour') - getvariable('adjustment_hours')
            AND hod < getvariable('event_start_hour') AS in_adjustment
    FROM mart_baseline_hourly
),
agg AS (
    SELECT
        analysis_id,
        any_value(kind) AS kind,
        device_id,
        count(*) FILTER (in_window) AS n_window_hours,
        count(*) FILTER (in_adjustment) AS n_adjustment_hours,
        avg(actual_kw) FILTER (in_window) AS actual_kw,
        avg(baseline_kw) FILTER (in_window) AS baseline_kw,
        sum(actual_kw) FILTER (in_adjustment) / sum(baseline_kw) FILTER (in_adjustment) AS raw_adj,
        min(n_baseline_days) AS n_baseline_days
    FROM h
    GROUP BY analysis_id, device_id
)
SELECT
    a.analysis_id,
    a.kind,
    a.device_id,
    d.grp,
    a.actual_kw,
    a.baseline_kw,
    a.baseline_kw * greatest(1 - getvariable('adjustment_cap'),
                             least(1 + getvariable('adjustment_cap'), a.raw_adj))
        AS adj_baseline_kw,
    a.raw_adj AS adjustment_factor,
    a.n_window_hours,
    a.n_baseline_days,
    -- Explicit, auditable inclusion rule: complete event window and adjustment window,
    -- and a full set of baseline days.
    a.n_window_hours >= getvariable('event_hours') * getvariable('min_coverage')
        AND a.n_adjustment_hours = getvariable('adjustment_hours')
        AND coalesce(a.n_baseline_days, 0) >= getvariable('baseline_days') AS eligible
FROM agg a
JOIN stg_devices d USING (device_id)
