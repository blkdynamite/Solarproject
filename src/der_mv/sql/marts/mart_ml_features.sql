-- Device-hour feature table for the ML counterfactual model.
--   * weather: current, 4 h and 12 h trailing means (thermal mass), daily max
--   * device fingerprint learned on non-analysis days only (no leakage):
--       typical load by hour, and sensitivity of afternoon load to daily max temp
--   * same-day morning load: captures "today's" household behaviour, and ends
--     before any pre-cooling, so it is never contaminated by the event itself
WITH wx AS (
    SELECT
        ts, temp_c,
        avg(temp_c) OVER (ORDER BY ts ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS temp_4h,
        avg(temp_c) OVER (ORDER BY ts ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) AS temp_12h,
        max(temp_c) OVER (PARTITION BY day) AS tmax_day
    FROM stg_weather
),
train AS (
    SELECT t.*
    FROM stg_telemetry t
    WHERE t.day NOT IN (SELECT day FROM int_analysis_days)
),
dev_hod AS (
    SELECT device_id, hod, avg(kw) AS dev_hod_kw
    FROM train
    GROUP BY device_id, hod
),
dev_daily AS (
    SELECT t.device_id, t.day, avg(t.kw) AS pm_kw, any_value(w.tmax_day) AS tmax_day
    FROM train t
    JOIN wx w USING (ts)
    WHERE t.hod BETWEEN 12 AND 20
    GROUP BY t.device_id, t.day
),
dev_sens AS (
    SELECT device_id, regr_slope(pm_kw, tmax_day) AS dev_temp_slope
    FROM dev_daily
    GROUP BY device_id
),
morning AS (
    SELECT device_id, day, avg(kw) AS morning_kw
    FROM stg_telemetry
    WHERE hod >= 6 AND hod < getvariable('morning_end_hour')
    GROUP BY device_id, day
)
SELECT
    t.device_id,
    t.ts,
    t.day,
    t.hod,
    dayofweek(t.day) AS dow,
    w.temp_c,
    w.temp_4h,
    w.temp_12h,
    w.tmax_day,
    h.dev_hod_kw,
    s.dev_temp_slope,
    m.morning_kw,
    t.kw,
    t.day IN (SELECT day FROM int_analysis_days) AS is_analysis_day
FROM stg_telemetry t
JOIN wx w USING (ts)
LEFT JOIN dev_hod h USING (device_id, hod)
LEFT JOIN dev_sens s USING (device_id)
LEFT JOIN morning m USING (device_id, day)
