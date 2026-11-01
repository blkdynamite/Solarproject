-- "X-of-Y" baseline (default 10-of-10): for each device and analysis day, average
-- the load at each hour over the N most recent eligible days before it.
-- Eligible = weekday, not a dispatch day, and a complete 24 h of telemetry.
WITH baseline_days AS (
    SELECT a.analysis_id, d.device_id, d.day
    FROM int_analysis_days a
    JOIN int_device_days d
      ON d.day < a.day
     AND d.is_weekday
     AND NOT d.is_event_day
     AND d.n_hours = 24
    QUALIFY row_number() OVER (PARTITION BY a.analysis_id, d.device_id ORDER BY d.day DESC)
            <= getvariable('baseline_days')
),
baseline AS (
    SELECT b.analysis_id, b.device_id, t.hod,
           avg(t.kw) AS baseline_kw,
           count(*) AS n_baseline_days
    FROM baseline_days b
    JOIN stg_telemetry t ON t.device_id = b.device_id AND t.day = b.day
    GROUP BY b.analysis_id, b.device_id, t.hod
)
SELECT
    a.analysis_id,
    a.kind,
    t.device_id,
    t.ts,
    t.hod,
    t.kw AS actual_kw,
    b.baseline_kw,
    b.n_baseline_days
FROM int_analysis_days a
JOIN stg_telemetry t ON t.day = a.day
LEFT JOIN baseline b
  ON b.analysis_id = a.analysis_id AND b.device_id = t.device_id AND b.hod = t.hod
