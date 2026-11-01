-- Device-day reporting coverage, used to pick valid baseline days.
SELECT
    t.device_id,
    t.day,
    count(*) AS n_hours,
    dayofweek(t.day) BETWEEN 1 AND 5 AS is_weekday,
    t.day IN (SELECT event_date FROM stg_events) AS is_event_day
FROM stg_telemetry t
GROUP BY t.device_id, t.day
