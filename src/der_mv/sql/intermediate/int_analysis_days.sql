-- Days the M&V methods are evaluated on:
--   * 'event'   - real dispatch days
--   * 'placebo' - the hottest *non-event* weekdays, treated as if an event had been
--                 called. Nothing happened, so every method should estimate ~0 kW;
--                 any non-zero answer is bias. This check works on real data too.
WITH daily AS (
    SELECT day, max(temp_c) AS tmax
    FROM stg_weather
    GROUP BY day
),
events AS (
    SELECT e.event_id AS analysis_id, 'event' AS kind, e.event_date AS day, d.tmax
    FROM stg_events e
    JOIN daily d ON d.day = e.event_date
),
placebo AS (
    SELECT
        1000 + row_number() OVER (ORDER BY tmax DESC) AS analysis_id,
        'placebo' AS kind,
        day,
        tmax
    FROM daily
    WHERE dayofweek(day) BETWEEN 1 AND 5
      AND day NOT IN (SELECT event_date FROM stg_events)
      AND day >= (SELECT min(event_date) FROM stg_events)  -- same lookback as events
    QUALIFY row_number() OVER (ORDER BY tmax DESC) <= getvariable('n_placebo_days')
)
SELECT * FROM events
UNION ALL
SELECT * FROM placebo
