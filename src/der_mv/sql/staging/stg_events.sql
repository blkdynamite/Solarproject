SELECT
    CAST(event_id AS INTEGER) AS event_id,
    CAST(event_date AS DATE) AS event_date,
    start_ts,
    end_ts
FROM raw_events
