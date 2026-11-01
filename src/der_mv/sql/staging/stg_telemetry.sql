-- One row per device-hour.
-- Ingestion retries create duplicate readings: keep the latest-received copy.
-- Physically impossible values are dropped (a home can't draw < 0 or > 25 kW).
SELECT
    device_id,
    ts,
    CAST(ts AS DATE) AS day,
    hour(ts) AS hod,
    kw
FROM raw_telemetry
WHERE kw IS NOT NULL AND kw >= 0 AND kw < 25
QUALIFY row_number() OVER (PARTITION BY device_id, ts ORDER BY received_at DESC) = 1
