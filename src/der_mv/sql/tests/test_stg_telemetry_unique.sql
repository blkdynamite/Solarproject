-- Fails if any device-hour appears more than once after deduplication.
SELECT device_id, ts, count(*) AS n
FROM stg_telemetry
GROUP BY device_id, ts
HAVING count(*) > 1
