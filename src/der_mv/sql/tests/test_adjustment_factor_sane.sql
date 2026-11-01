-- Fails if an uncapped adjustment factor is wildly off (sign of a unit or join bug).
SELECT analysis_id, device_id, adjustment_factor
FROM mart_event_device
WHERE eligible AND (adjustment_factor <= 0 OR adjustment_factor > 10)
