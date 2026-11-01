-- Fails for analysis days without enough eligible devices in either group
-- (too few to estimate anything with useful precision).
SELECT analysis_id, grp, count(*) FILTER (eligible) AS n_eligible
FROM mart_event_device
GROUP BY analysis_id, grp
HAVING count(*) FILTER (eligible) < 30
