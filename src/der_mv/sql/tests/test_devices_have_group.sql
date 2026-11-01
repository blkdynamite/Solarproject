-- Fails for devices without a valid treatment/control assignment.
SELECT device_id FROM stg_devices WHERE grp NOT IN ('treatment', 'control') OR grp IS NULL
