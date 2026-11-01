SELECT ts, CAST(ts AS DATE) AS day, hour(ts) AS hod, temp_c
FROM raw_weather
