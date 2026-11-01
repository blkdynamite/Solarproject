"""Weather loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pvlib

from .config import SiteConfig

WEATHER_COLUMNS = ["ghi", "dni", "dhi", "temp_air", "wind_speed", "albedo"]
DEFAULT_ALBEDO = 0.2


@dataclass(frozen=True)
class Weather:
    data: pd.DataFrame  # hourly, tz-aware, labelled at mid-hour
    location: pvlib.location.Location


def resolve_tmy3_path(name: str) -> Path:
    """Return an absolute path for a TMY3 file, falling back to pvlib's bundled data."""
    path = Path(name)
    if path.is_absolute():
        return path
    return Path(pvlib.__file__).parent / "data" / name


def load_weather(site: SiteConfig) -> Weather:
    """Load a TMY3 file and return mid-hour-labelled weather plus a site Location."""
    path = resolve_tmy3_path(site.tmy3_file)
    if not path.is_file():
        raise FileNotFoundError(f"TMY3 file not found: {path}")
    raw, meta = pvlib.iotools.read_tmy3(str(path), map_variables=True, coerce_year=site.model_year)

    data = raw[WEATHER_COLUMNS].copy()
    data["albedo"] = data["albedo"].fillna(DEFAULT_ALBEDO)
    # TMY3 values are integrals over the hour *ending* at the timestamp, so the
    # sun position that best represents the hour is at its midpoint.
    data.index = data.index - pd.Timedelta(30, unit="min")
    data.index.name = "time"

    location = pvlib.location.Location(
        latitude=meta["latitude"],
        longitude=meta["longitude"],
        tz=site.timezone,
        altitude=meta["altitude"],
        name=site.name,
    )
    return Weather(data=data, location=location)
