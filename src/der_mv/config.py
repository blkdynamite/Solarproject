"""Configuration for the fleet demand-response M&V study."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from solar_rooftop.config import ConfigError, SiteConfig, build_section, require


@dataclass(frozen=True)
class FleetConfig:
    """The simulated device fleet and the program's event design."""

    n_homes: int = 2000
    holdout_share: float = 0.20  # randomised control group (never dispatched)
    season_start: str = "2025-06-01"
    season_end: str = "2025-09-30"
    n_events: int = 15  # called on the hottest weekdays
    min_lookback_days: int = 21  # no events before this many season days
    event_start_hour: int = 15  # local standard time, hour-beginning
    event_hours: int = 3
    precool_hours: int = 1
    snapback_hours: int = 2
    override_rate: float = 0.15  # share of homes that opt out mid-event (hot day)
    offline_rate: float = 0.03  # devices that miss the dispatch signal
    telemetry_gap_rate: float = 0.01  # random missing hourly readings
    long_outage_share: float = 0.02  # devices that drop offline for days
    duplicate_rate: float = 0.002  # duplicated readings (ingestion retries)
    seed: int = 7

    def validate(self) -> None:
        require(self.n_homes >= 50, "fleet.n_homes must be >= 50")
        require(0 < self.holdout_share < 0.9, "fleet.holdout_share must be within 0-0.9")
        require(1 <= self.n_events <= 40, "fleet.n_events must be within 1-40")
        require(0 <= self.event_start_hour <= 23 - self.event_hours, "event window out of day")
        require(
            self.event_start_hour - self.precool_hours >= 9,
            "events (incl. pre-cooling) must start after 09:00 to leave a clean morning window",
        )
        for name in ("override_rate", "offline_rate", "telemetry_gap_rate", "long_outage_share"):
            require(0 <= getattr(self, name) < 0.5, f"fleet.{name} must be within 0-0.5")


@dataclass(frozen=True)
class MVConfig:
    """Measurement & verification method settings."""

    baseline_days: int = 10  # "10-of-10": average of the 10 most recent eligible days
    adjustment_hours: int = 3  # day-of adjustment window: hours just before the event
    adjustment_cap: float = 0.4  # adjustment factor capped to 1 +/- this
    min_coverage: float = 1.0  # share of event-window hours a device must report
    n_placebo_days: int = 10
    n_bootstrap: int = 500
    confidence: float = 0.90
    target_precision_pct: float = 2.0  # +/- per-event precision used to size the holdout
    # Value
    capacity_value_per_kw_yr: float = 100.0  # $/kW-yr, illustrative avoided capacity cost
    program_devices: int = 50_000  # scale results to a utility-sized program

    def validate(self) -> None:
        require(self.baseline_days >= 3, "mv.baseline_days must be >= 3")
        require(self.adjustment_hours >= 1, "mv.adjustment_hours must be >= 1")
        require(0 <= self.adjustment_cap < 1, "mv.adjustment_cap must be within 0-1")
        require(0 < self.min_coverage <= 1, "mv.min_coverage must be within 0-1")
        require(self.n_bootstrap >= 50, "mv.n_bootstrap must be >= 50")
        require(0.5 < self.confidence < 1, "mv.confidence must be within 0.5-1")
        require(0 < self.target_precision_pct < 50, "mv.target_precision_pct must be 0-50")


@dataclass(frozen=True)
class StudyConfig:
    site: SiteConfig = field(default_factory=SiteConfig)
    fleet: FleetConfig = field(default_factory=FleetConfig)
    mv: MVConfig = field(default_factory=MVConfig)

    def validate(self) -> StudyConfig:
        self.site.validate()
        self.fleet.validate()
        self.mv.validate()
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_study_config(path: str | Path | None = None) -> StudyConfig:
    if path is None:
        return StudyConfig().validate()
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    unknown = set(raw) - {"site", "fleet", "mv"}
    if unknown:
        raise ConfigError(f"Unknown section(s): {', '.join(sorted(unknown))}")
    return StudyConfig(
        site=build_section(SiteConfig, raw.get("site", {}), "site"),
        fleet=build_section(FleetConfig, raw.get("fleet", {}), "fleet"),
        mv=build_section(MVConfig, raw.get("mv", {}), "mv"),
    ).validate()
