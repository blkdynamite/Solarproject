"""Typed, validated project configuration.

All model inputs live in a TOML file (see ``config/greensboro_300kw.toml``) and
are loaded into frozen dataclasses. Keeping assumptions out of the code means a
new site or a sensitivity case is just a new config file, and every run writes
the exact inputs it used next to its results.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar


class ConfigError(ValueError):
    """Raised when a configuration file is missing keys or has invalid values."""


@dataclass(frozen=True)
class SiteConfig:
    name: str = "Greensboro, NC (Piedmont Triad Intl.)"
    # TMY3 file name inside pvlib's bundled data folder, or an absolute path.
    tmy3_file: str = "723170TYA.CSV"
    # Fixed-offset zone matching TMY3 local standard time (no DST).
    timezone: str = "Etc/GMT+5"
    model_year: int = 2025

    def validate(self) -> None:
        if not self.tmy3_file:
            raise ConfigError("site.tmy3_file must be set")


@dataclass(frozen=True)
class SystemConfig:
    dc_kw: float = 300.0
    dc_ac_ratio: float = 1.2
    n_inverters: int = 5
    tilt_deg: float = 10.0
    azimuth_deg: float = 180.0
    gamma_pdc: float = -0.0035
    losses_pct: float = 14.0
    inverter_eff_nom: float = 0.96
    racking: str = "open_rack_glass_polymer"
    transposition_model: str = "haydavies"

    @property
    def ac_kw(self) -> float:
        return self.dc_kw / self.dc_ac_ratio

    @property
    def inverter_kw(self) -> float:
        return self.ac_kw / self.n_inverters

    def validate(self) -> None:
        require(self.dc_kw > 0, "system.dc_kw must be positive")
        require(0.8 <= self.dc_ac_ratio <= 2.0, "system.dc_ac_ratio must be within 0.8-2.0")
        require(self.n_inverters >= 1, "system.n_inverters must be >= 1")
        require(0 <= self.tilt_deg <= 90, "system.tilt_deg must be within 0-90")
        require(0 <= self.azimuth_deg < 360, "system.azimuth_deg must be within 0-360")
        require(-0.01 < self.gamma_pdc < 0, "system.gamma_pdc should be a small negative number")
        require(0 <= self.losses_pct < 50, "system.losses_pct must be within 0-50")
        require(0.8 < self.inverter_eff_nom <= 1.0, "system.inverter_eff_nom must be within 0.8-1")


@dataclass(frozen=True)
class FinanceConfig:
    capex_per_w_dc: float = 1.90
    om_per_kw_yr: float = 18.0
    om_escalation: float = 0.025
    tariff_per_kwh: float = 0.12
    tariff_escalation: float = 0.025
    degradation: float = 0.005
    discount_rate: float = 0.07
    life_yrs: int = 25
    itc: float = 0.30
    inverter_replace_yr: int = 13
    inverter_replace_per_w_ac: float = 0.10
    # After-tax view (commercial owner with tax appetite)
    tax_rate: float = 0.23
    macrs_schedule: tuple[float, ...] = (0.20, 0.32, 0.192, 0.1152, 0.1152, 0.0576)
    # Energy-yield uncertainty (1-sigma, fraction) used for P90 estimate
    yield_uncertainty: float = 0.06

    def validate(self) -> None:
        require(self.capex_per_w_dc > 0, "finance.capex_per_w_dc must be positive")
        require(self.tariff_per_kwh > 0, "finance.tariff_per_kwh must be positive")
        require(0 < self.discount_rate < 0.5, "finance.discount_rate must be within 0-0.5")
        require(self.life_yrs >= 1, "finance.life_yrs must be >= 1")
        require(0 <= self.itc < 1, "finance.itc must be within 0-1")
        require(0 <= self.tax_rate < 1, "finance.tax_rate must be within 0-1")
        require(0 <= self.degradation < 0.05, "finance.degradation must be within 0-0.05")
        require(
            0 < self.inverter_replace_yr <= self.life_yrs,
            "finance.inverter_replace_yr must fall within the project life",
        )
        require(
            abs(sum(self.macrs_schedule) - 1.0) < 1e-6,
            "finance.macrs_schedule must sum to 1.0",
        )
        require(0 <= self.yield_uncertainty < 0.5, "finance.yield_uncertainty must be within 0-0.5")


@dataclass(frozen=True)
class MonitoringConfig:
    seed: int = 42
    # Synthetic "meter" error model
    hourly_noise: float = 0.02  # independent per-hour measurement noise (1-sigma)
    daily_noise: float = 0.015  # day-level model error, e.g. irradiance mismatch (1-sigma)
    # Detector settings
    min_expected_kwh_per_kwp: float = 1.0  # skip very overcast days (clear day is ~6-7)
    baseline_window_days: int = 60
    baseline_min_days: int = 14
    drift_window_days: int = 7
    drift_threshold: float = 0.03  # flag when rolling median is >3 % below baseline
    major_outage_threshold: float = 0.5  # flag when output drops by more than half

    def validate(self) -> None:
        require(self.hourly_noise >= 0 and self.daily_noise >= 0, "noise must be >= 0")
        require(self.min_expected_kwh_per_kwp >= 0, "min_expected_kwh_per_kwp must be >= 0")
        require(self.baseline_window_days >= self.baseline_min_days >= 1, "bad baseline window")
        require(self.drift_window_days >= 1, "drift_window_days must be >= 1")
        require(0 < self.drift_threshold < 0.5, "drift_threshold must be within 0-0.5")
        require(0 < self.major_outage_threshold < 1, "major_outage_threshold must be within 0-1")


@dataclass(frozen=True)
class ProjectConfig:
    site: SiteConfig = field(default_factory=SiteConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    finance: FinanceConfig = field(default_factory=FinanceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    def validate(self) -> ProjectConfig:
        for section in (self.site, self.system, self.finance, self.monitoring):
            section.validate()
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def build_section(cls: type[T], data: dict[str, Any], section: str) -> T:
    known = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(f"Unknown key(s) in [{section}]: {', '.join(sorted(unknown))}")
    kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in data.items()}
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> ProjectConfig:
    """Load and validate a TOML config. ``None`` returns the built-in defaults."""
    if path is None:
        return ProjectConfig().validate()
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    unknown = set(raw) - {"site", "system", "finance", "monitoring"}
    if unknown:
        raise ConfigError(f"Unknown section(s): {', '.join(sorted(unknown))}")
    return ProjectConfig(
        site=build_section(SiteConfig, raw.get("site", {}), "site"),
        system=build_section(SystemConfig, raw.get("system", {}), "system"),
        finance=build_section(FinanceConfig, raw.get("finance", {}), "finance"),
        monitoring=build_section(MonitoringConfig, raw.get("monitoring", {}), "monitoring"),
    ).validate()
