from dataclasses import replace
from pathlib import Path

import pytest

from der_mv.config import FleetConfig, StudyConfig, load_study_config
from solar_rooftop.config import ConfigError

REPO = Path(__file__).resolve().parents[2]


def test_shipped_config_matches_defaults() -> None:
    assert load_study_config(REPO / "config" / "fleet_thermostat_dr.toml") == StudyConfig()


@pytest.mark.parametrize(
    "overrides",
    [
        {"holdout_share": 0.0},
        {"n_homes": 10},
        {"event_start_hour": 9},  # pre-cooling would overlap the morning feature window
        {"override_rate": 0.9},
    ],
)
def test_invalid_fleet_rejected(overrides: dict[str, float]) -> None:
    with pytest.raises(ConfigError):
        replace(FleetConfig(), **overrides).validate()


def test_unknown_section_rejected(tmp_path: Path) -> None:
    p = tmp_path / "bad.toml"
    p.write_text("[flet]\nn_homes = 100\n")
    with pytest.raises(ConfigError, match="flet"):
        load_study_config(p)
