from pathlib import Path

import pytest

from solar_rooftop.config import ConfigError, ProjectConfig, load_config

REPO = Path(__file__).resolve().parents[2]


def test_shipped_config_matches_defaults() -> None:
    # The example config documents the defaults; keep the two in sync.
    assert load_config(REPO / "config" / "greensboro_300kw.toml") == ProjectConfig()


def test_derived_ratings() -> None:
    s = ProjectConfig().system
    assert s.ac_kw == pytest.approx(250.0)
    assert s.inverter_kw == pytest.approx(50.0)


def test_unknown_key_rejected(tmp_path: Path) -> None:
    p = tmp_path / "bad.toml"
    p.write_text("[system]\ndc_kw = 100\ntypo_key = 1\n")
    with pytest.raises(ConfigError, match="typo_key"):
        load_config(p)


def test_unknown_section_rejected(tmp_path: Path) -> None:
    p = tmp_path / "bad.toml"
    p.write_text("[sytem]\ndc_kw = 100\n")
    with pytest.raises(ConfigError, match="sytem"):
        load_config(p)


@pytest.mark.parametrize(
    "toml",
    [
        "[system]\ndc_kw = -5",
        "[system]\ndc_ac_ratio = 3.0",
        "[finance]\nitc = 1.5",
        "[finance]\nmacrs_schedule = [0.5, 0.4]",
        "[finance]\ninverter_replace_yr = 40",
    ],
)
def test_invalid_values_rejected(tmp_path: Path, toml: str) -> None:
    p = tmp_path / "bad.toml"
    p.write_text(toml)
    with pytest.raises(ConfigError):
        load_config(p)


def test_missing_file() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config("does/not/exist.toml")
