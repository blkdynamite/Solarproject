from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from der_mv.config import StudyConfig, load_study_config
from der_mv.pipeline import Analysis, analyze


@pytest.fixture(scope="session")
def small_cfg() -> StudyConfig:
    """A small fleet that still has enough homes for stable estimates (~10 s end to end)."""
    cfg = load_study_config()
    return replace(
        cfg,
        fleet=replace(cfg.fleet, n_homes=400, n_events=8, seed=11),
        mv=replace(cfg.mv, n_bootstrap=200, n_placebo_days=6),
    ).validate()


@pytest.fixture(scope="session")
def analysis(
    small_cfg: StudyConfig, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Analysis]:
    tmp: Path = tmp_path_factory.mktemp("fleet")
    a = analyze(small_cfg, ":memory:", tmp / "raw")
    yield a
    a.con.close()
