from __future__ import annotations

import pandas as pd
import pytest

from solar_rooftop import energy
from solar_rooftop.config import ProjectConfig, load_config
from solar_rooftop.weather import Weather, load_weather


@pytest.fixture(scope="session")
def cfg() -> ProjectConfig:
    return load_config()


@pytest.fixture(scope="session")
def weather(cfg: ProjectConfig) -> Weather:
    return load_weather(cfg.site)


@pytest.fixture(scope="session")
def sim(weather: Weather, cfg: ProjectConfig) -> pd.DataFrame:
    return energy.simulate(weather, cfg.system)
