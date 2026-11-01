"""A small dbt-style SQL layer on DuckDB.

Raw tables land as Parquet (the "lake"); SQL models in ``sql/`` are materialised
layer by layer (staging -> intermediate -> marts) as tables named after their
files, then ``sql/tests`` run as data tests: each test query returns the
offending rows, and any row is a failure. The SQL is plain ANSI-ish DuckDB, so
porting the models to dbt on Snowflake is mostly a matter of adding ``ref()``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from .config import StudyConfig
from .simulate import Fleet

log = logging.getLogger(__name__)

SQL_DIR = Path(__file__).parent / "sql"
LAYERS = ("staging", "intermediate", "marts")


class DataQualityError(RuntimeError):
    """Raised when a SQL data test returns failing rows."""


@dataclass(frozen=True)
class TestResult:
    name: str
    failing_rows: int


def connect(database: str | Path = ":memory:") -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(database))


def load_raw(con: duckdb.DuckDBPyConnection, fleet: Fleet, raw_dir: Path) -> None:
    """Write the observable tables to Parquet and expose them as raw_* views."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "raw_telemetry": fleet.telemetry,
        "raw_devices": fleet.devices,
        "raw_events": fleet.events,
        "raw_weather": fleet.weather,
    }
    for name, df in tables.items():
        path = raw_dir / f"{name}.parquet"
        con.register("_df", df)
        con.execute(f"COPY _df TO '{path}' (FORMAT PARQUET)")
        con.unregister("_df")
        con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{path}')")


def sql_variables(cfg: StudyConfig) -> dict[str, Any]:
    f, mv = cfg.fleet, cfg.mv
    return {
        "event_start_hour": f.event_start_hour,
        "event_hours": f.event_hours,
        "baseline_days": mv.baseline_days,
        "adjustment_hours": mv.adjustment_hours,
        "adjustment_cap": mv.adjustment_cap,
        "min_coverage": mv.min_coverage,
        "n_placebo_days": mv.n_placebo_days,
        # Morning-load feature window ends an hour before any pre-cooling starts.
        "morning_end_hour": f.event_start_hour - f.precool_hours - 2,
    }


def build(con: duckdb.DuckDBPyConnection, variables: dict[str, Any]) -> list[str]:
    """Materialise every model, layer by layer. Returns the model names in build order."""
    for key, value in variables.items():
        con.execute(f"SET VARIABLE {key} = {value!r}")
    built = []
    for layer in LAYERS:
        for path in sorted((SQL_DIR / layer).glob("*.sql")):
            log.debug("building %s.%s", layer, path.stem)
            con.execute(f"CREATE OR REPLACE TABLE {path.stem} AS {path.read_text()}")
            built.append(path.stem)
    return built


def run_tests(con: duckdb.DuckDBPyConnection, raise_on_failure: bool = True) -> list[TestResult]:
    results = []
    for path in sorted((SQL_DIR / "tests").glob("*.sql")):
        row = con.execute(f"SELECT count(*) FROM ({path.read_text()})").fetchone()
        results.append(TestResult(path.stem, int(row[0]) if row else 0))
    failed = [r for r in results if r.failing_rows]
    if failed and raise_on_failure:
        detail = ", ".join(f"{r.name} ({r.failing_rows} rows)" for r in failed)
        raise DataQualityError(f"Data tests failed: {detail}")
    return results


def data_quality_summary(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Headline data-quality numbers for the report."""

    def scalar(sql: str) -> Any:
        row = con.execute(sql).fetchone()
        return row[0] if row else None

    raw = scalar("SELECT count(*) FROM raw_telemetry")
    staged = scalar("SELECT count(*) FROM stg_telemetry")
    expected = scalar(
        "SELECT (SELECT count(*) FROM stg_devices) * (SELECT count(DISTINCT ts) FROM stg_weather)"
    )
    elig = con.execute(
        """
        SELECT kind, grp, avg(eligible::INT) AS share
        FROM mart_event_device GROUP BY kind, grp ORDER BY kind, grp
        """
    ).df()
    return {
        "raw_rows": raw,
        "duplicates_removed": raw - staged,
        "missing_device_hours_pct": round(100 * (1 - staged / expected), 2),
        "devices_with_multiday_outage": scalar(
            "SELECT count(DISTINCT device_id) FROM int_device_days WHERE n_hours < 12"
        ),
        "eligible_share": {
            f"{r.kind}/{r.grp}": round(float(r.share), 3) for r in elig.itertuples()
        },
    }


def query(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).df()
