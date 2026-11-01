"""Command-line interface for the fleet M&V study.

fleet-mv run [--config FILE] [--out DIR] [--homes N] [--no-plots] [--validate N]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

from solar_rooftop.config import ConfigError

from .config import load_study_config
from .pipeline import run
from .warehouse import DataQualityError

DEFAULT_CONFIG = Path("config/fleet_thermostat_dr.toml")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="fleet-mv", description="Demand-response M&V study for a smart-thermostat fleet."
    )
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="simulate, build the SQL layer, estimate and report")
    r.add_argument("-c", "--config", type=Path, default=None)
    r.add_argument("-o", "--out", type=Path, default=Path("results/fleet_mv"))
    r.add_argument("--homes", type=int, default=None, help="override fleet.n_homes")
    r.add_argument("--no-plots", action="store_true")
    r.add_argument(
        "--validate",
        type=int,
        default=0,
        metavar="N",
        help="also repeat the study on N independent fleets to check bias and CI calibration",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    path = args.config or (DEFAULT_CONFIG if DEFAULT_CONFIG.is_file() else None)
    try:
        cfg = load_study_config(path)
        if args.homes is not None:
            cfg = replace(cfg, fleet=replace(cfg.fleet, n_homes=args.homes)).validate()
    except ConfigError as exc:
        logging.error("Invalid configuration: %s", exc)
        return 2
    try:
        out = run(cfg, args.out, make_plots=not args.no_plots, validate_runs=args.validate)
    except DataQualityError as exc:
        logging.error("%s", exc)
        return 3
    headline = {
        k: {
            "season_kw": v["season_kw"],
            "season_error_pct": v["season_error_pct"],
            "event_ci_coverage": v["event_ci_coverage"],
        }
        for k, v in out["methods"].items()
    }
    print(json.dumps({"true_season_kw": out["true_season_kw_per_device"], **headline}, indent=2))
    logging.info("Report written to %s", (args.out / "REPORT.md").resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
