"""Command-line interface.

solar-rooftop run [--config FILE] [--out DIR] [--seed N] [--no-plots]
solar-rooftop evaluate-detector [--config FILE] [--seeds N]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

from .config import ConfigError, load_config
from .pipeline import evaluate_detector, run

DEFAULT_CONFIG = Path("config/greensboro_300kw.toml")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="solar-rooftop",
        description="Commercial rooftop PV: energy yield, financials and performance monitoring.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="command", required=True)

    def add_config(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "-c",
            "--config",
            type=Path,
            default=None,
            help=f"TOML config (default: {DEFAULT_CONFIG} if present, else built-in defaults)",
        )

    r = sub.add_parser("run", help="run the full analysis and write results")
    add_config(r)
    r.add_argument("-o", "--out", type=Path, default=Path("results"), help="output directory")
    r.add_argument("--seed", type=int, default=None, help="override monitoring.seed")
    r.add_argument("--no-plots", action="store_true", help="skip chart rendering")

    e = sub.add_parser("evaluate-detector", help="Monte Carlo test of the fault detector")
    add_config(e)
    e.add_argument("--seeds", type=int, default=50, help="number of random realisations")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    config_path = args.config or (DEFAULT_CONFIG if DEFAULT_CONFIG.is_file() else None)
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        logging.error("Invalid configuration: %s", exc)
        return 2

    if args.command == "run":
        if args.seed is not None:
            cfg = replace(cfg, monitoring=replace(cfg.monitoring, seed=args.seed))
        summary = run(cfg, args.out, make_plots=not args.no_plots)
        print(json.dumps({k: summary[k] for k in ("energy", "finance")}, indent=2, default=str))
        print(json.dumps(summary["monitoring"]["detection_score"], indent=2, default=str))
        logging.info("Results written to %s", args.out.resolve())
    elif args.command == "evaluate-detector":
        df = evaluate_detector(cfg, args.seeds)
        print(df.describe().loc[["mean", "min", "max"]].round(2).T.to_string())
        print(
            f"\nRealisations with any false-positive day: "
            f"{int((df['false_positive_days'] > 0).sum())}/{len(df)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
