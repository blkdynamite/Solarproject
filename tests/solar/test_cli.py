import json
from pathlib import Path

from solar_rooftop.cli import main


def test_run_writes_artifacts(tmp_path: Path) -> None:
    out = tmp_path / "results"
    assert main(["run", "--out", str(out)]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["energy"]["annual_mwh_p50"] > 0
    assert {"pretax_no_itc", "pretax_itc", "aftertax_itc_macrs"} <= set(summary["finance"])
    for name in (
        "daily_performance.csv",
        "cash_flows.csv",
        "sensitivity.csv",
        "1_monthly_production.png",
        "6_performance_monitoring.png",
    ):
        assert (out / name).is_file(), name


def test_bad_config_exit_code(tmp_path: Path) -> None:
    p = tmp_path / "bad.toml"
    p.write_text("[system]\ndc_kw = -1\n")
    assert main(["run", "--config", str(p), "--out", str(tmp_path), "--no-plots"]) == 2
