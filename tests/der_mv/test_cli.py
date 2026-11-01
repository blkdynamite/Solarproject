import json
from pathlib import Path

from der_mv.cli import main


def test_cli_end_to_end(tmp_path: Path) -> None:
    out = tmp_path / "fleet"
    assert main(["run", "--homes", "300", "--out", str(out)]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary["methods"]) == {"ten_of_ten", "ten_of_ten_adj", "ml", "ml_control", "rct"}
    report = (out / "REPORT.md").read_text()
    assert "Bottom line" in report and "{" not in report.split("## Bottom line")[1][:2000]
    for name in ("1_event_day_profile.png", "5_holdout_tradeoff.png", "estimates_by_event.csv"):
        assert (out / name).is_file()


def test_cli_rejects_bad_config(tmp_path: Path) -> None:
    p = tmp_path / "bad.toml"
    p.write_text("[fleet]\nholdout_share = 0.95\n")
    assert main(["run", "--config", str(p), "--out", str(tmp_path)]) == 2
