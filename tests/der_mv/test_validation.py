import json
from dataclasses import replace

from der_mv import report
from der_mv.config import StudyConfig
from der_mv.pipeline import validate, validation_summary


def test_multi_fleet_validation_and_report(small_cfg: StudyConfig) -> None:
    cfg = replace(small_cfg, fleet=replace(small_cfg.fleet, n_homes=250, n_events=5))
    runs = validate(cfg, n_runs=2)
    assert runs["run"].nunique() == 2
    assert set(runs["method"]) == {"ten_of_ten", "ten_of_ten_adj", "ml", "ml_control", "rct"}

    vs = validation_summary(runs)
    assert vs.attrs["n_runs"] == 2
    assert abs(vs.loc["ml_control", "mean_error_pct"]) < 5
    assert vs.loc["ten_of_ten", "mean_error_pct"] < -30

    # The memo's validation section renders from these numbers.
    section = report._validation_text(
        {"validation": {"n_runs": 2, "methods": json.loads(vs.to_json(orient="index"))}},
        cfg.mv.confidence,
    )[2]
    assert "Validation across 2 independent fleets" in section
    assert "{" not in section
