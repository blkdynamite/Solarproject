import numpy as np
import pandas as pd
import pytest

from der_mv import estimators as es


def _values(
    n_t: int = 400, n_c: int = 100, model_bias: float = 0.3, effect: float = 1.0, seed: int = 0
) -> pd.DataFrame:
    """Synthetic device values with a known effect and a known model bias."""
    rng = np.random.default_rng(seed)
    n = n_t + n_c
    counterfactual = rng.normal(3.0, 0.8, n)
    grp = np.array(["treatment"] * n_t + ["control"] * n_c)
    actual = counterfactual - np.where(grp == "treatment", effect, 0.0)
    pred = counterfactual + model_bias + rng.normal(0, 0.1, n)
    return pd.DataFrame(
        {
            "analysis_id": 1,
            "kind": "event",
            "device_id": np.arange(n),
            "grp": grp,
            "actual_kw": actual,
            "v_10of10": np.nan,
            "v_10of10_adj": np.nan,
            "v_ml": pred - actual,
            "v_rct": -actual,
        }
    )


def test_rct_is_difference_in_means() -> None:
    v = _values()
    est = es.estimate(v, n_boot=100, seed=0).point.loc[1, "rct"]
    t, c = v[v["grp"] == "treatment"], v[v["grp"] == "control"]
    assert est == pytest.approx(c["actual_kw"].mean() - t["actual_kw"].mean())


def test_control_group_cancels_model_bias() -> None:
    v = _values(model_bias=0.5)
    point = es.estimate(v, n_boot=100, seed=0).point.loc[1]
    assert point["ml"] == pytest.approx(1.5, abs=0.05)  # biased by the model error
    assert point["ml_control"] == pytest.approx(1.0, abs=0.05)  # bias removed


def test_ml_adjustment_shrinks_the_confidence_interval() -> None:
    est = es.estimate(_values(), n_boot=400, seed=1)
    width = {
        k: np.diff(np.quantile(est.replicates[k][:, 0], [0.05, 0.95]))[0]
        for k in ("rct", "ml_control")
    }
    assert width["ml_control"] < width["rct"] / 3


def test_bootstrap_interval_is_calibrated() -> None:
    covered = []
    for seed in range(40):
        est = es.estimate(_values(seed=seed), n_boot=300, seed=seed)
        lo, hi = np.nanquantile(est.replicates["rct"][:, 0], [0.05, 0.95])
        covered.append(lo <= 1.0 <= hi)
    assert 0.75 <= np.mean(covered) <= 1.0
