"""ML counterfactual: predict what each home *would* have used without the event.

A single pooled gradient-boosted model is trained on non-analysis days for all
devices. Pooling lets thousands of homes share what they have in common
(weather response, daily shape), while device-fingerprint features keep the
predictions home-specific. The model never sees an event day, so its
predictions there are genuinely out-of-sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

FEATURES = [
    "hod",
    "dow",
    "temp_c",
    "temp_4h",
    "temp_12h",
    "tmax_day",
    "dev_hod_kw",
    "dev_temp_slope",
    "morning_kw",
]
TARGET = "kw"


@dataclass(frozen=True)
class ModelReport:
    n_train_rows: int
    n_validation_days: int
    hourly_cv_rmse_pct: float  # device-hour level, event-window hours, held-out days
    fleet_bias_pct: float  # fleet-average error on held-out days, event-window hours
    feature_importance: dict[str, float]


def train(
    con: duckdb.DuckDBPyConnection,
    *,
    window_hours: tuple[int, int],
    sample_rows: int = 600_000,
    validation_share: float = 0.15,
    seed: int = 0,
) -> tuple[HistGradientBoostingRegressor, ModelReport]:
    """Fit on a sample of non-analysis days; validate on held-out *days*.

    Splitting by day (not by row) matters: rows from the same day share weather
    and behaviour, so a random row split would leak and flatter the model.
    """
    # Deterministic hash sampling (DuckDB's reservoir sampling is not reproducible
    # under multi-threading) and a fixed row order, so every run trains the same model.
    n_rows = con.execute("SELECT count(*) FROM mart_ml_features WHERE NOT is_analysis_day")
    total = int(n_rows.fetchone()[0])  # type: ignore[index]
    keep_per_million = min(1_000_000, int(1_000_000 * sample_rows / max(total, 1)))
    df = con.execute(
        f"""
        SELECT {", ".join(FEATURES)}, {TARGET}, day
        FROM mart_ml_features
        WHERE NOT is_analysis_day
          AND hash(device_id, ts, {int(seed)}) % 1000000 < {keep_per_million}
        ORDER BY device_id, ts
        """
    ).df()
    rng = np.random.default_rng(seed)
    days = np.array(sorted(df["day"].unique()))
    val_days = set(rng.choice(days, max(1, round(len(days) * validation_share)), replace=False))
    is_val = df["day"].isin(val_days).to_numpy()

    model = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.06,
        max_leaf_nodes=63,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=seed,
    )
    model.fit(df.loc[~is_val, FEATURES], df.loc[~is_val, TARGET])

    val = df.loc[is_val & df["hod"].between(window_hours[0], window_hours[1] - 1)].copy()
    val["pred"] = model.predict(val[FEATURES])
    err = val["pred"] - val[TARGET]
    report = ModelReport(
        n_train_rows=int((~is_val).sum()),
        n_validation_days=len(val_days),
        hourly_cv_rmse_pct=round(float(100 * np.sqrt((err**2).mean()) / val[TARGET].mean()), 1),
        fleet_bias_pct=round(float(100 * err.mean() / val[TARGET].mean()), 2),
        feature_importance=_permutation_importance(model, val, seed),
    )
    return model, report


def predict_analysis_days(
    con: duckdb.DuckDBPyConnection, model: HistGradientBoostingRegressor
) -> pd.DataFrame:
    """Hourly counterfactual predictions for every device on every analysis day."""
    df = con.execute(
        f"""
        SELECT a.analysis_id, f.device_id, f.ts, f.hod, {", ".join(FEATURES)}
        FROM mart_ml_features f
        JOIN int_analysis_days a ON a.day = f.day
        """
    ).df()
    df["pred_kw"] = model.predict(df[FEATURES])
    return df[["analysis_id", "device_id", "ts", "hod", "pred_kw"]]


def _permutation_importance(
    model: HistGradientBoostingRegressor, val: pd.DataFrame, seed: int, max_rows: int = 50_000
) -> dict[str, float]:
    """Increase in RMSE when each feature is shuffled (share of total increase)."""
    rng = np.random.default_rng(seed)
    sample = val.sample(min(len(val), max_rows), random_state=seed)
    x, y = sample[FEATURES].copy(), sample[TARGET].to_numpy()
    base = np.sqrt(np.mean((model.predict(x) - y) ** 2))
    gains = {}
    for col in FEATURES:
        shuffled = x.copy()
        shuffled[col] = rng.permutation(shuffled[col].to_numpy())
        gains[col] = max(0.0, float(np.sqrt(np.mean((model.predict(shuffled) - y) ** 2)) - base))
    total = sum(gains.values()) or 1.0
    return {k: round(v / total, 3) for k, v in sorted(gains.items(), key=lambda kv: -kv[1])}
