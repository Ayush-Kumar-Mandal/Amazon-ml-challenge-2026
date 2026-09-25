"""LightGBM helpers (MIT license)."""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

DEFAULT_PARAMS = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 127, "min_data_in_leaf": 200,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "metric": ["binary_logloss", "auc"], "verbose": -1, "num_threads": 0, "seed": 42,
}


def to_matrix(X: pl.DataFrame) -> np.ndarray:
    return X.select(pl.all().cast(pl.Float32)).to_numpy()


def train_binary(X_tr: pl.DataFrame, y_tr, X_va: pl.DataFrame, y_va, params: dict | None,
                 num_boost_round: int, early_stopping_rounds: int) -> lgb.Booster:
    dtr = lgb.Dataset(to_matrix(X_tr), label=y_tr, feature_name=list(X_tr.columns), free_raw_data=True)
    dva = lgb.Dataset(to_matrix(X_va.select(X_tr.columns)), label=y_va, reference=dtr)
    return lgb.train(
        {**DEFAULT_PARAMS, **(params or {})}, dtr, num_boost_round=num_boost_round, valid_sets=[dva],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=True, first_metric_only=True),
                   lgb.log_evaluation(100)],
    )


def predict(booster: lgb.Booster, X: pl.DataFrame) -> np.ndarray:
    return booster.predict(to_matrix(X.select(booster.feature_name())))


def load_booster(path: str | Path) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))
