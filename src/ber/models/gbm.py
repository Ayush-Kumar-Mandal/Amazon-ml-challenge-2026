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


def train_combiner(X: pl.DataFrame, y: np.ndarray, groups: np.ndarray, train_mask: np.ndarray, params: dict | None,
                   num_boost_round: int, n_folds: int) -> tuple[np.ndarray, list[lgb.Booster]]:
    """GroupKFold (by S1) out-of-fold predictions on train_mask rows; fold-mean on the rest.
    Fixed num_boost_round (no early stopping) so OOF predictions stay unbiased."""
    from sklearn.model_selection import GroupKFold

    m = to_matrix(X)
    oof = np.zeros(X.height, np.float32)
    idx_tr, idx_out = np.flatnonzero(train_mask), np.flatnonzero(~train_mask)
    boosters = []
    for a, b in GroupKFold(n_splits=n_folds).split(idx_tr, groups=groups[idx_tr]):
        tr, va = idx_tr[a], idx_tr[b]
        bst = lgb.train({**DEFAULT_PARAMS, **(params or {})},
                        lgb.Dataset(m[tr], label=y[tr], feature_name=list(X.columns)), num_boost_round=num_boost_round)
        oof[va] = bst.predict(m[va])
        boosters.append(bst)
    if len(idx_out):
        oof[idx_out] = np.mean([bst.predict(m[idx_out]) for bst in boosters], axis=0)
    return oof, boosters
