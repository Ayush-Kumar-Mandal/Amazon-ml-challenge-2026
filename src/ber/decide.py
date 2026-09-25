"""Turn pair probabilities into per-S1 match sets that maximize macro F0.5.

Expected-F0.5 rule (plug-in approximation, independence assumed): for an S1 with candidate
probabilities sorted p1>=p2>=..., choosing the top-k gives E[F] ~= 1.25*S_k / (0.25*S_all + k);
choosing nothing scores 1.0 only if every candidate is a non-match: prod(1 - p_i).
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from ber.metrics import macro_f05_frame

THRESHOLDS = np.round(np.arange(0.2, 0.9, 0.025), 3)
SCALES = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]


def fit_calibrator(p: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)


def calibrate(iso: IsotonicRegression, p: np.ndarray) -> np.ndarray:
    return iso.predict(p).astype(np.float32)


def crossfit_calibrate(pred: pl.DataFrame, seed: int) -> np.ndarray:
    """Calibrate each half of the S1 entities with an isotonic fit on the other half."""
    l_idx = pred["l_idx"].to_numpy().astype(np.int64)
    # Salt the seed: a bare `default_rng(seed)` shares its float-stream prefix with
    # `make_folds`'s own `default_rng(seed)` draw, so with the same seed every "valid"
    # l_idx (drawn at threshold valid_frac) would land in the same half here whenever
    # valid_frac < 0.5, collapsing one half to zero rows.
    half = (np.random.default_rng([seed, 1]).random(l_idx.max() + 1) < 0.5)[l_idx]
    p, y = pred["p"].to_numpy(), pred["y"].to_numpy()
    out = np.empty(len(p), np.float32)
    for h in (True, False):
        out[half == h] = calibrate(fit_calibrator(p[half != h], y[half != h]), p[half == h])
    return out


def one_owner(pred: pl.DataFrame) -> pl.DataFrame:
    return (pred.sort(["r_idx", "p", "l_idx"], descending=[False, True, False])
            .unique(subset="r_idx", keep="first", maintain_order=True))


def choose_threshold(pred: pl.DataFrame, t: float) -> pl.DataFrame:
    return pred.filter(pl.col("p") >= t).select("l_idx", "r_idx")


def choose_expected_f05(pred: pl.DataFrame) -> pl.DataFrame:
    d = (
        pred.select("l_idx", "r_idx", "p").sort(["l_idx", "p"], descending=[False, True])
        .with_columns(
            k=pl.int_range(1, pl.len() + 1).over("l_idx"),
            s_k=pl.col("p").cum_sum().over("l_idx"),
            s_all=pl.col("p").sum().over("l_idx"),
            log_empty=(1 - pl.col("p")).clip(1e-9, 1.0).log().sum().over("l_idx"),
        )
        .with_columns(ef=1.25 * pl.col("s_k") / (0.25 * pl.col("s_all") + pl.col("k")))
    )
    best = d.group_by("l_idx").agg(
        best_k=pl.col("k").sort_by("ef").last(),
        best_ef=pl.col("ef").max(),
        e0=pl.col("log_empty").first().exp(),
    ).with_columns(chosen=pl.when(pl.col("e0") >= pl.col("best_ef")).then(0).otherwise(pl.col("best_k")))
    return (d.join(best.select("l_idx", "chosen"), on="l_idx")
            .filter(pl.col("k") <= pl.col("chosen")).select("l_idx", "r_idx"))


def tune_threshold(pred: pl.DataFrame, gt: pl.DataFrame, eval_l: pl.Series, grid=THRESHOLDS) -> tuple[float, float]:
    best_t, best_f = float(grid[0]), -1.0
    for t in grid:
        f = macro_f05_frame(choose_threshold(pred, float(t)), gt, eval_l)
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t, best_f


def scale_unseen(pred: pl.DataFrame, s1_country: pl.DataFrame, seen: list[str], scale: float) -> pl.DataFrame:
    if scale == 1.0:
        return pred
    return (pred.join(s1_country, on="l_idx", how="left")
            .with_columns(p=pl.when(pl.col("country").is_in(seen)).then(pl.col("p"))
                          .otherwise((pl.col("p") * scale).clip(0.0, 1.0)))
            .select(pred.columns))


def select(pred: pl.DataFrame, decision: dict) -> pl.DataFrame:
    if decision["method"] == "expected_f05":
        return choose_expected_f05(pred)
    return choose_threshold(pred, decision["threshold"])
