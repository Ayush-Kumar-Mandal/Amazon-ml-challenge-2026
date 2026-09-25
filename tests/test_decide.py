import numpy as np
import polars as pl

from ber.decide import (choose_expected_f05, choose_threshold, crossfit_calibrate, one_owner,
                        scale_unseen, select, tune_threshold)

U = pl.UInt32


def _pred(rows):
    return pl.DataFrame(rows, schema={"l_idx": U, "r_idx": U, "p": pl.Float64}, orient="row")


def _pairs(df):
    return set(zip(df["l_idx"].to_list(), df["r_idx"].to_list()))


def test_expected_f05_picks_k_per_entity():
    pred = _pred([(0, 1, 0.95), (0, 2, 0.9), (0, 3, 0.1), (1, 4, 0.05), (1, 5, 0.02), (2, 6, 0.6)])
    assert _pairs(choose_expected_f05(pred)) == {(0, 1), (0, 2), (2, 6)}


def test_one_owner_keeps_best_claim():
    assert _pairs(one_owner(_pred([(0, 5, 0.9), (1, 5, 0.7), (1, 6, 0.8)]))) == {(0, 5), (1, 6)}


def test_threshold_tuning():
    pred = _pred([(0, 1, 0.8), (0, 2, 0.4), (1, 3, 0.3)])
    gt = pl.DataFrame({"l_idx": [0], "r_idx": [1]}, schema={"l_idx": U, "r_idx": U})
    t, f = tune_threshold(pred, gt, pl.Series("l_idx", [0, 1], dtype=U))
    assert f == 1.0 and 0.4 < t <= 0.8
    assert _pairs(choose_threshold(pred, 0.5)) == {(0, 1)}
    assert _pairs(select(pred, {"method": "threshold", "threshold": 0.5})) == {(0, 1)}


def test_crossfit_calibration_bounded_and_calibrated():
    rng = np.random.default_rng(0)
    n = 4000
    p = rng.random(n)
    y = (rng.random(n) < p ** 2).astype(int)
    pred = pl.DataFrame({"l_idx": rng.integers(0, 500, n).astype(np.uint32),
                         "r_idx": np.arange(n, dtype=np.uint32), "p": p, "y": y})
    cal = crossfit_calibrate(pred, seed=0)
    assert cal.min() >= 0 and cal.max() <= 1 and abs(cal.mean() - y.mean()) < 0.03


def test_scale_unseen_only_touches_unseen_countries():
    pred = _pred([(0, 1, 0.5), (1, 2, 0.5)])
    s1c = pl.DataFrame({"l_idx": [0, 1], "country": ["US", "France"]}, schema_overrides={"l_idx": U})
    out = scale_unseen(pred, s1c, ["US", "India"], 1.2).sort("l_idx")
    assert out["p"].to_list() == [0.5, 0.6] and out.columns == pred.columns
