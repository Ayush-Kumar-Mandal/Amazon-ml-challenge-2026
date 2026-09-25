import numpy as np
import polars as pl

from ber.models.gbm import load_booster, predict, train_binary


def test_train_predict_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    X = pl.DataFrame({"a": rng.random(2000), "b": rng.random(2000), "flag": rng.random(2000) > 0.5})
    y = (X["a"].to_numpy() > 0.5).astype(int)
    bst = train_binary(X[:1500], y[:1500], X[1500:], y[1500:], {"min_data_in_leaf": 5, "num_leaves": 7}, 200, 20)
    p = predict(bst, X[1500:])
    assert ((p > 0.5) == y[1500:]).mean() > 0.95
    path = tmp_path / "m.txt"
    bst.save_model(str(path))
    assert np.allclose(predict(load_booster(path), X[1500:].select("b", "a", "flag")), p)


from ber.models.gbm import train_combiner


def test_train_combiner_oof_and_outside_rows():
    rng = np.random.default_rng(1)
    n = 3000
    X = pl.DataFrame({"a": rng.random(n), "cross_p": np.where(rng.random(n) < 0.5, np.nan, rng.random(n))})
    y = (X["a"].to_numpy() > 0.5).astype(int)
    groups = rng.integers(0, 600, n)
    train_mask = np.arange(n) < 2500
    oof, boosters = train_combiner(X, y, groups, train_mask, {"min_data_in_leaf": 5, "num_leaves": 7}, 100, 5)
    assert len(boosters) == 5 and oof.shape == (n,)
    assert ((oof[train_mask] > 0.5) == y[train_mask]).mean() > 0.95
    assert ((oof[~train_mask] > 0.5) == y[~train_mask]).mean() > 0.95
