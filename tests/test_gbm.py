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
