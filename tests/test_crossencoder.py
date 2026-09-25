import numpy as np
import polars as pl

from ber.models.crossencoder import band_mask, pair_text, sample_training_pairs


def test_pair_text_and_band():
    df = pl.DataFrame({"business_name": ["Acme"], "business_address": ["Tyler"]})
    assert pair_text(df).to_list() == ["Acme | Tyler"]
    assert band_mask(np.array([0.05, 0.1, 0.5, 0.9, 0.95]), 0.1, 0.9).tolist() == [False, True, True, True, False]


def test_sample_training_pairs_mix():
    rng = np.random.default_rng(0)
    fp = pl.DataFrame({"l_idx": np.arange(1000, dtype=np.uint32), "r_idx": np.arange(1000, dtype=np.uint32),
                       "p": rng.random(1000), "y": rng.integers(0, 2, 1000)})
    out = sample_training_pairs(fp, 0.1, 0.9, n=100, seed=0)
    in_band = out.join(fp, on=["l_idx", "r_idx"]).filter(pl.col("p").is_between(0.1, 0.9)).height
    assert out.height == 100 and in_band == 80 and out.columns == ["l_idx", "r_idx", "y"]
