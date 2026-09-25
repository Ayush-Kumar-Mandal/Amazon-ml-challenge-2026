import polars as pl

from ber.models.biencoder import make_training_triplets, record_text

U = pl.UInt32


def test_record_text():
    df = pl.DataFrame({"business_name": ["राम Traders"], "business_address": ["Pune"]})
    assert record_text(df).to_list() == ["query: राम Traders | Pune"]


def test_triplets_use_hardest_non_match_and_fallback():
    left_text = pl.Series(["L0", "L1"])
    right_text = pl.Series(["R0", "R1", "R2", "R3"])
    gt = pl.DataFrame({"l_idx": [0, 1], "r_idx": [0, 2]}, schema={"l_idx": U, "r_idx": U})
    cands = pl.DataFrame({"l_idx": [0, 0, 0], "r_idx": [0, 1, 3], "cheap_score": [0.9, 0.2, 0.7]},
                         schema_overrides={"l_idx": U, "r_idx": U})
    fit_l = pl.DataFrame({"l_idx": [0, 1]}, schema={"l_idx": U})
    t = make_training_triplets(cands, gt, fit_l, left_text, right_text, n=10, seed=0).sort("anchor")
    assert t.columns == ["anchor", "positive", "negative"]
    assert t.row(0) == ("L0", "R0", "R3")          # hardest non-match by cheap_score
    assert t.row(1)[:2] == ("L1", "R2") and t.row(1)[2] in {"R0", "R1", "R2", "R3"}  # random fallback
