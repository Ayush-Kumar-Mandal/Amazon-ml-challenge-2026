import polars as pl

from ber.blocking.merge import merge_candidates, top_k_per_left

U = pl.UInt32


def test_merge_takes_max_and_fills_defaults():
    keys = pl.DataFrame({"l_idx": [0, 1], "r_idx": [5, 6], "from_keys": [True, True]},
                        schema_overrides={"l_idx": U, "r_idx": U})
    tf = pl.DataFrame({"l_idx": [0, 0, 2], "r_idx": [5, 5, 7], "tfidf_name": [0.4, 0.9, 0.5]},
                      schema_overrides={"l_idx": U, "r_idx": U, "tfidf_name": pl.Float32})
    out = merge_candidates([keys, tf]).sort("l_idx", "r_idx")
    assert out.columns == ["l_idx", "r_idx", "tfidf_name", "tfidf_name_addr", "embed_cos", "from_keys"]
    assert out["tfidf_name"].to_list() == [0.8999999761581421, 0.0, 0.5]
    assert out["from_keys"].to_list() == [True, True, False]
    assert out["embed_cos"].to_list() == [0.0, 0.0, 0.0]


def test_top_k_per_left():
    df = pl.DataFrame({"l_idx": [0, 0, 0, 1], "r_idx": [1, 2, 3, 4], "s": [0.1, 0.9, 0.5, 0.2]})
    out = top_k_per_left(df, "s", 2).sort("l_idx", "r_idx")
    assert list(zip(out["l_idx"], out["r_idx"])) == [(0, 2), (0, 3), (1, 4)]
