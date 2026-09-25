import polars as pl
import pytest

pytest.importorskip("sparse_dot_topn")
from ber.blocking.tfidf_ann import tfidf_candidates  # noqa: E402

TCFG = {"ngram": 3, "min_df": 1, "max_df_frac": 1.0, "max_df_abs": 10**9, "top_k_fwd": 1, "top_k_rev": 1,
        "threshold": 0.1, "chunk_size": 1, "n_threads": 1}


def _frame(countries, texts):
    return pl.DataFrame({"idx": list(range(len(texts))), "country": countries, "_text": texts},
                        schema_overrides={"idx": pl.UInt32})


def test_finds_best_match_in_both_directions_within_country():
    left = _frame(["US", "US"], ["sunrise bakery", "delta motors"])
    right = _frame(["US", "US", "France"], ["delta motor", "sunrise bakeries", "sunrise bakery"])
    out = tfidf_candidates(left, right, "_text", "tfidf_name", TCFG, same_country=True)
    pairs = set(zip(out["l_idx"].to_list(), out["r_idx"].to_list()))
    assert (0, 1) in pairs and (1, 0) in pairs
    assert (0, 2) not in pairs
    assert out["tfidf_name"].dtype == pl.Float32 and out["tfidf_name"].max() <= 1.0001


def test_reverse_direction_adds_pairs_beyond_forward_top_k():
    left = _frame(["US"], ["acme pizza"])
    right = _frame(["US", "US"], ["acme pizza", "acme pizzas"])
    out = tfidf_candidates(left, right, "_text", "s", TCFG, same_country=True)
    assert out.height == 2


def test_no_terms_after_pruning_returns_empty():
    left = _frame(["US"], ["ab"])
    right = _frame(["US"], ["cd"])
    out = tfidf_candidates(left, right, "_text", "s", {**TCFG, "min_df": 5}, same_country=True)
    assert out.height == 0 and out.columns == ["l_idx", "r_idx", "s"]
