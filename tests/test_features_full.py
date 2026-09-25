import numpy as np
import pytest

from ber.features import FEATURE_COLUMNS, context_features, fit_vectorizers, full_features
from tests.test_features_cheap import CANDS, LEFT, RIGHT


def test_context_features():
    out = context_features(CANDS).sort("l_idx", "r_idx")
    assert out["rank_l"].to_list() == [1, 2, 1]
    assert out["n_claims_r"].to_list() == [1, 1, 1]
    assert out["other_r_idx"].to_list() == [1, 0, None]
    assert out["mutual_best"].to_list() == [1, 0, 1]


def test_full_features_schema_and_values():
    vecs = fit_vectorizers(LEFT, RIGHT, fit_rows=100, seed=0)
    out = full_features(context_features(CANDS), LEFT, RIGHT, vecs).sort("l_idx", "r_idx")
    assert out.columns == ["l_idx", "r_idx", *FEATURE_COLUMNS]
    r0 = out.row(0, named=True)
    assert r0["name_ratio"] == 100 and r0["legal_eq"] == 1 and r0["street_eq"] == 1
    assert r0["name_char_cos"] == pytest.approx(1.0, abs=1e-5)
    r2 = out.row(2, named=True)
    assert r2["l_addr_empty"] == 1 and r2["legal_eq"] == -1 and np.isnan(r2["sim_other_name"])
