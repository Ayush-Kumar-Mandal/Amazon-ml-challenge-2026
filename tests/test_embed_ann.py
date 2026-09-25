import numpy as np
import polars as pl
import pytest

pytest.importorskip("faiss")
from ber.blocking.embed_ann import embed_candidates, knn, pair_cosine  # noqa: E402


def _unit(rng, n, d=16):
    v = rng.normal(size=(n, d)).astype(np.float32)
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float16)


def test_knn_cpu_finds_identical_vectors():
    rng = np.random.default_rng(0)
    base = _unit(rng, 50)
    D, I = knn(base, base[[3, 7]], k=2, use_gpu=False)
    assert I[:, 0].tolist() == [3, 7] and np.allclose(D[:, 0], 1.0, atol=1e-2)


def test_embed_candidates_within_country_both_directions():
    rng = np.random.default_rng(1)
    emb_l = _unit(rng, 4)
    emb_r = np.concatenate([emb_l[[2, 0]], _unit(rng, 2)])  # r0==l2, r1==l0
    lm = pl.DataFrame({"idx": [0, 1, 2, 3], "country": ["US", "US", "US", "France"]}, schema_overrides={"idx": pl.UInt32})
    rm = pl.DataFrame({"idx": [0, 1, 2, 3], "country": ["US", "US", "France", "US"]}, schema_overrides={"idx": pl.UInt32})
    out = embed_candidates(lm, rm, emb_l, emb_r, k_fwd=1, k_rev=1, same_country=True, use_gpu=False)
    pairs = set(zip(out["l_idx"].to_list(), out["r_idx"].to_list()))
    assert {(2, 0), (0, 1)} <= pairs
    assert all(not (l == 3 and r in (0, 1, 3)) for l, r in pairs)  # France S1 only sees France right
    assert out["embed_cos"].dtype == pl.Float32


def test_pair_cosine():
    rng = np.random.default_rng(2)
    a, b = _unit(rng, 5), _unit(rng, 6)
    got = pair_cosine(a, b, np.array([0, 4]), np.array([5, 1]))
    want = [float(a[0].astype(np.float32) @ b[5].astype(np.float32)), float(a[4].astype(np.float32) @ b[1].astype(np.float32))]
    assert np.allclose(got, want, atol=1e-3)
