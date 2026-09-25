"""Char n-gram TF-IDF cosine top-k in both directions (S1->S2/S3 and S2/S3->S1), per country."""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

_SCHEMA = {"l_idx": pl.UInt32, "r_idx": pl.UInt32}


def _topn(a, bt, top_n: int, threshold: float, n_threads: int, chunk: int):
    rows, cols, vals = [], [], []
    for s in range(0, a.shape[0], chunk):
        c = sp_matmul_topn(a[s:s + chunk], bt, top_n=top_n, threshold=threshold,
                           sort=False, n_threads=n_threads).tocoo()
        rows.append(c.row + s)
        cols.append(c.col)
        vals.append(c.data)
    if not rows:
        return np.array([], np.int64), np.array([], np.int64), np.array([], np.float32)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def tfidf_candidates(left: pl.DataFrame, right: pl.DataFrame, text_col: str, score_name: str,
                     tcfg: dict, same_country: bool = True) -> pl.DataFrame:
    schema = {**_SCHEMA, score_name: pl.Float32}
    groups = sorted(set(left["country"].unique()) | set(right["country"].unique())) if same_country else [None]
    frames = []
    for country in groups:
        L = left if country is None else left.filter(pl.col("country") == country)
        R = right if country is None else right.filter(pl.col("country") == country)
        if L.height == 0 or R.height == 0:
            continue
        n = tcfg["ngram"]
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(n, n), min_df=tcfg["min_df"],
                              max_df=tcfg["max_df"], sublinear_tf=True, dtype=np.float32)
        try:
            vec.fit(pl.concat([L[text_col], R[text_col]]).to_list())
        except ValueError:  # no terms left after min_df/max_df pruning
            continue
        a = vec.transform(L[text_col].to_list()).tocsr()
        b = vec.transform(R[text_col].to_list()).tocsr()
        l_ids, r_ids = L["idx"].to_numpy(), R["idx"].to_numpy()
        args = (tcfg["threshold"], tcfg["n_threads"], tcfg["chunk_size"])
        r1, c1, v1 = _topn(a, b.T.tocsr(), tcfg["top_k_fwd"], *args)
        r2, c2, v2 = _topn(b, a.T.tocsr(), tcfg["top_k_rev"], *args)
        frames.append(pl.DataFrame({
            "l_idx": np.concatenate([l_ids[r1], l_ids[c2]]),
            "r_idx": np.concatenate([r_ids[c1], r_ids[r2]]),
            score_name: np.concatenate([v1, v2]).astype(np.float32),
        }, schema=schema))
        print(f"[tfidf:{score_name}] {country}: L={L.height:,} R={R.height:,} pairs={frames[-1].height:,}")
    if not frames:
        return pl.DataFrame(schema=schema)
    return pl.concat(frames).group_by("l_idx", "r_idx").agg(pl.col(score_name).max())
