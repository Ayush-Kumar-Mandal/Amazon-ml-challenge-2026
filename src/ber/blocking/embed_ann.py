"""Exact inner-product k-NN over L2-normalized embeddings (cosine), per country, both directions.
GPU: FAISS GPU if available, else a chunked torch matmul fallback. CPU: FAISS IndexFlatIP.
"""
from __future__ import annotations

import numpy as np
import polars as pl

_SCHEMA = {"l_idx": pl.UInt32, "r_idx": pl.UInt32, "embed_cos": pl.Float32}


def _knn_faiss(index_vecs, query_vecs, k: int, gpu: bool, add_chunk: int = 1_000_000, batch: int = 65_536):
    import faiss

    d = index_vecs.shape[1]
    if gpu:
        res = faiss.StandardGpuResources()
        conf = faiss.GpuIndexFlatConfig()
        conf.useFloat16 = True
        index = faiss.GpuIndexFlatIP(res, d, conf)
    else:
        index = faiss.IndexFlatIP(d)
    for s in range(0, len(index_vecs), add_chunk):
        index.add(np.ascontiguousarray(index_vecs[s:s + add_chunk], dtype=np.float32))
    dists, ids = [], []
    for s in range(0, len(query_vecs), batch):
        d_, i_ = index.search(np.ascontiguousarray(query_vecs[s:s + batch], dtype=np.float32), k)
        dists.append(d_)
        ids.append(i_)
    return np.vstack(dists), np.vstack(ids)


def _knn_torch(index_vecs, query_vecs, k: int, batch: int = 1024, chunk: int = 1_000_000):
    import torch

    xs = [torch.from_numpy(np.ascontiguousarray(index_vecs[s:s + chunk], dtype=np.float16)).cuda()
          for s in range(0, len(index_vecs), chunk)]
    d_out = np.empty((len(query_vecs), k), np.float32)
    i_out = np.empty((len(query_vecs), k), np.int64)
    with torch.no_grad():
        for s in range(0, len(query_vecs), batch):
            q = torch.from_numpy(np.ascontiguousarray(query_vecs[s:s + batch], dtype=np.float16)).cuda()
            best_d = best_i = None
            for ci, x in enumerate(xs):
                d_, i_ = (q @ x.T).float().topk(min(k, x.shape[0]), dim=1)
                i_ = i_ + ci * chunk
                if best_d is None:
                    best_d, best_i = d_, i_
                else:
                    cat_d, cat_i = torch.cat([best_d, d_], 1), torch.cat([best_i, i_], 1)
                    best_d, pos = cat_d.topk(k, dim=1)
                    best_i = cat_i.gather(1, pos)
            d_out[s:s + len(q)] = best_d.cpu().numpy()
            i_out[s:s + len(q)] = best_i.cpu().numpy()
    return d_out, i_out


def knn(index_vecs, query_vecs, k: int, use_gpu: bool):
    k = min(k, len(index_vecs))
    if not use_gpu:
        return _knn_faiss(index_vecs, query_vecs, k, gpu=False)
    try:
        import faiss
        if faiss.get_num_gpus() > 0:
            return _knn_faiss(index_vecs, query_vecs, k, gpu=True)
    except ImportError:
        pass
    return _knn_torch(index_vecs, query_vecs, k)


def embed_candidates(left_meta: pl.DataFrame, right_meta: pl.DataFrame, emb_l, emb_r, k_fwd: int, k_rev: int,
                     same_country: bool, use_gpu: bool) -> pl.DataFrame:
    groups = (sorted(set(left_meta["country"].unique()) | set(right_meta["country"].unique()))
              if same_country else [None])
    frames = []
    for country in groups:
        lm = left_meta if country is None else left_meta.filter(pl.col("country") == country)
        rm = right_meta if country is None else right_meta.filter(pl.col("country") == country)
        li, ri = lm["idx"].to_numpy(), rm["idx"].to_numpy()
        if len(li) == 0 or len(ri) == 0:
            continue
        el, er = np.asarray(emb_l[li]), np.asarray(emb_r[ri])
        d1, i1 = knn(er, el, k_fwd, use_gpu)
        d2, i2 = knn(el, er, k_rev, use_gpu)
        ok1, ok2 = i1 >= 0, i2 >= 0
        l1 = np.broadcast_to(li[:, None], i1.shape)[ok1]
        r1 = ri[i1[ok1]]
        l2 = li[i2[ok2]]
        r2 = np.broadcast_to(ri[:, None], i2.shape)[ok2]
        frames.append(pl.DataFrame({"l_idx": np.concatenate([l1, l2]), "r_idx": np.concatenate([r1, r2]),
                                    "embed_cos": np.concatenate([d1[ok1], d2[ok2]]).astype(np.float32)},
                                   schema=_SCHEMA))
        print(f"[embed_ann] {country}: L={len(li):,} R={len(ri):,} pairs={frames[-1].height:,}")
    if not frames:
        return pl.DataFrame(schema=_SCHEMA)
    return pl.concat(frames).group_by("l_idx", "r_idx").agg(pl.col("embed_cos").max())


def pair_cosine(emb_l, emb_r, li: np.ndarray, ri: np.ndarray, chunk: int = 500_000) -> np.ndarray:
    out = np.empty(len(li), np.float32)
    for s in range(0, len(li), chunk):
        a = np.asarray(emb_l[li[s:s + chunk]], dtype=np.float32)
        b = np.asarray(emb_r[ri[s:s + chunk]], dtype=np.float32)
        out[s:s + chunk] = (a * b).sum(axis=1)
    return out
