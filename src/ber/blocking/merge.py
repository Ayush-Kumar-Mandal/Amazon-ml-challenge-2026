"""Union of candidate sources and per-S1 top-k cut."""
from __future__ import annotations

import polars as pl

SCORE_COLUMNS = ["tfidf_name", "tfidf_name_addr", "embed_cos"]


def merge_candidates(frames: list[pl.DataFrame]) -> pl.DataFrame:
    df = pl.concat([f for f in frames if f.height > 0] or frames[:1], how="diagonal_relaxed")
    extra = [c for c in df.columns if c not in ("l_idx", "r_idx")]
    out = df.group_by("l_idx", "r_idx").agg([pl.col(c).max() for c in extra])
    fills = [
        (pl.col(c).fill_null(0.0) if c in out.columns else pl.lit(0.0)).cast(pl.Float32).alias(c)
        for c in SCORE_COLUMNS
    ]
    keys = (pl.col("from_keys").fill_null(False) if "from_keys" in out.columns else pl.lit(False)).alias("from_keys")
    return out.select("l_idx", "r_idx", *fills, keys)


def top_k_per_left(df: pl.DataFrame, score_col: str, k: int) -> pl.DataFrame:
    return df.filter(pl.col(score_col).rank("ordinal", descending=True).over("l_idx") <= k)
