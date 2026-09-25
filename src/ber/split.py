"""Validation folds and the local dev slice."""
from __future__ import annotations

import re

import numpy as np
import polars as pl


def make_folds(s1: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    v = cfg["validation"]
    rng = np.random.default_rng(cfg["seed"])
    draw = rng.random(s1.height) < v["valid_frac"]
    if v["scheme"] == "s1_random":
        fold = np.where(draw, "valid", "fit")
    elif v["scheme"] == "country_holdout":
        country = s1["country"]
        is_train = country.is_in(v["holdout_train_countries"]).to_numpy()
        is_eval = country.is_in(v["holdout_eval_countries"]).to_numpy()
        fold = np.where(is_train, np.where(draw, "valid_seen", "fit"), np.where(is_eval, "valid", "unused"))
    else:
        raise ValueError(f"unknown validation scheme {v['scheme']!r}")
    return pl.DataFrame({"l_idx": s1["idx"], "fold": fold})


def _reindex(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort("idx").rename({"idx": "old_idx"}).with_row_index("idx")


def make_dev_slice(s1: pl.DataFrame, right: pl.DataFrame, gt: pl.DataFrame, localities: list[str]):
    pat = "|".join(re.escape(x.lower()) for x in localities)
    in_area = pl.col("business_address").str.to_lowercase().str.contains(pat)
    s1_sel = s1.filter(in_area)
    gt_sel = gt.join(s1_sel.select(pl.col("idx").alias("l_idx")), on="l_idx", how="semi")
    keep = pl.concat([gt_sel.select(pl.col("r_idx").alias("idx")),
                      right.filter(in_area).select("idx")]).unique()
    right_sel = right.join(keep, on="idx", how="semi")
    s1_new, right_new = _reindex(s1_sel), _reindex(right_sel)
    gt_new = (
        gt_sel.join(s1_new.select(pl.col("old_idx").alias("l_idx"), pl.col("idx").alias("l_new")), on="l_idx")
        .join(right_new.select(pl.col("old_idx").alias("r_idx"), pl.col("idx").alias("r_new")), on="r_idx")
        .select(pl.col("l_new").alias("l_idx"), pl.col("r_new").alias("r_idx"))
    )
    return s1_new.drop("old_idx"), right_new.drop("old_idx"), gt_new
