"""Pair features. Country-agnostic: country is never a feature (script type is)."""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

CHEAP_COLUMNS = ["tfidf_name", "tfidf_name_addr", "embed_cos", "from_keys", "name_jw", "name_tset",
                 "addr_tset", "postcode_eq", "house_eq", "source_s3"]


def attach_sides(c: pl.DataFrame, left: pl.DataFrame, right: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    lft = left.select(pl.col(cols).gather(c["l_idx"])).rename({x: f"l_{x}" for x in cols})
    rcols = cols + ["source"]
    rgt = right.select(pl.col(rcols).gather(c["r_idx"])).rename({x: f"r_{x}" for x in rcols})
    return pl.concat([c, lft, rgt], how="horizontal_extend")


def sim(a: pl.Series, b: pl.Series, scorer) -> np.ndarray:
    return process.cpdist(a.to_list(), b.to_list(), scorer=scorer, workers=-1, dtype=np.float32)


def tri(a: str, b: str) -> pl.Expr:
    """-1 if either side missing, 1 if equal, 0 if conflicting."""
    return (pl.when((pl.col(a) == "") | (pl.col(b) == "")).then(-1)
            .when(pl.col(a) == pl.col(b)).then(1).otherwise(0).cast(pl.Int8))


def cheap_features(c: pl.DataFrame, left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
    x = attach_sides(c, left, right, ["name_core", "addr_norm", "postcode", "house_no"])
    x = x.with_columns(
        name_jw=pl.Series(sim(x["l_name_core"], x["r_name_core"], JaroWinkler.normalized_similarity)),
        name_tset=pl.Series(sim(x["l_name_core"], x["r_name_core"], fuzz.token_set_ratio)),
        addr_tset=pl.Series(sim(x["l_addr_norm"], x["r_addr_norm"], fuzz.token_set_ratio)),
        postcode_eq=tri("l_postcode", "r_postcode"),
        house_eq=tri("l_house_no", "r_house_no"),
        source_s3=(pl.col("r_source") == "S3").cast(pl.Int8),
    )
    return x.select(c.columns + [col for col in CHEAP_COLUMNS if col not in c.columns])
