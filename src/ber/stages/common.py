"""Helpers shared by stage functions."""
from __future__ import annotations

import numpy as np
import polars as pl

from ber.config import read_path, write_path


def load(cfg: dict, split: str, name: str) -> pl.DataFrame:
    return pl.read_parquet(read_path(cfg, split, name))


def save(df: pl.DataFrame, cfg: dict, split: str, name: str) -> None:
    df.write_parquet(write_path(cfg, split, name))


def load_norm(cfg: dict, split: str, side: str) -> pl.DataFrame:
    df = load(cfg, split, f"{side}_norm.parquet")
    if not np.array_equal(df["idx"].to_numpy(), np.arange(df.height)):
        raise ValueError(f"{split}/{side}_norm.parquet: idx must equal row position")
    return df


def sample_l(folds: pl.DataFrame, fold: str, n: int | None, seed: int) -> pl.DataFrame:
    ids = folds.filter(pl.col("fold") == fold).select("l_idx")
    return ids.sample(n, seed=seed) if n and ids.height > n else ids


def eval_folds(cfg: dict) -> list[str]:
    return ["valid", "valid_seen"] if cfg["validation"]["scheme"] == "country_holdout" else ["valid"]


def label_pairs(cands: pl.DataFrame, gt: pl.DataFrame) -> np.ndarray:
    lab = (
        cands.select("l_idx", "r_idx").with_row_index("_row")
        .join(gt.select("l_idx", "r_idx").with_columns(y=pl.lit(1, pl.Int8)), on=["l_idx", "r_idx"], how="left")
        .sort("_row")
    )
    return lab["y"].fill_null(0).to_numpy()


def es_fold(cfg: dict) -> str:
    return "valid_seen" if cfg["validation"]["scheme"] == "country_holdout" else "valid"
