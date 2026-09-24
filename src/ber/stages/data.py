"""Data stages: ingest, dev slice, folds, lexicon, normalize."""
from __future__ import annotations

import json

import polars as pl

from ber.config import write_path
from ber.io import ingest
from ber.lexicon import mine_lexicon
from ber.split import make_dev_slice, make_folds
from ber.stages.common import load, save


def stage_ingest(cfg: dict, split: str) -> None:
    ingest(cfg, split)


def stage_dev_slice(cfg: dict, split: str) -> None:
    s1, right, gt = make_dev_slice(load(cfg, "train", "s1.parquet"), load(cfg, "train", "right.parquet"),
                                   load(cfg, "train", "gt.parquet"), cfg["dev"]["localities"])
    save(s1, cfg, "dev", "s1.parquet")
    save(right, cfg, "dev", "right.parquet")
    save(gt, cfg, "dev", "gt.parquet")
    print(f"[dev_slice] s1={s1.height:,} right={right.height:,} gt={gt.height:,}")


def stage_split(cfg: dict, split: str) -> None:
    folds = make_folds(load(cfg, split, "s1.parquet"), cfg)
    save(folds, cfg, split, "folds.parquet")
    counts = folds.group_by("fold").len().sort("fold")
    print("[split] " + ", ".join(f"{r['fold']}={r['len']:,}" for r in counts.to_dicts()))


def stage_lexicon(cfg: dict, split: str) -> None:
    s1, right = load(cfg, split, "s1.parquet"), load(cfg, split, "right.parquet")
    fit = load(cfg, split, "folds.parquet").filter(pl.col("fold") == "fit").select("l_idx")
    pairs = load(cfg, split, "gt.parquet").join(fit, on="l_idx", how="semi")
    n = cfg["lexicon"]["sample_pairs"]
    if pairs.height > n:
        pairs = pairs.sample(n, seed=cfg["seed"])
    pairs = pairs.join(
        s1.select(pl.col("idx").alias("l_idx"), pl.col("business_name").alias("l_name"),
                  pl.col("business_address").alias("l_addr")), on="l_idx",
    ).join(
        right.select(pl.col("idx").alias("r_idx"), pl.col("business_name").alias("r_name"),
                     pl.col("business_address").alias("r_addr")), on="r_idx",
    )
    lc = cfg["lexicon"]
    lex = mine_lexicon(pairs, lc["min_count"], lc["abbrev_min_count"], lc["min_share"])
    write_path(cfg, split, "lexicon.json").write_text(
        json.dumps(lex, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    print(f"[lexicon] name={len(lex['name'])} addr={len(lex['addr'])} entries")
