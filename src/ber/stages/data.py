"""Data stages: ingest, dev slice, folds, lexicon, normalize."""
from __future__ import annotations

from ber.io import ingest
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
