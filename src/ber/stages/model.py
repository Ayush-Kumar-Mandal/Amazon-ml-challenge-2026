"""Model stages: features, train, evaluate, predict."""
from __future__ import annotations

import polars as pl

from ber.config import read_path, write_path
from ber.features import context_features, fit_vectorizers, full_features
from ber.stages.common import eval_folds, label_pairs, load, load_norm, sample_l


def load_features(cfg: dict, split: str) -> pl.LazyFrame:
    return pl.scan_parquet(str(read_path(cfg, split, "features") / "*.parquet"))


def stage_features(cfg: dict, split: str) -> None:
    left, right = load_norm(cfg, split, "s1"), load_norm(cfg, split, "right")
    cands = context_features(load(cfg, split, "cands_final.parquet"))  # context on the FULL candidate set
    if split != "test":
        folds = load(cfg, split, "folds.parquet")
        keep = pl.concat([sample_l(folds, "fit", cfg["features"]["max_train_s1"], cfg["seed"])]
                         + [folds.filter(pl.col("fold") == f).select("l_idx") for f in eval_folds(cfg)])
        cands = cands.join(keep, on="l_idx", how="semi")
        cands = cands.with_columns(y=pl.Series(label_pairs(cands, load(cfg, split, "gt.parquet"))))
    vecs = fit_vectorizers(left, right, cfg["features"]["vectorizer_fit_rows"], cfg["seed"])
    out_dir = write_path(cfg, split, "features/.keep").parent
    for old in out_dir.glob("part-*.parquet"):
        old.unlink()
    step = cfg["features"]["chunk_rows"]
    for n, s in enumerate(range(0, cands.height, step)):
        chunk = cands.slice(s, step)
        feats = full_features(chunk, left, right, vecs)
        if "y" in chunk.columns:
            feats = feats.with_columns(y=chunk["y"])
        feats.write_parquet(out_dir / f"part-{n:04d}.parquet")
        print(f"[features] {split}: part {n} rows={feats.height:,}")
