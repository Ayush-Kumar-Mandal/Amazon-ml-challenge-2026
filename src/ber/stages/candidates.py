"""Candidate stages: block, cheap_train, cheap_apply."""
from __future__ import annotations

import numpy as np
import polars as pl

from ber.blocking.keys import join_keys, record_keys
from ber.blocking.merge import merge_candidates, top_k_per_left
from ber.blocking.tfidf_ann import tfidf_candidates
from ber.config import MODEL_SPLIT, read_path, write_path
from ber.features import CHEAP_COLUMNS, cheap_features
from ber.metrics import blocking_recall, cands_per_s1
from ber.models.gbm import load_booster, predict, train_binary
from ber.stages.common import es_fold, label_pairs, load, load_norm, sample_l, save

VIEWS = {
    "name": pl.col("name_core"),
    "name_addr": pl.concat_str([pl.col("name_core"), pl.col("addr_norm")], separator=" "),
}


def stage_block(cfg: dict, split: str) -> None:
    left, right = load_norm(cfg, split, "s1"), load_norm(cfg, split, "right")
    b = cfg["blocking"]
    same = b["same_country"]
    frames = [join_keys(record_keys(left, same), record_keys(right, same), b["max_block_pairs"])]
    print(f"[block] keys: {frames[0].height:,} pairs")
    for view in b["tfidf"]["views"]:
        text = VIEWS[view].alias("_text")
        frames.append(tfidf_candidates(left.select("idx", "country", text), right.select("idx", "country", text),
                                       "_text", f"tfidf_{view}", b["tfidf"], same))
    if b["embed"]["enabled"]:
        frames.append(load(cfg, split, "cands_embed.parquet"))
    cands = merge_candidates(frames)
    save(cands, cfg, split, "cands_all.parquet")
    print(f"[block] {split}: {cands.height:,} pairs, {cands.height / max(left.height, 1):.1f} per S1")


def stage_cheap_train(cfg: dict, split: str) -> None:
    left, right = load_norm(cfg, split, "s1"), load_norm(cfg, split, "right")
    cands, gt, folds = load(cfg, split, "cands_all.parquet"), load(cfg, split, "gt.parquet"), load(cfg, split, "folds.parquet")
    cc = cfg["cheap"]
    tr = cands.join(sample_l(folds, "fit", cc["max_train_s1"], cfg["seed"]), on="l_idx", how="semi")
    va = cands.join(sample_l(folds, es_fold(cfg), 50_000, cfg["seed"]), on="l_idx", how="semi")
    x_tr, x_va = cheap_features(tr, left, right), cheap_features(va, left, right)
    booster = train_binary(x_tr.select(CHEAP_COLUMNS), label_pairs(tr, gt), x_va.select(CHEAP_COLUMNS),
                           label_pairs(va, gt), cc["params"], cc["num_boost_round"], cc["early_stopping_rounds"])
    booster.save_model(str(write_path(cfg, split, "cheap_model.txt")))


def stage_cheap_apply(cfg: dict, split: str) -> None:
    left, right = load_norm(cfg, split, "s1"), load_norm(cfg, split, "right")
    cands = load(cfg, split, "cands_all.parquet")
    booster = load_booster(read_path(cfg, MODEL_SPLIT[split], "cheap_model.txt"))
    step = cfg["cheap"]["chunk_rows"]
    scores = [predict(booster, cheap_features(cands.slice(s, step), left, right).select(CHEAP_COLUMNS))
              for s in range(0, cands.height, step)]
    cands = cands.with_columns(cheap_score=pl.Series(np.concatenate(scores) if scores else [], dtype=pl.Float32))
    final = top_k_per_left(cands, "cheap_score", cfg["cheap"]["keep_top"])
    save(final, cfg, split, "cands_final.parquet")
    print(f"[cheap_apply] {split}: {cands.height:,} -> {final.height:,} pairs")
    if split != "test":
        gt, folds = load(cfg, split, "gt.parquet"), load(cfg, split, "folds.parquet")
        ev = folds.filter(pl.col("fold") == es_fold(cfg))["l_idx"]
        print(f"[cheap_apply] recall all={blocking_recall(cands, gt, ev):.4f} "
              f"final={blocking_recall(final, gt, ev):.4f} per_s1={cands_per_s1(final, ev):.1f}")
