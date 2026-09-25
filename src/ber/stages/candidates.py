"""Candidate stages: block, cheap_train, cheap_apply."""
from __future__ import annotations

import polars as pl

from ber.blocking.keys import join_keys, record_keys
from ber.blocking.merge import merge_candidates
from ber.blocking.tfidf_ann import tfidf_candidates
from ber.stages.common import load, load_norm, save

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
