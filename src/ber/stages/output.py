"""Output stages: submit (and package, added in Task 25)."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import polars as pl

from ber.config import MODEL_SPLIT, read_path
from ber.decide import calibrate, one_owner, scale_unseen, select
from ber.stages.common import load
from ber.submit import ensure_test_dir, run_validator, write_id_lists


def stage_submit(cfg: dict, split: str) -> int:
    if split != "test":
        raise ValueError("submit runs on the test split only")
    ms = MODEL_SPLIT[split]
    iso = joblib.load(read_path(cfg, ms, "calibrator.joblib"))
    decision = json.loads(read_path(cfg, ms, "decision.json").read_text())
    s1, right = load(cfg, split, "s1.parquet"), load(cfg, split, "right.parquet")
    pred = load(cfg, split, cfg["submit"]["pred_file"])
    pred = pred.with_columns(p=pl.Series(calibrate(iso, pred["p"].to_numpy())))
    d = cfg["decide"]
    pred = scale_unseen(pred, s1.select(pl.col("idx").alias("l_idx"), "country"),
                        d["seen_countries"], d["unseen_country_scale"])
    if d["one_owner"]:
        pred = one_owner(pred)
    matches = select(pred, decision)
    cands = load(cfg, split, "cands_final.parquet").select("l_idx", "r_idx")
    out = Path(cfg["paths"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    write_id_lists(out / "matching_results.tsv", "matched_entity_ids", s1, right, matches)
    write_id_lists(out / "candidate_pairs.tsv", "candidate_entity_ids", s1, right, cands)
    per = (s1.select(pl.col("idx").alias("l_idx"), "country")
           .join(matches.group_by("l_idx").len("n"), on="l_idx", how="left").fill_null(0)
           .group_by("country").agg(mean_matches=pl.col("n").mean(), empty_share=(pl.col("n") == 0).mean()))
    print(per)
    code = run_validator(out / "matching_results.tsv", out / "candidate_pairs.tsv", ensure_test_dir(cfg))
    if code != 0:
        raise RuntimeError("validate_submission FAILED; do not upload")
    return code
