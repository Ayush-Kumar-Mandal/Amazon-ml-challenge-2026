"""Validation error analysis + test-time France monitoring.
Usage: python scripts/error_analysis.py --config configs/kaggle.yaml [--set evaluate.pred_file=valid_pred_combined.parquet]
Writes reports/error_analysis_<run_name>.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import polars as pl

from ber.config import apply_override, load_config, read_path
from ber.decide import calibrate, one_owner, select
from ber.stages.common import load


def block(df: pl.DataFrame) -> str:
    with pl.Config(tbl_rows=60, fmt_str_lengths=60, tbl_width_chars=250):
        return f"```\n{df}\n```"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--split", default="train")
    args = ap.parse_args()
    cfg = load_config(args.config)
    for expr in args.set:
        apply_override(cfg, expr)
    split = args.split
    iso = joblib.load(read_path(cfg, split, "calibrator.joblib"))
    decision = json.loads(read_path(cfg, split, "decision.json").read_text())
    pred = load(cfg, split, cfg["evaluate"]["pred_file"]).filter(pl.col("fold") == "valid")
    pred = pred.with_columns(p=pl.Series(calibrate(iso, pred["p"].to_numpy())))
    if cfg["decide"]["one_owner"]:
        pred = one_owner(pred)
    sel = select(pred, decision)
    gt = load(cfg, split, "gt.parquet")
    cands = load(cfg, split, "cands_final.parquet").select("l_idx", "r_idx").with_columns(in_cands=pl.lit(True))
    s1 = load(cfg, split, "s1_norm.parquet").select(
        pl.col("idx").alias("l_idx"), "country", pl.col("name_script").alias("l_script"),
        pl.col("business_name").alias("l_name"), pl.col("business_address").alias("l_addr"))
    right = load(cfg, split, "right_norm.parquet").select(
        pl.col("idx").alias("r_idx"), "source", pl.col("business_name").alias("r_name"),
        pl.col("business_address").alias("r_addr"))
    valid_l = load(cfg, split, "folds.parquet").filter(pl.col("fold") == "valid").select("l_idx")
    fp = sel.join(gt, on=["l_idx", "r_idx"], how="anti").join(s1, on="l_idx").join(right, on="r_idx")
    fn = (gt.join(valid_l, on="l_idx", how="semi").join(sel, on=["l_idx", "r_idx"], how="anti")
          .join(cands, on=["l_idx", "r_idx"], how="left")
          .with_columns(reason=pl.when(pl.col("in_cands")).then(pl.lit("scored_below"))
                        .otherwise(pl.lit("not_in_candidates")))
          .join(s1, on="l_idx").join(right, on="r_idx"))
    cols = ["country", "source", "l_name", "r_name", "l_addr", "r_addr"]
    lines = [f"# Error analysis: {cfg['run_name']} ({cfg['evaluate']['pred_file']})", "",
             f"decision: `{decision}`", "", f"- false positives: {fp.height:,}", f"- false negatives: {fn.height:,}",
             "", "## False positives by country / script / source",
             block(fp.group_by("country", "l_script", "source").len().sort("len", descending=True)),
             "## False negatives by reason / country",
             block(fn.group_by("reason", "country").len().sort("len", descending=True)),
             "## 40 sample false positives", block(fp.sample(min(40, fp.height), seed=0).select(cols)),
             "## 40 sample false negatives (scored below)",
             block(fn.filter(pl.col("reason") == "scored_below").pipe(lambda d: d.sample(min(40, d.height), seed=0)).select(cols)),
             "## 40 sample false negatives (blocking misses)",
             block(fn.filter(pl.col("reason") == "not_in_candidates").pipe(lambda d: d.sample(min(40, d.height), seed=0)).select(cols))]
    out_tsv = Path(cfg["paths"]["output_dir"]) / "matching_results.tsv"
    if out_tsv.exists():
        m = pl.read_csv(out_tsv, separator="\t", quote_char=None, infer_schema=False, empty_string_is_null=False)
        s1t = load(cfg, "test", "s1.parquet").select(pl.col("entity_id").alias("source1_entity_id"), "country")
        stats = (m.join(s1t, on="source1_entity_id")
                 .with_columns(n=pl.when(pl.col("matched_entity_ids").fill_null("") == "").then(0)
                               .otherwise(pl.col("matched_entity_ids").str.count_matches(",") + 1))
                 .group_by("country").agg(n_s1=pl.len(), mean_matches=pl.col("n").mean(),
                                          empty_share=(pl.col("n") == 0).mean()))
        lines += ["## Test predictions by country (France is unseen in training)", block(stats),
                  "Reference from train ground truth: about 3.5 matches per S1, 5.6% singletons."]
    Path("reports").mkdir(exist_ok=True)
    path = Path("reports") / f"error_analysis_{cfg['run_name']}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
