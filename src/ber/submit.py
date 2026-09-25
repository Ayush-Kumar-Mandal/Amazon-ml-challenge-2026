"""Write the two submission TSVs, run the official validator, build the final zip."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import polars as pl

from ber.io import open_raw


def write_id_lists(path: str | Path, value_col: str, s1: pl.DataFrame, right: pl.DataFrame,
                   pairs: pl.DataFrame) -> None:
    ids = (
        pairs.select("l_idx", "r_idx").unique()
        .join(s1.select(pl.col("idx").alias("l_idx"), pl.col("entity_id").alias("l_id")), on="l_idx")
        .join(right.select(pl.col("idx").alias("r_idx"), pl.col("entity_id").alias("r_id")), on="r_idx")
        .group_by("l_id").agg(pl.col("r_id").unique().sort())
    )
    table = (
        s1.select(pl.col("entity_id").alias("source1_entity_id"))
        .join(ids.rename({"l_id": "source1_entity_id"}), on="source1_entity_id", how="left")
        .with_columns(pl.col("r_id").list.join(",").fill_null("").alias(value_col))
        .select("source1_entity_id", value_col)
    )
    table.write_csv(path, separator="\t", quote_style="never")


def ensure_test_dir(cfg: dict) -> Path:
    raw_dir = cfg["paths"].get("raw_dir")
    if raw_dir:
        return Path(raw_dir) / "test"
    out = Path(cfg["paths"]["scratch_dir"]) / "test_raw"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "test_source1.tsv"
    if not target.exists():
        target.write_bytes(open_raw(cfg, "test", "test_source1.tsv"))
    return out


def run_validator(matching: str | Path, candidate: str | Path, test_dir: str | Path) -> int:
    cmd = [sys.executable, "-m", "ber.vendor.validate_submission", "--matching", str(matching),
           "--candidate", str(candidate), "--test-dir", str(test_dir)]
    return subprocess.run(cmd).returncode
