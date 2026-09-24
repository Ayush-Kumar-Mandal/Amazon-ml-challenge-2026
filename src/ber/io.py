"""Read the organizer TSVs (from the zip locally, or an extracted dir on Kaggle) into parquet."""
from __future__ import annotations

import zipfile
from pathlib import Path

import polars as pl

from ber.config import write_path


def read_tsv(source: bytes | str | Path) -> pl.DataFrame:
    df = pl.read_csv(
        source, separator="\t", quote_char=None, infer_schema=False,
        empty_string_is_null=False,
    )
    return df.with_columns(pl.all().fill_null(""))


def open_raw(cfg: dict, split: str, filename: str) -> bytes:
    raw_dir = cfg["paths"].get("raw_dir")
    if raw_dir:
        return (Path(raw_dir) / split / filename).read_bytes()
    with zipfile.ZipFile(cfg["paths"]["raw_zip"]) as z:
        return z.read(f"student_resource/dataset/{split}/{filename}")


def parse_ground_truth(gt_raw: pl.DataFrame, s1: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
    pairs = (
        gt_raw.rename({"source1_entity_id": "l_id", "matched_entity_ids": "r_id"})
        .with_columns(pl.col("r_id").str.split(","))
        .explode("r_id", empty_as_null=True)
        .filter(pl.col("r_id") != "")
    )
    joined = pairs.join(
        s1.select(pl.col("entity_id").alias("l_id"), pl.col("idx").alias("l_idx")), on="l_id"
    ).join(
        right.select(pl.col("entity_id").alias("r_id"), pl.col("idx").alias("r_idx")), on="r_id"
    )
    if joined.height != pairs.height:
        print(f"[ingest] WARNING: {pairs.height - joined.height} ground-truth ids not found in sources")
    return joined.select("l_idx", "r_idx").unique()


def ingest(cfg: dict, split: str) -> None:
    s1 = read_tsv(open_raw(cfg, split, f"{split}_source1.tsv")).with_row_index("idx")
    s2 = read_tsv(open_raw(cfg, split, f"{split}_source2.tsv")).with_columns(source=pl.lit("S2"))
    s3 = read_tsv(open_raw(cfg, split, f"{split}_source3.tsv")).with_columns(source=pl.lit("S3"))
    right = pl.concat([s2, s3]).with_row_index("idx")
    s1.write_parquet(write_path(cfg, split, "s1.parquet"))
    right.write_parquet(write_path(cfg, split, "right.parquet"))
    print(f"[ingest] {split}: s1={s1.height:,} right={right.height:,}")
    if split == "train":
        gt = parse_ground_truth(read_tsv(open_raw(cfg, split, "train_ground_truth.tsv")), s1, right)
        gt.write_parquet(write_path(cfg, split, "gt.parquet"))
        print(f"[ingest] gt pairs={gt.height:,}")
