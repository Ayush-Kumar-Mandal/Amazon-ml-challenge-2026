"""M0 EDA on the train split. Usage: python scripts/eda_m0.py --config configs/dev.yaml"""
from __future__ import annotations

import argparse
import sys

import polars as pl

from ber.config import load_config, read_path

# Windows console defaults to cp1252, which can't print the non-ASCII
# business names/addresses this EDA prints (Hindi/Kannada script, etc.).
# Force UTF-8 stdout so the script doesn't crash on those rows.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

pl.Config.set_tbl_rows(40)
pl.Config.set_fmt_str_lengths(80)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    s1 = pl.read_parquet(read_path(cfg, "train", "s1.parquet"))
    right = pl.read_parquet(read_path(cfg, "train", "right.parquet"))
    gt = pl.read_parquet(read_path(cfg, "train", "gt.parquet"))

    print("== field quality by source/country")
    both = pl.concat([s1.with_columns(source=pl.lit("S1")), right], how="diagonal")
    print(both.group_by("source", "country").agg(
        n=pl.len(),
        empty_name=(pl.col("business_name") == "").mean(),
        empty_addr=(pl.col("business_address") == "").mean(),
        nonascii_name=pl.col("business_name").str.contains(r"[^\x00-\x7F]").mean(),
        nonascii_addr=pl.col("business_address").str.contains(r"[^\x00-\x7F]").mean(),
        has_6digit=pl.col("business_address").str.contains(r"\b\d{6}\b").mean(),
        has_5digit=pl.col("business_address").str.contains(r"\b\d{5}\b").mean(),
    ).sort("source", "country"))

    pairs = gt.join(
        s1.select(pl.col("idx").alias("l_idx"), pl.col("country").alias("lc"),
                  pl.col("business_name").alias("l_name"), pl.col("business_address").alias("l_addr")),
        on="l_idx",
    ).join(
        right.select(pl.col("idx").alias("r_idx"), pl.col("country").alias("rc"), "source",
                     pl.col("business_name").alias("r_name"), pl.col("business_address").alias("r_addr")),
        on="r_idx",
    )
    print("== cross-country match share:", (pairs["lc"] != pairs["rc"]).mean())

    owners = gt.group_by("r_idx").len("n_owners")
    print("== right records with >1 S1 owner:", (owners["n_owners"] > 1).sum())
    print("== share of right records that match some S1:", owners.height / right.height)

    sizes = s1.select(pl.col("idx").alias("l_idx"), "country").join(
        gt.group_by("l_idx").len("n"), on="l_idx", how="left").fill_null(0)
    print(sizes.group_by("country").agg(
        singleton=(pl.col("n") == 0).mean(), mean=pl.col("n").mean(), p99=pl.col("n").quantile(0.99)))
    print(pairs.group_by("lc", "source").len().sort("lc", "source"))

    print("== 40 random matched pairs")
    print(pairs.sample(40, seed=0).select("lc", "source", "l_name", "r_name", "l_addr", "r_addr"))


if __name__ == "__main__":
    main()
