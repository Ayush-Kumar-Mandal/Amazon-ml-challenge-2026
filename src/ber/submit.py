"""Write the two submission TSVs, run the official validator, build the final zip."""
from __future__ import annotations

import subprocess
import sys
import zipfile
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


CODE_ITEMS = ["src", "configs", "notebooks", "scripts", "tests", "README.md", "requirements.txt",
              "requirements.in", "pyproject.toml"]


def build_package(repo_root: Path, output_dir: Path, doc_path: Path, team: str, dest: Path) -> Path:
    zip_path = Path(dest) / f"{team}_submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(Path(output_dir) / name, f"output/{name}")
        for item in CODE_ITEMS:
            path = Path(repo_root) / item
            if not path.exists():
                continue
            files = [path] if path.is_file() else sorted(
                f for f in path.rglob("*") if f.is_file() and "__pycache__" not in f.parts)
            for f in files:
                z.write(f, f"code/business_entity_resolution/{f.relative_to(repo_root).as_posix()}")
        z.write(doc_path, "Documentation_template.md")
    return zip_path
