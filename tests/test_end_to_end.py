from pathlib import Path

import polars as pl
import pytest

from ber.pipeline import run

pytest.importorskip("sparse_dot_topn")

STEPS = [("ingest", "train"), ("ingest", "test"), ("split", "train"), ("lexicon", "train"),
         ("normalize", "train"), ("normalize", "test"), ("block", "train"), ("block", "test"),
         ("cheap_train", "train"), ("cheap_apply", "train"), ("cheap_apply", "test"),
         ("features", "train"), ("features", "test"), ("train", "train"), ("evaluate", "train"),
         ("predict", "test"), ("submit", "test")]


def test_pipeline_end_to_end(small_cfg):
    results = {stage: run(small_cfg, stage, split) for stage, split in STEPS}
    metrics = results["evaluate"]
    assert metrics["recall_cands_final"] > 0.9
    assert metrics["f05"] > 0.6
    assert results["submit"] == 0
    out = pl.read_csv(Path(small_cfg["paths"]["output_dir"]) / "matching_results.tsv", separator="\t",
                      quote_char=None, infer_schema=False)
    assert out.height == 60  # every test S1, France included
