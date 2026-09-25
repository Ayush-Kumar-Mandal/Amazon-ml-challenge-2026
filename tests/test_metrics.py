import polars as pl
import pytest

from ber.metrics import blocking_recall, cands_per_s1, f05, log_experiment, macro_f05_frame


def test_pdf_worked_example():
    assert f05({"S2-47", "S2-193", "S3-812"}, {"S2-47", "S3-812"}) == pytest.approx(0.714, abs=1e-3)


def test_singleton_rules():
    assert f05(set(), set()) == 1.0
    assert f05({"S2-1"}, set()) == 0.0
    assert f05(set(), {"S2-1"}) == 0.0


def _pairs(rows):
    return pl.DataFrame(rows, schema={"l_idx": pl.UInt32, "r_idx": pl.UInt32}, orient="row")


def test_frame_matches_scalar():
    gt = _pairs([(0, 10), (0, 11), (1, 12)])           # l=2 is a singleton
    pred = _pairs([(0, 10), (0, 11), (0, 13), (2, 14)])  # l=1 predicted empty, l=2 false merge
    eval_l = pl.Series("l_idx", [0, 1, 2], dtype=pl.UInt32)
    expected = (f05({10, 11, 13}, {10, 11}) + f05(set(), {12}) + f05({14}, set())) / 3
    assert macro_f05_frame(pred, gt, eval_l) == pytest.approx(expected)


def test_frame_only_scores_eval_ids():
    gt = _pairs([(0, 10), (1, 12)])
    pred = _pairs([(0, 10)])
    assert macro_f05_frame(pred, gt, pl.Series("l_idx", [0], dtype=pl.UInt32)) == 1.0


def test_blocking_recall_and_density():
    gt = _pairs([(0, 10), (0, 11), (1, 12)])
    cands = _pairs([(0, 10), (0, 99), (1, 12), (1, 98)])
    eval_l = pl.Series("l_idx", [0, 1], dtype=pl.UInt32)
    assert blocking_recall(cands, gt, eval_l) == pytest.approx(2 / 3)
    assert cands_per_s1(cands, eval_l) == 2.0


def test_log_experiment_appends_with_new_columns(tmp_path):
    path = tmp_path / "exp.csv"
    log_experiment(path, {"run": "a", "f05": 0.5})
    log_experiment(path, {"run": "b", "f05": 0.6, "recall": 0.9})
    df = pl.read_csv(path, infer_schema=False)
    assert df["run"].to_list() == ["a", "b"] and df["recall"].to_list() == [None, "0.9"]
