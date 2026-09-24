"""Challenge metric (macro F0.5 per S1 entity) and blocking diagnostics.

F_beta = (1+b^2) TP / (b^2 |T| + |P|); with b = 0.5: 1.25 TP / (0.25 |T| + |P|).
Singletons: 1.0 if nothing predicted, else 0.0.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl


def f05(pred: set, true: set) -> float:
    if not true:
        return 1.0 if not pred else 0.0
    tp = len(pred & true)
    return 1.25 * tp / (0.25 * len(true) + len(pred))


def macro_f05_frame(pred: pl.DataFrame, gt: pl.DataFrame, eval_l: pl.Series) -> float:
    ev = eval_l.alias("l_idx").to_frame().unique()
    pred = pred.select("l_idx", "r_idx").unique().join(ev, on="l_idx", how="semi")
    gt = gt.select("l_idx", "r_idx").join(ev, on="l_idx", how="semi")
    tp = pred.join(gt, on=["l_idx", "r_idx"]).group_by("l_idx").len("tp")
    table = (
        ev.join(pred.group_by("l_idx").len("n_pred"), on="l_idx", how="left")
        .join(gt.group_by("l_idx").len("n_true"), on="l_idx", how="left")
        .join(tp, on="l_idx", how="left")
        .fill_null(0)
    )
    score = (
        pl.when(pl.col("n_true") == 0)
        .then((pl.col("n_pred") == 0).cast(pl.Float64))
        .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("n_true") + pl.col("n_pred")))
    )
    return float(table.select(score.mean()).item())


def blocking_recall(cands: pl.DataFrame, gt: pl.DataFrame, eval_l: pl.Series) -> float:
    ev = eval_l.alias("l_idx").to_frame().unique()
    truth = gt.select("l_idx", "r_idx").join(ev, on="l_idx", how="semi")
    if truth.height == 0:
        return 1.0
    found = truth.join(cands.select("l_idx", "r_idx").unique(), on=["l_idx", "r_idx"], how="semi")
    return found.height / truth.height


def cands_per_s1(cands: pl.DataFrame, eval_l: pl.Series) -> float:
    ev = eval_l.alias("l_idx").to_frame().unique()
    return cands.join(ev, on="l_idx", how="semi").height / max(ev.height, 1)


def log_experiment(path: str | Path, row: dict) -> None:
    row = {"time": datetime.now().isoformat(timespec="seconds"), **row}
    new = pl.DataFrame([{k: str(v) for k, v in row.items()}])
    path = Path(path)
    if path.exists():
        new = pl.concat([pl.read_csv(path, infer_schema=False), new], how="diagonal")
    new.write_csv(path)
