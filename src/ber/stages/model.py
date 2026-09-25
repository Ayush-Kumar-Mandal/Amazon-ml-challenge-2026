"""Model stages: features, train, evaluate, predict."""
from __future__ import annotations

import json

import joblib
import numpy as np
import polars as pl

from ber.config import MODEL_SPLIT, read_path, write_path
from ber.decide import (SCALES, calibrate, choose_expected_f05, choose_threshold, crossfit_calibrate,
                        fit_calibrator, one_owner, tune_threshold)
from ber.features import FEATURE_COLUMNS, context_features, fit_vectorizers, full_features
from ber.metrics import blocking_recall, cands_per_s1, log_experiment, macro_f05_frame
from ber.models.gbm import load_booster, predict, train_binary
from ber.stages.common import es_fold, eval_folds, label_pairs, load, load_norm, sample_l, save


def load_features(cfg: dict, split: str) -> pl.LazyFrame:
    return pl.scan_parquet(str(read_path(cfg, split, "features") / "*.parquet"))


def stage_features(cfg: dict, split: str) -> None:
    left, right = load_norm(cfg, split, "s1"), load_norm(cfg, split, "right")
    cands = context_features(load(cfg, split, "cands_final.parquet"))  # context on the FULL candidate set
    if split != "test":
        folds = load(cfg, split, "folds.parquet")
        keep = pl.concat([sample_l(folds, "fit", cfg["features"]["max_train_s1"], cfg["seed"])]
                         + [folds.filter(pl.col("fold") == f).select("l_idx") for f in eval_folds(cfg)])
        cands = cands.join(keep, on="l_idx", how="semi")
        cands = cands.with_columns(y=pl.Series(label_pairs(cands, load(cfg, split, "gt.parquet"))))
    vecs = fit_vectorizers(left, right, cfg["features"]["vectorizer_fit_rows"], cfg["seed"])
    out_dir = write_path(cfg, split, "features/.keep").parent
    for old in out_dir.glob("part-*.parquet"):
        old.unlink()
    step = cfg["features"]["chunk_rows"]
    for n, s in enumerate(range(0, cands.height, step)):
        chunk = cands.slice(s, step)
        feats = full_features(chunk, left, right, vecs)
        if "y" in chunk.columns:
            feats = feats.with_columns(y=chunk["y"])
        feats.write_parquet(out_dir / f"part-{n:04d}.parquet")
        print(f"[features] {split}: part {n} rows={feats.height:,}")


def stage_train(cfg: dict, split: str) -> None:
    feats = load_features(cfg, split).collect().join(load(cfg, split, "folds.parquet"), on="l_idx")
    cols = [c for c in FEATURE_COLUMNS if c in feats.columns]
    fit = feats.filter(pl.col("fold") == "fit")
    es = feats.join(sample_l(load(cfg, split, "folds.parquet"), es_fold(cfg), cfg["gbm"]["max_es_s1"], cfg["seed"]),
                    on="l_idx", how="semi")
    g = cfg["gbm"]
    booster = train_binary(fit.select(cols), fit["y"].to_numpy(), es.select(cols), es["y"].to_numpy(),
                           g["params"], g["num_boost_round"], g["early_stopping_rounds"])
    booster.save_model(str(write_path(cfg, split, "gbm_model.txt")))
    ev = feats.filter(pl.col("fold").is_in(eval_folds(cfg)))
    for name, part in (("valid_pred.parquet", ev), ("fit_pred.parquet", fit)):
        out = part.select("l_idx", "r_idx", "y", "fold").with_columns(p=pl.Series(predict(booster, part.select(cols))))
        save(out, cfg, split, name)
    imp = sorted(zip(booster.feature_name(), booster.feature_importance("gain")), key=lambda t: -t[1])
    print("[train] top features:", [f for f, _ in imp[:15]])


def _decide(pred: pl.DataFrame, gt: pl.DataFrame, eval_l: pl.Series):
    sel_e = choose_expected_f05(pred)
    f_e = macro_f05_frame(sel_e, gt, eval_l)
    t, f_t = tune_threshold(pred, gt, eval_l)
    scores = {"f05_expected": f_e, "f05_threshold": f_t, "threshold": t}
    if f_e >= f_t:
        return {"method": "expected_f05", "threshold": t}, sel_e, scores
    return {"method": "threshold", "threshold": t}, choose_threshold(pred, t), scores


def stage_evaluate(cfg: dict, split: str) -> dict:
    pred = load(cfg, split, cfg["evaluate"]["pred_file"])
    gt, folds = load(cfg, split, "gt.parquet"), load(cfg, split, "folds.parquet")
    s1c = load(cfg, split, "s1.parquet").select(pl.col("idx").alias("l_idx"), "country")
    eval_l = folds.filter(pl.col("fold") == "valid")["l_idx"]
    owner = one_owner if cfg["decide"]["one_owner"] else (lambda df: df)
    res = {"run": cfg["run_name"], "split": split, "scheme": cfg["validation"]["scheme"],
           "pred_file": cfg["evaluate"]["pred_file"]}
    if cfg["validation"]["scheme"] == "s1_random":
        ev = pred.filter(pl.col("fold") == "valid")
        iso = fit_calibrator(ev["p"].to_numpy(), ev["y"].to_numpy())
        cal = ev.with_columns(p=pl.Series(crossfit_calibrate(ev, cfg["seed"])))
        decision, sel, scores = _decide(owner(cal), gt, eval_l)
        res.update(scores)
        res["f05"] = max(scores["f05_expected"], scores["f05_threshold"])
    else:  # country_holdout: calibrate on seen-country rows, tune the unseen scale on the held-out country
        seen, unseen = pred.filter(pl.col("fold") == "valid_seen"), pred.filter(pl.col("fold") == "valid")
        iso = fit_calibrator(seen["p"].to_numpy(), seen["y"].to_numpy())
        cal = unseen.with_columns(p=pl.Series(calibrate(iso, unseen["p"].to_numpy())))
        best = None
        for s in SCALES:
            d, sel_s, sc = _decide(owner(cal.with_columns(p=(pl.col("p") * s).clip(0.0, 1.0))), gt, eval_l)
            f = max(sc["f05_expected"], sc["f05_threshold"])
            res[f"f05_scale_{s}"] = f
            if best is None or f > best[0]:
                best = (f, s, d, sel_s, sc)
        _, res["best_unseen_scale"], decision, sel, scores = best
        res.update(scores)
        res["f05"] = res["f05_scale_1.0"]
    ev_c = s1c.join(eval_l.to_frame(), on="l_idx", how="semi")
    for country in sorted(ev_c["country"].unique()):
        res[f"f05_{country}"] = macro_f05_frame(sel, gt, ev_c.filter(pl.col("country") == country)["l_idx"])
    singles = ev_c.join(gt.select("l_idx").unique(), on="l_idx", how="anti")["l_idx"]
    res["singleton_acc"] = macro_f05_frame(sel, gt, singles) if singles.len() else float("nan")
    for name in ("cands_all", "cands_final"):
        cands = load(cfg, split, f"{name}.parquet")
        res[f"recall_{name}"] = blocking_recall(cands, gt, eval_l)
        res[f"per_s1_{name}"] = cands_per_s1(cands, eval_l)
    joblib.dump(iso, write_path(cfg, split, "calibrator.joblib"))
    write_path(cfg, split, "decision.json").write_text(json.dumps(decision))
    log_experiment(cfg["experiments_csv"], res)
    print(json.dumps(res, indent=1, default=str))
    return res


def stage_predict(cfg: dict, split: str) -> None:
    booster = load_booster(read_path(cfg, MODEL_SPLIT[split], "gbm_model.txt"))
    lf = load_features(cfg, split)
    n = lf.select(pl.len()).collect().item()
    step = cfg["gbm"]["predict_chunk_rows"]
    outs = []
    for s in range(0, n, step):
        part = lf.slice(s, step).collect()
        outs.append(part.select("l_idx", "r_idx").with_columns(p=pl.Series(predict(booster, part))))
    save(pl.concat(outs), cfg, split, "pred.parquet")
