"""GPU stages: bi-encoder (M2) and cross-encoder + combiner (M3). Heavy imports stay inside functions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from ber.blocking.embed_ann import embed_candidates, pair_cosine
from ber.config import MODEL_SPLIT, read_path, write_path
from ber.features import FEATURE_COLUMNS
from ber.models.biencoder import encode_to_memmap, make_training_triplets, record_text, train_biencoder
from ber.models.crossencoder import band_mask, pair_text, sample_training_pairs, score_pairs, train_cross_encoder
from ber.models.gbm import load_booster, predict, train_combiner
from ber.stages.common import es_fold, load, save
from ber.stages.model import load_features


def _scratch(cfg: dict, split: str) -> Path:
    path = Path(cfg["paths"]["scratch_dir"]) / split
    path.mkdir(parents=True, exist_ok=True)
    return path


def stage_embed_train(cfg: dict, split: str) -> None:
    bc = cfg["biencoder"]
    s1, right = load(cfg, split, "s1.parquet"), load(cfg, split, "right.parquet")
    fit_l = load(cfg, split, "folds.parquet").filter(pl.col("fold") == "fit").select("l_idx")
    trip = make_training_triplets(load(cfg, split, "cands_final.parquet"), load(cfg, split, "gt.parquet"), fit_l,
                                  record_text(s1), record_text(right), bc["n_triplets"], cfg["seed"])
    out = Path(cfg["paths"]["work_dir"]) / split / "biencoder"
    train_biencoder(trip, bc["model_name"], out, bc["epochs"], bc["batch_size"], bc["lr"], bc["max_seq_len"], cfg["seed"])


def stage_embed_encode(cfg: dict, split: str) -> None:
    bc, e = cfg["biencoder"], cfg["blocking"]["embed"]
    model_dir = read_path(cfg, MODEL_SPLIT[split], "biencoder")
    s1, right = load(cfg, split, "s1.parquet"), load(cfg, split, "right.parquet")
    scratch = _scratch(cfg, split)
    emb_l = encode_to_memmap(model_dir, record_text(s1).to_list(), scratch / "emb_s1.npy",
                             bc["encode_batch_size"], bc["max_seq_len"])
    emb_r = encode_to_memmap(model_dir, record_text(right).to_list(), scratch / "emb_right.npy",
                             bc["encode_batch_size"], bc["max_seq_len"])
    cands = embed_candidates(s1.select("idx", "country"), right.select("idx", "country"), emb_l, emb_r,
                             e["top_k_fwd"], e["top_k_rev"], cfg["blocking"]["same_country"], e["use_gpu"])
    save(cands, cfg, split, "cands_embed.parquet")


def stage_embed_attach(cfg: dict, split: str) -> None:
    scratch = _scratch(cfg, split)
    emb_l, emb_r = np.load(scratch / "emb_s1.npy"), np.load(scratch / "emb_right.npy")
    cands = load(cfg, split, "cands_all.parquet")
    cos = pair_cosine(emb_l, emb_r, cands["l_idx"].to_numpy(), cands["r_idx"].to_numpy())
    save(cands.with_columns(embed_cos=pl.Series(cos, dtype=pl.Float32)), cfg, split, "cands_all.parquet")
    print(f"[embed_attach] {split}: {cands.height:,} pairs")


def _texts(cfg: dict, split: str, pairs: pl.DataFrame) -> tuple[list[str], list[str]]:
    lt = pair_text(load(cfg, split, "s1.parquet"))
    rt = pair_text(load(cfg, split, "right.parquet"))
    return lt.gather(pairs["l_idx"]).to_list(), rt.gather(pairs["r_idx"]).to_list()


def stage_cross_train(cfg: dict, split: str) -> None:
    cc = cfg["cross"]
    pairs = sample_training_pairs(load(cfg, split, "fit_pred.parquet"), cc["band_lo"], cc["band_hi"],
                                  cc["n_train_pairs"], cfg["seed"])
    ta, tb = _texts(cfg, split, pairs)
    out = Path(cfg["paths"]["work_dir"]) / split / "crossencoder"
    train_cross_encoder(ta, tb, pairs["y"].to_list(), cc["model_name"], out, cc["max_len"], cc["epochs"],
                        cc["batch_size"], cc["lr"], cfg["seed"])


def stage_cross_score(cfg: dict, split: str) -> None:
    cc = cfg["cross"]
    pred = load(cfg, split, "pred.parquet" if split == "test" else "valid_pred.parquet")
    band = pred.filter(pl.Series(band_mask(pred["p"].to_numpy(), cc["band_lo"], cc["band_hi"])))
    print(f"[cross_score] {split}: {band.height:,} of {pred.height:,} pairs in band")
    ta, tb = _texts(cfg, split, band)
    scores = score_pairs(read_path(cfg, MODEL_SPLIT[split], "crossencoder"), ta, tb, cc["max_len"],
                         cc["score_batch_size"])
    save(band.select("l_idx", "r_idx").with_columns(cross_p=pl.Series(scores, dtype=pl.Float32)),
         cfg, split, "cross_scores.parquet")


COMBINER_EXTRA = ["p_stage1", "cross_p"]


def stage_combine_train(cfg: dict, split: str) -> None:
    feats = load_features(cfg, split).collect().drop("y")
    df = (load(cfg, split, "valid_pred.parquet").rename({"p": "p_stage1"})
          .join(feats, on=["l_idx", "r_idx"])
          .join(load(cfg, split, "cross_scores.parquet"), on=["l_idx", "r_idx"], how="left"))
    cols = [c for c in FEATURE_COLUMNS if c in df.columns] + COMBINER_EXTRA
    train_mask = (df["fold"] == es_fold(cfg)).to_numpy()  # random: all "valid"; holdout: "valid_seen" only
    cb = cfg["combiner"]
    oof, boosters = train_combiner(df.select(cols), df["y"].to_numpy(), df["l_idx"].to_numpy(), train_mask,
                                   cb["params"], cb["num_boost_round"], cb["n_folds"])
    for k, bst in enumerate(boosters):
        bst.save_model(str(write_path(cfg, split, f"combiner_{k}.txt")))
    save(df.select("l_idx", "r_idx", "y", "fold").with_columns(p=pl.Series(oof)), cfg, split,
         "valid_pred_combined.parquet")


def stage_combine_predict(cfg: dict, split: str) -> None:
    cb = cfg["combiner"]
    boosters = [load_booster(read_path(cfg, MODEL_SPLIT[split], f"combiner_{k}.txt")) for k in range(cb["n_folds"])]
    stage1 = load(cfg, split, "pred.parquet").rename({"p": "p_stage1"})
    cross = load(cfg, split, "cross_scores.parquet")
    lf = load_features(cfg, split)
    n = lf.select(pl.len()).collect().item()
    outs = []
    for s in range(0, n, cb["predict_chunk_rows"]):
        part = (lf.slice(s, cb["predict_chunk_rows"]).collect()
                .join(stage1, on=["l_idx", "r_idx"]).join(cross, on=["l_idx", "r_idx"], how="left"))
        p = np.mean([predict(bst, part) for bst in boosters], axis=0)
        outs.append(part.select("l_idx", "r_idx").with_columns(p=pl.Series(p)))
    save(pl.concat(outs), cfg, split, "pred_combined.parquet")
