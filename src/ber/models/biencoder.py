"""Bi-encoder: fine-tune intfloat/multilingual-e5-small (MIT, 118M) with in-batch + hard negatives.
Heavy imports (torch, sentence-transformers, datasets) happen inside functions so the module is
importable locally without GPU packages.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


def record_text(df: pl.DataFrame) -> pl.Series:
    return df.select(pl.concat_str([pl.lit("query: "), pl.col("business_name"), pl.lit(" | "),
                                    pl.col("business_address")]).alias("text")).to_series()


def make_training_triplets(cands: pl.DataFrame, gt: pl.DataFrame, fit_l: pl.DataFrame, left_text: pl.Series,
                           right_text: pl.Series, n: int, seed: int) -> pl.DataFrame:
    pos = gt.join(fit_l, on="l_idx", how="semi")
    if pos.height > n:
        pos = pos.sample(n, seed=seed)
    neg = (cands.join(gt, on=["l_idx", "r_idx"], how="anti").join(fit_l, on="l_idx", how="semi")
           .sort("cheap_score", descending=True)
           .unique(subset="l_idx", keep="first", maintain_order=True)
           .select("l_idx", pl.col("r_idx").alias("neg_idx")))
    t = pos.join(neg, on="l_idx", how="left")
    fallback = np.random.default_rng(seed).integers(0, right_text.len(), t.height)
    neg_idx = np.where(t["neg_idx"].is_null().to_numpy(), fallback, t["neg_idx"].fill_null(0).to_numpy())
    return pl.DataFrame({
        "anchor": left_text.gather(t["l_idx"]),
        "positive": right_text.gather(t["r_idx"]),
        "negative": right_text.gather(pl.Series(neg_idx)),
    })


def train_biencoder(triplets: pl.DataFrame, model_name: str, out_dir: str | Path, epochs: int,
                    batch_size: int, lr: float, max_seq_len: int, seed: int) -> None:
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.training_args import BatchSamplers

    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_len
    ds = Dataset.from_dict({c: triplets[c].to_list() for c in ("anchor", "positive", "negative")})
    args = SentenceTransformerTrainingArguments(
        output_dir=str(Path(out_dir) / "_ckpt"), num_train_epochs=epochs, per_device_train_batch_size=batch_size,
        learning_rate=lr, warmup_ratio=0.05, fp16=True, batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no", logging_steps=200, report_to="none", seed=seed,
    )
    SentenceTransformerTrainer(model=model, args=args, train_dataset=ds,
                               loss=losses.MultipleNegativesRankingLoss(model)).train()
    model.save(str(out_dir))


def encode_to_memmap(model_dir: str | Path, texts: list[str], path: str | Path, batch_size: int,
                     max_seq_len: int, chunk: int = 500_000) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_dir), device="cuda")
    model.max_seq_length = max_seq_len
    model.half()
    dim = model.get_sentence_embedding_dimension()
    arr = np.lib.format.open_memmap(str(path), mode="w+", dtype=np.float16, shape=(len(texts), dim))
    for s in range(0, len(texts), chunk):
        arr[s:s + chunk] = model.encode(texts[s:s + chunk], batch_size=batch_size, normalize_embeddings=True,
                                        convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
        print(f"[encode] {min(s + chunk, len(texts)):,}/{len(texts):,}")
    arr.flush()
    return arr
