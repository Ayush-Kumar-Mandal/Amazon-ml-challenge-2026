"""Cross-encoder reranker: xlm-roberta-base (MIT) pair classifier on uncertain pairs only."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


def pair_text(df: pl.DataFrame) -> pl.Series:
    return df.select(pl.concat_str([pl.col("business_name"), pl.lit(" | "), pl.col("business_address")])
                     .alias("text")).to_series()


def band_mask(p: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (p >= lo) & (p <= hi)


def sample_training_pairs(fit_pred: pl.DataFrame, lo: float, hi: float, n: int, seed: int) -> pl.DataFrame:
    in_band = fit_pred.filter(pl.col("p").is_between(lo, hi))
    rest = fit_pred.join(in_band.select("l_idx", "r_idx"), on=["l_idx", "r_idx"], how="anti")
    n_band = min(in_band.height, int(n * 0.8))
    n_rest = min(rest.height, n - n_band)
    return pl.concat([in_band.sample(n_band, seed=seed), rest.sample(n_rest, seed=seed)]).select("l_idx", "r_idx", "y")


def train_cross_encoder(text_a: list[str], text_b: list[str], labels: list[int], model_name: str,
                        out_dir: str | Path, max_len: int, epochs: int, batch_size: int, lr: float, seed: int) -> None:
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments)

    tok = AutoTokenizer.from_pretrained(model_name)
    ds = Dataset.from_dict({"a": text_a, "b": text_b, "label": labels}).map(
        lambda x: tok(x["a"], x["b"], truncation=True, max_length=max_len), batched=True, remove_columns=["a", "b"])
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    args = TrainingArguments(
        output_dir=str(Path(out_dir) / "_ckpt"), per_device_train_batch_size=batch_size, learning_rate=lr,
        num_train_epochs=epochs, warmup_ratio=0.06, weight_decay=0.01, fp16=True, logging_steps=500,
        save_strategy="no", report_to="none", seed=seed, dataloader_num_workers=2,
    )
    Trainer(model=model, args=args, train_dataset=ds, data_collator=DataCollatorWithPadding(tok)).train()
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))


def score_pairs(model_dir: str | Path, text_a: list[str], text_b: list[str], max_len: int,
                batch_size: int) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).half().cuda().eval()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    order = np.argsort([len(a) + len(b) for a, b in zip(text_a, text_b)])
    out = np.empty(len(text_a), np.float32)
    with torch.no_grad():
        for s in range(0, len(order), batch_size):
            ids = order[s:s + batch_size]
            enc = tok([text_a[i] for i in ids], [text_b[i] for i in ids], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt").to("cuda")
            logits = model(**enc, return_dict=False)[0].float()
            out[ids] = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    return out
