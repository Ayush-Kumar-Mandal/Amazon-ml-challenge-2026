"""CLI: python -m ber.pipeline --config configs/dev.yaml --stage ingest --split train [--set a.b=v]"""
from __future__ import annotations

import argparse
import sys
import time

from ber.config import apply_override, load_config
from ber.stages import candidates, data, model

STAGES = {
    "ingest": data.stage_ingest,
    "dev_slice": data.stage_dev_slice,
    "split": data.stage_split,
    "lexicon": data.stage_lexicon,
    "normalize": data.stage_normalize,
    "block": candidates.stage_block,
    "cheap_train": candidates.stage_cheap_train,
    "cheap_apply": candidates.stage_cheap_apply,
    "features": model.stage_features,
}


def run(cfg: dict, stage: str, split: str):
    start = time.time()
    result = STAGES[stage](cfg, split)
    print(f"[pipeline] {stage} --split {split} done in {time.time() - start:,.0f}s")
    return result


def main(argv: list[str] | None = None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    ap.add_argument("--split", default="train", choices=["train", "test", "dev"])
    ap.add_argument("--set", action="append", default=[], help="override, e.g. blocking.tfidf.top_k_fwd=60")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    for expr in args.set:
        apply_override(cfg, expr)
    return run(cfg, args.stage, args.split)


if __name__ == "__main__":
    main()
