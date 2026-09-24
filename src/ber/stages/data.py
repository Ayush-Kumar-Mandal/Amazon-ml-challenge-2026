"""Data stages: ingest, dev slice, folds, lexicon, normalize."""
from __future__ import annotations

from ber.io import ingest


def stage_ingest(cfg: dict, split: str) -> None:
    ingest(cfg, split)
