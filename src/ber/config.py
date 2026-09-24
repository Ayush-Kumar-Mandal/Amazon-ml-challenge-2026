"""Config loading (YAML with `extends` + env vars), CLI overrides and artifact paths."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

# Which split's artifacts (lexicon, models, calibrator) a split uses.
LEXICON_SPLIT = {"train": "train", "dev": "dev", "test": "train"}
MODEL_SPLIT = {"train": "train", "dev": "dev", "test": "train"}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _expand(obj):
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(os.path.expanduser(obj))
    return obj


def load_config(path: str | Path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = _merge(load_config(path.parent / parent), cfg)
    return _expand(cfg)


def apply_override(cfg: dict, expr: str) -> None:
    """Apply `a.b.c=value`, parsing value as YAML (numbers, bools, lists)."""
    dotted, _, raw = expr.partition("=")
    keys = dotted.strip().split(".")
    node = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = yaml.safe_load(raw)


def write_path(cfg: dict, split: str, name: str) -> Path:
    path = Path(cfg["paths"]["work_dir"]) / split / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_path(cfg: dict, split: str, name: str) -> Path:
    bases = [cfg["paths"]["work_dir"], *cfg["paths"].get("prev_work_dirs", [])]
    for base in bases:
        path = Path(base) / split / name
        if path.exists():
            return path
    raise FileNotFoundError(f"{split}/{name} not found under {bases}")
