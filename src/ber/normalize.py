"""Apply text normalization to whole frames (multiprocess, spawn-safe on Windows)."""
from __future__ import annotations

import multiprocessing as mp

import polars as pl

from ber.text import detect_script, normalize_address, normalize_name

NORM_COLUMNS = ["name_norm", "name_core", "name_alt", "legal_form", "is_web", "name_key", "name_script",
                "addr_norm", "postcode", "house_no", "street_tok", "num_tokens", "localities"]
_LEX: dict = {}


def normalize_record(name: str, address: str, lex: dict) -> tuple:
    n = normalize_name(name, lex.get("name", {}))
    a = normalize_address(address, lex.get("addr", {}))
    return (n["name_norm"], n["name_core"], n["name_alt"], n["legal_form"], n["is_web"], n["name_key"],
            detect_script(name), a["addr_norm"], a["postcode"], a["house_no"], a["street_tok"],
            a["num_tokens"], a["localities"])


def _init(lex: dict) -> None:
    global _LEX
    _LEX = lex


def _run(rows: list[tuple[str, str]]) -> list[tuple]:
    return [normalize_record(n, a, _LEX) for n, a in rows]


def normalize_frame(df: pl.DataFrame, lex: dict, n_jobs: int = 1, chunk: int = 50_000) -> pl.DataFrame:
    rows = list(zip(df["business_name"].to_list(), df["business_address"].to_list()))
    batches = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    if n_jobs <= 1:
        _init(lex)
        results = [_run(b) for b in batches]
    else:
        with mp.get_context("spawn").Pool(n_jobs, initializer=_init, initargs=(lex,)) as pool:
            results = pool.map(_run, batches)
    flat = [r for res in results for r in res]
    columns = list(zip(*flat)) if flat else [()] * len(NORM_COLUMNS)
    schema = {c: (pl.Int8 if c == "is_web" else pl.String) for c in NORM_COLUMNS}
    norm = pl.DataFrame({c: list(v) for c, v in zip(NORM_COLUMNS, columns)}, schema=schema)
    return pl.concat([df, norm], how="horizontal_extend")
