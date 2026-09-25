"""Exact-key blocking. Keys (all prefixed with country unless same_country=False):
pc|postcode|long-name-token, loc|locality|name-prefix4, ph|metaphone(name)|last-locality, hn|house|street.
"""
from __future__ import annotations

import polars as pl

_COLS = ["idx", "country", "name_core", "postcode", "localities", "name_key", "house_no", "street_tok"]


def _keys_for_slice(df: pl.DataFrame, same_country: bool) -> pl.DataFrame:
    idx_out: list[int] = []
    keys: list[str] = []
    for i, c, core, pc, locs, nk, hn, st in zip(*(df[col].to_list() for col in _COLS)):
        c = c if same_country else "*"
        toks = sorted(core.split(), key=len, reverse=True)[:2]
        prefix = core.replace(" ", "")[:4]
        loc_list = [x for x in locs.split("|") if x][:3]
        ks = [f"pc|{c}|{pc}|{t}" for t in toks] if pc else []
        if prefix:
            ks += [f"loc|{c}|{x}|{prefix}" for x in loc_list]
        if nk and loc_list:
            ks.append(f"ph|{c}|{nk}|{loc_list[-1]}")
        if hn and st:
            ks.append(f"hn|{c}|{hn}|{st}")
        idx_out += [i] * len(ks)
        keys += ks
    frame = pl.DataFrame({"idx": idx_out, "key": keys}, schema={"idx": pl.UInt32, "key": pl.String})
    return frame.select("idx", pl.col("key").hash(seed=0))


def record_keys(df: pl.DataFrame, same_country: bool = True, slice_rows: int = 1_000_000) -> pl.DataFrame:
    parts = [_keys_for_slice(df.slice(s, slice_rows), same_country) for s in range(0, df.height, slice_rows)]
    return pl.concat(parts) if parts else _keys_for_slice(df, same_country)


def join_keys(lk: pl.DataFrame, rk: pl.DataFrame, max_block_pairs: int) -> pl.DataFrame:
    lc = lk.group_by("key").len("nl")
    rc = rk.group_by("key").len("nr")
    ok = (lc.join(rc, on="key")
          .filter(pl.col("nl").cast(pl.Int64) * pl.col("nr").cast(pl.Int64) <= max_block_pairs)
          .select("key"))
    left = lk.join(ok, on="key", how="semi").rename({"idx": "l_idx"})
    right = rk.join(ok, on="key", how="semi").rename({"idx": "r_idx"})
    return (left.join(right, on="key").select("l_idx", "r_idx").unique()
            .with_columns(from_keys=pl.lit(True)))
