"""Mine token maps from training matches (allowed: uses only the provided training data).

- transliteration: tokens of a non-Latin name/address aligned to the best Jaro-Winkler token
  on the Latin side of the same true match (e.g. praivet -> private)
- abbreviations: Latin tokens present on one side only that are an in-order subsequence of a
  longer token on the other side, sharing the first letter (e.g. mfg -> manufacturing)
"""
from __future__ import annotations

from collections import Counter

import polars as pl
from rapidfuzz.distance import JaroWinkler

from ber.text import basic_clean, detect_script

ALIGN_MIN_SIM = 0.7


def _is_subseq(a: str, b: str) -> bool:
    it = iter(b)
    return all(ch in it for ch in a)


def is_abbrev(a: str, b: str) -> bool:
    return 2 <= len(a) < len(b) and a[0] == b[0] and _is_subseq(a, b)


def _align(src: list[str], dst: list[str]) -> list[tuple[str, str]]:
    out = []
    for a in src:
        best, score = None, ALIGN_MIN_SIM
        for b in dst:
            s = JaroWinkler.normalized_similarity(a, b)
            if a != b and s >= score:
                best, score = b, s
        if best:
            out.append((a, best))
    return out


def _select(counts: Counter, totals: Counter, min_count: int, min_share: float) -> dict[str, str]:
    best: dict[str, tuple[str, int]] = {}
    for (a, b), n in counts.items():
        if n >= min_count and n / totals[a] >= min_share and n > best.get(a, ("", 0))[1]:
            best[a] = (b, n)
    return {a: b for a, (b, _) in best.items()}


def mine_lexicon(pairs: pl.DataFrame, min_count: int, abbrev_min_count: int, min_share: float) -> dict:
    result = {}
    for field, lcol, rcol in (("name", "l_name", "r_name"), ("addr", "l_addr", "r_addr")):
        t_counts, t_totals, a_counts, a_totals = Counter(), Counter(), Counter(), Counter()
        for left, right in zip(pairs[lcol].to_list(), pairs[rcol].to_list()):
            for x, y in ((left, right), (right, left)):
                x_latin, y_latin = detect_script(x) == "latin", detect_script(y) == "latin"
                xt = [t for t in basic_clean(x).split() if not any(c.isdigit() for c in t)]
                yt = [t for t in basic_clean(y).split() if not any(c.isdigit() for c in t)]
                if not x_latin and y_latin:
                    t_totals.update(xt)
                    t_counts.update(_align(xt, yt))
                elif x_latin and y_latin:
                    only_x, only_y = set(xt) - set(yt), set(yt) - set(xt)
                    a_totals.update(only_x)
                    a_counts.update((a, b) for a in only_x for b in only_y if is_abbrev(a, b))
        merged = _select(t_counts, t_totals, min_count, min_share)
        merged.update(_select(a_counts, a_totals, abbrev_min_count, min_share))
        result[field] = merged
    return result
