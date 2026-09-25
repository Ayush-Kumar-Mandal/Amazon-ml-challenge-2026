"""Pair features. Country-agnostic: country is never a feature (script type is)."""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer

CHEAP_COLUMNS = ["tfidf_name", "tfidf_name_addr", "embed_cos", "from_keys", "name_jw", "name_tset",
                 "addr_tset", "postcode_eq", "house_eq", "source_s3"]


def attach_sides(c: pl.DataFrame, left: pl.DataFrame, right: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    lft = left.select(pl.col(cols).gather(c["l_idx"])).rename({x: f"l_{x}" for x in cols})
    rcols = cols + ["source"]
    rgt = right.select(pl.col(rcols).gather(c["r_idx"])).rename({x: f"r_{x}" for x in rcols})
    return pl.concat([c, lft, rgt], how="horizontal_extend")


def sim(a: pl.Series, b: pl.Series, scorer) -> np.ndarray:
    return process.cpdist(a.to_list(), b.to_list(), scorer=scorer, workers=-1, dtype=np.float32)


def tri(a: str, b: str) -> pl.Expr:
    """-1 if either side missing, 1 if equal, 0 if conflicting."""
    return (pl.when((pl.col(a) == "") | (pl.col(b) == "")).then(-1)
            .when(pl.col(a) == pl.col(b)).then(1).otherwise(0).cast(pl.Int8))


def cheap_features(c: pl.DataFrame, left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
    x = attach_sides(c, left, right, ["name_core", "addr_norm", "postcode", "house_no"])
    x = x.with_columns(
        name_jw=pl.Series(sim(x["l_name_core"], x["r_name_core"], JaroWinkler.normalized_similarity)),
        name_tset=pl.Series(sim(x["l_name_core"], x["r_name_core"], fuzz.token_set_ratio)),
        addr_tset=pl.Series(sim(x["l_addr_norm"], x["r_addr_norm"], fuzz.token_set_ratio)),
        postcode_eq=tri("l_postcode", "r_postcode"),
        house_eq=tri("l_house_no", "r_house_no"),
        source_s3=(pl.col("r_source") == "S3").cast(pl.Int8),
    )
    return x.select(c.columns + [col for col in CHEAP_COLUMNS if col not in c.columns])


CONTEXT_COLUMNS = ["cheap_score", "rank_l", "gap_l", "n_cands_l", "n_claims_r", "rank_r", "mutual_best",
                   "sim_other_name", "sim_other_addr"]
NAME_COLUMNS = ["name_ratio", "name_tsort", "name_partial", "name_lev", "name_norm_tset", "name_nospace_ratio",
                "name_alt_tset", "legal_eq", "first_tok_eq", "len_ratio", "name_num_eq", "name_char_cos",
                "name_word_cos", "l_nonlatin", "r_nonlatin", "l_web", "r_web"]
ADDR_COLUMNS = ["addr_ratio", "addr_partial", "addr_word_cos", "loc_jaccard", "num_jaccard", "street_eq",
                "l_addr_empty", "r_addr_empty"]
FEATURE_COLUMNS = CHEAP_COLUMNS + CONTEXT_COLUMNS + NAME_COLUMNS + ADDR_COLUMNS

_TEXT = ["name_norm", "name_core", "name_alt", "legal_form", "is_web", "name_script", "addr_norm",
         "postcode", "house_no", "street_tok", "num_tokens", "localities"]
VEC_SPECS = {
    "name_char": ("name_core", {"analyzer": "char_wb", "ngram_range": (3, 3), "min_df": 2}),
    "name_word": ("name_core", {"analyzer": "word", "token_pattern": r"\S+", "min_df": 2}),
    "addr_word": ("addr_norm", {"analyzer": "word", "token_pattern": r"\S+", "min_df": 2}),
}


def context_features(c: pl.DataFrame) -> pl.DataFrame:
    c = c.with_columns(
        rank_l=pl.col("cheap_score").rank("ordinal", descending=True).over("l_idx").cast(pl.Int32),
        gap_l=(pl.col("cheap_score").max().over("l_idx") - pl.col("cheap_score")).cast(pl.Float32),
        n_cands_l=pl.len().over("l_idx").cast(pl.Int32),
        n_claims_r=pl.len().over("r_idx").cast(pl.Int32),
        rank_r=pl.col("cheap_score").rank("ordinal", descending=True).over("r_idx").cast(pl.Int32),
    ).with_columns(mutual_best=((pl.col("rank_l") == 1) & (pl.col("rank_r") == 1)).cast(pl.Int8))
    best1 = c.filter(pl.col("rank_l") == 1).select("l_idx", pl.col("r_idx").alias("_b1"))
    best2 = c.filter(pl.col("rank_l") == 2).select("l_idx", pl.col("r_idx").alias("_b2"))
    return (c.join(best1, on="l_idx", how="left").join(best2, on="l_idx", how="left")
            .with_columns(other_r_idx=pl.when(pl.col("rank_l") == 1).then(pl.col("_b2")).otherwise(pl.col("_b1")))
            .drop("_b1", "_b2"))


def fit_vectorizers(left: pl.DataFrame, right: pl.DataFrame, fit_rows: int, seed: int) -> dict:
    out = {}
    for name, (col, kw) in VEC_SPECS.items():
        texts = pl.concat([left[col], right[col]])
        sample = texts.sample(min(fit_rows, texts.len()), seed=seed).to_list()
        vec = TfidfVectorizer(dtype=np.float32, sublinear_tf=True, **kw)
        try:
            vec.fit(sample)
        except ValueError:  # empty vocabulary at min_df=2 on tiny data
            vec = TfidfVectorizer(dtype=np.float32, sublinear_tf=True, **{**kw, "min_df": 1}).fit(sample)
        out[name] = (vec.transform(left[col].to_list()).tocsr(), vec.transform(right[col].to_list()).tocsr())
    return out


def _rowdot(a, b, li: np.ndarray, ri: np.ndarray, chunk: int = 1_000_000) -> np.ndarray:
    out = np.empty(len(li), np.float32)
    for s in range(0, len(li), chunk):
        out[s:s + chunk] = np.asarray(a[li[s:s + chunk]].multiply(b[ri[s:s + chunk]]).sum(axis=1)).ravel()
    return out


def _jaccard(a: str, b: str, sep: str) -> pl.Expr:
    la = pl.col(a).str.split(sep).list.eval(pl.element().filter(pl.element() != ""))
    lb = pl.col(b).str.split(sep).list.eval(pl.element().filter(pl.element() != ""))
    union = la.list.set_union(lb).list.len()
    return pl.when(union == 0).then(None).otherwise(la.list.set_intersection(lb).list.len() / union).cast(pl.Float32)


def full_features(c: pl.DataFrame, left: pl.DataFrame, right: pl.DataFrame, vecs: dict) -> pl.DataFrame:
    c = cheap_features(c, left, right)
    x = attach_sides(c, left, right, _TEXT)
    li, ri = c["l_idx"].to_numpy(), c["r_idx"].to_numpy()
    has_other = c["other_r_idx"].is_not_null().to_numpy()
    oi = np.where(has_other, c["other_r_idx"].fill_null(0).to_numpy(), ri)
    lc, rc = x["l_name_core"], x["r_name_core"]
    l_alt = x.select(pl.when(pl.col("l_name_alt") != "").then(pl.col("l_name_alt"))
                     .otherwise(pl.col("l_name_core"))).to_series()
    r_alt = x.select(pl.when(pl.col("r_name_alt") != "").then(pl.col("r_name_alt"))
                     .otherwise(pl.col("r_name_core"))).to_series()
    other_core, other_addr = right["name_core"].gather(oi), right["addr_norm"].gather(oi)
    tset = fuzz.token_set_ratio
    arrays = {
        "name_ratio": sim(lc, rc, fuzz.ratio),
        "name_tsort": sim(lc, rc, fuzz.token_sort_ratio),
        "name_partial": sim(lc, rc, fuzz.partial_ratio),
        "name_lev": sim(lc, rc, Levenshtein.normalized_similarity),
        "name_norm_tset": sim(x["l_name_norm"], x["r_name_norm"], tset),
        "name_nospace_ratio": sim(lc.str.replace_all(" ", ""), rc.str.replace_all(" ", ""), fuzz.ratio),
        "name_alt_tset": np.maximum(sim(l_alt, rc, tset), sim(lc, r_alt, tset)),
        "name_char_cos": _rowdot(*vecs["name_char"], li, ri),
        "name_word_cos": _rowdot(*vecs["name_word"], li, ri),
        "addr_ratio": sim(x["l_addr_norm"], x["r_addr_norm"], fuzz.ratio),
        "addr_partial": sim(x["l_addr_norm"], x["r_addr_norm"], fuzz.partial_ratio),
        "addr_word_cos": _rowdot(*vecs["addr_word"], li, ri),
        "sim_other_name": np.where(has_other, sim(rc, other_core, tset), np.nan).astype(np.float32),
        "sim_other_addr": np.where(has_other, sim(x["r_addr_norm"], other_addr, tset), np.nan).astype(np.float32),
    }
    x = x.with_columns(**{k: pl.Series(v) for k, v in arrays.items()})
    ln, rn = pl.col("l_name_core").str.len_chars(), pl.col("r_name_core").str.len_chars()
    first_l = pl.col("l_name_core").str.split(" ").list.first()
    first_r = pl.col("r_name_core").str.split(" ").list.first()
    x = x.with_columns(
        _l_num=pl.col("l_name_core").str.extract_all(r"\d+").list.join(" "),
        _r_num=pl.col("r_name_core").str.extract_all(r"\d+").list.join(" "),
    ).with_columns(
        legal_eq=tri("l_legal_form", "r_legal_form"),
        first_tok_eq=((first_l == first_r) & (pl.col("l_name_core") != "")).cast(pl.Int8),
        len_ratio=(pl.min_horizontal(ln, rn) / pl.max_horizontal(ln, rn, pl.lit(1))).cast(pl.Float32),
        name_num_eq=tri("_l_num", "_r_num"),
        l_nonlatin=(pl.col("l_name_script") != "latin").cast(pl.Int8),
        r_nonlatin=(pl.col("r_name_script") != "latin").cast(pl.Int8),
        l_web=pl.col("l_is_web"),
        r_web=pl.col("r_is_web"),
        loc_jaccard=_jaccard("l_localities", "r_localities", "|"),
        num_jaccard=_jaccard("l_num_tokens", "r_num_tokens", " "),
        street_eq=tri("l_street_tok", "r_street_tok"),
        l_addr_empty=(pl.col("l_addr_norm") == "").cast(pl.Int8),
        r_addr_empty=(pl.col("r_addr_norm") == "").cast(pl.Int8),
    )
    return x.select("l_idx", "r_idx", *FEATURE_COLUMNS)
