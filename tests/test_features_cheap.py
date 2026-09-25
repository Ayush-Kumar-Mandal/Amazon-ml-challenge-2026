import polars as pl
import pytest

from ber.features import CHEAP_COLUMNS, cheap_features
from ber.normalize import normalize_frame

U, F = pl.UInt32, pl.Float32
LEX = {"name": {}, "addr": {}}


def norm(rows, sources=None):
    df = pl.DataFrame(rows, schema=["business_name", "business_address", "country"], orient="row").with_row_index("idx")
    if sources:
        df = df.with_columns(source=pl.Series(sources))
    return normalize_frame(df, LEX)


LEFT = norm([("Sunrise Bakery Inc", "12 Main St, Tyler TX 75701", "US"), ("Delta Motors LLC", "", "US")])
RIGHT = norm([("SUNRISE BAKERY INC", "12 MAIN STREET, TYLER", "US"), ("Sunrise Labs", "40 Oak Ave, Tyler", "US"),
              ("Delta Motors", "5 Elm Rd, Austin", "US")], sources=["S2", "S3", "S3"])
CANDS = pl.DataFrame({
    "l_idx": [0, 0, 1], "r_idx": [0, 1, 2], "tfidf_name": [0.9, 0.4, 0.8], "tfidf_name_addr": [0.8, 0.3, 0.5],
    "embed_cos": [0.0, 0.0, 0.0], "from_keys": [True, False, False], "cheap_score": [0.9, 0.2, 0.7],
}, schema_overrides={"l_idx": U, "r_idx": U, "tfidf_name": F, "tfidf_name_addr": F, "embed_cos": F, "cheap_score": F})


def test_cheap_features_values_and_order():
    out = cheap_features(CANDS.drop("cheap_score"), LEFT, RIGHT)
    assert out["r_idx"].to_list() == [0, 1, 2] and set(CHEAP_COLUMNS) <= set(out.columns)
    r0 = out.row(0, named=True)
    assert r0["name_jw"] == pytest.approx(1.0) and r0["name_tset"] == pytest.approx(100.0)
    assert (r0["house_eq"], r0["postcode_eq"], r0["source_s3"]) == (1, -1, 0)
    assert out.row(1, named=True)["source_s3"] == 1
