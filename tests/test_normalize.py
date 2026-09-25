import polars as pl

from ber.normalize import NORM_COLUMNS, normalize_frame, normalize_record

RAW = pl.DataFrame({
    "idx": [0, 1, 2],
    "business_name": ["Pvt. EFS Print Ventures Ltd.", "राम मार्केटिंग", "Zephay Labs Inc"],
    "business_address": ["Bangalore, 560011", "", "2621 Cotten Road, Tyler, TX"],
    "country": ["India", "India", "US"],
}, schema_overrides={"idx": pl.UInt32})


def test_record_order_matches_columns():
    rec = normalize_record("Zephay Labs Inc", "2621 Cotten Road, Tyler, TX", {})
    row = dict(zip(NORM_COLUMNS, rec))
    assert row["name_core"] == "zephay labs" and row["house_no"] == "2621" and row["name_script"] == "latin"


def test_frame_serial_equals_parallel():
    a = normalize_frame(RAW, {"name": {}, "addr": {}}, n_jobs=1)
    b = normalize_frame(RAW, {"name": {}, "addr": {}}, n_jobs=2, chunk=1)
    assert a.equals(b)
    assert a.columns == RAW.columns + NORM_COLUMNS
    assert a["is_web"].dtype == pl.Int8 and a["name_script"].to_list() == ["latin", "devanagari", "latin"]
