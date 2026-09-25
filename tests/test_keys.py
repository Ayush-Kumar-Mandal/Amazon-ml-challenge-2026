import polars as pl

from ber.blocking.keys import join_keys, record_keys
from ber.normalize import normalize_frame

LEX = {"name": {}, "addr": {}}


def _norm(rows):
    df = pl.DataFrame(rows, schema=["business_name", "business_address", "country"], orient="row")
    return normalize_frame(df.with_row_index("idx"), LEX)


def test_house_number_street_key_links_variants_within_country():
    left = _norm([("Sunrise Bakery Inc", "12 Main St, Tyler TX 75701", "US")])
    right = _norm([("SUNRISE BAKERY", "12 MAIN STREET, TYLER", "US"),
                   ("Sunrise Bakery", "12 Main St, Tyler TX 75701", "France"),
                   ("Other Shop", "99 Oak Ave, Austin", "US")])
    pairs = join_keys(record_keys(left), record_keys(right), max_block_pairs=100)
    assert set(zip(pairs["l_idx"].to_list(), pairs["r_idx"].to_list())) == {(0, 0)}
    assert pairs["from_keys"].to_list() == [True]


def test_same_country_off_allows_cross_country():
    left = _norm([("Sunrise Bakery Inc", "12 Main St, Tyler TX 75701", "US")])
    right = _norm([("Sunrise Bakery", "12 Main St, Tyler TX 75701", "France")])
    pairs = join_keys(record_keys(left, same_country=False), record_keys(right, same_country=False), 100)
    assert pairs.height == 1


def test_block_cap_drops_huge_blocks():
    left = _norm([("Sunrise Bakery Inc", "12 Main St, Tyler TX 75701", "US")])
    right = _norm([("SUNRISE BAKERY", "12 MAIN STREET, TYLER", "US")])
    assert join_keys(record_keys(left), record_keys(right), max_block_pairs=0).height == 0
