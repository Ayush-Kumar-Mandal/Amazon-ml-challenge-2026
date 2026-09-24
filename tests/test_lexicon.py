import polars as pl

from ber.lexicon import is_abbrev, mine_lexicon


def test_is_abbrev():
    assert is_abbrev("mfg", "manufacturing") and is_abbrev("blvd", "boulevard")
    assert not is_abbrev("manufacturing", "mfg") and not is_abbrev("xyz", "boulevard")


def test_mines_transliteration_map():
    pairs = pl.DataFrame({"l_name": ["प्राइवेट सन"] * 6, "r_name": ["Private Sun"] * 6,
                          "l_addr": ["x"] * 6, "r_addr": ["x"] * 6})
    lex = mine_lexicon(pairs, min_count=5, abbrev_min_count=5, min_share=0.6)
    assert lex["name"]["praivet"] == "private"


def test_mines_abbreviations_per_field():
    pairs = pl.DataFrame({"l_name": ["Acme Mfg"] * 25, "r_name": ["Acme Manufacturing"] * 25,
                          "l_addr": ["5 Oak Blvd"] * 25, "r_addr": ["5 Oak Boulevard"] * 25})
    lex = mine_lexicon(pairs, min_count=5, abbrev_min_count=20, min_share=0.6)
    assert lex["name"]["mfg"] == "manufacturing"
    assert lex["addr"]["blvd"] == "boulevard"
    assert "blvd" not in lex["name"]


def test_rare_pairs_are_dropped():
    pairs = pl.DataFrame({"l_name": ["Acme Mfg"] * 3, "r_name": ["Acme Manufacturing"] * 3,
                          "l_addr": [""] * 3, "r_addr": [""] * 3})
    assert mine_lexicon(pairs, min_count=5, abbrev_min_count=20, min_share=0.6)["name"] == {}
