import polars as pl

from ber.split import make_dev_slice, make_folds


def _s1(countries):
    return pl.DataFrame({"idx": list(range(len(countries))), "country": countries},
                        schema_overrides={"idx": pl.UInt32})


def test_random_folds_fraction_and_determinism():
    s1 = _s1(["US"] * 10000)
    cfg = {"seed": 1, "validation": {"scheme": "s1_random", "valid_frac": 0.2}}
    a, b = make_folds(s1, cfg), make_folds(s1, cfg)
    assert a.equals(b)
    share = (a["fold"] == "valid").mean()
    assert 0.18 < share < 0.22 and set(a["fold"]) == {"fit", "valid"}


def test_country_holdout_folds():
    s1 = _s1(["US"] * 1000 + ["India"] * 10 + ["France"] * 5)
    cfg = {"seed": 1, "validation": {"scheme": "country_holdout", "valid_frac": 0.2,
                                     "holdout_train_countries": ["US"], "holdout_eval_countries": ["India"]}}
    f = make_folds(s1, cfg).join(s1.rename({"idx": "l_idx"}), on="l_idx")
    assert set(f.filter(pl.col("country") == "US")["fold"]) == {"fit", "valid_seen"}
    assert set(f.filter(pl.col("country") == "India")["fold"]) == {"valid"}
    assert set(f.filter(pl.col("country") == "France")["fold"]) == {"unused"}


def test_dev_slice_reindexes_and_keeps_matches():
    s1 = pl.DataFrame({"idx": [0, 1, 2], "business_address": ["1 Main, Tyler", "2 Oak, Austin", "Kolkata"]},
                      schema_overrides={"idx": pl.UInt32})
    right = pl.DataFrame({"idx": [0, 1, 2, 3], "business_address": ["Austin", "x", "TYLER TX", "y"]},
                         schema_overrides={"idx": pl.UInt32})
    gt = pl.DataFrame({"l_idx": [0, 1, 2], "r_idx": [1, 0, 3]}, schema={"l_idx": pl.UInt32, "r_idx": pl.UInt32})
    s1d, rd, gtd = make_dev_slice(s1, right, gt, ["tyler", "kolkata"])
    assert s1d["idx"].to_list() == [0, 1] and rd["idx"].to_list() == [0, 1, 2]
    assert rd["business_address"].to_list() == ["x", "TYLER TX", "y"]
    assert sorted(zip(gtd["l_idx"].to_list(), gtd["r_idx"].to_list())) == [(0, 0), (1, 2)]
