from ber.text import normalize_address


def test_reordered_us_address():
    out = normalize_address("GREENSBORO, NC, 19 1/2 STARDUST TRAIL", {})
    assert out == {
        "addr_norm": "greensboro north carolina 19 1 2 stardust trail",
        "postcode": "", "house_no": "19", "street_tok": "stardust",
        "num_tokens": "1 19 2", "localities": "greensboro|north carolina",
    }


def test_street_abbrev_and_type_excluded_from_street_tok():
    out = normalize_address("105 ELM ST, MORGANTON, NC", {})
    assert out["addr_norm"] == "105 elm street morganton north carolina"
    assert (out["house_no"], out["street_tok"]) == ("105", "elm")


def test_indian_pin_split_and_noise_words():
    out = normalize_address(
        "Door No 183, 41St Cross, 22Nd Main 9Th Block Jayanagar, Bengaluru Urban, Bangalore, 560 011", {})
    assert out["postcode"] == "560011" and out["house_no"] == "183"
    assert "bangalore" in out["localities"].split("|")
    assert "560" not in out["num_tokens"].split()


def test_house_number_not_mistaken_for_zip():
    out = normalize_address("17560 Ellis Road, Tahlequah, OK", {})
    assert (out["postcode"], out["house_no"], out["street_tok"]) == ("", "17560", "ellis")
    assert out["localities"] == "tahlequah|oklahoma"


def test_trailing_five_digit_is_postcode():
    out = normalize_address("Bordeaux, 33000", {})
    assert (out["postcode"], out["house_no"]) == ("33000", "")


def test_postcode_inside_locality_part_keeps_city():
    out = normalize_address("12 Main St, Tyler TX 75701", {})
    assert out["postcode"] == "75701" and out["localities"] == "tyler tx"


def test_empty_address():
    assert normalize_address("", {})["addr_norm"] == ""


def test_placeholder_tokens_dropped():
    out = normalize_address("null, N/A, 12 Main St, Tyler", {})
    assert out["addr_norm"] == "12 main street tyler"


def test_all_placeholder_address_is_empty():
    assert normalize_address("NULL", {})["addr_norm"] == ""


def test_single_a_token_kept_not_treated_as_placeholder():
    out = normalize_address("Lake Town Block A", {})
    assert "a" in out["addr_norm"].split()
