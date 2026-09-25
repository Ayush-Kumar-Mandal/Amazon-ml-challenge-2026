import pytest

from ber.text import basic_clean, detect_script, normalize_name, transliterate


@pytest.mark.parametrize("s,expected", [
    ("ಕರ್ನಾಟಕ", "kannada"), ("राम Traders", "devanagari"), ("Café Rouge", "latin"), ("", "latin"),
])
def test_detect_script(s, expected):
    assert detect_script(s) == expected


def test_transliterate_hindi_to_common_spelling():
    # If this fails only on spelling details, print the actual output and adjust the
    # schwa/anusvara handling in transliterate(); keep the intent (common English spelling).
    assert transliterate("राम मार्केटिंग प्राइवेट लिमिटेड") == "ram marketing praivet limited"


def test_basic_clean():
    assert basic_clean("Orelee's  Barber-Shop & Co.") == "orelees barber shop and co"
    assert basic_clean("G-3/571, Gulmohar", keep_commas=True) == "g 3 571, gulmohar"
    assert basic_clean("Moncada Léarning") == "moncada learning"


@pytest.mark.parametrize("raw,core,legal", [
    ("Pvt. EFS Print Ventures Ltd.", "efs print ventures", "ltd pvt"),
    ("LLC Moncada Léarning Center", "moncada learning center", "llc"),
    ("-- Holloway Peak Inc Seafood", "holloway peak seafood", "inc"),
    ("The Sunrise & Sons Corporation", "sunrise and sons", "corp"),
    ("Delta Intl Mfg", "delta international manufacturing", ""),
])
def test_normalize_name_core_and_legal(raw, core, legal):
    out = normalize_name(raw, {})
    assert (out["name_core"], out["legal_form"]) == (core, legal)


def test_website_name():
    out = normalize_name("wilfordhancock.com", {})
    assert out["name_core"] == "wilfordhancock" and out["is_web"] == 1


def test_dba_split():
    out = normalize_name("Acme Holdings LLC dba Acme Pizza", {})
    assert (out["name_core"], out["name_alt"], out["legal_form"]) == ("acme holdings", "acme pizza", "llc")


def test_lexicon_applied_before_legal_split():
    out = normalize_name("राम मार्केटिंग प्राइवेट लिमिटेड", {"praivet": "private"})
    assert (out["name_core"], out["legal_form"]) == ("ram marketing", "ltd pvt")
