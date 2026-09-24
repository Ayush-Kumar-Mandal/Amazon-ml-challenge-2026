"""Text normalization primitives for business names and addresses.

Country-agnostic: dictionaries hold general abbreviations/legal forms; nothing branches on country.
"""
from __future__ import annotations

import re
import unicodedata
from itertools import groupby

import jellyfish
from indic_transliteration import sanscript
from unidecode import unidecode

_INDIC = [
    (0x0900, 0x097F, sanscript.DEVANAGARI, "devanagari"),
    (0x0980, 0x09FF, sanscript.BENGALI, "bengali"),
    (0x0A00, 0x0A7F, sanscript.GURMUKHI, "gurmukhi"),
    (0x0A80, 0x0AFF, sanscript.GUJARATI, "gujarati"),
    (0x0B00, 0x0B7F, sanscript.ORIYA, "oriya"),
    (0x0B80, 0x0BFF, sanscript.TAMIL, "tamil"),
    (0x0C00, 0x0C7F, sanscript.TELUGU, "telugu"),
    (0x0C80, 0x0CFF, sanscript.KANNADA, "kannada"),
    (0x0D00, 0x0D7F, sanscript.MALAYALAM, "malayalam"),
]

LEGAL_CANON = {
    "inc": "inc", "incorporated": "inc", "llc": "llc", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "ltd": "ltd", "limited": "ltd", "pvt": "pvt", "private": "pvt",
    "llp": "llp", "lp": "lp", "plc": "plc", "pllc": "pllc", "opc": "opc", "gmbh": "gmbh",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl", "sci": "sci", "snc": "snc",
}
NAME_ABBREV = {
    "intl": "international", "mfg": "manufacturing", "mfrs": "manufacturers", "svc": "service",
    "svcs": "services", "assoc": "associates", "assn": "association", "bros": "brothers",
    "ctr": "center", "centre": "center", "dept": "department", "natl": "national",
    "mgmt": "management", "grp": "group", "sys": "systems", "engg": "engineering",
    "inds": "industries", "hosp": "hospital", "univ": "university", "trdg": "trading",
}
ADDR_ABBREV = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court",
    "pl": "place", "hwy": "highway", "pkwy": "parkway", "sq": "square", "ter": "terrace",
    "cir": "circle", "trl": "trail", "apt": "apartment", "ste": "suite", "fl": "floor",
    "bldg": "building", "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "nr": "near", "opp": "opposite", "sec": "sector", "ph": "phase", "extn": "extension",
    "ext": "extension", "ngr": "nagar", "mkt": "market", "dist": "district",
    "r": "rue", "rte": "route", "chem": "chemin", "imp": "impasse", "fbg": "faubourg",
}
STREET_TYPES = {
    "street", "road", "avenue", "boulevard", "drive", "lane", "court", "place", "highway",
    "parkway", "square", "terrace", "circle", "trail", "way", "rue", "route", "chemin",
    "impasse", "allee", "marg", "path", "sector", "nagar", "colony", "block", "main", "cross",
}
NON_STREET_WORDS = {
    "door", "no", "number", "plot", "flat", "house", "shop", "unit", "suite", "apartment",
    "floor", "building", "near", "opposite", "bis", "ter", "po", "box", "north", "south",
    "east", "west", "northeast", "northwest", "southeast", "southwest",
}
US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "new hampshire",
    "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee",
    "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
}
_EDGE_STOP = {"the", "and", "of"}
_SCHWA = re.compile(r"(?<=[bcdfghjklmnpqrstvwxyzḍṅñṇṣśṭ])a\b")
_APOS = re.compile(r"['’`´]")
_WEB_TLD = re.compile(r"\.(?:com|net|org|in|co|fr|biz|info|io|us)\b")
_DBA = re.compile(r"\b(?:dba|d b a|doing business as)\b")


def _block(ch: str):
    o = ord(ch)
    if o < 0x0900:
        return None
    for lo, hi, scheme, name in _INDIC:
        if lo <= o <= hi:
            return scheme, name
    return None


def detect_script(s: str) -> str:
    counts: dict[str, int] = {}
    for ch in s:
        if ord(ch) <= 0x024F:
            continue
        block = _block(ch)
        name = block[1] if block else ("other" if ch.isalpha() else None)
        if name:
            counts[name] = counts.get(name, 0) + 1
    return max(counts, key=counts.get) if counts else "latin"


def _iast_to_plain(seg: str) -> str:
    seg = _SCHWA.sub("", seg)                  # drop inherent final schwa: rāma -> rām
    seg = re.sub(r"ṃ(?=[pbm])", "m", seg)      # anusvara before labials
    return seg.replace("ṃ", "n").replace("m̐", "n")


def transliterate(s: str) -> str:
    parts = []
    for block, chars in groupby(s, key=_block):
        seg = "".join(chars)
        if block is not None:
            seg = _iast_to_plain(sanscript.transliterate(seg, block[0], sanscript.IAST))
        parts.append(seg)
    return unidecode("".join(parts))


def basic_clean(s: str, keep_commas: bool = False) -> str:
    s = transliterate(unicodedata.normalize("NFKC", s))
    s = _APOS.sub("", s).lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9,]+" if keep_commas else r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def map_tokens(tokens: list[str], seed: dict, lex: dict) -> list[str]:
    return " ".join(seed.get(t) or lex.get(t, t) for t in tokens).split()


def _strip_edges(tokens: list[str]) -> list[str]:
    while tokens and tokens[0] in _EDGE_STOP:
        tokens = tokens[1:]
    while tokens and tokens[-1] in _EDGE_STOP:
        tokens = tokens[:-1]
    return tokens


def normalize_name(raw: str, lex: dict) -> dict:
    low = raw.strip().lower()
    is_web = bool(_WEB_TLD.search(low))
    if is_web:
        low = _WEB_TLD.sub(" ", re.sub(r"\b(?:https?://)?www\.", " ", low))
    s = " ".join(map_tokens(basic_clean(low).split(), NAME_ABBREV, lex))
    pieces = _DBA.split(s, maxsplit=1)
    main = pieces[0].split()
    alt = pieces[1].split() if len(pieces) > 1 else []
    legal = sorted({LEGAL_CANON[t] for t in main + alt if t in LEGAL_CANON})
    core = _strip_edges([t for t in main if t not in LEGAL_CANON]) or main
    alt_core = _strip_edges([t for t in alt if t not in LEGAL_CANON])
    return {
        "name_norm": s,
        "name_core": " ".join(core),
        "name_alt": " ".join(alt_core),
        "legal_form": " ".join(legal),
        "is_web": int(is_web),
        "name_key": " ".join(jellyfish.metaphone(t) for t in core[:2]),
    }


def _has_digit(t: str) -> bool:
    return any(c.isdigit() for c in t)


# R8 (M0 EDA): literal placeholder tokens seen embedded in real addresses.
# "N/A" survives basic_clean as the two adjacent tokens "n", "a" (slash -> space).
_ADDR_PLACEHOLDERS = {"null", "na", "none", "nil"}


def _drop_placeholders(toks: list[str]) -> list[str]:
    out, i = [], 0
    while i < len(toks):
        if toks[i:i + 2] == ["n", "a"]:
            i += 2
        elif toks[i] in _ADDR_PLACEHOLDERS:
            i += 1
        else:
            out.append(toks[i])
            i += 1
    return out


def _find_postcode(parts: list[list[str]]) -> tuple[str, set[str]]:
    """Last 6-digit token, 3+3 digit pair, or 5-digit token that ends its part / precedes digits."""
    pc, used = "", set()
    for toks in parts:
        for i, t in enumerate(toks):
            nxt = toks[i + 1] if i + 1 < len(toks) else ""
            if t.isdigit() and len(t) == 6:
                pc, used = t, {t}
            elif t.isdigit() and len(t) == 3 and nxt.isdigit() and len(nxt) == 3:
                pc, used = t + nxt, {t, nxt}
            elif t.isdigit() and len(t) == 5 and (not nxt or nxt.isdigit()):
                pc, used = t, {t}
    return pc, used


def _house_and_street(parts: list[list[str]], pc_toks: set[str]) -> tuple[str, str, int]:
    excluded = STREET_TYPES | NON_STREET_WORDS
    for pi, toks in enumerate(parts):
        nums = [t for t in toks if _has_digit(t) and t not in pc_toks]
        if nums:
            for part in parts[pi:pi + 2]:
                words = [t for t in part if t.isalpha() and len(t) > 1 and t not in excluded]
                if words:
                    return nums[0], max(words, key=len), pi
            return nums[0], "", pi
    return "", "", -1


def normalize_address(raw: str, lex: dict) -> dict:
    parts: list[list[str]] = []
    for chunk in basic_clean(raw, keep_commas=True).split(","):
        toks = _drop_placeholders(chunk.split())
        if len(toks) == 1 and toks[0] in US_STATES:
            toks = US_STATES[toks[0]].split()
        toks = map_tokens(toks, ADDR_ABBREV, lex)
        if toks:
            parts.append(toks)
    postcode, pc_toks = _find_postcode(parts)
    house_no, street_tok, house_part = _house_and_street(parts, pc_toks)
    nums = sorted({t for p in parts for t in p if _has_digit(t)} - pc_toks)
    localities = []
    for i, p in enumerate(parts):
        words = [t for t in p if not _has_digit(t)]
        if i != house_part and words:
            localities.append(" ".join(words))
    return {
        "addr_norm": " ".join(t for p in parts for t in p),
        "postcode": postcode,
        "house_no": house_no,
        "street_tok": street_tok,
        "num_tokens": " ".join(nums),
        "localities": "|".join(localities),
    }
