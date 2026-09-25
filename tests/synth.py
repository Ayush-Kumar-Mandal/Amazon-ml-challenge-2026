"""Synthetic ER dataset in the organizer file layout (for the end-to-end test)."""
from __future__ import annotations

import random
from pathlib import Path

import polars as pl

ADJ = ["sunrise", "prime", "delta", "vision", "summit", "zephyr", "golden", "silver", "royal", "united",
       "global", "metro", "green", "apex", "nova", "pioneer", "eagle", "crown", "harbor", "maple"]
NOUN = ["bakery", "trading", "labs", "partners", "consulting", "motors", "textiles", "foods", "builders", "pharma"]
STREETS = {"US": ["Main Street", "Oak Avenue", "Elm Road", "Park Lane"],
           "India": ["MG Road", "Station Road", "Lake Town", "Gandhi Nagar"],
           "France": ["Rue Victor Hugo", "Boulevard Voltaire", "Avenue Jean Jaures", "Rue de la Paix"]}
CITIES = {"US": ["Phoenix", "Cleveland", "Tyler"], "India": ["Kolkata", "Bhopal", "Pune"],
          "France": ["Bordeaux", "Lyon", "Nantes"]}
LEGAL = {"US": ["Inc", "LLC", "Corporation"], "India": ["Private Limited", "LLP", "Limited"],
         "France": ["SARL", "SAS", "SA"]}
SHORT = {"Corporation": "Corp", "Private Limited": "Pvt Ltd", "Limited": "Ltd", "Street": "St",
         "Avenue": "Ave", "Road": "Rd", "Boulevard": "Bd"}
COLS = ["entity_id", "business_name", "business_address", "country"]


def _typo(s: str, rng: random.Random) -> str:
    if len(s) < 5:
        return s
    i = rng.randrange(1, len(s) - 1)
    return s[:i] + rng.choice("abcdefghijklmnopqrstuvwxyz") + s[i + 1:]


def _variant(name: str, legal: str, parts: list[str], rng: random.Random) -> tuple[str, str]:
    n = f"{legal} {name}" if rng.random() < 0.15 else f"{name} {legal}"
    addr = list(parts)
    for long, short in SHORT.items():
        if rng.random() < 0.5:
            n = n.replace(long, short)
            addr = [a.replace(long, short) for a in addr]
    if rng.random() < 0.3:
        n = _typo(n, rng)
    if rng.random() < 0.2:
        n, addr = n.upper(), [a.upper() for a in addr]
    if rng.random() < 0.3:
        rng.shuffle(addr)
    return n, ", ".join(addr)


def make_split(split: str, n_entities: int, countries: list[str], seed: int) -> dict:
    rng = random.Random(seed)
    names = [f"{a} {b}".title() for a in ADJ for b in NOUN]
    rng.shuffle(names)
    rows = {"S1": [], "S2": [], "S3": []}
    gt = []

    def add(src: str, name: str, addr: str, country: str) -> str:
        eid = f"{src}-{split}{len(rows[src])}"
        rows[src].append((eid, name, addr, country))
        return eid

    for i in range(n_entities):
        c = countries[i % len(countries)]
        name, legal = names[i], rng.choice(LEGAL[c])
        parts = [f"{rng.randint(1, 999)} {rng.choice(STREETS[c])}", rng.choice(CITIES[c])]
        s1_id = add("S1", f"{name} {legal}", ", ".join(parts), c)
        singleton = rng.random() < 0.1
        k2 = 0 if singleton else rng.choice([0, 1, 1, 2])
        k3 = 0 if singleton else rng.choice([1, 1, 2])
        ids = [add("S2", *_variant(name, legal, parts, rng), c) for _ in range(k2)]
        ids += [add("S3", *_variant(name, legal, parts, rng), c) for _ in range(k3)]
        gt.append((s1_id, ",".join(ids)))
    for j in range(n_entities // 3):  # unmatched distractors
        c = countries[j % len(countries)]
        parts = [f"{rng.randint(1, 999)} {rng.choice(STREETS[c])}", rng.choice(CITIES[c])]
        add(rng.choice(["S2", "S3"]), *_variant(names[n_entities + j], rng.choice(LEGAL[c]), parts, rng), c)
    return {"source1": rows["S1"], "source2": rows["S2"], "source3": rows["S3"], "gt": gt}


def write_raw(root: Path, seed: int = 0) -> None:
    for split, n, countries in (("train", 120, ["US", "India"]), ("test", 60, ["US", "India", "France"])):
        data = make_split(split, n, countries, seed + (split == "test"))
        out = root / split
        out.mkdir(parents=True, exist_ok=True)
        for key in ("source1", "source2", "source3"):
            pl.DataFrame(data[key], schema=COLS, orient="row").write_csv(
                out / f"{split}_{key}.tsv", separator="\t", quote_style="never")
        if split == "train":
            pl.DataFrame(data["gt"], schema=["source1_entity_id", "matched_entity_ids"], orient="row").write_csv(
                out / "train_ground_truth.tsv", separator="\t", quote_style="never")
