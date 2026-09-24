# Business Entity Resolution: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **After finishing each task, update `memory.md`** (Status, Results and Pitfalls sections).

**Goal:** Build a reproducible pipeline that takes the challenge's three business-record sources and produces `matching_results.tsv` and `candidate_pairs.tsv`. Every matched ID is an S2/S3 record that refers to the same business as an S1 record. The target is the best possible macro F0.5 on the private leaderboard.

**Architecture:**
- **Stages:** a CLI of stages (`python -m ber.pipeline --stage X --split Y`). Each stage reads the parquet files of earlier stages from `work_dir/<split>/` and writes its own.
- **Resuming on Kaggle:** a new session reads the previous session's outputs through `prev_work_dirs`.
- **Stage 1 (M1):** normalize, then key and TF-IDF blocking, then a cheap LightGBM cut to top-25 per S1, then full features, then LightGBM, then calibration, then the one-owner rule, then the expected-F0.5 decision.
- **Stage 2 (M2):** adds a fine-tuned multilingual bi-encoder, used for blocking and as a feature.
- **Stage 3 (M3):** adds an xlm-roberta cross-encoder on uncertain pairs, plus a combining LightGBM.

**Tech Stack:**
- Python 3.10+
- Data: polars, pyarrow, numpy, scipy
- Text: rapidfuzz, jellyfish, unidecode, indic-transliteration
- Retrieval and models: scikit-learn, sparse_dot_topn, lightgbm, faiss
- Neural models: torch, transformers, sentence-transformers, datasets
- Tests: pytest
- Compute: Kaggle notebooks (about 29 GB RAM, 4 CPU cores, T4 or P100 GPUs, 12-hour sessions)

**Spec:** `docs/superpowers/specs/2026-09-25-business-entity-resolution-design.md`

## Global Constraints

- **Model licensing:** the final model must be MIT or Apache-2.0 licensed with at most 8B parameters. Only these are used:
  - `intfloat/multilingual-e5-small` (MIT)
  - `xlm-roberta-base` (MIT)
  - LightGBM (MIT)
- **No external data:** no external databases, APIs, geocoding or internet data augmentation. Only the provided train and test files.
- **Countries:** treat `country` as an open set. Never hard-code, filter or one-hot it to {US, India}. France appears only in test, and every test S1 must appear in the output.
- **TSV I/O:** all TSV reads use `separator="\t"` and `quote_char=None`, with every column as a string and missing values as `""`. All TSV writes are tab-separated, UTF-8 and unquoted.
- **`matching_results.tsv`:**
  - header `source1_entity_id\tmatched_entity_ids`
  - exactly one row per test S1
  - empty string for no match
  - comma-joined S2/S3 IDs, with no duplicates
- **`candidate_pairs.tsv`:** header `source1_entity_id\tcandidate_entity_ids`, same rules. Matches ⊆ candidates.
- **Validator gate:** `python -m ber.vendor.validate_submission ...` must print `PASS` (exit 0) before any upload.
- **Features:** country is never a model feature. Script type is used instead.
- **Stage rule:** a stage (M2, M3, M4 changes) is kept only if it improves held-out macro F0.5 on **both** the `s1_random` and `country_holdout` schemes (logged in `experiments.csv`).
- **Local data location:** outside OneDrive, at `C:/Users/AYUSH/ber_data/`. Never commit data, the dataset zip or the PDF.
- **Local Python:** a 3.10 venv (`py -3.10 -m venv .venv`). Local commands use `.venv/Scripts/python`. On Kaggle, use `python`.
- **Commits:** every commit message ends with a blank line and then `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. The commit commands below show only the subject; always add `-m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`.
- **Logbook:** update `memory.md` at the end of every task.

---

## File Structure

| Path | Responsibility |
|---|---|
| `pyproject.toml`, `requirements.in`, `requirements.txt` | package `ber` (src layout); unpinned inputs; pinned freeze from Kaggle |
| `.gitignore` | keep data, zip, PDF, venv and work dirs out of git |
| `configs/base.yaml` | every tunable, with defaults |
| `configs/dev.yaml` | local paths and small sample sizes |
| `configs/kaggle.yaml` | Kaggle paths |
| `configs/kaggle_holdout.yaml` | the US→India scheme |
| `src/ber/config.py` | YAML loading (`extends`, env vars), `--set` overrides, `read_path`/`write_path`, split-mapping tables |
| `src/ber/io.py` | raw TSV (zip or directory) → `s1.parquet`, `right.parquet`, `gt.parquet` |
| `src/ber/metrics.py` | scalar and frame-based macro F0.5, blocking recall, experiment log |
| `src/ber/split.py` | folds (`fit` / `valid` / `valid_seen` / `unused`), dev slice |
| `src/ber/text.py` | script detection, transliteration, cleaning, legal forms, abbreviations, address parsing |
| `src/ber/lexicon.py` | mines transliteration and abbreviation token maps from training matches |
| `src/ber/normalize.py` | applies `text.py` to whole frames (multiprocess, in slices) |
| `src/ber/blocking/keys.py` | exact-key blocking |
| `src/ber/blocking/tfidf_ann.py` | char-trigram TF-IDF top-k search in both directions, within each country |
| `src/ber/blocking/embed_ann.py` | FAISS k-nearest-neighbours over embeddings; pair cosine |
| `src/ber/blocking/merge.py` | union of candidate sources; top-k per S1 |
| `src/ber/features.py` | cheap features, context features, full features, vectorizers |
| `src/ber/models/gbm.py` | LightGBM train, predict, save and load |
| `src/ber/models/biencoder.py` | triplet builder, e5 fine-tuning, encoding to a memmap |
| `src/ber/models/crossencoder.py` | xlm-r pair classifier training and scoring |
| `src/ber/decide.py` | calibration, one-owner rule, expected-F0.5 subset, threshold tuning, unseen-country scaling |
| `src/ber/submit.py` | write ID-list TSVs, run the validator, build the submission zip |
| `src/ber/vendor/validate_submission.py` | exact copy of the official validator |
| `src/ber/stages/*.py` | stage functions `(cfg, split) -> result`, grouped by area |
| `src/ber/pipeline.py` | CLI, stage registry, timing |
| `tests/` | unit tests per module, `synth.py` (synthetic dataset), end-to-end integration test |
| `scripts/eda_m0.py`, `scripts/error_analysis.py` | analysis scripts (not part of the pipeline) |
| `notebooks/kaggle_driver.py` | Kaggle notebook cells (`# %%`) |
| `docs/Documentation.md` | the filled-in organizer template, updated from M1 on |
| `plan.md`, `memory.md` | this plan; the living project logbook |

**Artifacts** in `work_dir/<split>/` (split ∈ `train`, `test`, `dev`):
- data: `s1.parquet`, `right.parquet`, `gt.parquet`, `folds.parquet`, `lexicon.json`, `s1_norm.parquet`, `right_norm.parquet`
- candidates: `cands_keys.parquet`, `cands_embed.parquet`, `cands_all.parquet`, `cands_final.parquet`
- features and predictions: `features/part-*.parquet`, `valid_pred.parquet`, `fit_pred.parquet`, `pred.parquet`
- models and decisions: `cheap_model.txt`, `gbm_model.txt`, `calibrator.joblib`, `decision.json`

**Split mapping:**
- The `test` split uses the lexicon, models and calibrator from `train`.
- The `dev` split uses its own.

## Milestone M0: Data and validation

### Task 1: Repo scaffold, environment, config loader, vendored validator, GitHub

**Files:**
- Create: `.gitignore`, `pyproject.toml`, `requirements.in`, `README.md` (stub), `configs/base.yaml`, `configs/dev.yaml`, `configs/kaggle.yaml`, `configs/kaggle_holdout.yaml`
- Create: `src/ber/__init__.py`, `src/ber/config.py`, `src/ber/vendor/__init__.py`, `src/ber/vendor/validate_submission.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `load_config(path) -> dict`
  - `apply_override(cfg: dict, expr: str) -> None`
  - `read_path(cfg, split, name) -> Path`
  - `write_path(cfg, split, name) -> Path`
  - `LEXICON_SPLIT`, `MODEL_SPLIT`: dict[str, str]
  - the `ber.vendor.validate_submission` module (the official `main()`)

- [ ] **Step 1: Git repo and ignore file.** Run in the project root:

```bash
git init -b main
```

Create `.gitignore`:

```gitignore
*.zip
*.pdf
.venv/
__pycache__/
*.egg-info/
.pytest_cache/
work/
scratch/
output/
*.parquet
*.npy
*.joblib
.DS_Store
```

- [ ] **Step 2: Packaging files.**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "ber"
version = "0.1.0"
description = "Business entity resolution pipeline (Amazon ML Challenge 2026)"
requires-python = ">=3.10"
dependencies = [
  "polars>=1.0", "pyarrow", "numpy", "scipy", "scikit-learn", "joblib",
  "rapidfuzz>=3.6", "jellyfish", "unidecode", "indic-transliteration",
  "sparse_dot_topn>=1.1", "lightgbm>=4.0", "pyyaml",
]

[project.optional-dependencies]
dev = ["pytest"]
ann = ["faiss-cpu"]
nn = ["torch", "transformers", "sentence-transformers>=3.0", "datasets", "accelerate"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Create `requirements.in` (unpinned; the pinned `requirements.txt` is generated on Kaggle in Task 17):

```text
polars>=1.0
pyarrow
numpy
scipy
scikit-learn
joblib
rapidfuzz>=3.6
jellyfish
unidecode
indic-transliteration
sparse_dot_topn>=1.1
lightgbm>=4.0
pyyaml
faiss-cpu
torch
transformers
sentence-transformers>=3.0
datasets
accelerate
```

Create `README.md` stub: `# ber — Business Entity Resolution (Amazon ML Challenge 2026)\n\nSee plan.md and memory.md. Full reproduction steps are added in Task 24.\n`

- [ ] **Step 3: Create the venv and install.**

```bash
py -3.10 -m venv .venv
.venv/Scripts/python -m pip install -U pip
.venv/Scripts/python -m pip install -e ".[dev,ann]"
```

Expected: installs without errors. If `sparse_dot_topn` has no wheel for this Python/Windows combination, record it in `memory.md` → Pitfalls and continue. The TF-IDF tests will be skipped locally and run on Kaggle.

- [ ] **Step 4: Vendor the official validator.** Copy it out of the dataset zip byte-for-byte:

```bash
mkdir -p src/ber/vendor && touch src/ber/vendor/__init__.py src/ber/__init__.py
.venv/Scripts/python -c "import zipfile; z=zipfile.ZipFile('6ab10eb3b23ba_student_resource.zip'); open('src/ber/vendor/validate_submission.py','wb').write(z.read('student_resource/utils/validate_submission.py'))"
.venv/Scripts/python -c "import zipfile; z=zipfile.ZipFile('6ab10eb3b23ba_student_resource.zip'); open('docs/Documentation_template_original.md','wb').write(z.read('student_resource/Documentation_template.md'))"
```

- [ ] **Step 5: Write the failing config tests.** Create `tests/test_config.py`:

```python
from pathlib import Path

import pytest

from ber.config import apply_override, load_config, read_path, write_path


def test_extends_merges_nested(tmp_path: Path):
    (tmp_path / "base.yaml").write_text("a: {x: 1, y: 2}\nb: 5\n", encoding="utf-8")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\na: {y: 3}\n", encoding="utf-8")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg == {"a": {"x": 1, "y": 3}, "b": 5}


def test_env_vars_expanded(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BER_TEST_DIR", "/data/x")
    (tmp_path / "c.yaml").write_text("paths: {work_dir: $BER_TEST_DIR/work}\n", encoding="utf-8")
    assert load_config(tmp_path / "c.yaml")["paths"]["work_dir"] == "/data/x/work"


def test_apply_override_parses_yaml_values():
    cfg = {"blocking": {"tfidf": {"top_k_fwd": 40}, "embed": {"enabled": False}}}
    apply_override(cfg, "blocking.tfidf.top_k_fwd=60")
    apply_override(cfg, "blocking.embed.enabled=true")
    apply_override(cfg, "new.key=[a, b]")
    assert cfg["blocking"]["tfidf"]["top_k_fwd"] == 60
    assert cfg["blocking"]["embed"]["enabled"] is True
    assert cfg["new"]["key"] == ["a", "b"]


def test_read_path_falls_back_to_prev_work_dirs(tmp_path: Path):
    prev = tmp_path / "prev"
    (prev / "train").mkdir(parents=True)
    (prev / "train" / "s1.parquet").write_bytes(b"x")
    cfg = {"paths": {"work_dir": str(tmp_path / "work"), "prev_work_dirs": [str(prev)]}}
    assert read_path(cfg, "train", "s1.parquet") == prev / "train" / "s1.parquet"
    out = write_path(cfg, "train", "s1.parquet")
    assert out == tmp_path / "work" / "train" / "s1.parquet" and out.parent.is_dir()
    with pytest.raises(FileNotFoundError):
        read_path(cfg, "train", "missing.parquet")
```

- [ ] **Step 6: Run the tests and confirm they fail.**

Run: `.venv/Scripts/python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ber.config'`

- [ ] **Step 7: Implement `src/ber/config.py`.**

```python
"""Config loading (YAML with `extends` + env vars), CLI overrides and artifact paths."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

# Which split's artifacts (lexicon, models, calibrator) a split uses.
LEXICON_SPLIT = {"train": "train", "dev": "dev", "test": "train"}
MODEL_SPLIT = {"train": "train", "dev": "dev", "test": "train"}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _expand(obj):
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(os.path.expanduser(obj))
    return obj


def load_config(path: str | Path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = _merge(load_config(path.parent / parent), cfg)
    return _expand(cfg)


def apply_override(cfg: dict, expr: str) -> None:
    """Apply `a.b.c=value`, parsing value as YAML (numbers, bools, lists)."""
    dotted, _, raw = expr.partition("=")
    keys = dotted.strip().split(".")
    node = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = yaml.safe_load(raw)


def write_path(cfg: dict, split: str, name: str) -> Path:
    path = Path(cfg["paths"]["work_dir"]) / split / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_path(cfg: dict, split: str, name: str) -> Path:
    bases = [cfg["paths"]["work_dir"], *cfg["paths"].get("prev_work_dirs", [])]
    for base in bases:
        path = Path(base) / split / name
        if path.exists():
            return path
    raise FileNotFoundError(f"{split}/{name} not found under {bases}")
```

- [ ] **Step 8: Run the tests and confirm they pass.**

Run: `.venv/Scripts/python -m pytest tests/test_config.py -v`
Expected: 4 passed

- [ ] **Step 9: Write the configs.** Create `configs/base.yaml`:

```yaml
paths:
  raw_zip: ""               # local: path to the student_resource zip
  raw_dir: ""               # Kaggle: extracted .../student_resource/dataset (wins over raw_zip)
  work_dir: ./work
  prev_work_dirs: []        # read-only fallbacks, e.g. a previous Kaggle session's output
  scratch_dir: ./scratch    # large temporary files (embeddings)
  output_dir: ./output
seed: 42
run_name: default
experiments_csv: ./experiments.csv
validation:
  scheme: s1_random         # s1_random | country_holdout
  valid_frac: 0.2
  holdout_train_countries: [US]
  holdout_eval_countries: [India]
dev:
  localities: [phoenix, cleveland, tyler, kolkata, bhopal]
lexicon:
  sample_pairs: 500000
  min_count: 5
  abbrev_min_count: 20
  min_share: 0.6
normalize:
  n_jobs: 4
  slice_rows: 1000000
blocking:
  same_country: true
  max_block_pairs: 5000
  tfidf:
    views: [name, name_addr]
    ngram: 3
    min_df: 2
    max_df: 0.02
    top_k_fwd: 40
    top_k_rev: 3
    threshold: 0.3
    chunk_size: 20000
    n_threads: 4
  embed:
    enabled: false
    top_k_fwd: 30
    top_k_rev: 3
    use_gpu: true
cheap:
  keep_top: 25
  max_train_s1: 300000
  num_boost_round: 300
  early_stopping_rounds: 30
  chunk_rows: 5000000
  params: {num_leaves: 63, learning_rate: 0.1, min_data_in_leaf: 100}
features:
  chunk_rows: 2000000
  max_train_s1: 800000
  vectorizer_fit_rows: 2000000
gbm:
  num_boost_round: 3000
  early_stopping_rounds: 100
  max_es_s1: 100000
  predict_chunk_rows: 5000000
  params: {}
decide:
  one_owner: true
  seen_countries: [US, India]
  unseen_country_scale: 1.0
evaluate:
  pred_file: valid_pred.parquet
submit:
  pred_file: pred.parquet
  team_name: team
biencoder:
  model_name: intfloat/multilingual-e5-small
  max_seq_len: 64
  n_triplets: 2000000
  epochs: 1
  batch_size: 256
  lr: 5.0e-5
  encode_batch_size: 1024
cross:
  model_name: xlm-roberta-base
  max_len: 96
  band_lo: 0.1
  band_hi: 0.9
  n_train_pairs: 1000000
  epochs: 1
  batch_size: 64
  lr: 2.0e-5
  score_batch_size: 512
combiner:
  n_folds: 5
```

Create `configs/dev.yaml`:

```yaml
extends: base.yaml
paths:
  raw_zip: "C:/Users/AYUSH/OneDrive/Desktop/Amazon mlChallenge 20226/6ab10eb3b23ba_student_resource.zip"
  work_dir: C:/Users/AYUSH/ber_data/work
  scratch_dir: C:/Users/AYUSH/ber_data/scratch
  output_dir: C:/Users/AYUSH/ber_data/output
run_name: dev
normalize: {n_jobs: 6}
blocking: {tfidf: {n_threads: 8, max_df: 0.05}}
cheap: {max_train_s1: 50000}
features: {max_train_s1: 100000}
gbm: {max_es_s1: 20000}
```

Create `configs/kaggle.yaml`:

```yaml
extends: base.yaml
paths:
  raw_dir: /kaggle/input/ber-raw/student_resource/dataset   # verify with `ls` in Task 17
  work_dir: /kaggle/working/work
  prev_work_dirs: []        # set per session, e.g. [/kaggle/input/ber-m1-a/work]
  scratch_dir: /kaggle/tmp/ber
  output_dir: /kaggle/working/output
run_name: kaggle
experiments_csv: /kaggle/working/experiments.csv
```

Create `configs/kaggle_holdout.yaml`:

```yaml
extends: kaggle.yaml
paths:
  work_dir: /kaggle/working/work_holdout
run_name: kaggle_holdout
validation:
  scheme: country_holdout
```

- [ ] **Step 10: GitHub remote.** The user chose a private GitHub repo.
  1. Check `gh --version`.
  2. If `gh` is present and authenticated: **confirm the repo name with the user**, then run `gh repo create ber-2026 --private --source . --remote origin`.
  3. Otherwise, ask the user to create the private repo `ber-2026` on github.com and run `git remote add origin https://github.com/<user>/ber-2026.git`.
  4. Record the repo URL in `memory.md` → Environment.

- [ ] **Step 11: Commit and push.**

```bash
git add .gitignore pyproject.toml requirements.in README.md configs src tests docs plan.md memory.md
git commit -m "chore: scaffold ber package, configs, vendored validator"
git push -u origin main
```

---

### Task 2: Raw ingestion to parquet and pipeline CLI skeleton

**Files:**
- Create: `src/ber/io.py`, `src/ber/stages/__init__.py`, `src/ber/stages/common.py`, `src/ber/stages/data.py`, `src/ber/pipeline.py`
- Test: `tests/test_io.py`

**Interfaces:**
- Consumes: `read_path`, `write_path`, `apply_override`, `load_config` (Task 1)
- Produces:
  - `read_tsv(source: bytes | str | Path) -> pl.DataFrame`
  - `parse_ground_truth(gt_raw, s1, right) -> pl.DataFrame[l_idx: u32, r_idx: u32]`
  - `ingest(cfg, split) -> None`, which writes:
    - `s1.parquet` [idx u32, entity_id, business_name, business_address, country]
    - `right.parquet` (same columns plus `source` "S2"/"S3")
    - `gt.parquet` (train only)
  - `stages.common`: `load(cfg, split, name)`, `save(df, cfg, split, name)`, `load_norm(cfg, split, side)`, `sample_l(folds, fold, n, seed)`, `eval_folds(cfg)`, `label_pairs(cands, gt) -> np.ndarray[int8]`
  - `pipeline.main(argv)` and `pipeline.STAGES: dict[str, Callable[[dict, str], object]]`

- [ ] **Step 1: Write the failing tests.** Create `tests/test_io.py`:

```python
import zipfile
from pathlib import Path

import polars as pl

from ber.io import ingest, read_tsv

S1 = "entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-1\tAcme \"Best\" Inc\t1 Main St, Tyler, TX\tUS\nS1-2\tSolo Co\t\tUS\n"
S2 = "entity_id\tbusiness_name\tbusiness_address\tcountry\nS2-1\tACME BEST INC\t1 MAIN ST, TYLER\tUS\n"
S3 = "entity_id\tbusiness_name\tbusiness_address\tcountry\nS3-1\tAcme Best\tTyler, TX\tUS\nS3-2\tOther\tX\tUS\n"
GT = "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S3-1\nS1-2\t\n"


def _zip(tmp_path: Path) -> Path:
    path = tmp_path / "raw.zip"
    with zipfile.ZipFile(path, "w") as z:
        base = "student_resource/dataset/train/"
        z.writestr(base + "train_source1.tsv", S1)
        z.writestr(base + "train_source2.tsv", S2)
        z.writestr(base + "train_source3.tsv", S3)
        z.writestr(base + "train_ground_truth.tsv", GT)
    return path


def test_read_tsv_keeps_quotes_and_empty_strings():
    df = read_tsv(S1.encode())
    assert df["business_name"][0] == 'Acme "Best" Inc'
    assert df["business_address"][1] == ""


def test_ingest_from_zip(tmp_path: Path):
    cfg = {"paths": {"raw_zip": str(_zip(tmp_path)), "raw_dir": "", "work_dir": str(tmp_path / "w")}}
    ingest(cfg, "train")
    s1 = pl.read_parquet(tmp_path / "w/train/s1.parquet")
    right = pl.read_parquet(tmp_path / "w/train/right.parquet")
    gt = pl.read_parquet(tmp_path / "w/train/gt.parquet")
    assert s1["idx"].to_list() == [0, 1] and s1["idx"].dtype == pl.UInt32
    assert right["source"].to_list() == ["S2", "S3", "S3"]
    assert right["entity_id"].to_list() == ["S2-1", "S3-1", "S3-2"]
    assert sorted(zip(gt["l_idx"].to_list(), gt["r_idx"].to_list())) == [(0, 0), (0, 1)]
```

- [ ] **Step 2: Run the tests and confirm they fail.**

Run: `.venv/Scripts/python -m pytest tests/test_io.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ber.io'`

- [ ] **Step 3: Implement `src/ber/io.py`.**

```python
"""Read the organizer TSVs (from the zip locally, or an extracted dir on Kaggle) into parquet."""
from __future__ import annotations

import zipfile
from pathlib import Path

import polars as pl

from ber.config import write_path


def read_tsv(source: bytes | str | Path) -> pl.DataFrame:
    df = pl.read_csv(
        source, separator="\t", quote_char=None, infer_schema=False,
        missing_utf8_is_empty_string=True,
    )
    return df.with_columns(pl.all().fill_null(""))


def open_raw(cfg: dict, split: str, filename: str) -> bytes:
    raw_dir = cfg["paths"].get("raw_dir")
    if raw_dir:
        return (Path(raw_dir) / split / filename).read_bytes()
    with zipfile.ZipFile(cfg["paths"]["raw_zip"]) as z:
        return z.read(f"student_resource/dataset/{split}/{filename}")


def parse_ground_truth(gt_raw: pl.DataFrame, s1: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
    pairs = (
        gt_raw.rename({"source1_entity_id": "l_id", "matched_entity_ids": "r_id"})
        .with_columns(pl.col("r_id").str.split(","))
        .explode("r_id")
        .filter(pl.col("r_id") != "")
    )
    joined = pairs.join(
        s1.select(pl.col("entity_id").alias("l_id"), pl.col("idx").alias("l_idx")), on="l_id"
    ).join(
        right.select(pl.col("entity_id").alias("r_id"), pl.col("idx").alias("r_idx")), on="r_id"
    )
    if joined.height != pairs.height:
        print(f"[ingest] WARNING: {pairs.height - joined.height} ground-truth ids not found in sources")
    return joined.select("l_idx", "r_idx").unique()


def ingest(cfg: dict, split: str) -> None:
    s1 = read_tsv(open_raw(cfg, split, f"{split}_source1.tsv")).with_row_index("idx")
    s2 = read_tsv(open_raw(cfg, split, f"{split}_source2.tsv")).with_columns(source=pl.lit("S2"))
    s3 = read_tsv(open_raw(cfg, split, f"{split}_source3.tsv")).with_columns(source=pl.lit("S3"))
    right = pl.concat([s2, s3]).with_row_index("idx")
    s1.write_parquet(write_path(cfg, split, "s1.parquet"))
    right.write_parquet(write_path(cfg, split, "right.parquet"))
    print(f"[ingest] {split}: s1={s1.height:,} right={right.height:,}")
    if split == "train":
        gt = parse_ground_truth(read_tsv(open_raw(cfg, split, "train_ground_truth.tsv")), s1, right)
        gt.write_parquet(write_path(cfg, split, "gt.parquet"))
        print(f"[ingest] gt pairs={gt.height:,}")
```

- [ ] **Step 4: Implement `src/ber/stages/common.py`, `stages/__init__.py` (empty), `stages/data.py` (ingest only for now) and `pipeline.py`.**

`src/ber/stages/common.py`:

```python
"""Helpers shared by stage functions."""
from __future__ import annotations

import numpy as np
import polars as pl

from ber.config import read_path, write_path


def load(cfg: dict, split: str, name: str) -> pl.DataFrame:
    return pl.read_parquet(read_path(cfg, split, name))


def save(df: pl.DataFrame, cfg: dict, split: str, name: str) -> None:
    df.write_parquet(write_path(cfg, split, name))


def load_norm(cfg: dict, split: str, side: str) -> pl.DataFrame:
    df = load(cfg, split, f"{side}_norm.parquet")
    if not np.array_equal(df["idx"].to_numpy(), np.arange(df.height)):
        raise ValueError(f"{split}/{side}_norm.parquet: idx must equal row position")
    return df


def sample_l(folds: pl.DataFrame, fold: str, n: int | None, seed: int) -> pl.DataFrame:
    ids = folds.filter(pl.col("fold") == fold).select("l_idx")
    return ids.sample(n, seed=seed) if n and ids.height > n else ids


def eval_folds(cfg: dict) -> list[str]:
    return ["valid", "valid_seen"] if cfg["validation"]["scheme"] == "country_holdout" else ["valid"]


def label_pairs(cands: pl.DataFrame, gt: pl.DataFrame) -> np.ndarray:
    lab = (
        cands.select("l_idx", "r_idx").with_row_index("_row")
        .join(gt.select("l_idx", "r_idx").with_columns(y=pl.lit(1, pl.Int8)), on=["l_idx", "r_idx"], how="left")
        .sort("_row")
    )
    return lab["y"].fill_null(0).to_numpy()
```

`src/ber/stages/data.py`:

```python
"""Data stages: ingest, dev slice, folds, lexicon, normalize."""
from __future__ import annotations

from ber.io import ingest


def stage_ingest(cfg: dict, split: str) -> None:
    ingest(cfg, split)
```

`src/ber/pipeline.py`:

```python
"""CLI: python -m ber.pipeline --config configs/dev.yaml --stage ingest --split train [--set a.b=v]"""
from __future__ import annotations

import argparse
import time

from ber.config import apply_override, load_config
from ber.stages import data

STAGES = {
    "ingest": data.stage_ingest,
}


def run(cfg: dict, stage: str, split: str):
    start = time.time()
    result = STAGES[stage](cfg, split)
    print(f"[pipeline] {stage} --split {split} done in {time.time() - start:,.0f}s")
    return result


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    ap.add_argument("--split", default="train", choices=["train", "test", "dev"])
    ap.add_argument("--set", action="append", default=[], help="override, e.g. blocking.tfidf.top_k_fwd=60")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    for expr in args.set:
        apply_override(cfg, expr)
    return run(cfg, args.stage, args.split)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests and confirm they pass.**

Run: `.venv/Scripts/python -m pytest tests/test_io.py tests/test_config.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit.**

```bash
git add src/ber/io.py src/ber/stages src/ber/pipeline.py tests/test_io.py
git commit -m "feat: ingest raw TSVs to parquet; pipeline CLI skeleton"
```

---

### Task 3: Metrics (macro F0.5, blocking recall, experiment log)

**Files:**
- Create: `src/ber/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `f05(pred: set, true: set) -> float`
  - `macro_f05_frame(pred: pl.DataFrame[l_idx, r_idx], gt: pl.DataFrame[l_idx, r_idx], eval_l: pl.Series) -> float`
  - `blocking_recall(cands, gt, eval_l) -> float`
  - `cands_per_s1(cands, eval_l) -> float`
  - `log_experiment(path, row: dict) -> None`

- [ ] **Step 1: Write the failing tests.** Create `tests/test_metrics.py`:

```python
import polars as pl
import pytest

from ber.metrics import blocking_recall, cands_per_s1, f05, log_experiment, macro_f05_frame


def test_pdf_worked_example():
    assert f05({"S2-47", "S2-193", "S3-812"}, {"S2-47", "S3-812"}) == pytest.approx(0.714, abs=1e-3)


def test_singleton_rules():
    assert f05(set(), set()) == 1.0
    assert f05({"S2-1"}, set()) == 0.0
    assert f05(set(), {"S2-1"}) == 0.0


def _pairs(rows):
    return pl.DataFrame(rows, schema={"l_idx": pl.UInt32, "r_idx": pl.UInt32}, orient="row")


def test_frame_matches_scalar():
    gt = _pairs([(0, 10), (0, 11), (1, 12)])           # l=2 is a singleton
    pred = _pairs([(0, 10), (0, 11), (0, 13), (2, 14)])  # l=1 predicted empty, l=2 false merge
    eval_l = pl.Series("l_idx", [0, 1, 2], dtype=pl.UInt32)
    expected = (f05({10, 11, 13}, {10, 11}) + f05(set(), {12}) + f05({14}, set())) / 3
    assert macro_f05_frame(pred, gt, eval_l) == pytest.approx(expected)


def test_frame_only_scores_eval_ids():
    gt = _pairs([(0, 10), (1, 12)])
    pred = _pairs([(0, 10)])
    assert macro_f05_frame(pred, gt, pl.Series("l_idx", [0], dtype=pl.UInt32)) == 1.0


def test_blocking_recall_and_density():
    gt = _pairs([(0, 10), (0, 11), (1, 12)])
    cands = _pairs([(0, 10), (0, 99), (1, 12), (1, 98)])
    eval_l = pl.Series("l_idx", [0, 1], dtype=pl.UInt32)
    assert blocking_recall(cands, gt, eval_l) == pytest.approx(2 / 3)
    assert cands_per_s1(cands, eval_l) == 2.0


def test_log_experiment_appends_with_new_columns(tmp_path):
    path = tmp_path / "exp.csv"
    log_experiment(path, {"run": "a", "f05": 0.5})
    log_experiment(path, {"run": "b", "f05": 0.6, "recall": 0.9})
    df = pl.read_csv(path, infer_schema=False)
    assert df["run"].to_list() == ["a", "b"] and df["recall"].to_list() == [None, "0.9"]
```

- [ ] **Step 2: Run the tests and confirm they fail.**

Run: `.venv/Scripts/python -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ber.metrics'`

- [ ] **Step 3: Implement `src/ber/metrics.py`.**

```python
"""Challenge metric (macro F0.5 per S1 entity) and blocking diagnostics.

F_beta = (1+b^2) TP / (b^2 |T| + |P|); with b = 0.5: 1.25 TP / (0.25 |T| + |P|).
Singletons: 1.0 if nothing predicted, else 0.0.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl


def f05(pred: set, true: set) -> float:
    if not true:
        return 1.0 if not pred else 0.0
    tp = len(pred & true)
    return 1.25 * tp / (0.25 * len(true) + len(pred))


def macro_f05_frame(pred: pl.DataFrame, gt: pl.DataFrame, eval_l: pl.Series) -> float:
    ev = eval_l.alias("l_idx").to_frame().unique()
    pred = pred.select("l_idx", "r_idx").unique().join(ev, on="l_idx", how="semi")
    gt = gt.select("l_idx", "r_idx").join(ev, on="l_idx", how="semi")
    tp = pred.join(gt, on=["l_idx", "r_idx"]).group_by("l_idx").len("tp")
    table = (
        ev.join(pred.group_by("l_idx").len("n_pred"), on="l_idx", how="left")
        .join(gt.group_by("l_idx").len("n_true"), on="l_idx", how="left")
        .join(tp, on="l_idx", how="left")
        .fill_null(0)
    )
    score = (
        pl.when(pl.col("n_true") == 0)
        .then((pl.col("n_pred") == 0).cast(pl.Float64))
        .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("n_true") + pl.col("n_pred")))
    )
    return float(table.select(score.mean()).item())


def blocking_recall(cands: pl.DataFrame, gt: pl.DataFrame, eval_l: pl.Series) -> float:
    ev = eval_l.alias("l_idx").to_frame().unique()
    truth = gt.select("l_idx", "r_idx").join(ev, on="l_idx", how="semi")
    if truth.height == 0:
        return 1.0
    found = truth.join(cands.select("l_idx", "r_idx").unique(), on=["l_idx", "r_idx"], how="semi")
    return found.height / truth.height


def cands_per_s1(cands: pl.DataFrame, eval_l: pl.Series) -> float:
    ev = eval_l.alias("l_idx").to_frame().unique()
    return cands.join(ev, on="l_idx", how="semi").height / max(ev.height, 1)


def log_experiment(path: str | Path, row: dict) -> None:
    row = {"time": datetime.now().isoformat(timespec="seconds"), **row}
    new = pl.DataFrame([{k: str(v) for k, v in row.items()}])
    path = Path(path)
    if path.exists():
        new = pl.concat([pl.read_csv(path, infer_schema=False), new], how="diagonal")
    new.write_csv(path)
```

- [ ] **Step 4: Run the tests and confirm they pass.**

Run: `.venv/Scripts/python -m pytest tests/test_metrics.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit.**

```bash
git add src/ber/metrics.py tests/test_metrics.py
git commit -m "feat: macro F0.5, blocking recall, experiment log"
```

---

### Task 4: Ingest the real train data locally and run the M0 EDA

**Files:**
- Create: `scripts/eda_m0.py`
- Modify: `memory.md` (EDA findings)

**Interfaces:**
- Consumes: the `ingest` stage and `read_path`
- Produces: facts recorded in `memory.md` that later tasks depend on:
  - Is the cross-country match share 0? (This decides `blocking.same_country`.)
  - Do any right records have more than one S1 owner? (This checks the one-owner rule.)
  - Script mix, and missing-address rates.

- [ ] **Step 1: Ingest train locally.** Disk check first: `C:` needs at least 3 GB free.

```bash
.venv/Scripts/python -m ber.pipeline --config configs/dev.yaml --stage ingest --split train
```

Expected: `[ingest] train: s1=2,206,821 right=10,320,219` and `gt pairs=7,638,365`. Record any "ids not found" warning in `memory.md`.

- [ ] **Step 2: Write `scripts/eda_m0.py`.**

```python
"""M0 EDA on the train split. Usage: python scripts/eda_m0.py --config configs/dev.yaml"""
from __future__ import annotations

import argparse

import polars as pl

from ber.config import load_config, read_path

pl.Config.set_tbl_rows(40)
pl.Config.set_fmt_str_lengths(80)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    s1 = pl.read_parquet(read_path(cfg, "train", "s1.parquet"))
    right = pl.read_parquet(read_path(cfg, "train", "right.parquet"))
    gt = pl.read_parquet(read_path(cfg, "train", "gt.parquet"))

    print("== field quality by source/country")
    both = pl.concat([s1.with_columns(source=pl.lit("S1")), right], how="diagonal")
    print(both.group_by("source", "country").agg(
        n=pl.len(),
        empty_name=(pl.col("business_name") == "").mean(),
        empty_addr=(pl.col("business_address") == "").mean(),
        nonascii_name=pl.col("business_name").str.contains(r"[^\x00-\x7F]").mean(),
        nonascii_addr=pl.col("business_address").str.contains(r"[^\x00-\x7F]").mean(),
        has_6digit=pl.col("business_address").str.contains(r"\b\d{6}\b").mean(),
        has_5digit=pl.col("business_address").str.contains(r"\b\d{5}\b").mean(),
    ).sort("source", "country"))

    pairs = gt.join(
        s1.select(pl.col("idx").alias("l_idx"), pl.col("country").alias("lc"),
                  pl.col("business_name").alias("l_name"), pl.col("business_address").alias("l_addr")),
        on="l_idx",
    ).join(
        right.select(pl.col("idx").alias("r_idx"), pl.col("country").alias("rc"), "source",
                     pl.col("business_name").alias("r_name"), pl.col("business_address").alias("r_addr")),
        on="r_idx",
    )
    print("== cross-country match share:", (pairs["lc"] != pairs["rc"]).mean())

    owners = gt.group_by("r_idx").len("n_owners")
    print("== right records with >1 S1 owner:", (owners["n_owners"] > 1).sum())
    print("== share of right records that match some S1:", owners.height / right.height)

    sizes = s1.select(pl.col("idx").alias("l_idx"), "country").join(
        gt.group_by("l_idx").len("n"), on="l_idx", how="left").fill_null(0)
    print(sizes.group_by("country").agg(
        singleton=(pl.col("n") == 0).mean(), mean=pl.col("n").mean(), p99=pl.col("n").quantile(0.99)))
    print(pairs.group_by("lc", "source").len().sort("lc", "source"))

    print("== 40 random matched pairs")
    print(pairs.sample(40, seed=0).select("lc", "source", "l_name", "r_name", "l_addr", "r_addr"))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the script and write the findings into `memory.md` → "EDA findings".**

Run: `.venv/Scripts/python scripts/eda_m0.py --config configs/dev.yaml`

Write down:
- the cross-country share
- the number of right records with more than one owner
- the per-country singleton rate
- the non-ASCII rates
- how often 5- and 6-digit numbers appear
- 5–10 noteworthy noise examples

**Decisions to record:**
- If cross-country share == 0, keep `blocking.same_country: true`. Otherwise set it to `false` in `base.yaml` and note it.
- If some right records have more than one owner, record the count. If it is above 0.1% of matched right records, set `decide.one_owner: false` in `base.yaml`.

- [ ] **Step 4: Commit.**

```bash
git add scripts/eda_m0.py memory.md configs/base.yaml
git commit -m "chore: M0 EDA script and findings"
```

---

### Task 5: Folds and dev slice

**Files:**
- Create: `src/ber/split.py`
- Modify: `src/ber/stages/data.py`, `src/ber/pipeline.py`
- Test: `tests/test_split.py`

**Interfaces:**
- Consumes: `load`, `save` (Task 2)
- Produces:
  - `make_folds(s1, cfg) -> pl.DataFrame[l_idx u32, fold str]`. `fold` is one of `fit`, `valid`, `valid_seen`, `unused`.
  - `make_dev_slice(s1, right, gt, localities) -> (s1, right, gt)`, re-indexed with idx = row position.
  - stages `split` (writes `folds.parquet`) and `dev_slice` (reads train, writes dev `s1`/`right`/`gt`)

- [ ] **Step 1: Write the failing tests.** Create `tests/test_split.py`:

```python
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
```

- [ ] **Step 2: Run the tests and confirm they fail.**

Run: `.venv/Scripts/python -m pytest tests/test_split.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ber.split'`

- [ ] **Step 3: Implement `src/ber/split.py`.**

```python
"""Validation folds and the local dev slice."""
from __future__ import annotations

import re

import numpy as np
import polars as pl


def make_folds(s1: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    v = cfg["validation"]
    rng = np.random.default_rng(cfg["seed"])
    draw = rng.random(s1.height) < v["valid_frac"]
    if v["scheme"] == "s1_random":
        fold = np.where(draw, "valid", "fit")
    elif v["scheme"] == "country_holdout":
        country = s1["country"]
        is_train = country.is_in(v["holdout_train_countries"]).to_numpy()
        is_eval = country.is_in(v["holdout_eval_countries"]).to_numpy()
        fold = np.where(is_train, np.where(draw, "valid_seen", "fit"), np.where(is_eval, "valid", "unused"))
    else:
        raise ValueError(f"unknown validation scheme {v['scheme']!r}")
    return pl.DataFrame({"l_idx": s1["idx"], "fold": fold})


def _reindex(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort("idx").rename({"idx": "old_idx"}).with_row_index("idx")


def make_dev_slice(s1: pl.DataFrame, right: pl.DataFrame, gt: pl.DataFrame, localities: list[str]):
    pat = "|".join(re.escape(x.lower()) for x in localities)
    in_area = pl.col("business_address").str.to_lowercase().str.contains(pat)
    s1_sel = s1.filter(in_area)
    gt_sel = gt.join(s1_sel.select(pl.col("idx").alias("l_idx")), on="l_idx", how="semi")
    keep = pl.concat([gt_sel.select(pl.col("r_idx").alias("idx")),
                      right.filter(in_area).select("idx")]).unique()
    right_sel = right.join(keep, on="idx", how="semi")
    s1_new, right_new = _reindex(s1_sel), _reindex(right_sel)
    gt_new = (
        gt_sel.join(s1_new.select(pl.col("old_idx").alias("l_idx"), pl.col("idx").alias("l_new")), on="l_idx")
        .join(right_new.select(pl.col("old_idx").alias("r_idx"), pl.col("idx").alias("r_new")), on="r_idx")
        .select(pl.col("l_new").alias("l_idx"), pl.col("r_new").alias("r_idx"))
    )
    return s1_new.drop("old_idx"), right_new.drop("old_idx"), gt_new
```

- [ ] **Step 4: Add the stages.** Append to `src/ber/stages/data.py`:

```python
from ber.split import make_dev_slice, make_folds
from ber.stages.common import load, save


def stage_dev_slice(cfg: dict, split: str) -> None:
    s1, right, gt = make_dev_slice(load(cfg, "train", "s1.parquet"), load(cfg, "train", "right.parquet"),
                                   load(cfg, "train", "gt.parquet"), cfg["dev"]["localities"])
    save(s1, cfg, "dev", "s1.parquet")
    save(right, cfg, "dev", "right.parquet")
    save(gt, cfg, "dev", "gt.parquet")
    print(f"[dev_slice] s1={s1.height:,} right={right.height:,} gt={gt.height:,}")


def stage_split(cfg: dict, split: str) -> None:
    folds = make_folds(load(cfg, split, "s1.parquet"), cfg)
    save(folds, cfg, split, "folds.parquet")
    print(folds.group_by("fold").len())
```

(Move the new imports to the top of the file, next to the existing `from ber.io import ingest`.)

In `pipeline.py`, add these to `STAGES`: `"dev_slice": data.stage_dev_slice, "split": data.stage_split`.

- [ ] **Step 5: Run the tests and confirm they pass, then build the dev slice and folds.**

Run: `.venv/Scripts/python -m pytest tests/test_split.py -v`
Expected: 3 passed

```bash
.venv/Scripts/python -m ber.pipeline --config configs/dev.yaml --stage dev_slice
.venv/Scripts/python -m ber.pipeline --config configs/dev.yaml --stage split --split dev
```

Expected: the dev slice has roughly 10k–60k S1 records. If it has more than 80k, drop a locality from `configs/dev.yaml`. Record the sizes in `memory.md`.

- [ ] **Step 6: Commit.**

```bash
git add src/ber/split.py src/ber/stages/data.py src/ber/pipeline.py tests/test_split.py configs/dev.yaml memory.md
git commit -m "feat: validation folds (random + country holdout) and dev slice"
```

## Milestone M1: First real submission

### Task 6: Text primitives (script, transliteration, cleaning, name normalization)

**Files:**
- Create: `src/ber/text.py`
- Test: `tests/test_text.py`

**Interfaces:**
- Produces:
  - `detect_script(s) -> str` ("latin", "devanagari", "kannada", …, "other")
  - `transliterate(s) -> str` (ASCII output)
  - `basic_clean(s, keep_commas=False) -> str`
  - `normalize_name(raw, lex_name: dict) -> dict` with keys `name_norm, name_core, name_alt, legal_form, is_web, name_key`
  - constants `LEGAL_CANON`, `NAME_ABBREV`, `ADDR_ABBREV`, `STREET_TYPES`, `NON_STREET_WORDS`, `US_STATES`

- [ ] **Step 1: Write the failing tests.** Create `tests/test_text.py`:

```python
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
```

- [ ] **Step 2: Run the tests and confirm they fail.**

Run: `.venv/Scripts/python -m pytest tests/test_text.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ber.text'`

- [ ] **Step 3: Implement `src/ber/text.py` (name part; the address part is added in Task 7).**

```python
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
```

- [ ] **Step 4: Run the tests and confirm they pass.**

Run: `.venv/Scripts/python -m pytest tests/test_text.py -v`
Expected: all passed. If only the transliteration spelling asserts fail, print `sanscript.transliterate(...)` for the inputs and fix `_iast_to_plain`. Do not weaken the test intent.

- [ ] **Step 5: Commit.**

```bash
git add src/ber/text.py tests/test_text.py
git commit -m "feat: script detection, transliteration, name normalization"
```

---

### Task 7: Address parsing

**Files:**
- Modify: `src/ber/text.py` (append)
- Test: `tests/test_text_address.py`

**Interfaces:**
- Consumes: `basic_clean`, `map_tokens`, `ADDR_ABBREV`, `US_STATES`, `STREET_TYPES`, `NON_STREET_WORDS` (Task 6)
- Produces: `normalize_address(raw, lex_addr: dict) -> dict` with keys `addr_norm, postcode, house_no, street_tok, num_tokens, localities`
  - `num_tokens` is space-joined and sorted.
  - `localities` is "|"-joined, one entry per comma part that isn't the house-number part, with digit tokens removed.

- [ ] **Step 1: Write the failing tests.** Create `tests/test_text_address.py`:

```python
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
```

- [ ] **Step 2: Run the tests and confirm they fail.**

Run: `.venv/Scripts/python -m pytest tests/test_text_address.py -v`
Expected: FAIL with `ImportError: cannot import name 'normalize_address'`

- [ ] **Step 3: Append to `src/ber/text.py`.**

```python
def _has_digit(t: str) -> bool:
    return any(c.isdigit() for c in t)


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
        toks = chunk.split()
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
```

- [ ] **Step 4: Run the tests and confirm they pass.**

Run: `.venv/Scripts/python -m pytest tests/test_text_address.py tests/test_text.py -v`
Expected: all passed

- [ ] **Step 5: Commit.**

```bash
git add src/ber/text.py tests/test_text_address.py
git commit -m "feat: address parsing (postcode, house number, street, localities)"
```

---

### Task 8: Lexicon mining (transliteration and abbreviation maps from training matches)

**Files:**
- Create: `src/ber/lexicon.py`
- Modify: `src/ber/stages/data.py`, `src/ber/pipeline.py`
- Test: `tests/test_lexicon.py`

**Interfaces:**
- Consumes: `basic_clean`, `detect_script` (Task 6); `load` (Task 2); `write_path` (Task 1)
- Produces:
  - `mine_lexicon(pairs: pl.DataFrame[l_name, r_name, l_addr, r_addr], min_count, abbrev_min_count, min_share) -> {"name": {tok: tok}, "addr": {tok: tok}}`
  - stage `lexicon`, which writes `lexicon.json`, mined only from the `fit` fold

- [ ] **Step 1: Write the failing tests.** Create `tests/test_lexicon.py`:

```python
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
```

- [ ] **Step 2: Run the tests and confirm they fail.**

Run: `.venv/Scripts/python -m pytest tests/test_lexicon.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ber.lexicon'`

- [ ] **Step 3: Implement `src/ber/lexicon.py`.**

```python
"""Mine token maps from training matches (allowed: uses only the provided training data).

- transliteration: tokens of a non-Latin name/address aligned to the best Jaro-Winkler token
  on the Latin side of the same true match (e.g. praivet -> private)
- abbreviations: Latin tokens present on one side only that are an in-order subsequence of a
  longer token on the other side, sharing the first letter (e.g. mfg -> manufacturing)
"""
from __future__ import annotations

from collections import Counter

import polars as pl
from rapidfuzz.distance import JaroWinkler

from ber.text import basic_clean, detect_script

ALIGN_MIN_SIM = 0.7


def _is_subseq(a: str, b: str) -> bool:
    it = iter(b)
    return all(ch in it for ch in a)


def is_abbrev(a: str, b: str) -> bool:
    return 2 <= len(a) < len(b) and a[0] == b[0] and _is_subseq(a, b)


def _align(src: list[str], dst: list[str]) -> list[tuple[str, str]]:
    out = []
    for a in src:
        best, score = None, ALIGN_MIN_SIM
        for b in dst:
            s = JaroWinkler.normalized_similarity(a, b)
            if a != b and s >= score:
                best, score = b, s
        if best:
            out.append((a, best))
    return out


def _select(counts: Counter, totals: Counter, min_count: int, min_share: float) -> dict[str, str]:
    best: dict[str, tuple[str, int]] = {}
    for (a, b), n in counts.items():
        if n >= min_count and n / totals[a] >= min_share and n > best.get(a, ("", 0))[1]:
            best[a] = (b, n)
    return {a: b for a, (b, _) in best.items()}


def mine_lexicon(pairs: pl.DataFrame, min_count: int, abbrev_min_count: int, min_share: float) -> dict:
    result = {}
    for field, lcol, rcol in (("name", "l_name", "r_name"), ("addr", "l_addr", "r_addr")):
        t_counts, t_totals, a_counts, a_totals = Counter(), Counter(), Counter(), Counter()
        for left, right in zip(pairs[lcol].to_list(), pairs[rcol].to_list()):
            for x, y in ((left, right), (right, left)):
                x_latin, y_latin = detect_script(x) == "latin", detect_script(y) == "latin"
                xt = [t for t in basic_clean(x).split() if not any(c.isdigit() for c in t)]
                yt = [t for t in basic_clean(y).split() if not any(c.isdigit() for c in t)]
                if not x_latin and y_latin:
                    t_totals.update(xt)
                    t_counts.update(_align(xt, yt))
                elif x_latin and y_latin:
                    only_x, only_y = set(xt) - set(yt), set(yt) - set(xt)
                    a_totals.update(only_x)
                    a_counts.update((a, b) for a in only_x for b in only_y if is_abbrev(a, b))
        merged = _select(t_counts, t_totals, min_count, min_share)
        merged.update(_select(a_counts, a_totals, abbrev_min_count, min_share))
        result[field] = merged
    return result
```

- [ ] **Step 4: Add the stage.** Append to `src/ber/stages/data.py` (with `import json`, `import polars as pl` and `from ber.lexicon import mine_lexicon` at the top, and `from ber.config import write_path`):

```python
def stage_lexicon(cfg: dict, split: str) -> None:
    s1, right = load(cfg, split, "s1.parquet"), load(cfg, split, "right.parquet")
    fit = load(cfg, split, "folds.parquet").filter(pl.col("fold") == "fit").select("l_idx")
    pairs = load(cfg, split, "gt.parquet").join(fit, on="l_idx", how="semi")
    n = cfg["lexicon"]["sample_pairs"]
    if pairs.height > n:
        pairs = pairs.sample(n, seed=cfg["seed"])
    pairs = pairs.join(
        s1.select(pl.col("idx").alias("l_idx"), pl.col("business_name").alias("l_name"),
                  pl.col("business_address").alias("l_addr")), on="l_idx",
    ).join(
        right.select(pl.col("idx").alias("r_idx"), pl.col("business_name").alias("r_name"),
                     pl.col("business_address").alias("r_addr")), on="r_idx",
    )
    lc = cfg["lexicon"]
    lex = mine_lexicon(pairs, lc["min_count"], lc["abbrev_min_count"], lc["min_share"])
    write_path(cfg, split, "lexicon.json").write_text(
        json.dumps(lex, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    print(f"[lexicon] name={len(lex['name'])} addr={len(lex['addr'])} entries")
```

Register `"lexicon": data.stage_lexicon` in `pipeline.STAGES`.

- [ ] **Step 5: Run the tests, then mine the dev lexicon and inspect it.**

Run: `.venv/Scripts/python -m pytest tests/test_lexicon.py -v`
Expected: 4 passed

```bash
.venv/Scripts/python -m ber.pipeline --config configs/dev.yaml --stage lexicon --split dev
```

Open `C:/Users/AYUSH/ber_data/work/dev/lexicon.json`. Record 10 good and 10 bad entries in `memory.md`. If bad entries dominate, raise `min_share` to 0.75 in `base.yaml`.

- [ ] **Step 6: Commit.**

```bash
git add src/ber/lexicon.py src/ber/stages/data.py src/ber/pipeline.py tests/test_lexicon.py memory.md
git commit -m "feat: mine transliteration/abbreviation lexicon from training matches"
```

---

### Task 9: Frame normalization and the `normalize` stage

**Files:**
- Create: `src/ber/normalize.py`
- Modify: `src/ber/stages/data.py`, `src/ber/pipeline.py`
- Test: `tests/test_normalize.py`

**Interfaces:**
- Consumes: `normalize_name`, `normalize_address`, `detect_script` (Tasks 6–7); `LEXICON_SPLIT`, `read_path` (Task 1)
- Produces:
  - `NORM_COLUMNS: list[str]` = `["name_norm","name_core","name_alt","legal_form","is_web","name_key","name_script","addr_norm","postcode","house_no","street_tok","num_tokens","localities"]`
  - `normalize_record(name, address, lex) -> tuple`, in `NORM_COLUMNS` order
  - `normalize_frame(df, lex, n_jobs=1, chunk=50_000) -> pl.DataFrame`: the input columns plus `NORM_COLUMNS` (`is_web` is Int8, the rest are String)
  - stage `normalize`, which writes `s1_norm.parquet` and `right_norm.parquet`

- [ ] **Step 1: Write the failing tests.** Create `tests/test_normalize.py`:

```python
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
```

- [ ] **Step 2: Run the tests and confirm they fail.**

Run: `.venv/Scripts/python -m pytest tests/test_normalize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ber.normalize'`

- [ ] **Step 3: Implement `src/ber/normalize.py`.**

```python
"""Apply text normalization to whole frames (multiprocess, spawn-safe on Windows)."""
from __future__ import annotations

import multiprocessing as mp

import polars as pl

from ber.text import detect_script, normalize_address, normalize_name

NORM_COLUMNS = ["name_norm", "name_core", "name_alt", "legal_form", "is_web", "name_key", "name_script",
                "addr_norm", "postcode", "house_no", "street_tok", "num_tokens", "localities"]
_LEX: dict = {}


def normalize_record(name: str, address: str, lex: dict) -> tuple:
    n = normalize_name(name, lex.get("name", {}))
    a = normalize_address(address, lex.get("addr", {}))
    return (n["name_norm"], n["name_core"], n["name_alt"], n["legal_form"], n["is_web"], n["name_key"],
            detect_script(name), a["addr_norm"], a["postcode"], a["house_no"], a["street_tok"],
            a["num_tokens"], a["localities"])


def _init(lex: dict) -> None:
    global _LEX
    _LEX = lex


def _run(rows: list[tuple[str, str]]) -> list[tuple]:
    return [normalize_record(n, a, _LEX) for n, a in rows]


def normalize_frame(df: pl.DataFrame, lex: dict, n_jobs: int = 1, chunk: int = 50_000) -> pl.DataFrame:
    rows = list(zip(df["business_name"].to_list(), df["business_address"].to_list()))
    batches = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    if n_jobs <= 1:
        _init(lex)
        results = [_run(b) for b in batches]
    else:
        with mp.get_context("spawn").Pool(n_jobs, initializer=_init, initargs=(lex,)) as pool:
            results = pool.map(_run, batches)
    flat = [r for res in results for r in res]
    columns = list(zip(*flat)) if flat else [()] * len(NORM_COLUMNS)
    schema = {c: (pl.Int8 if c == "is_web" else pl.String) for c in NORM_COLUMNS}
    norm = pl.DataFrame({c: list(v) for c, v in zip(NORM_COLUMNS, columns)}, schema=schema)
    return pl.concat([df, norm], how="horizontal")
```

- [ ] **Step 4: Add the stage.** Append to `src/ber/stages/data.py` (imports: `from ber.config import LEXICON_SPLIT, read_path`, `from ber.normalize import normalize_frame`):

```python
def stage_normalize(cfg: dict, split: str) -> None:
    lex = json.loads(read_path(cfg, LEXICON_SPLIT[split], "lexicon.json").read_text(encoding="utf-8"))
    nc = cfg["normalize"]
    for side in ("s1", "right"):
        df = load(cfg, split, f"{side}.parquet")
        step = nc["slice_rows"]
        parts = [normalize_frame(df.slice(s, step), lex, nc["n_jobs"]) for s in range(0, df.height, step)]
        out = pl.concat(parts) if parts else normalize_frame(df, lex, 1)
        save(out, cfg, split, f"{side}_norm.parquet")
        print(f"[normalize] {split}/{side}: {out.height:,} rows")
```

Register `"normalize": data.stage_normalize`.

- [ ] **Step 5: Run the tests, then normalize dev and eyeball the result.**

Run: `.venv/Scripts/python -m pytest tests/test_normalize.py -v`
Expected: 2 passed

```bash
.venv/Scripts/python -m ber.pipeline --config configs/dev.yaml --stage normalize --split dev
.venv/Scripts/python -c "import polars as pl; pl.Config.set_fmt_str_lengths(60); print(pl.read_parquet('C:/Users/AYUSH/ber_data/work/dev/right_norm.parquet').sample(15, seed=1).select('business_name','name_core','legal_form','business_address','postcode','house_no','localities'))"
```

Record the rows/second and any systematic parsing failures in `memory.md`.

- [ ] **Step 6: Commit.**

```bash
git add src/ber/normalize.py src/ber/stages/data.py src/ber/pipeline.py tests/test_normalize.py memory.md
git commit -m "feat: parallel frame normalization stage"
```

---

### Task 10: Exact-key blocking

**Files:**
- Create: `src/ber/blocking/__init__.py` (empty), `src/ber/blocking/keys.py`
- Test: `tests/test_keys.py`

**Interfaces:**
- Consumes: the normalized frame columns (Task 9)
- Produces:
  - `record_keys(df_norm, same_country=True, slice_rows=1_000_000) -> pl.DataFrame[idx u32, key u64]`
  - `join_keys(lk, rk, max_block_pairs) -> pl.DataFrame[l_idx u32, r_idx u32, from_keys bool]`

- [ ] **Step 1: Write the failing tests.** Create `tests/test_keys.py`:

```python
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
```

- [ ] **Step 2: Run the tests and confirm they fail.**

Run: `.venv/Scripts/python -m pytest tests/test_keys.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ber.blocking'`

- [ ] **Step 3: Implement `src/ber/blocking/keys.py`.**

```python
"""Exact-key blocking. Keys (all prefixed with country unless same_country=False):
pc|postcode|long-name-token, loc|locality|name-prefix4, ph|metaphone(name)|last-locality, hn|house|street.
"""
from __future__ import annotations

import polars as pl

_COLS = ["idx", "country", "name_core", "postcode", "localities", "name_key", "house_no", "street_tok"]


def _keys_for_slice(df: pl.DataFrame, same_country: bool) -> pl.DataFrame:
    idx_out: list[int] = []
    keys: list[str] = []
    for i, c, core, pc, locs, nk, hn, st in zip(*(df[col].to_list() for col in _COLS)):
        c = c if same_country else "*"
        toks = sorted(core.split(), key=len, reverse=True)[:2]
        prefix = core.replace(" ", "")[:4]
        loc_list = [x for x in locs.split("|") if x][:3]
        ks = [f"pc|{c}|{pc}|{t}" for t in toks] if pc else []
        if prefix:
            ks += [f"loc|{c}|{x}|{prefix}" for x in loc_list]
        if nk and loc_list:
            ks.append(f"ph|{c}|{nk}|{loc_list[-1]}")
        if hn and st:
            ks.append(f"hn|{c}|{hn}|{st}")
        idx_out += [i] * len(ks)
        keys += ks
    frame = pl.DataFrame({"idx": idx_out, "key": keys}, schema={"idx": pl.UInt32, "key": pl.String})
    return frame.select("idx", pl.col("key").hash(seed=0))


def record_keys(df: pl.DataFrame, same_country: bool = True, slice_rows: int = 1_000_000) -> pl.DataFrame:
    parts = [_keys_for_slice(df.slice(s, slice_rows), same_country) for s in range(0, df.height, slice_rows)]
    return pl.concat(parts) if parts else _keys_for_slice(df, same_country)


def join_keys(lk: pl.DataFrame, rk: pl.DataFrame, max_block_pairs: int) -> pl.DataFrame:
    lc = lk.group_by("key").len("nl")
    rc = rk.group_by("key").len("nr")
    ok = (lc.join(rc, on="key")
          .filter(pl.col("nl").cast(pl.Int64) * pl.col("nr").cast(pl.Int64) <= max_block_pairs)
          .select("key"))
    left = lk.join(ok, on="key", how="semi").rename({"idx": "l_idx"})
    right = rk.join(ok, on="key", how="semi").rename({"idx": "r_idx"})
    return (left.join(right, on="key").select("l_idx", "r_idx").unique()
            .with_columns(from_keys=pl.lit(True)))
```

- [ ] **Step 4: Run the tests and confirm they pass.**

Run: `.venv/Scripts/python -m pytest tests/test_keys.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit.**

```bash
git add src/ber/blocking tests/test_keys.py
git commit -m "feat: exact-key blocking with block-size cap"
```

---

### Task 11: TF-IDF top-k blocking (both directions, within each country)

**Files:**
- Create: `src/ber/blocking/tfidf_ann.py`
- Test: `tests/test_tfidf_ann.py`

**Interfaces:**
- Produces: `tfidf_candidates(left, right, text_col, score_name, tcfg: dict, same_country=True) -> pl.DataFrame[l_idx u32, r_idx u32, <score_name> f32]`
  - `left` and `right` have columns `idx`, `country`, `text_col`.
  - `tcfg` keys: `ngram, min_df, max_df, top_k_fwd, top_k_rev, threshold, chunk_size, n_threads`.

- [ ] **Step 1: Write the failing tests.** Create `tests/test_tfidf_ann.py`:

```python
import polars as pl
import pytest

pytest.importorskip("sparse_dot_topn")
from ber.blocking.tfidf_ann import tfidf_candidates  # noqa: E402

TCFG = {"ngram": 3, "min_df": 1, "max_df": 1.0, "top_k_fwd": 1, "top_k_rev": 1,
        "threshold": 0.1, "chunk_size": 1, "n_threads": 1}


def _frame(countries, texts):
    return pl.DataFrame({"idx": list(range(len(texts))), "country": countries, "_text": texts},
                        schema_overrides={"idx": pl.UInt32})


def test_finds_best_match_in_both_directions_within_country():
    left = _frame(["US", "US"], ["sunrise bakery", "delta motors"])
    right = _frame(["US", "US", "France"], ["delta motor", "sunrise bakeries", "sunrise bakery"])
    out = tfidf_candidates(left, right, "_text", "tfidf_name", TCFG, same_country=True)
    pairs = set(zip(out["l_idx"].to_list(), out["r_idx"].to_list()))
    assert (0, 1) in pairs and (1, 0) in pairs
    assert (0, 2) not in pairs
    assert out["tfidf_name"].dtype == pl.Float32 and out["tfidf_name"].max() <= 1.0001


def test_reverse_direction_adds_pairs_beyond_forward_top_k():
    left = _frame(["US"], ["acme pizza"])
    right = _frame(["US", "US"], ["acme pizza", "acme pizzas"])
    out = tfidf_candidates(left, right, "_text", "s", TCFG, same_country=True)
    assert out.height == 2


def test_no_terms_after_pruning_returns_empty():
    left = _frame(["US"], ["ab"])
    right = _frame(["US"], ["cd"])
    out = tfidf_candidates(left, right, "_text", "s", {**TCFG, "min_df": 5}, same_country=True)
    assert out.height == 0 and out.columns == ["l_idx", "r_idx", "s"]
```

- [ ] **Step 2: Run the tests and confirm they fail.**

Run: `.venv/Scripts/python -m pytest tests/test_tfidf_ann.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ber.blocking.tfidf_ann'` (or all tests skipped if `sparse_dot_topn` is not installed; then run this task's tests on Kaggle in Task 17)

- [ ] **Step 3: Implement `src/ber/blocking/tfidf_ann.py`.**

```python
"""Char n-gram TF-IDF cosine top-k in both directions (S1->S2/S3 and S2/S3->S1), per country."""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

_SCHEMA = {"l_idx": pl.UInt32, "r_idx": pl.UInt32}


def _topn(a, bt, top_n: int, threshold: float, n_threads: int, chunk: int):
    rows, cols, vals = [], [], []
    for s in range(0, a.shape[0], chunk):
        c = sp_matmul_topn(a[s:s + chunk], bt, top_n=top_n, threshold=threshold,
                           sort=False, n_threads=n_threads).tocoo()
        rows.append(c.row + s)
        cols.append(c.col)
        vals.append(c.data)
    if not rows:
        return np.array([], np.int64), np.array([], np.int64), np.array([], np.float32)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def tfidf_candidates(left: pl.DataFrame, right: pl.DataFrame, text_col: str, score_name: str,
                     tcfg: dict, same_country: bool = True) -> pl.DataFrame:
    schema = {**_SCHEMA, score_name: pl.Float32}
    groups = sorted(set(left["country"].unique()) | set(right["country"].unique())) if same_country else [None]
    frames = []
    for country in groups:
        L = left if country is None else left.filter(pl.col("country") == country)
        R = right if country is None else right.filter(pl.col("country") == country)
        if L.height == 0 or R.height == 0:
            continue
        n = tcfg["ngram"]
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(n, n), min_df=tcfg["min_df"],
                              max_df=tcfg["max_df"], sublinear_tf=True, dtype=np.float32)
        try:
            vec.fit(pl.concat([L[text_col], R[text_col]]).to_list())
        except ValueError:  # no terms left after min_df/max_df pruning
            continue
        a = vec.transform(L[text_col].to_list()).tocsr()
        b = vec.transform(R[text_col].to_list()).tocsr()
        l_ids, r_ids = L["idx"].to_numpy(), R["idx"].to_numpy()
        args = (tcfg["threshold"], tcfg["n_threads"], tcfg["chunk_size"])
        r1, c1, v1 = _topn(a, b.T.tocsr(), tcfg["top_k_fwd"], *args)
        r2, c2, v2 = _topn(b, a.T.tocsr(), tcfg["top_k_rev"], *args)
        frames.append(pl.DataFrame({
            "l_idx": np.concatenate([l_ids[r1], l_ids[c2]]),
            "r_idx": np.concatenate([r_ids[c1], r_ids[r2]]),
            score_name: np.concatenate([v1, v2]).astype(np.float32),
        }, schema=schema))
        print(f"[tfidf:{score_name}] {country}: L={L.height:,} R={R.height:,} pairs={frames[-1].height:,}")
    if not frames:
        return pl.DataFrame(schema=schema)
    return pl.concat(frames).group_by("l_idx", "r_idx").agg(pl.col(score_name).max())
```

- [ ] **Step 4: Run the tests and confirm they pass.**

Run: `.venv/Scripts/python -m pytest tests/test_tfidf_ann.py -v`
Expected: 3 passed (or skipped locally; see Step 2)

- [ ] **Step 5: Commit.**

```bash
git add src/ber/blocking/tfidf_ann.py tests/test_tfidf_ann.py
git commit -m "feat: bidirectional char-trigram TF-IDF top-k blocking"
```

---

### Task 12: Merge candidates and the `block` stage

**Files:**
- Create: `src/ber/blocking/merge.py`, `src/ber/stages/candidates.py`
- Modify: `src/ber/pipeline.py`
- Test: `tests/test_merge.py`

**Interfaces:**
- Consumes: `record_keys`, `join_keys` (Task 10); `tfidf_candidates` (Task 11); `load_norm`, `load`, `save` (Task 2)
- Produces:
  - `SCORE_COLUMNS = ["tfidf_name", "tfidf_name_addr", "embed_cos"]`
  - `merge_candidates(frames) -> pl.DataFrame[l_idx, r_idx, tfidf_name, tfidf_name_addr, embed_cos (f32, 0 when absent), from_keys (bool)]`
  - `top_k_per_left(df, score_col, k) -> pl.DataFrame`
  - stage `block`, which writes `cands_all.parquet` and includes `cands_embed.parquet` when `blocking.embed.enabled`

- [ ] **Step 1: Write the failing tests.** Create `tests/test_merge.py`:

```python
import polars as pl

from ber.blocking.merge import merge_candidates, top_k_per_left

U = pl.UInt32


def test_merge_takes_max_and_fills_defaults():
    keys = pl.DataFrame({"l_idx": [0, 1], "r_idx": [5, 6], "from_keys": [True, True]},
                        schema_overrides={"l_idx": U, "r_idx": U})
    tf = pl.DataFrame({"l_idx": [0, 0, 2], "r_idx": [5, 5, 7], "tfidf_name": [0.4, 0.9, 0.5]},
                      schema_overrides={"l_idx": U, "r_idx": U, "tfidf_name": pl.Float32})
    out = merge_candidates([keys, tf]).sort("l_idx", "r_idx")
    assert out.columns == ["l_idx", "r_idx", "tfidf_name", "tfidf_name_addr", "embed_cos", "from_keys"]
    assert out["tfidf_name"].to_list() == [0.8999999761581421, 0.0, 0.5]
    assert out["from_keys"].to_list() == [True, True, False]
    assert out["embed_cos"].to_list() == [0.0, 0.0, 0.0]


def test_top_k_per_left():
    df = pl.DataFrame({"l_idx": [0, 0, 0, 1], "r_idx": [1, 2, 3, 4], "s": [0.1, 0.9, 0.5, 0.2]})
    out = top_k_per_left(df, "s", 2).sort("l_idx", "r_idx")
    assert list(zip(out["l_idx"], out["r_idx"])) == [(0, 2), (0, 3), (1, 4)]
```

- [ ] **Step 2: Run the tests and confirm they fail.**

Run: `.venv/Scripts/python -m pytest tests/test_merge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ber.blocking.merge'`

- [ ] **Step 3: Implement `src/ber/blocking/merge.py`.**

```python
"""Union of candidate sources and per-S1 top-k cut."""
from __future__ import annotations

import polars as pl

SCORE_COLUMNS = ["tfidf_name", "tfidf_name_addr", "embed_cos"]


def merge_candidates(frames: list[pl.DataFrame]) -> pl.DataFrame:
    df = pl.concat([f for f in frames if f.height > 0] or frames[:1], how="diagonal_relaxed")
    extra = [c for c in df.columns if c not in ("l_idx", "r_idx")]
    out = df.group_by("l_idx", "r_idx").agg([pl.col(c).max() for c in extra])
    fills = [
        (pl.col(c).fill_null(0.0) if c in out.columns else pl.lit(0.0)).cast(pl.Float32).alias(c)
        for c in SCORE_COLUMNS
    ]
    keys = (pl.col("from_keys").fill_null(False) if "from_keys" in out.columns else pl.lit(False)).alias("from_keys")
    return out.select("l_idx", "r_idx", *fills, keys)


def top_k_per_left(df: pl.DataFrame, score_col: str, k: int) -> pl.DataFrame:
    return df.filter(pl.col(score_col).rank("ordinal", descending=True).over("l_idx") <= k)
```

- [ ] **Step 4: Implement the `block` stage.** Create `src/ber/stages/candidates.py`:

```python
"""Candidate stages: block, cheap_train, cheap_apply."""
from __future__ import annotations

import polars as pl

from ber.blocking.keys import join_keys, record_keys
from ber.blocking.merge import merge_candidates
from ber.blocking.tfidf_ann import tfidf_candidates
from ber.stages.common import load, load_norm, save

VIEWS = {
    "name": pl.col("name_core"),
    "name_addr": pl.concat_str([pl.col("name_core"), pl.col("addr_norm")], separator=" "),
}


def stage_block(cfg: dict, split: str) -> None:
    left, right = load_norm(cfg, split, "s1"), load_norm(cfg, split, "right")
    b = cfg["blocking"]
    same = b["same_country"]
    frames = [join_keys(record_keys(left, same), record_keys(right, same), b["max_block_pairs"])]
    print(f"[block] keys: {frames[0].height:,} pairs")
    for view in b["tfidf"]["views"]:
        text = VIEWS[view].alias("_text")
        frames.append(tfidf_candidates(left.select("idx", "country", text), right.select("idx", "country", text),
                                       "_text", f"tfidf_{view}", b["tfidf"], same))
    if b["embed"]["enabled"]:
        frames.append(load(cfg, split, "cands_embed.parquet"))
    cands = merge_candidates(frames)
    save(cands, cfg, split, "cands_all.parquet")
    print(f"[block] {split}: {cands.height:,} pairs, {cands.height / max(left.height, 1):.1f} per S1")
```

In `pipeline.py`: `from ber.stages import candidates, data`, then register `"block": candidates.stage_block`.

- [ ] **Step 5: Run the tests, then block dev and measure recall.**

Run: `.venv/Scripts/python -m pytest tests/test_merge.py -v`
Expected: 2 passed

```bash
.venv/Scripts/python -m ber.pipeline --config configs/dev.yaml --stage block --split dev
.venv/Scripts/python -c "
import polars as pl
from ber.metrics import blocking_recall, cands_per_s1
w='C:/Users/AYUSH/ber_data/work/dev/'
c=pl.read_parquet(w+'cands_all.parquet'); gt=pl.read_parquet(w+'gt.parquet'); f=pl.read_parquet(w+'folds.parquet')
ev=f.filter(pl.col('fold')=='valid')['l_idx']
print('recall', blocking_recall(c, gt, ev), 'per_s1', cands_per_s1(c, ev))
for col in ['from_keys']: print(col, blocking_recall(c.filter(pl.col(col)), gt, ev))
for col in ['tfidf_name','tfidf_name_addr']: print(col, blocking_recall(c.filter(pl.col(col)>0), gt, ev))
"
```

Expected: overall recall of at least 0.97 on dev. If it is below that, raise `top_k_fwd` (40→60) or lower `threshold` (0.3→0.2) with `--set`, re-run, and record each attempt in `memory.md`. Also record the elapsed time per view; it is used to extrapolate Kaggle runtime.

- [ ] **Step 6: Commit.**

```bash
git add src/ber/blocking/merge.py src/ber/stages/candidates.py src/ber/pipeline.py tests/test_merge.py memory.md
git commit -m "feat: merge candidate sources; block stage"
```

### Task 13: Cheap features, LightGBM helpers, and the cheap-cut stages

**Files:**
- Create: `src/ber/features.py` (cheap part), `src/ber/models/__init__.py` (empty), `src/ber/models/gbm.py`
- Modify: `src/ber/stages/common.py` (add `es_fold`), `src/ber/stages/candidates.py`, `src/ber/pipeline.py`
- Test: `tests/test_features_cheap.py`, `tests/test_gbm.py`

**Interfaces:**
- Consumes: `cands_all.parquet` (Task 12); `load_norm`, `sample_l`, `label_pairs` (Task 2); `top_k_per_left` (Task 12); `MODEL_SPLIT`, `read_path`, `write_path` (Task 1)
- Produces:
  - `CHEAP_COLUMNS = ["tfidf_name","tfidf_name_addr","embed_cos","from_keys","name_jw","name_tset","addr_tset","postcode_eq","house_eq","source_s3"]`
  - `attach_sides(c, left, right, cols) -> pl.DataFrame` (adds `l_<col>`, `r_<col>` and `r_source`)
  - `cheap_features(c, left, right) -> pl.DataFrame` (the columns of `c` plus the missing `CHEAP_COLUMNS`, same row order)
  - `gbm.DEFAULT_PARAMS`, `to_matrix(X) -> np.ndarray[float32]`, `train_binary(X_tr, y_tr, X_va, y_va, params, num_boost_round, early_stopping_rounds) -> lgb.Booster`, `predict(booster, X) -> np.ndarray` (selects `booster.feature_name()` columns), `load_booster(path) -> lgb.Booster`
  - `common.es_fold(cfg) -> str`: `"valid_seen"` under `country_holdout`, `"valid"` otherwise
  - stages `cheap_train` (writes `cheap_model.txt`) and `cheap_apply` (writes `cands_final.parquet` with `cheap_score`)

- [ ] **Step 1: Write the failing tests.** Create `tests/test_features_cheap.py`:

```python
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
```

Create `tests/test_gbm.py`:

```python
import numpy as np
import polars as pl

from ber.models.gbm import load_booster, predict, train_binary


def test_train_predict_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    X = pl.DataFrame({"a": rng.random(2000), "b": rng.random(2000), "flag": rng.random(2000) > 0.5})
    y = (X["a"].to_numpy() > 0.5).astype(int)
    bst = train_binary(X[:1500], y[:1500], X[1500:], y[1500:], {"min_data_in_leaf": 5, "num_leaves": 7}, 200, 20)
    p = predict(bst, X[1500:])
    assert ((p > 0.5) == y[1500:]).mean() > 0.95
    path = tmp_path / "m.txt"
    bst.save_model(str(path))
    assert np.allclose(predict(load_booster(path), X[1500:].select("b", "a", "flag")), p)
```

- [ ] **Step 2: Run the tests and confirm they fail.**

Run: `.venv/Scripts/python -m pytest tests/test_features_cheap.py tests/test_gbm.py -v`
Expected: FAIL with `ModuleNotFoundError` for `ber.features` / `ber.models`

- [ ] **Step 3: Implement `src/ber/features.py` (cheap part).**

```python
"""Pair features. Country-agnostic: country is never a feature (script type is)."""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

CHEAP_COLUMNS = ["tfidf_name", "tfidf_name_addr", "embed_cos", "from_keys", "name_jw", "name_tset",
                 "addr_tset", "postcode_eq", "house_eq", "source_s3"]


def attach_sides(c: pl.DataFrame, left: pl.DataFrame, right: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    lft = left.select(pl.col(cols).gather(c["l_idx"])).rename({x: f"l_{x}" for x in cols})
    rcols = cols + ["source"]
    rgt = right.select(pl.col(rcols).gather(c["r_idx"])).rename({x: f"r_{x}" for x in rcols})
    return pl.concat([c, lft, rgt], how="horizontal")


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
```

- [ ] **Step 4: Implement `src/ber/models/gbm.py`.**

```python
"""LightGBM helpers (MIT license)."""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

DEFAULT_PARAMS = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 127, "min_data_in_leaf": 200,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "metric": ["binary_logloss", "auc"], "verbose": -1, "num_threads": 0, "seed": 42,
}


def to_matrix(X: pl.DataFrame) -> np.ndarray:
    return X.select(pl.all().cast(pl.Float32)).to_numpy()


def train_binary(X_tr: pl.DataFrame, y_tr, X_va: pl.DataFrame, y_va, params: dict | None,
                 num_boost_round: int, early_stopping_rounds: int) -> lgb.Booster:
    dtr = lgb.Dataset(to_matrix(X_tr), label=y_tr, feature_name=list(X_tr.columns), free_raw_data=True)
    dva = lgb.Dataset(to_matrix(X_va.select(X_tr.columns)), label=y_va, reference=dtr)
    return lgb.train(
        {**DEFAULT_PARAMS, **(params or {})}, dtr, num_boost_round=num_boost_round, valid_sets=[dva],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=True), lgb.log_evaluation(100)],
    )


def predict(booster: lgb.Booster, X: pl.DataFrame) -> np.ndarray:
    return booster.predict(to_matrix(X.select(booster.feature_name())))


def load_booster(path: str | Path) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))
```

- [ ] **Step 5: Add `es_fold` and the stages.**

Append to `src/ber/stages/common.py`:

```python
def es_fold(cfg: dict) -> str:
    return "valid_seen" if cfg["validation"]["scheme"] == "country_holdout" else "valid"
```

Append to `src/ber/stages/candidates.py` (imports at the top: `numpy as np`; `from ber.blocking.merge import top_k_per_left`; `from ber.config import MODEL_SPLIT, read_path, write_path`; `from ber.features import CHEAP_COLUMNS, cheap_features`; `from ber.metrics import blocking_recall, cands_per_s1`; `from ber.models.gbm import load_booster, predict, train_binary`; `from ber.stages.common import es_fold, label_pairs, sample_l`):

```python
def stage_cheap_train(cfg: dict, split: str) -> None:
    left, right = load_norm(cfg, split, "s1"), load_norm(cfg, split, "right")
    cands, gt, folds = load(cfg, split, "cands_all.parquet"), load(cfg, split, "gt.parquet"), load(cfg, split, "folds.parquet")
    cc = cfg["cheap"]
    tr = cands.join(sample_l(folds, "fit", cc["max_train_s1"], cfg["seed"]), on="l_idx", how="semi")
    va = cands.join(sample_l(folds, es_fold(cfg), 50_000, cfg["seed"]), on="l_idx", how="semi")
    x_tr, x_va = cheap_features(tr, left, right), cheap_features(va, left, right)
    booster = train_binary(x_tr.select(CHEAP_COLUMNS), label_pairs(tr, gt), x_va.select(CHEAP_COLUMNS),
                           label_pairs(va, gt), cc["params"], cc["num_boost_round"], cc["early_stopping_rounds"])
    booster.save_model(str(write_path(cfg, split, "cheap_model.txt")))


def stage_cheap_apply(cfg: dict, split: str) -> None:
    left, right = load_norm(cfg, split, "s1"), load_norm(cfg, split, "right")
    cands = load(cfg, split, "cands_all.parquet")
    booster = load_booster(read_path(cfg, MODEL_SPLIT[split], "cheap_model.txt"))
    step = cfg["cheap"]["chunk_rows"]
    scores = [predict(booster, cheap_features(cands.slice(s, step), left, right).select(CHEAP_COLUMNS))
              for s in range(0, cands.height, step)]
    cands = cands.with_columns(cheap_score=pl.Series(np.concatenate(scores) if scores else [], dtype=pl.Float32))
    final = top_k_per_left(cands, "cheap_score", cfg["cheap"]["keep_top"])
    save(final, cfg, split, "cands_final.parquet")
    print(f"[cheap_apply] {split}: {cands.height:,} -> {final.height:,} pairs")
    if split != "test":
        gt, folds = load(cfg, split, "gt.parquet"), load(cfg, split, "folds.parquet")
        ev = folds.filter(pl.col("fold") == es_fold(cfg))["l_idx"]
        print(f"[cheap_apply] recall all={blocking_recall(cands, gt, ev):.4f} "
              f"final={blocking_recall(final, gt, ev):.4f} per_s1={cands_per_s1(final, ev):.1f}")
```

Register `"cheap_train"` and `"cheap_apply"` in `pipeline.STAGES`.

- [ ] **Step 6: Run the tests and confirm they pass, then run on dev.**

Run: `.venv/Scripts/python -m pytest tests/test_features_cheap.py tests/test_gbm.py -v`
Expected: 2 passed

```bash
.venv/Scripts/python -m ber.pipeline --config configs/dev.yaml --stage cheap_train --split dev
.venv/Scripts/python -m ber.pipeline --config configs/dev.yaml --stage cheap_apply --split dev
```

Expected: final recall at least 0.965 and about 25 candidates per S1. Record both in `memory.md`.

- [ ] **Step 7: Commit.**

```bash
git add src/ber/features.py src/ber/models src/ber/stages tests/test_features_cheap.py tests/test_gbm.py src/ber/pipeline.py memory.md
git commit -m "feat: cheap features, LightGBM helpers, cheap cut to top-k"
```

---

### Task 14: Context and full features, and the `features` stage

**Files:**
- Modify: `src/ber/features.py` (append)
- Create: `src/ber/stages/model.py`
- Modify: `src/ber/pipeline.py`
- Test: `tests/test_features_full.py`

**Interfaces:**
- Consumes: `cheap_features`, `attach_sides`, `sim`, `tri` (Task 13); `cands_final.parquet` (Task 13)
- Produces:
  - `CONTEXT_COLUMNS`, `NAME_COLUMNS`, `ADDR_COLUMNS`, `FEATURE_COLUMNS` (lists of str)
  - `context_features(c) -> pl.DataFrame`: adds `rank_l, gap_l, n_cands_l, n_claims_r, rank_r, mutual_best, other_r_idx`; needs `cheap_score`
  - `fit_vectorizers(left, right, fit_rows, seed) -> dict[str, tuple[csr_left, csr_right]]` with keys `name_char`, `name_word`, `addr_word`
  - `full_features(c, left, right, vecs) -> pl.DataFrame[l_idx, r_idx, *FEATURE_COLUMNS]`
  - stage `features`, which writes `features/part-*.parquet`. On train/dev it keeps sampled `fit` S1 plus all eval folds and adds `y`.

- [ ] **Step 1: Write the failing tests.** Create `tests/test_features_full.py`:

```python
import numpy as np
import pytest

from ber.features import FEATURE_COLUMNS, context_features, fit_vectorizers, full_features
from tests.test_features_cheap import CANDS, LEFT, RIGHT


def test_context_features():
    out = context_features(CANDS).sort("l_idx", "r_idx")
    assert out["rank_l"].to_list() == [1, 2, 1]
    assert out["n_claims_r"].to_list() == [1, 1, 1]
    assert out["other_r_idx"].to_list() == [1, 0, None]
    assert out["mutual_best"].to_list() == [1, 0, 1]


def test_full_features_schema_and_values():
    vecs = fit_vectorizers(LEFT, RIGHT, fit_rows=100, seed=0)
    out = full_features(context_features(CANDS), LEFT, RIGHT, vecs).sort("l_idx", "r_idx")
    assert out.columns == ["l_idx", "r_idx", *FEATURE_COLUMNS]
    r0 = out.row(0, named=True)
    assert r0["name_ratio"] == 100 and r0["legal_eq"] == 1 and r0["street_eq"] == 1
    assert r0["name_char_cos"] == pytest.approx(1.0, abs=1e-5)
    r2 = out.row(2, named=True)
    assert r2["l_addr_empty"] == 1 and r2["legal_eq"] == -1 and np.isnan(r2["sim_other_name"])
```

Add an empty `tests/__init__.py` so `from tests.test_features_cheap import ...` resolves.

- [ ] **Step 2: Run the tests and confirm they fail.**

Run: `.venv/Scripts/python -m pytest tests/test_features_full.py -v`
Expected: FAIL with `ImportError: cannot import name 'FEATURE_COLUMNS'`

- [ ] **Step 3: Append to `src/ber/features.py`** (add these imports at the top: `from rapidfuzz.distance import Levenshtein`, `from sklearn.feature_extraction.text import TfidfVectorizer`).

```python
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
```

- [ ] **Step 4: Implement the `features` stage.** Create `src/ber/stages/model.py`:

```python
"""Model stages: features, train, evaluate, predict."""
from __future__ import annotations

import polars as pl

from ber.config import read_path, write_path
from ber.features import context_features, fit_vectorizers, full_features
from ber.stages.common import eval_folds, label_pairs, load, load_norm, sample_l


def load_features(cfg: dict, split: str) -> pl.LazyFrame:
    return pl.scan_parquet(str(read_path(cfg, split, "features") / "*.parquet"))


def stage_features(cfg: dict, split: str) -> None:
    left, right = load_norm(cfg, split, "s1"), load_norm(cfg, split, "right")
    cands = context_features(load(cfg, split, "cands_final.parquet"))  # context on the FULL candidate set
    if split != "test":
        folds = load(cfg, split, "folds.parquet")
        keep = pl.concat([sample_l(folds, "fit", cfg["features"]["max_train_s1"], cfg["seed"])]
                         + [folds.filter(pl.col("fold") == f).select("l_idx") for f in eval_folds(cfg)])
        cands = cands.join(keep, on="l_idx", how="semi")
        cands = cands.with_columns(y=pl.Series(label_pairs(cands, load(cfg, split, "gt.parquet"))))
    vecs = fit_vectorizers(left, right, cfg["features"]["vectorizer_fit_rows"], cfg["seed"])
    out_dir = write_path(cfg, split, "features/.keep").parent
    for old in out_dir.glob("part-*.parquet"):
        old.unlink()
    step = cfg["features"]["chunk_rows"]
    for n, s in enumerate(range(0, cands.height, step)):
        chunk = cands.slice(s, step)
        feats = full_features(chunk, left, right, vecs)
        if "y" in chunk.columns:
            feats = feats.with_columns(y=chunk["y"])
        feats.write_parquet(out_dir / f"part-{n:04d}.parquet")
        print(f"[features] {split}: part {n} rows={feats.height:,}")
```

Register `"features": model.stage_features` (`from ber.stages import candidates, data, model`).

- [ ] **Step 5: Run the tests and confirm they pass, then run on dev.**

Run: `.venv/Scripts/python -m pytest tests/test_features_full.py tests/test_features_cheap.py -v`
Expected: 3 passed

```bash
.venv/Scripts/python -m ber.pipeline --config configs/dev.yaml --stage features --split dev
```

Record the rows per second in `memory.md`; it is used to extrapolate Kaggle time.

- [ ] **Step 6: Commit.**

```bash
git add src/ber/features.py src/ber/stages/model.py src/ber/pipeline.py tests/__init__.py tests/test_features_full.py memory.md
git commit -m "feat: context + full pair features, features stage"
```

---

### Task 15: Decision rules (calibration, one owner, expected-F0.5, threshold, unseen-country scale)

**Files:**
- Create: `src/ber/decide.py`
- Test: `tests/test_decide.py`

**Interfaces:**
- Consumes: `macro_f05_frame` (Task 3)
- Produces:
  - `THRESHOLDS: np.ndarray`, `SCALES: list[float]`
  - `fit_calibrator(p, y) -> IsotonicRegression`, `calibrate(iso, p) -> np.ndarray[float32]`
  - `crossfit_calibrate(pred[l_idx, r_idx, p, y], seed) -> np.ndarray`
  - `one_owner(pred) -> pred`
  - `choose_threshold(pred, t) -> pl.DataFrame[l_idx, r_idx]`, `choose_expected_f05(pred) -> pl.DataFrame[l_idx, r_idx]`
  - `tune_threshold(pred, gt, eval_l, grid=THRESHOLDS) -> (t, f)`
  - `scale_unseen(pred, s1_country[l_idx, country], seen, scale) -> pred`
  - `select(pred, decision: dict) -> pl.DataFrame[l_idx, r_idx]`

- [ ] **Step 1: Write the failing tests.** Create `tests/test_decide.py`:

```python
import numpy as np
import polars as pl

from ber.decide import (choose_expected_f05, choose_threshold, crossfit_calibrate, one_owner,
                        scale_unseen, select, tune_threshold)

U = pl.UInt32


def _pred(rows):
    return pl.DataFrame(rows, schema={"l_idx": U, "r_idx": U, "p": pl.Float64}, orient="row")


def _pairs(df):
    return set(zip(df["l_idx"].to_list(), df["r_idx"].to_list()))


def test_expected_f05_picks_k_per_entity():
    pred = _pred([(0, 1, 0.95), (0, 2, 0.9), (0, 3, 0.1), (1, 4, 0.05), (1, 5, 0.02), (2, 6, 0.6)])
    assert _pairs(choose_expected_f05(pred)) == {(0, 1), (0, 2), (2, 6)}


def test_one_owner_keeps_best_claim():
    assert _pairs(one_owner(_pred([(0, 5, 0.9), (1, 5, 0.7), (1, 6, 0.8)]))) == {(0, 5), (1, 6)}


def test_threshold_tuning():
    pred = _pred([(0, 1, 0.8), (0, 2, 0.4), (1, 3, 0.3)])
    gt = pl.DataFrame({"l_idx": [0], "r_idx": [1]}, schema={"l_idx": U, "r_idx": U})
    t, f = tune_threshold(pred, gt, pl.Series("l_idx", [0, 1], dtype=U))
    assert f == 1.0 and 0.4 < t <= 0.8
    assert _pairs(choose_threshold(pred, 0.5)) == {(0, 1)}
    assert _pairs(select(pred, {"method": "threshold", "threshold": 0.5})) == {(0, 1)}


def test_crossfit_calibration_bounded_and_calibrated():
    rng = np.random.default_rng(0)
    n = 4000
    p = rng.random(n)
    y = (rng.random(n) < p ** 2).astype(int)
    pred = pl.DataFrame({"l_idx": rng.integers(0, 500, n).astype(np.uint32),
                         "r_idx": np.arange(n, dtype=np.uint32), "p": p, "y": y})
    cal = crossfit_calibrate(pred, seed=0)
    assert cal.min() >= 0 and cal.max() <= 1 and abs(cal.mean() - y.mean()) < 0.03


def test_scale_unseen_only_touches_unseen_countries():
    pred = _pred([(0, 1, 0.5), (1, 2, 0.5)])
    s1c = pl.DataFrame({"l_idx": [0, 1], "country": ["US", "France"]}, schema_overrides={"l_idx": U})
    out = scale_unseen(pred, s1c, ["US", "India"], 1.2).sort("l_idx")
    assert out["p"].to_list() == [0.5, 0.6] and out.columns == pred.columns
```

- [ ] **Step 2: Run the tests and confirm they fail.**

Run: `.venv/Scripts/python -m pytest tests/test_decide.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ber.decide'`

- [ ] **Step 3: Implement `src/ber/decide.py`.**

```python
"""Turn pair probabilities into per-S1 match sets that maximize macro F0.5.

Expected-F0.5 rule (plug-in approximation, independence assumed): for an S1 with candidate
probabilities sorted p1>=p2>=..., choosing the top-k gives E[F] ~= 1.25*S_k / (0.25*S_all + k);
choosing nothing scores 1.0 only if every candidate is a non-match: prod(1 - p_i).
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from ber.metrics import macro_f05_frame

THRESHOLDS = np.round(np.arange(0.2, 0.9, 0.025), 3)
SCALES = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]


def fit_calibrator(p: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)


def calibrate(iso: IsotonicRegression, p: np.ndarray) -> np.ndarray:
    return iso.predict(p).astype(np.float32)


def crossfit_calibrate(pred: pl.DataFrame, seed: int) -> np.ndarray:
    """Calibrate each half of the S1 entities with an isotonic fit on the other half."""
    l_idx = pred["l_idx"].to_numpy().astype(np.int64)
    half = (np.random.default_rng(seed).random(l_idx.max() + 1) < 0.5)[l_idx]
    p, y = pred["p"].to_numpy(), pred["y"].to_numpy()
    out = np.empty(len(p), np.float32)
    for h in (True, False):
        out[half == h] = calibrate(fit_calibrator(p[half != h], y[half != h]), p[half == h])
    return out


def one_owner(pred: pl.DataFrame) -> pl.DataFrame:
    return (pred.sort(["r_idx", "p", "l_idx"], descending=[False, True, False])
            .unique(subset="r_idx", keep="first", maintain_order=True))


def choose_threshold(pred: pl.DataFrame, t: float) -> pl.DataFrame:
    return pred.filter(pl.col("p") >= t).select("l_idx", "r_idx")


def choose_expected_f05(pred: pl.DataFrame) -> pl.DataFrame:
    d = (
        pred.select("l_idx", "r_idx", "p").sort(["l_idx", "p"], descending=[False, True])
        .with_columns(
            k=pl.int_range(1, pl.len() + 1).over("l_idx"),
            s_k=pl.col("p").cum_sum().over("l_idx"),
            s_all=pl.col("p").sum().over("l_idx"),
            log_empty=(1 - pl.col("p")).clip(1e-9, 1.0).log().sum().over("l_idx"),
        )
        .with_columns(ef=1.25 * pl.col("s_k") / (0.25 * pl.col("s_all") + pl.col("k")))
    )
    best = d.group_by("l_idx").agg(
        best_k=pl.col("k").sort_by("ef").last(),
        best_ef=pl.col("ef").max(),
        e0=pl.col("log_empty").first().exp(),
    ).with_columns(chosen=pl.when(pl.col("e0") >= pl.col("best_ef")).then(0).otherwise(pl.col("best_k")))
    return (d.join(best.select("l_idx", "chosen"), on="l_idx")
            .filter(pl.col("k") <= pl.col("chosen")).select("l_idx", "r_idx"))


def tune_threshold(pred: pl.DataFrame, gt: pl.DataFrame, eval_l: pl.Series, grid=THRESHOLDS) -> tuple[float, float]:
    best_t, best_f = float(grid[0]), -1.0
    for t in grid:
        f = macro_f05_frame(choose_threshold(pred, float(t)), gt, eval_l)
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t, best_f


def scale_unseen(pred: pl.DataFrame, s1_country: pl.DataFrame, seen: list[str], scale: float) -> pl.DataFrame:
    if scale == 1.0:
        return pred
    return (pred.join(s1_country, on="l_idx", how="left")
            .with_columns(p=pl.when(pl.col("country").is_in(seen)).then(pl.col("p"))
                          .otherwise((pl.col("p") * scale).clip(0.0, 1.0)))
            .select(pred.columns))


def select(pred: pl.DataFrame, decision: dict) -> pl.DataFrame:
    if decision["method"] == "expected_f05":
        return choose_expected_f05(pred)
    return choose_threshold(pred, decision["threshold"])
```

- [ ] **Step 4: Run the tests and confirm they pass.**

Run: `.venv/Scripts/python -m pytest tests/test_decide.py -v`
Expected: 5 passed. Also check in the test output that `[0.5, 0.6]` compares exactly. If floating-point noise breaks equality, change that assert to `pytest.approx`.

- [ ] **Step 5: Commit.**

```bash
git add src/ber/decide.py tests/test_decide.py
git commit -m "feat: calibration, one-owner, expected-F0.5 and threshold decisions"
```

---

### Task 16: Submission writer and validator runner

**Files:**
- Create: `src/ber/submit.py`
- Test: `tests/test_submit.py`

**Interfaces:**
- Consumes: `open_raw` (Task 2); the vendored validator (Task 1)
- Produces:
  - `write_id_lists(path, value_col, s1[idx, entity_id], right[idx, entity_id], pairs[l_idx, r_idx]) -> None`
  - `ensure_test_dir(cfg) -> Path` (a directory containing `test_source1.tsv`)
  - `run_validator(matching, candidate, test_dir) -> int`

- [ ] **Step 1: Write the failing tests.** Create `tests/test_submit.py`:

```python
import polars as pl

from ber.submit import run_validator, write_id_lists

U = pl.UInt32
S1 = pl.DataFrame({"idx": [0, 1, 2], "entity_id": ["S1-a", "S1-b", "S1-c"]}, schema_overrides={"idx": U})
RIGHT = pl.DataFrame({"idx": [0, 1, 2], "entity_id": ["S2-x", "S3-y", "S2-z"]}, schema_overrides={"idx": U})
PAIRS = pl.DataFrame({"l_idx": [0, 0, 0, 1], "r_idx": [1, 0, 1, 2]}, schema={"l_idx": U, "r_idx": U})
SRC1 = "entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-a\ta\tb\tUS\nS1-b\ta\tb\tUS\nS1-c\ta\tb\tFrance\n"


def test_writer_format(tmp_path):
    m = tmp_path / "matching_results.tsv"
    write_id_lists(m, "matched_entity_ids", S1, RIGHT, PAIRS)
    lines = m.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "source1_entity_id\tmatched_entity_ids"
    assert sorted(lines[1:]) == ["S1-a\tS2-x,S3-y", "S1-b\tS2-z", "S1-c\t"]


def test_validator_pass_and_fail(tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    (test_dir / "test_source1.tsv").write_text(SRC1, encoding="utf-8")
    m, c = tmp_path / "matching_results.tsv", tmp_path / "candidate_pairs.tsv"
    write_id_lists(m, "matched_entity_ids", S1, RIGHT, PAIRS)
    write_id_lists(c, "candidate_entity_ids", S1, RIGHT, PAIRS)
    assert run_validator(m, c, test_dir) == 0
    write_id_lists(m, "matched_entity_ids", S1.filter(pl.col("idx") < 2), RIGHT, PAIRS)  # drops S1-c
    assert run_validator(m, c, test_dir) == 1
```

- [ ] **Step 2: Run the tests and confirm they fail.**

Run: `.venv/Scripts/python -m pytest tests/test_submit.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ber.submit'`

- [ ] **Step 3: Implement `src/ber/submit.py`.**

```python
"""Write the two submission TSVs, run the official validator, build the final zip."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import polars as pl

from ber.io import open_raw


def write_id_lists(path: str | Path, value_col: str, s1: pl.DataFrame, right: pl.DataFrame,
                   pairs: pl.DataFrame) -> None:
    ids = (
        pairs.select("l_idx", "r_idx").unique()
        .join(s1.select(pl.col("idx").alias("l_idx"), pl.col("entity_id").alias("l_id")), on="l_idx")
        .join(right.select(pl.col("idx").alias("r_idx"), pl.col("entity_id").alias("r_id")), on="r_idx")
        .group_by("l_id").agg(pl.col("r_id").unique().sort())
    )
    table = (
        s1.select(pl.col("entity_id").alias("source1_entity_id"))
        .join(ids.rename({"l_id": "source1_entity_id"}), on="source1_entity_id", how="left")
        .with_columns(pl.col("r_id").list.join(",").fill_null("").alias(value_col))
        .select("source1_entity_id", value_col)
    )
    table.write_csv(path, separator="\t", quote_style="never")


def ensure_test_dir(cfg: dict) -> Path:
    raw_dir = cfg["paths"].get("raw_dir")
    if raw_dir:
        return Path(raw_dir) / "test"
    out = Path(cfg["paths"]["scratch_dir"]) / "test_raw"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "test_source1.tsv"
    if not target.exists():
        target.write_bytes(open_raw(cfg, "test", "test_source1.tsv"))
    return out


def run_validator(matching: str | Path, candidate: str | Path, test_dir: str | Path) -> int:
    cmd = [sys.executable, "-m", "ber.vendor.validate_submission", "--matching", str(matching),
           "--candidate", str(candidate), "--test-dir", str(test_dir)]
    return subprocess.run(cmd).returncode
```

- [ ] **Step 4: Run the tests and confirm they pass.**

Run: `.venv/Scripts/python -m pytest tests/test_submit.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit.**

```bash
git add src/ber/submit.py tests/test_submit.py
git commit -m "feat: submission TSV writer and validator runner"
```

---

### Task 17: Model stages (train, evaluate, predict, submit) and the synthetic end-to-end test

**Files:**
- Modify: `src/ber/stages/model.py`, `src/ber/pipeline.py`
- Create: `src/ber/stages/output.py`, `tests/synth.py`, `tests/conftest.py`, `tests/test_end_to_end.py`

**Interfaces:**
- Consumes: everything above
- Produces:
  - stage `train` (train/dev) → `gbm_model.txt`, `valid_pred.parquet` [l_idx, r_idx, y, fold, p], `fit_pred.parquet`
  - stage `evaluate` → returns a dict of metrics, writes `calibrator.joblib` and `decision.json`, appends to `experiments.csv`
  - stage `predict` (test) → `pred.parquet` [l_idx, r_idx, p]
  - stage `submit` (test) → `output_dir/matching_results.tsv` and `candidate_pairs.tsv`, returns the validator exit code (raises if non-zero)
  - `tests/synth.py: write_raw(root, seed=0)`, which creates `root/{train,test}/*.tsv` in the organizer layout

- [ ] **Step 1: Write the synthetic data generator.** Create `tests/synth.py`:

```python
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
```

- [ ] **Step 2: Write the fixture and the failing end-to-end test.**

`tests/conftest.py`:

```python
from pathlib import Path

import pytest

from ber.config import load_config
from tests.synth import write_raw

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def small_cfg(tmp_path: Path) -> dict:
    raw = tmp_path / "raw"
    write_raw(raw)
    cfg = load_config(REPO / "configs" / "base.yaml")
    cfg["paths"].update(raw_dir=str(raw), raw_zip="", work_dir=str(tmp_path / "work"),
                        scratch_dir=str(tmp_path / "scratch"), output_dir=str(tmp_path / "output"))
    cfg["experiments_csv"] = str(tmp_path / "exp.csv")
    cfg["validation"]["valid_frac"] = 0.3
    cfg["lexicon"].update(min_count=2, abbrev_min_count=3)
    cfg["normalize"]["n_jobs"] = 1
    cfg["blocking"]["tfidf"].update(min_df=1, max_df=1.0, threshold=0.1, n_threads=1, chunk_size=50)
    cfg["cheap"].update(max_train_s1=None, params={"num_leaves": 15, "learning_rate": 0.1, "min_data_in_leaf": 5})
    cfg["features"].update(max_train_s1=None, vectorizer_fit_rows=100_000)
    cfg["gbm"].update(num_boost_round=300, early_stopping_rounds=30, max_es_s1=None,
                      params={"num_leaves": 15, "min_data_in_leaf": 5})
    return cfg
```

`tests/test_end_to_end.py`:

```python
from pathlib import Path

import polars as pl
import pytest

from ber.pipeline import run

pytest.importorskip("sparse_dot_topn")

STEPS = [("ingest", "train"), ("ingest", "test"), ("split", "train"), ("lexicon", "train"),
         ("normalize", "train"), ("normalize", "test"), ("block", "train"), ("block", "test"),
         ("cheap_train", "train"), ("cheap_apply", "train"), ("cheap_apply", "test"),
         ("features", "train"), ("features", "test"), ("train", "train"), ("evaluate", "train"),
         ("predict", "test"), ("submit", "test")]


def test_pipeline_end_to_end(small_cfg):
    results = {stage: run(small_cfg, stage, split) for stage, split in STEPS}
    metrics = results["evaluate"]
    assert metrics["recall_cands_final"] > 0.9
    assert metrics["f05"] > 0.6
    assert results["submit"] == 0
    out = pl.read_csv(Path(small_cfg["paths"]["output_dir"]) / "matching_results.tsv", separator="\t",
                      quote_char=None, infer_schema=False)
    assert out.height == 60  # every test S1, France included
```

Run: `.venv/Scripts/python -m pytest tests/test_end_to_end.py -v`
Expected: FAIL with `KeyError: 'train'` (the stage is not registered yet)

- [ ] **Step 3: Add the train, evaluate and predict stages.** Append to `src/ber/stages/model.py`. Imports at the top of the file:

```python
import json

import joblib
import numpy as np

from ber.config import MODEL_SPLIT
from ber.decide import SCALES, calibrate, choose_expected_f05, choose_threshold, crossfit_calibrate, fit_calibrator, one_owner, tune_threshold
from ber.features import FEATURE_COLUMNS
from ber.metrics import blocking_recall, cands_per_s1, log_experiment, macro_f05_frame
from ber.models.gbm import load_booster, predict, train_binary
from ber.stages.common import es_fold, save
```

The stage code:

```python
def stage_train(cfg: dict, split: str) -> None:
    feats = load_features(cfg, split).collect().join(load(cfg, split, "folds.parquet"), on="l_idx")
    cols = [c for c in FEATURE_COLUMNS if c in feats.columns]
    fit = feats.filter(pl.col("fold") == "fit")
    es = feats.join(sample_l(load(cfg, split, "folds.parquet"), es_fold(cfg), cfg["gbm"]["max_es_s1"], cfg["seed"]),
                    on="l_idx", how="semi")
    g = cfg["gbm"]
    booster = train_binary(fit.select(cols), fit["y"].to_numpy(), es.select(cols), es["y"].to_numpy(),
                           g["params"], g["num_boost_round"], g["early_stopping_rounds"])
    booster.save_model(str(write_path(cfg, split, "gbm_model.txt")))
    ev = feats.filter(pl.col("fold").is_in(eval_folds(cfg)))
    for name, part in (("valid_pred.parquet", ev), ("fit_pred.parquet", fit)):
        out = part.select("l_idx", "r_idx", "y", "fold").with_columns(p=pl.Series(predict(booster, part.select(cols))))
        save(out, cfg, split, name)
    imp = sorted(zip(booster.feature_name(), booster.feature_importance("gain")), key=lambda t: -t[1])
    print("[train] top features:", [f for f, _ in imp[:15]])


def _decide(pred: pl.DataFrame, gt: pl.DataFrame, eval_l: pl.Series):
    sel_e = choose_expected_f05(pred)
    f_e = macro_f05_frame(sel_e, gt, eval_l)
    t, f_t = tune_threshold(pred, gt, eval_l)
    scores = {"f05_expected": f_e, "f05_threshold": f_t, "threshold": t}
    if f_e >= f_t:
        return {"method": "expected_f05", "threshold": t}, sel_e, scores
    return {"method": "threshold", "threshold": t}, choose_threshold(pred, t), scores


def stage_evaluate(cfg: dict, split: str) -> dict:
    pred = load(cfg, split, cfg["evaluate"]["pred_file"])
    gt, folds = load(cfg, split, "gt.parquet"), load(cfg, split, "folds.parquet")
    s1c = load(cfg, split, "s1.parquet").select(pl.col("idx").alias("l_idx"), "country")
    eval_l = folds.filter(pl.col("fold") == "valid")["l_idx"]
    owner = one_owner if cfg["decide"]["one_owner"] else (lambda df: df)
    res = {"run": cfg["run_name"], "split": split, "scheme": cfg["validation"]["scheme"],
           "pred_file": cfg["evaluate"]["pred_file"]}
    if cfg["validation"]["scheme"] == "s1_random":
        ev = pred.filter(pl.col("fold") == "valid")
        iso = fit_calibrator(ev["p"].to_numpy(), ev["y"].to_numpy())
        cal = ev.with_columns(p=pl.Series(crossfit_calibrate(ev, cfg["seed"])))
        decision, sel, scores = _decide(owner(cal), gt, eval_l)
        res.update(scores)
        res["f05"] = max(scores["f05_expected"], scores["f05_threshold"])
    else:  # country_holdout: calibrate on seen-country rows, tune the unseen scale on the held-out country
        seen, unseen = pred.filter(pl.col("fold") == "valid_seen"), pred.filter(pl.col("fold") == "valid")
        iso = fit_calibrator(seen["p"].to_numpy(), seen["y"].to_numpy())
        cal = unseen.with_columns(p=pl.Series(calibrate(iso, unseen["p"].to_numpy())))
        best = None
        for s in SCALES:
            d, sel_s, sc = _decide(owner(cal.with_columns(p=(pl.col("p") * s).clip(0.0, 1.0))), gt, eval_l)
            f = max(sc["f05_expected"], sc["f05_threshold"])
            res[f"f05_scale_{s}"] = f
            if best is None or f > best[0]:
                best = (f, s, d, sel_s, sc)
        _, res["best_unseen_scale"], decision, sel, scores = best
        res.update(scores)
        res["f05"] = res["f05_scale_1.0"]
    ev_c = s1c.join(eval_l.to_frame(), on="l_idx", how="semi")
    for country in sorted(ev_c["country"].unique()):
        res[f"f05_{country}"] = macro_f05_frame(sel, gt, ev_c.filter(pl.col("country") == country)["l_idx"])
    singles = ev_c.join(gt.select("l_idx").unique(), on="l_idx", how="anti")["l_idx"]
    res["singleton_acc"] = macro_f05_frame(sel, gt, singles) if singles.len() else float("nan")
    for name in ("cands_all", "cands_final"):
        cands = load(cfg, split, f"{name}.parquet")
        res[f"recall_{name}"] = blocking_recall(cands, gt, eval_l)
        res[f"per_s1_{name}"] = cands_per_s1(cands, eval_l)
    joblib.dump(iso, write_path(cfg, split, "calibrator.joblib"))
    write_path(cfg, split, "decision.json").write_text(json.dumps(decision))
    log_experiment(cfg["experiments_csv"], res)
    print(json.dumps(res, indent=1, default=str))
    return res


def stage_predict(cfg: dict, split: str) -> None:
    booster = load_booster(read_path(cfg, MODEL_SPLIT[split], "gbm_model.txt"))
    lf = load_features(cfg, split)
    n = lf.select(pl.len()).collect().item()
    step = cfg["gbm"]["predict_chunk_rows"]
    outs = []
    for s in range(0, n, step):
        part = lf.slice(s, step).collect()
        outs.append(part.select("l_idx", "r_idx").with_columns(p=pl.Series(predict(booster, part))))
    save(pl.concat(outs), cfg, split, "pred.parquet")
```

- [ ] **Step 4: Add the submit stage.** Create `src/ber/stages/output.py`:

```python
"""Output stages: submit (and package, added in Task 25)."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import polars as pl

from ber.config import MODEL_SPLIT, read_path
from ber.decide import calibrate, one_owner, scale_unseen, select
from ber.stages.common import load
from ber.submit import ensure_test_dir, run_validator, write_id_lists


def stage_submit(cfg: dict, split: str) -> int:
    if split != "test":
        raise ValueError("submit runs on the test split only")
    ms = MODEL_SPLIT[split]
    iso = joblib.load(read_path(cfg, ms, "calibrator.joblib"))
    decision = json.loads(read_path(cfg, ms, "decision.json").read_text())
    s1, right = load(cfg, split, "s1.parquet"), load(cfg, split, "right.parquet")
    pred = load(cfg, split, cfg["submit"]["pred_file"])
    pred = pred.with_columns(p=pl.Series(calibrate(iso, pred["p"].to_numpy())))
    d = cfg["decide"]
    pred = scale_unseen(pred, s1.select(pl.col("idx").alias("l_idx"), "country"),
                        d["seen_countries"], d["unseen_country_scale"])
    if d["one_owner"]:
        pred = one_owner(pred)
    matches = select(pred, decision)
    cands = load(cfg, split, "cands_final.parquet").select("l_idx", "r_idx")
    out = Path(cfg["paths"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    write_id_lists(out / "matching_results.tsv", "matched_entity_ids", s1, right, matches)
    write_id_lists(out / "candidate_pairs.tsv", "candidate_entity_ids", s1, right, cands)
    per = (s1.select(pl.col("idx").alias("l_idx"), "country")
           .join(matches.group_by("l_idx").len("n"), on="l_idx", how="left").fill_null(0)
           .group_by("country").agg(mean_matches=pl.col("n").mean(), empty_share=(pl.col("n") == 0).mean()))
    print(per)
    code = run_validator(out / "matching_results.tsv", out / "candidate_pairs.tsv", ensure_test_dir(cfg))
    if code != 0:
        raise RuntimeError("validate_submission FAILED; do not upload")
    return code
```

Register in `pipeline.STAGES`: `"train": model.stage_train, "evaluate": model.stage_evaluate, "predict": model.stage_predict, "submit": output.stage_submit` (with `from ber.stages import candidates, data, model, output`).

- [ ] **Step 5: Run the end-to-end test and the full suite.**

Run: `.venv/Scripts/python -m pytest -v`
Expected: all pass, and `test_pipeline_end_to_end` passes in under 2 minutes. If `f05` is at most 0.6, inspect `exp.csv` in the pytest tmp dir. The likely causes are recall (blocking config) or a label join bug. Debug with superpowers:systematic-debugging; do not loosen the assert.

- [ ] **Step 6: Commit.**

```bash
git add src/ber/stages src/ber/pipeline.py tests/synth.py tests/conftest.py tests/test_end_to_end.py
git commit -m "feat: train/evaluate/predict/submit stages; synthetic end-to-end test"
git push
```

---

### Task 18: Local dev run, Kaggle driver, first full Kaggle run, leaderboard upload (completes M1)

**Files:**
- Create: `notebooks/kaggle_driver.py`, `docs/Documentation.md`, `requirements.txt`
- Modify: `memory.md`, `configs/kaggle.yaml` (the confirmed `raw_dir`)

**Interfaces:**
- Consumes: all stages
- Produces:
  - the first leaderboard score
  - a pinned `requirements.txt`
  - Kaggle session outputs (`ber-m1-a`, `ber-m1-b`) for later sessions to reuse through `prev_work_dirs`

- [ ] **Step 1: Run the full dev loop locally.**

```bash
.venv/Scripts/python -m ber.pipeline --config configs/dev.yaml --stage train --split dev
.venv/Scripts/python -m ber.pipeline --config configs/dev.yaml --stage evaluate --split dev
```

Expected: the printed JSON has `f05` and `recall_cands_final`. Record the dev `f05`, per-country F0.5, singleton accuracy and stage timings in `memory.md` → Results. Extrapolate the Kaggle timings: full train is about 50× the dev slice, but divide by the thread ratio. If any stage is estimated at more than 10 hours, lower that stage's size knobs first (`blocking.tfidf.top_k_fwd`, `views`, `features.max_train_s1`) and note the change.

- [ ] **Step 2: Write `notebooks/kaggle_driver.py`.**

```python
# %% [markdown]
# # ber: Kaggle driver
# Inputs to attach: dataset `ber-raw` (the organizer zip; Kaggle auto-extracts it) and, when resuming,
# the output of an earlier committed version of this notebook (set PREV below).
# Secrets: GITHUB_TOKEN (fine-grained, read-only "Contents" on the private repo).

# %% setup
import os
import subprocess
import sys

from kaggle_secrets import UserSecretsClient

GH_REPO = "GITHUB_USER/ber-2026"  # replace GITHUB_USER with the account recorded in memory.md (Task 1)
BRANCH = "main"
CODE = "/kaggle/working/ber"
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
if not os.path.exists(CODE):
    subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH,
                    f"https://{token}@github.com/{GH_REPO}.git", CODE], check=True)
    # never persist the token inside the saved notebook output
    subprocess.run(["git", "-C", CODE, "remote", "set-url", "origin", f"https://github.com/{GH_REPO}.git"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", CODE], check=True)
os.chdir(CODE)
subprocess.run(["df", "-h", "/kaggle/working", "/kaggle/tmp"], check=False)
subprocess.run(["ls", "/kaggle/input"], check=False)

# %% config
CONFIG = "configs/kaggle.yaml"
PREV: list[str] = []  # e.g. ["/kaggle/input/ber-m1-a/work"] (the work dir of an earlier session's output)


def stage(name: str, split: str = "train", *overrides: str) -> None:
    args = [sys.executable, "-m", "ber.pipeline", "--config", CONFIG, "--stage", name, "--split", split]
    if PREV:
        args += ["--set", f"paths.prev_work_dirs=[{','.join(PREV)}]"]
    for o in overrides:
        args += ["--set", o]
    subprocess.run(args, check=True)


# %% session A (CPU notebook): data, lexicon, normalization, blocking, cheap cut
for sp in ("train", "test"):
    stage("ingest", sp)
stage("split", "train")
stage("lexicon", "train")
for sp in ("train", "test"):
    stage("normalize", sp)
for sp in ("train", "test"):
    stage("block", sp)
stage("cheap_train", "train")
for sp in ("train", "test"):
    stage("cheap_apply", sp)

# %% session B (CPU notebook, PREV = session A output): features, model, evaluation, submission
for sp in ("train", "test"):
    stage("features", sp)
stage("train", "train")
stage("evaluate", "train")
stage("predict", "test")
stage("submit", "test")

# %% pinned requirements (run once per milestone; download and commit as requirements.txt)
subprocess.run(
    f"{sys.executable} -m pip freeze | grep -iE '^(polars|pyarrow|numpy|scipy|scikit-learn|joblib|rapidfuzz|"
    "jellyfish|unidecode|indic.transliteration|sparse.dot.topn|lightgbm|pyyaml|faiss.*|torch|transformers|"
    "sentence.transformers|datasets|accelerate)==' > /kaggle/working/requirements.txt", shell=True, check=True)
```

- [ ] **Step 3: One-time Kaggle setup.** The user does this with guidance.
  1. Kaggle → Datasets → New Dataset: upload `6ab10eb3b23ba_student_resource.zip`, name it `ber-raw`, keep it **Private**.
  2. On github.com, create a fine-grained token with read-only Contents access to `ber-2026`. Add it in the Kaggle notebook under Add-ons → Secrets as `GITHUB_TOKEN`.
  3. Create a new notebook, paste `notebooks/kaggle_driver.py` into it (one cell per `# %%`), and attach `ber-raw`.
  4. Run the setup cell and confirm that `ls /kaggle/input` and the path of `test_source1.tsv` match `configs/kaggle.yaml → paths.raw_dir`. If they differ, fix the yaml, commit, push, and re-clone.
  5. Record the `df -h` output in `memory.md`, especially the space free on `/kaggle/tmp`.

- [ ] **Step 4: Run session A.** Use "Save Version → Save & Run All" so the output is kept. Name the version `ber-m1-a`, and run only the setup, config and session-A cells. Record the stage timings from the log in `memory.md`.

- [ ] **Step 5: Run session B.**
  1. Start a new notebook version with the output of `ber-m1-a` attached as input.
  2. Set `PREV = ["/kaggle/input/<ber-m1-a-slug>/work"]` (check the path with `ls`).
  3. Run the session-B cells as "Save & Run All" with version name `ber-m1-b`.

  Expected: `evaluate` prints `f05` and `submit` prints `PASS — no blocking issues found`.

- [ ] **Step 6: Leaderboard.**
  1. Download `output/matching_results.tsv` from the `ber-m1-b` output.
  2. The user uploads it in the challenge Portal.
  3. Record the public LB score next to the local `f05` in `memory.md` → Results. The gap between them is the calibration of our local validation.

- [ ] **Step 7: Freeze the requirements and start the documentation.**
  1. Run the requirements cell, download `requirements.txt`, and put it in the repo root.
  2. Create `docs/Documentation.md` by copying `docs/Documentation_template_original.md`.
  3. Fill in sections 2.1 (EDA facts from `memory.md`), 3 (blocking keys, TF-IDF views, candidate counts, recall) and 4 (feature list, LightGBM, decision rule) with the numbers measured so far.

- [ ] **Step 8: Commit and push.**

```bash
git add notebooks/kaggle_driver.py docs/Documentation.md requirements.txt configs/kaggle.yaml memory.md
git commit -m "chore: Kaggle driver, pinned requirements, M1 results"
git push
```

### Task 19: Country-holdout baseline (US → India), the France stand-in

**Files:**
- Modify: `memory.md`, and `configs/kaggle.yaml` only if the scale rule below fires

**Interfaces:**
- Consumes: `configs/kaggle_holdout.yaml`, the `ber-m1-a` output (train parquet reused through `PREV`)
- Produces:
  - holdout rows in `experiments.csv`: `f05` (scale 1.0), `f05_scale_*`, `best_unseen_scale`
  - the baseline that every later stage must beat on this scheme

- [ ] **Step 1: Run the holdout sessions on Kaggle.**
  - Driver setup: `CONFIG = "configs/kaggle_holdout.yaml"`, `PREV = ["/kaggle/input/<ber-m1-a-slug>/work"]`.
  - Run these **in this order**, all on `train`: `split`, `lexicon`, `normalize`, `block`, `cheap_train`, `cheap_apply`, `features`, `train`, `evaluate`.
  - Skip `ingest`: `s1/right/gt.parquet` are read from `PREV`.
  - **Never skip a later stage.** Otherwise `read_path` silently falls back to M1's artifacts in `PREV`.
  - Split the stages across two "Save & Run All" versions, `ber-m1-holdout-a` and `ber-m1-holdout-b`, if needed.

- [ ] **Step 2: Record the results and apply the France rule.**
  1. Copy the holdout row of `experiments.csv` into `memory.md` → Results.
  2. If `f05_scale_<best>` − `f05_scale_1.0` ≥ 0.005, set `decide.unseen_country_scale: <best_unseen_scale>` in `configs/kaggle.yaml`. Otherwise keep 1.0.
  3. Write the decision and its numbers under `memory.md` → Decisions.

- [ ] **Step 3: Commit.**

```bash
git add memory.md configs/kaggle.yaml
git commit -m "chore: country-holdout baseline and unseen-country scale decision"
git push
```

---

## Milestone M2: Fine-tuned multilingual bi-encoder

### Task 20: Bi-encoder module (triplets, fine-tuning, encoding)

**Files:**
- Create: `src/ber/models/biencoder.py`
- Test: `tests/test_biencoder.py`

**Interfaces:**
- Consumes: `cands_final.parquet` (with `cheap_score`), `gt.parquet`, `folds.parquet`
- Produces:
  - `record_text(df[business_name, business_address]) -> pl.Series`, giving `"query: <name> | <address>"` (raw text, so the multilingual model sees the original script)
  - `make_training_triplets(cands, gt, fit_l, left_text, right_text, n, seed) -> pl.DataFrame[anchor, positive, negative]`
  - `train_biencoder(triplets, model_name, out_dir, epochs, batch_size, lr, max_seq_len, seed) -> None`
  - `encode_to_memmap(model_dir, texts, path, batch_size, max_seq_len) -> np.memmap[float16, (n, 384)]` (L2-normalized)

- [ ] **Step 1: Write the failing tests.** Create `tests/test_biencoder.py`:

```python
import polars as pl

from ber.models.biencoder import make_training_triplets, record_text

U = pl.UInt32


def test_record_text():
    df = pl.DataFrame({"business_name": ["राम Traders"], "business_address": ["Pune"]})
    assert record_text(df).to_list() == ["query: राम Traders | Pune"]


def test_triplets_use_hardest_non_match_and_fallback():
    left_text = pl.Series(["L0", "L1"])
    right_text = pl.Series(["R0", "R1", "R2", "R3"])
    gt = pl.DataFrame({"l_idx": [0, 1], "r_idx": [0, 2]}, schema={"l_idx": U, "r_idx": U})
    cands = pl.DataFrame({"l_idx": [0, 0, 0], "r_idx": [0, 1, 3], "cheap_score": [0.9, 0.2, 0.7]},
                         schema_overrides={"l_idx": U, "r_idx": U})
    fit_l = pl.DataFrame({"l_idx": [0, 1]}, schema={"l_idx": U})
    t = make_training_triplets(cands, gt, fit_l, left_text, right_text, n=10, seed=0).sort("anchor")
    assert t.columns == ["anchor", "positive", "negative"]
    assert t.row(0) == ("L0", "R0", "R3")          # hardest non-match by cheap_score
    assert t.row(1)[:2] == ("L1", "R2") and t.row(1)[2] in {"R0", "R1", "R2", "R3"}  # random fallback
```

- [ ] **Step 2: Run the tests and confirm they fail.**

Run: `.venv/Scripts/python -m pytest tests/test_biencoder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ber.models.biencoder'`

- [ ] **Step 3: Implement `src/ber/models/biencoder.py`.**

```python
"""Bi-encoder: fine-tune intfloat/multilingual-e5-small (MIT, 118M) with in-batch + hard negatives.
Heavy imports (torch, sentence-transformers, datasets) happen inside functions so the module is
importable locally without GPU packages.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


def record_text(df: pl.DataFrame) -> pl.Series:
    return df.select(pl.concat_str([pl.lit("query: "), pl.col("business_name"), pl.lit(" | "),
                                    pl.col("business_address")]).alias("text")).to_series()


def make_training_triplets(cands: pl.DataFrame, gt: pl.DataFrame, fit_l: pl.DataFrame, left_text: pl.Series,
                           right_text: pl.Series, n: int, seed: int) -> pl.DataFrame:
    pos = gt.join(fit_l, on="l_idx", how="semi")
    if pos.height > n:
        pos = pos.sample(n, seed=seed)
    neg = (cands.join(gt, on=["l_idx", "r_idx"], how="anti").join(fit_l, on="l_idx", how="semi")
           .sort("cheap_score", descending=True)
           .unique(subset="l_idx", keep="first", maintain_order=True)
           .select("l_idx", pl.col("r_idx").alias("neg_idx")))
    t = pos.join(neg, on="l_idx", how="left")
    fallback = np.random.default_rng(seed).integers(0, right_text.len(), t.height)
    neg_idx = np.where(t["neg_idx"].is_null().to_numpy(), fallback, t["neg_idx"].fill_null(0).to_numpy())
    return pl.DataFrame({
        "anchor": left_text.gather(t["l_idx"]),
        "positive": right_text.gather(t["r_idx"]),
        "negative": right_text.gather(pl.Series(neg_idx)),
    })


def train_biencoder(triplets: pl.DataFrame, model_name: str, out_dir: str | Path, epochs: int,
                    batch_size: int, lr: float, max_seq_len: int, seed: int) -> None:
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.training_args import BatchSamplers

    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_len
    ds = Dataset.from_dict({c: triplets[c].to_list() for c in ("anchor", "positive", "negative")})
    args = SentenceTransformerTrainingArguments(
        output_dir=str(Path(out_dir) / "_ckpt"), num_train_epochs=epochs, per_device_train_batch_size=batch_size,
        learning_rate=lr, warmup_ratio=0.05, fp16=True, batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no", logging_steps=200, report_to="none", seed=seed,
    )
    SentenceTransformerTrainer(model=model, args=args, train_dataset=ds,
                               loss=losses.MultipleNegativesRankingLoss(model)).train()
    model.save(str(out_dir))


def encode_to_memmap(model_dir: str | Path, texts: list[str], path: str | Path, batch_size: int,
                     max_seq_len: int, chunk: int = 500_000) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_dir), device="cuda")
    model.max_seq_length = max_seq_len
    model.half()
    dim = model.get_sentence_embedding_dimension()
    arr = np.lib.format.open_memmap(str(path), mode="w+", dtype=np.float16, shape=(len(texts), dim))
    for s in range(0, len(texts), chunk):
        arr[s:s + chunk] = model.encode(texts[s:s + chunk], batch_size=batch_size, normalize_embeddings=True,
                                        convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
        print(f"[encode] {min(s + chunk, len(texts)):,}/{len(texts):,}")
    arr.flush()
    return arr
```

- [ ] **Step 4: Run the tests and confirm they pass.**

Run: `.venv/Scripts/python -m pytest tests/test_biencoder.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit.**

```bash
git add src/ber/models/biencoder.py tests/test_biencoder.py
git commit -m "feat: bi-encoder triplet builder, fine-tuning and encoding"
```

---

### Task 21: Embedding k-NN blocking, embedding stages, and the M2 Kaggle run

**Files:**
- Create: `src/ber/blocking/embed_ann.py`, `src/ber/stages/nn.py`
- Modify: `src/ber/pipeline.py`, `notebooks/kaggle_driver.py`, `memory.md`, `docs/Documentation.md`
- Test: `tests/test_embed_ann.py`

**Interfaces:**
- Consumes: `record_text`, `make_training_triplets`, `train_biencoder`, `encode_to_memmap` (Task 20); the `block` stage (Task 12), which already merges `cands_embed.parquet` when `blocking.embed.enabled`
- Produces:
  - `knn(index_vecs, query_vecs, k, use_gpu) -> (D float32[n,k], I int64[n,k])`
  - `embed_candidates(left_meta[idx,country], right_meta[idx,country], emb_l, emb_r, k_fwd, k_rev, same_country, use_gpu) -> pl.DataFrame[l_idx, r_idx, embed_cos]`
  - `pair_cosine(emb_l, emb_r, li, ri) -> np.ndarray[float32]`
  - stages:
    - `embed_train` (train/dev) → `work/<split>/biencoder/`
    - `embed_encode` → `scratch/<split>/emb_{s1,right}.npy` and `cands_embed.parquet`
    - `embed_attach` → rewrites `cands_all.parquet` with the exact `embed_cos` for every pair

- [ ] **Step 1: Write the failing tests.** Create `tests/test_embed_ann.py`:

```python
import numpy as np
import polars as pl
import pytest

pytest.importorskip("faiss")
from ber.blocking.embed_ann import embed_candidates, knn, pair_cosine  # noqa: E402


def _unit(rng, n, d=16):
    v = rng.normal(size=(n, d)).astype(np.float32)
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float16)


def test_knn_cpu_finds_identical_vectors():
    rng = np.random.default_rng(0)
    base = _unit(rng, 50)
    D, I = knn(base, base[[3, 7]], k=2, use_gpu=False)
    assert I[:, 0].tolist() == [3, 7] and np.allclose(D[:, 0], 1.0, atol=1e-2)


def test_embed_candidates_within_country_both_directions():
    rng = np.random.default_rng(1)
    emb_l = _unit(rng, 4)
    emb_r = np.concatenate([emb_l[[2, 0]], _unit(rng, 2)])  # r0==l2, r1==l0
    lm = pl.DataFrame({"idx": [0, 1, 2, 3], "country": ["US", "US", "US", "France"]}, schema_overrides={"idx": pl.UInt32})
    rm = pl.DataFrame({"idx": [0, 1, 2, 3], "country": ["US", "US", "France", "US"]}, schema_overrides={"idx": pl.UInt32})
    out = embed_candidates(lm, rm, emb_l, emb_r, k_fwd=1, k_rev=1, same_country=True, use_gpu=False)
    pairs = set(zip(out["l_idx"].to_list(), out["r_idx"].to_list()))
    assert {(2, 0), (0, 1)} <= pairs
    assert all(not (l == 3 and r in (0, 1, 3)) for l, r in pairs)  # France S1 only sees France right
    assert out["embed_cos"].dtype == pl.Float32


def test_pair_cosine():
    rng = np.random.default_rng(2)
    a, b = _unit(rng, 5), _unit(rng, 6)
    got = pair_cosine(a, b, np.array([0, 4]), np.array([5, 1]))
    want = [float(a[0].astype(np.float32) @ b[5].astype(np.float32)), float(a[4].astype(np.float32) @ b[1].astype(np.float32))]
    assert np.allclose(got, want, atol=1e-3)
```

- [ ] **Step 2: Run the tests and confirm they fail.**

Run: `.venv/Scripts/python -m pytest tests/test_embed_ann.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ber.blocking.embed_ann'`

- [ ] **Step 3: Implement `src/ber/blocking/embed_ann.py`.**

```python
"""Exact inner-product k-NN over L2-normalized embeddings (cosine), per country, both directions.
GPU: FAISS GPU if available, else a chunked torch matmul fallback. CPU: FAISS IndexFlatIP.
"""
from __future__ import annotations

import numpy as np
import polars as pl

_SCHEMA = {"l_idx": pl.UInt32, "r_idx": pl.UInt32, "embed_cos": pl.Float32}


def _knn_faiss(index_vecs, query_vecs, k: int, gpu: bool, add_chunk: int = 1_000_000, batch: int = 65_536):
    import faiss

    d = index_vecs.shape[1]
    if gpu:
        res = faiss.StandardGpuResources()
        conf = faiss.GpuIndexFlatConfig()
        conf.useFloat16 = True
        index = faiss.GpuIndexFlatIP(res, d, conf)
    else:
        index = faiss.IndexFlatIP(d)
    for s in range(0, len(index_vecs), add_chunk):
        index.add(np.ascontiguousarray(index_vecs[s:s + add_chunk], dtype=np.float32))
    dists, ids = [], []
    for s in range(0, len(query_vecs), batch):
        d_, i_ = index.search(np.ascontiguousarray(query_vecs[s:s + batch], dtype=np.float32), k)
        dists.append(d_)
        ids.append(i_)
    return np.vstack(dists), np.vstack(ids)


def _knn_torch(index_vecs, query_vecs, k: int, batch: int = 1024, chunk: int = 1_000_000):
    import torch

    xs = [torch.from_numpy(np.ascontiguousarray(index_vecs[s:s + chunk], dtype=np.float16)).cuda()
          for s in range(0, len(index_vecs), chunk)]
    d_out = np.empty((len(query_vecs), k), np.float32)
    i_out = np.empty((len(query_vecs), k), np.int64)
    with torch.no_grad():
        for s in range(0, len(query_vecs), batch):
            q = torch.from_numpy(np.ascontiguousarray(query_vecs[s:s + batch], dtype=np.float16)).cuda()
            best_d = best_i = None
            for ci, x in enumerate(xs):
                d_, i_ = (q @ x.T).float().topk(min(k, x.shape[0]), dim=1)
                i_ = i_ + ci * chunk
                if best_d is None:
                    best_d, best_i = d_, i_
                else:
                    cat_d, cat_i = torch.cat([best_d, d_], 1), torch.cat([best_i, i_], 1)
                    best_d, pos = cat_d.topk(k, dim=1)
                    best_i = cat_i.gather(1, pos)
            d_out[s:s + len(q)] = best_d.cpu().numpy()
            i_out[s:s + len(q)] = best_i.cpu().numpy()
    return d_out, i_out


def knn(index_vecs, query_vecs, k: int, use_gpu: bool):
    k = min(k, len(index_vecs))
    if not use_gpu:
        return _knn_faiss(index_vecs, query_vecs, k, gpu=False)
    try:
        import faiss
        if faiss.get_num_gpus() > 0:
            return _knn_faiss(index_vecs, query_vecs, k, gpu=True)
    except ImportError:
        pass
    return _knn_torch(index_vecs, query_vecs, k)


def embed_candidates(left_meta: pl.DataFrame, right_meta: pl.DataFrame, emb_l, emb_r, k_fwd: int, k_rev: int,
                     same_country: bool, use_gpu: bool) -> pl.DataFrame:
    groups = (sorted(set(left_meta["country"].unique()) | set(right_meta["country"].unique()))
              if same_country else [None])
    frames = []
    for country in groups:
        lm = left_meta if country is None else left_meta.filter(pl.col("country") == country)
        rm = right_meta if country is None else right_meta.filter(pl.col("country") == country)
        li, ri = lm["idx"].to_numpy(), rm["idx"].to_numpy()
        if len(li) == 0 or len(ri) == 0:
            continue
        el, er = np.asarray(emb_l[li]), np.asarray(emb_r[ri])
        d1, i1 = knn(er, el, k_fwd, use_gpu)
        d2, i2 = knn(el, er, k_rev, use_gpu)
        ok1, ok2 = i1 >= 0, i2 >= 0
        l1 = np.broadcast_to(li[:, None], i1.shape)[ok1]
        r1 = ri[i1[ok1]]
        l2 = li[i2[ok2]]
        r2 = np.broadcast_to(ri[:, None], i2.shape)[ok2]
        frames.append(pl.DataFrame({"l_idx": np.concatenate([l1, l2]), "r_idx": np.concatenate([r1, r2]),
                                    "embed_cos": np.concatenate([d1[ok1], d2[ok2]]).astype(np.float32)},
                                   schema=_SCHEMA))
        print(f"[embed_ann] {country}: L={len(li):,} R={len(ri):,} pairs={frames[-1].height:,}")
    if not frames:
        return pl.DataFrame(schema=_SCHEMA)
    return pl.concat(frames).group_by("l_idx", "r_idx").agg(pl.col("embed_cos").max())


def pair_cosine(emb_l, emb_r, li: np.ndarray, ri: np.ndarray, chunk: int = 500_000) -> np.ndarray:
    out = np.empty(len(li), np.float32)
    for s in range(0, len(li), chunk):
        a = np.asarray(emb_l[li[s:s + chunk]], dtype=np.float32)
        b = np.asarray(emb_r[ri[s:s + chunk]], dtype=np.float32)
        out[s:s + chunk] = (a * b).sum(axis=1)
    return out
```

- [ ] **Step 4: Implement the embedding stages.** Create `src/ber/stages/nn.py`:

```python
"""GPU stages: bi-encoder (M2) and cross-encoder + combiner (M3). Heavy imports stay inside functions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from ber.blocking.embed_ann import embed_candidates, pair_cosine
from ber.config import MODEL_SPLIT, read_path
from ber.models.biencoder import encode_to_memmap, make_training_triplets, record_text, train_biencoder
from ber.stages.common import load, save


def _scratch(cfg: dict, split: str) -> Path:
    path = Path(cfg["paths"]["scratch_dir"]) / split
    path.mkdir(parents=True, exist_ok=True)
    return path


def stage_embed_train(cfg: dict, split: str) -> None:
    bc = cfg["biencoder"]
    s1, right = load(cfg, split, "s1.parquet"), load(cfg, split, "right.parquet")
    fit_l = load(cfg, split, "folds.parquet").filter(pl.col("fold") == "fit").select("l_idx")
    trip = make_training_triplets(load(cfg, split, "cands_final.parquet"), load(cfg, split, "gt.parquet"), fit_l,
                                  record_text(s1), record_text(right), bc["n_triplets"], cfg["seed"])
    out = Path(cfg["paths"]["work_dir"]) / split / "biencoder"
    train_biencoder(trip, bc["model_name"], out, bc["epochs"], bc["batch_size"], bc["lr"], bc["max_seq_len"], cfg["seed"])


def stage_embed_encode(cfg: dict, split: str) -> None:
    bc, e = cfg["biencoder"], cfg["blocking"]["embed"]
    model_dir = read_path(cfg, MODEL_SPLIT[split], "biencoder")
    s1, right = load(cfg, split, "s1.parquet"), load(cfg, split, "right.parquet")
    scratch = _scratch(cfg, split)
    emb_l = encode_to_memmap(model_dir, record_text(s1).to_list(), scratch / "emb_s1.npy",
                             bc["encode_batch_size"], bc["max_seq_len"])
    emb_r = encode_to_memmap(model_dir, record_text(right).to_list(), scratch / "emb_right.npy",
                             bc["encode_batch_size"], bc["max_seq_len"])
    cands = embed_candidates(s1.select("idx", "country"), right.select("idx", "country"), emb_l, emb_r,
                             e["top_k_fwd"], e["top_k_rev"], cfg["blocking"]["same_country"], e["use_gpu"])
    save(cands, cfg, split, "cands_embed.parquet")


def stage_embed_attach(cfg: dict, split: str) -> None:
    scratch = _scratch(cfg, split)
    emb_l, emb_r = np.load(scratch / "emb_s1.npy"), np.load(scratch / "emb_right.npy")
    cands = load(cfg, split, "cands_all.parquet")
    cos = pair_cosine(emb_l, emb_r, cands["l_idx"].to_numpy(), cands["r_idx"].to_numpy())
    save(cands.with_columns(embed_cos=pl.Series(cos, dtype=pl.Float32)), cfg, split, "cands_all.parquet")
    print(f"[embed_attach] {split}: {cands.height:,} pairs")
```

Register `"embed_train"`, `"embed_encode"` and `"embed_attach"` in `pipeline.STAGES` (with `from ber.stages import candidates, data, model, nn, output`).

- [ ] **Step 5: Run the tests and confirm they pass, then commit and push.**

Run: `.venv/Scripts/python -m pytest tests/test_embed_ann.py -v`
Expected: 3 passed

```bash
git add src/ber/blocking/embed_ann.py src/ber/stages/nn.py src/ber/pipeline.py tests/test_embed_ann.py
git commit -m "feat: embedding kNN blocking and bi-encoder stages"
git push
```

- [ ] **Step 6: Add the M2 cells to `notebooks/kaggle_driver.py`.**

```python
# %% M2 GPU smoke test on the dev slice (PREV = ber-m1-a output)
EMB = "blocking.embed.enabled=true"
subprocess.run([sys.executable, "-c", "import faiss; print('faiss gpus', faiss.get_num_gpus())"], check=False)
for st in ("dev_slice", "split", "lexicon", "normalize", "block", "cheap_train", "cheap_apply"):
    stage(st, "train" if st == "dev_slice" else "dev")
stage("embed_train", "dev", "biencoder.n_triplets=20000")
stage("embed_encode", "dev", EMB)
stage("block", "dev", EMB)
stage("embed_attach", "dev")

# %% M2 GPU session (PREV = [ber-m1-a work, ber-m1-b work]): bi-encoder, embedding blocking, exact cosine
stage("embed_train", "train")
for sp in ("train", "test"):
    stage("embed_encode", sp, EMB)
    stage("block", sp, EMB)
    stage("embed_attach", sp)

# %% M2 CPU session (PREV = M2 GPU output + M1 outputs): rest of the pipeline
stage("cheap_train", "train")
for sp in ("train", "test"):
    stage("cheap_apply", sp)
for sp in ("train", "test"):
    stage("features", sp)
stage("train", "train", "run_name=kaggle_m2")
stage("evaluate", "train", "run_name=kaggle_m2")
stage("predict", "test")
stage("submit", "test")
```

- [ ] **Step 7: Run M2 on Kaggle.** Both schemes are required by the stage rule.
  1. Run the smoke cell on a GPU notebook. It must finish without errors.
  2. Run the GPU session (version `ber-m2-a`, accelerator "GPU T4 x2"), then the CPU session (`ber-m2-b`).
  3. Repeat both for `configs/kaggle_holdout.yaml` (`ber-m2-holdout-a/b`, `PREV` = the holdout M1 outputs).

  In the holdout run, `embed_train` uses only US `fit` S1 records, so India stays unseen.

- [ ] **Step 8: Apply the stage rule and record the result.**
  1. Compare `f05` and `recall_cands_all` for M2 against M1 in both schemes.
  2. **Keep M2 only if `f05` improved in both schemes.**
     - If kept: set `blocking.embed.enabled: true` in `configs/kaggle.yaml` and `configs/kaggle_holdout.yaml`, upload the M2 `matching_results.tsv` to the leaderboard, and record the public LB score.
     - If not kept: leave it disabled and write down why in `memory.md` → Decisions.
  3. Update `docs/Documentation.md` sections 3 and 4.

```bash
git add notebooks/kaggle_driver.py configs memory.md docs/Documentation.md
git commit -m "chore: M2 bi-encoder results"
git push
```

---

## Milestone M3: Cross-encoder reranker and combiner

### Task 22: Cross-encoder module and stages

**Files:**
- Create: `src/ber/models/crossencoder.py`
- Modify: `src/ber/stages/nn.py`, `src/ber/pipeline.py`
- Test: `tests/test_crossencoder.py`

**Interfaces:**
- Consumes: `fit_pred.parquet`, `valid_pred.parquet`, `pred.parquet` (Task 17)
- Produces:
  - `pair_text(df[business_name, business_address]) -> pl.Series` (`"<name> | <address>"`)
  - `band_mask(p, lo, hi) -> np.ndarray[bool]`
  - `sample_training_pairs(fit_pred, lo, hi, n, seed) -> pl.DataFrame[l_idx, r_idx, y]` (80% from the band, 20% from outside it)
  - `train_cross_encoder(text_a, text_b, labels, model_name, out_dir, max_len, epochs, batch_size, lr, seed) -> None`
  - `score_pairs(model_dir, text_a, text_b, max_len, batch_size) -> np.ndarray[float32]`
  - stages `cross_train` (train/dev → `work/<split>/crossencoder/`) and `cross_score` (→ `cross_scores.parquet` [l_idx, r_idx, cross_p], band pairs only)

- [ ] **Step 1: Write the failing tests.** Create `tests/test_crossencoder.py`:

```python
import numpy as np
import polars as pl

from ber.models.crossencoder import band_mask, pair_text, sample_training_pairs


def test_pair_text_and_band():
    df = pl.DataFrame({"business_name": ["Acme"], "business_address": ["Tyler"]})
    assert pair_text(df).to_list() == ["Acme | Tyler"]
    assert band_mask(np.array([0.05, 0.1, 0.5, 0.9, 0.95]), 0.1, 0.9).tolist() == [False, True, True, True, False]


def test_sample_training_pairs_mix():
    rng = np.random.default_rng(0)
    fp = pl.DataFrame({"l_idx": np.arange(1000, dtype=np.uint32), "r_idx": np.arange(1000, dtype=np.uint32),
                       "p": rng.random(1000), "y": rng.integers(0, 2, 1000)})
    out = sample_training_pairs(fp, 0.1, 0.9, n=100, seed=0)
    in_band = out.join(fp, on=["l_idx", "r_idx"]).filter(pl.col("p").is_between(0.1, 0.9)).height
    assert out.height == 100 and in_band == 80 and out.columns == ["l_idx", "r_idx", "y"]
```

- [ ] **Step 2: Run the tests and confirm they fail.**

Run: `.venv/Scripts/python -m pytest tests/test_crossencoder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ber.models.crossencoder'`

- [ ] **Step 3: Implement `src/ber/models/crossencoder.py`.**

```python
"""Cross-encoder reranker: xlm-roberta-base (MIT) pair classifier on uncertain pairs only."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


def pair_text(df: pl.DataFrame) -> pl.Series:
    return df.select(pl.concat_str([pl.col("business_name"), pl.lit(" | "), pl.col("business_address")])
                     .alias("text")).to_series()


def band_mask(p: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (p >= lo) & (p <= hi)


def sample_training_pairs(fit_pred: pl.DataFrame, lo: float, hi: float, n: int, seed: int) -> pl.DataFrame:
    in_band = fit_pred.filter(pl.col("p").is_between(lo, hi))
    rest = fit_pred.join(in_band.select("l_idx", "r_idx"), on=["l_idx", "r_idx"], how="anti")
    n_band = min(in_band.height, int(n * 0.8))
    n_rest = min(rest.height, n - n_band)
    return pl.concat([in_band.sample(n_band, seed=seed), rest.sample(n_rest, seed=seed)]).select("l_idx", "r_idx", "y")


def train_cross_encoder(text_a: list[str], text_b: list[str], labels: list[int], model_name: str,
                        out_dir: str | Path, max_len: int, epochs: int, batch_size: int, lr: float, seed: int) -> None:
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments)

    tok = AutoTokenizer.from_pretrained(model_name)
    ds = Dataset.from_dict({"a": text_a, "b": text_b, "label": labels}).map(
        lambda x: tok(x["a"], x["b"], truncation=True, max_length=max_len), batched=True, remove_columns=["a", "b"])
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    args = TrainingArguments(
        output_dir=str(Path(out_dir) / "_ckpt"), per_device_train_batch_size=batch_size, learning_rate=lr,
        num_train_epochs=epochs, warmup_ratio=0.06, weight_decay=0.01, fp16=True, logging_steps=500,
        save_strategy="no", report_to="none", seed=seed, dataloader_num_workers=2,
    )
    Trainer(model=model, args=args, train_dataset=ds, data_collator=DataCollatorWithPadding(tok)).train()
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))


def score_pairs(model_dir: str | Path, text_a: list[str], text_b: list[str], max_len: int,
                batch_size: int) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).half().cuda().eval()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    order = np.argsort([len(a) + len(b) for a, b in zip(text_a, text_b)])
    out = np.empty(len(text_a), np.float32)
    with torch.no_grad():
        for s in range(0, len(order), batch_size):
            ids = order[s:s + batch_size]
            enc = tok([text_a[i] for i in ids], [text_b[i] for i in ids], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt").to("cuda")
            logits = model(**enc).logits.float()
            out[ids] = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    return out
```

- [ ] **Step 4: Add the stages.** Append to `src/ber/stages/nn.py` (imports: `from ber.models.crossencoder import band_mask, pair_text, sample_training_pairs, score_pairs, train_cross_encoder`):

```python
def _texts(cfg: dict, split: str, pairs: pl.DataFrame) -> tuple[list[str], list[str]]:
    lt = pair_text(load(cfg, split, "s1.parquet"))
    rt = pair_text(load(cfg, split, "right.parquet"))
    return lt.gather(pairs["l_idx"]).to_list(), rt.gather(pairs["r_idx"]).to_list()


def stage_cross_train(cfg: dict, split: str) -> None:
    cc = cfg["cross"]
    pairs = sample_training_pairs(load(cfg, split, "fit_pred.parquet"), cc["band_lo"], cc["band_hi"],
                                  cc["n_train_pairs"], cfg["seed"])
    ta, tb = _texts(cfg, split, pairs)
    out = Path(cfg["paths"]["work_dir"]) / split / "crossencoder"
    train_cross_encoder(ta, tb, pairs["y"].to_list(), cc["model_name"], out, cc["max_len"], cc["epochs"],
                        cc["batch_size"], cc["lr"], cfg["seed"])


def stage_cross_score(cfg: dict, split: str) -> None:
    cc = cfg["cross"]
    pred = load(cfg, split, "pred.parquet" if split == "test" else "valid_pred.parquet")
    band = pred.filter(pl.Series(band_mask(pred["p"].to_numpy(), cc["band_lo"], cc["band_hi"])))
    print(f"[cross_score] {split}: {band.height:,} of {pred.height:,} pairs in band")
    ta, tb = _texts(cfg, split, band)
    scores = score_pairs(read_path(cfg, MODEL_SPLIT[split], "crossencoder"), ta, tb, cc["max_len"],
                         cc["score_batch_size"])
    save(band.select("l_idx", "r_idx").with_columns(cross_p=pl.Series(scores, dtype=pl.Float32)),
         cfg, split, "cross_scores.parquet")
```

Register `"cross_train"` and `"cross_score"`.

- [ ] **Step 5: Run the tests, then commit.**

Run: `.venv/Scripts/python -m pytest tests/test_crossencoder.py -v`
Expected: 2 passed

```bash
git add src/ber/models/crossencoder.py src/ber/stages/nn.py src/ber/pipeline.py tests/test_crossencoder.py
git commit -m "feat: cross-encoder reranker module and stages"
```

---

### Task 23: Combiner LightGBM and the M3 Kaggle run

**Files:**
- Modify: `src/ber/models/gbm.py`, `src/ber/stages/nn.py`, `src/ber/pipeline.py`, `configs/base.yaml`, `notebooks/kaggle_driver.py`, `memory.md`, `docs/Documentation.md`
- Test: `tests/test_gbm.py` (append)

**Interfaces:**
- Consumes: `features/`, `valid_pred.parquet`, `pred.parquet`, `cross_scores.parquet`
- Produces:
  - `train_combiner(X, y, groups, train_mask, params, num_boost_round, n_folds) -> (oof: np.ndarray, boosters: list[lgb.Booster])`. Rows in `train_mask` get out-of-fold predictions; other rows get the mean over fold models.
  - stages `combine_train` (→ `valid_pred_combined.parquet` [l_idx, r_idx, y, fold, p], `combiner_{k}.txt`) and `combine_predict` (test → `pred_combined.parquet`)

- [ ] **Step 1: Write the failing test.** Append to `tests/test_gbm.py`:

```python
from ber.models.gbm import train_combiner


def test_train_combiner_oof_and_outside_rows():
    rng = np.random.default_rng(1)
    n = 3000
    X = pl.DataFrame({"a": rng.random(n), "cross_p": np.where(rng.random(n) < 0.5, np.nan, rng.random(n))})
    y = (X["a"].to_numpy() > 0.5).astype(int)
    groups = rng.integers(0, 600, n)
    train_mask = np.arange(n) < 2500
    oof, boosters = train_combiner(X, y, groups, train_mask, {"min_data_in_leaf": 5, "num_leaves": 7}, 100, 5)
    assert len(boosters) == 5 and oof.shape == (n,)
    assert ((oof[train_mask] > 0.5) == y[train_mask]).mean() > 0.95
    assert ((oof[~train_mask] > 0.5) == y[~train_mask]).mean() > 0.95
```

Run: `.venv/Scripts/python -m pytest tests/test_gbm.py -v`
Expected: FAIL with `ImportError: cannot import name 'train_combiner'`

- [ ] **Step 2: Implement `train_combiner`.** Append to `src/ber/models/gbm.py`:

```python
def train_combiner(X: pl.DataFrame, y: np.ndarray, groups: np.ndarray, train_mask: np.ndarray, params: dict | None,
                   num_boost_round: int, n_folds: int) -> tuple[np.ndarray, list[lgb.Booster]]:
    """GroupKFold (by S1) out-of-fold predictions on train_mask rows; fold-mean on the rest.
    Fixed num_boost_round (no early stopping) so OOF predictions stay unbiased."""
    from sklearn.model_selection import GroupKFold

    m = to_matrix(X)
    oof = np.zeros(X.height, np.float32)
    idx_tr, idx_out = np.flatnonzero(train_mask), np.flatnonzero(~train_mask)
    boosters = []
    for a, b in GroupKFold(n_splits=n_folds).split(idx_tr, groups=groups[idx_tr]):
        tr, va = idx_tr[a], idx_tr[b]
        bst = lgb.train({**DEFAULT_PARAMS, **(params or {})},
                        lgb.Dataset(m[tr], label=y[tr], feature_name=list(X.columns)), num_boost_round=num_boost_round)
        oof[va] = bst.predict(m[va])
        boosters.append(bst)
    if len(idx_out):
        oof[idx_out] = np.mean([bst.predict(m[idx_out]) for bst in boosters], axis=0)
    return oof, boosters
```

In `configs/base.yaml`, replace the existing `combiner:` block with:

```yaml
combiner:
  n_folds: 5
  num_boost_round: 600
  predict_chunk_rows: 5000000
  params: {learning_rate: 0.05, num_leaves: 63, min_data_in_leaf: 200}
```

- [ ] **Step 3: Add the stages.** Append to `src/ber/stages/nn.py` (imports: `from ber.features import FEATURE_COLUMNS`, `from ber.config import write_path`, `from ber.models.gbm import load_booster, predict, train_combiner`, `from ber.stages.common import es_fold`, `from ber.stages.model import load_features`):

```python
COMBINER_EXTRA = ["p_stage1", "cross_p"]


def stage_combine_train(cfg: dict, split: str) -> None:
    feats = load_features(cfg, split).collect().drop("y")
    df = (load(cfg, split, "valid_pred.parquet").rename({"p": "p_stage1"})
          .join(feats, on=["l_idx", "r_idx"])
          .join(load(cfg, split, "cross_scores.parquet"), on=["l_idx", "r_idx"], how="left"))
    cols = [c for c in FEATURE_COLUMNS if c in df.columns] + COMBINER_EXTRA
    train_mask = (df["fold"] == es_fold(cfg)).to_numpy()  # random: all "valid"; holdout: "valid_seen" only
    cb = cfg["combiner"]
    oof, boosters = train_combiner(df.select(cols), df["y"].to_numpy(), df["l_idx"].to_numpy(), train_mask,
                                   cb["params"], cb["num_boost_round"], cb["n_folds"])
    for k, bst in enumerate(boosters):
        bst.save_model(str(write_path(cfg, split, f"combiner_{k}.txt")))
    save(df.select("l_idx", "r_idx", "y", "fold").with_columns(p=pl.Series(oof)), cfg, split,
         "valid_pred_combined.parquet")


def stage_combine_predict(cfg: dict, split: str) -> None:
    cb = cfg["combiner"]
    boosters = [load_booster(read_path(cfg, MODEL_SPLIT[split], f"combiner_{k}.txt")) for k in range(cb["n_folds"])]
    stage1 = load(cfg, split, "pred.parquet").rename({"p": "p_stage1"})
    cross = load(cfg, split, "cross_scores.parquet")
    lf = load_features(cfg, split)
    n = lf.select(pl.len()).collect().item()
    outs = []
    for s in range(0, n, cb["predict_chunk_rows"]):
        part = (lf.slice(s, cb["predict_chunk_rows"]).collect()
                .join(stage1, on=["l_idx", "r_idx"]).join(cross, on=["l_idx", "r_idx"], how="left"))
        p = np.mean([predict(bst, part) for bst in boosters], axis=0)
        outs.append(part.select("l_idx", "r_idx").with_columns(p=pl.Series(p)))
    save(pl.concat(outs), cfg, split, "pred_combined.parquet")
```

Register `"combine_train"` and `"combine_predict"`.

- [ ] **Step 4: Run the unit tests, then commit and push.**

Run: `.venv/Scripts/python -m pytest -v`
Expected: all pass

```bash
git add src/ber/models/gbm.py src/ber/stages/nn.py src/ber/pipeline.py configs/base.yaml tests/test_gbm.py
git commit -m "feat: out-of-fold combiner over stage-1 features + cross-encoder score"
git push
```

- [ ] **Step 5: Add the M3 cells to `notebooks/kaggle_driver.py`.**

```python
# %% M3 GPU smoke test (dev) — needs the dev pipeline through `train` first
for st in ("features", "train"):
    stage(st, "dev")
stage("cross_train", "dev", "cross.n_train_pairs=5000")
stage("cross_score", "dev")

# %% M3 GPU session (PREV = latest train/test outputs): cross-encoder train + score
stage("cross_train", "train")
stage("cross_score", "train")
stage("cross_score", "test")

# %% M3 CPU session: combiner, evaluation on combined predictions, submission
COMB = ["evaluate.pred_file=valid_pred_combined.parquet", "submit.pred_file=pred_combined.parquet"]
stage("combine_train", "train")
stage("evaluate", "train", "run_name=kaggle_m3", *COMB)
stage("combine_predict", "test")
stage("submit", "test", *COMB)
```

- [ ] **Step 6: Run M3 on Kaggle for both schemes.**
  1. Run the smoke test.
  2. Run the GPU session `ber-m3-a` (T4 x2, about 2–3 hours), then the CPU session `ber-m3-b`.
  3. Repeat for `kaggle_holdout.yaml`. There, `cross_train` learns only from US `fit` pairs, and the combiner trains on `valid_seen`.
  4. **Keep M3 only if `f05` improves over the best previous stage in both schemes.**
     - If kept: set `evaluate.pred_file`/`submit.pred_file` to the combined files in `configs/kaggle.yaml`, upload to the leaderboard, and record the score.
     - Also write down the band share: the target is 10–15% of pairs. If it is off, adjust `cross.band_lo`/`band_hi`.
  5. Update `memory.md` and `docs/Documentation.md` section 4.

```bash
git add notebooks/kaggle_driver.py configs memory.md docs/Documentation.md
git commit -m "chore: M3 cross-encoder + combiner results"
git push
```

---

## Milestone M4: Error analysis, tuning, France check, packaging

### Task 24: Error analysis and France monitoring script, then tuning sweeps

**Files:**
- Create: `scripts/error_analysis.py`
- Modify: `configs/kaggle.yaml`, `memory.md`, `docs/Documentation.md` (section 5)

**Interfaces:**
- Consumes: the `calibrator.joblib`, `decision.json` and `valid_pred*.parquet` of the chosen configuration, the norm frames, and the latest `output/matching_results.tsv`
- Produces: `reports/error_analysis_<run>.md` (committed), plus the final tuned config

- [ ] **Step 1: Write `scripts/error_analysis.py`.**

```python
"""Validation error analysis + test-time France monitoring.
Usage: python scripts/error_analysis.py --config configs/kaggle.yaml [--set evaluate.pred_file=valid_pred_combined.parquet]
Writes reports/error_analysis_<run_name>.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import polars as pl

from ber.config import apply_override, load_config, read_path
from ber.decide import calibrate, one_owner, select
from ber.stages.common import load


def block(df: pl.DataFrame) -> str:
    with pl.Config(tbl_rows=60, fmt_str_lengths=60, tbl_width_chars=250):
        return f"```\n{df}\n```"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config)
    for expr in args.set:
        apply_override(cfg, expr)
    split = "train"
    iso = joblib.load(read_path(cfg, split, "calibrator.joblib"))
    decision = json.loads(read_path(cfg, split, "decision.json").read_text())
    pred = load(cfg, split, cfg["evaluate"]["pred_file"]).filter(pl.col("fold") == "valid")
    pred = pred.with_columns(p=pl.Series(calibrate(iso, pred["p"].to_numpy())))
    if cfg["decide"]["one_owner"]:
        pred = one_owner(pred)
    sel = select(pred, decision)
    gt = load(cfg, split, "gt.parquet")
    cands = load(cfg, split, "cands_final.parquet").select("l_idx", "r_idx").with_columns(in_cands=pl.lit(True))
    s1 = load(cfg, split, "s1_norm.parquet").select(
        pl.col("idx").alias("l_idx"), "country", pl.col("name_script").alias("l_script"),
        pl.col("business_name").alias("l_name"), pl.col("business_address").alias("l_addr"))
    right = load(cfg, split, "right_norm.parquet").select(
        pl.col("idx").alias("r_idx"), "source", pl.col("business_name").alias("r_name"),
        pl.col("business_address").alias("r_addr"))
    valid_l = load(cfg, split, "folds.parquet").filter(pl.col("fold") == "valid").select("l_idx")
    fp = sel.join(gt, on=["l_idx", "r_idx"], how="anti").join(s1, on="l_idx").join(right, on="r_idx")
    fn = (gt.join(valid_l, on="l_idx", how="semi").join(sel, on=["l_idx", "r_idx"], how="anti")
          .join(cands, on=["l_idx", "r_idx"], how="left")
          .with_columns(reason=pl.when(pl.col("in_cands")).then(pl.lit("scored_below"))
                        .otherwise(pl.lit("not_in_candidates")))
          .join(s1, on="l_idx").join(right, on="r_idx"))
    cols = ["country", "source", "l_name", "r_name", "l_addr", "r_addr"]
    lines = [f"# Error analysis: {cfg['run_name']} ({cfg['evaluate']['pred_file']})", "",
             f"decision: `{decision}`", "", f"- false positives: {fp.height:,}", f"- false negatives: {fn.height:,}",
             "", "## False positives by country / script / source",
             block(fp.group_by("country", "l_script", "source").len().sort("len", descending=True)),
             "## False negatives by reason / country",
             block(fn.group_by("reason", "country").len().sort("len", descending=True)),
             "## 40 sample false positives", block(fp.sample(min(40, fp.height), seed=0).select(cols)),
             "## 40 sample false negatives (scored below)",
             block(fn.filter(pl.col("reason") == "scored_below").pipe(lambda d: d.sample(min(40, d.height), seed=0)).select(cols)),
             "## 40 sample false negatives (blocking misses)",
             block(fn.filter(pl.col("reason") == "not_in_candidates").pipe(lambda d: d.sample(min(40, d.height), seed=0)).select(cols))]
    out_tsv = Path(cfg["paths"]["output_dir"]) / "matching_results.tsv"
    if out_tsv.exists():
        m = pl.read_csv(out_tsv, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
        s1t = load(cfg, "test", "s1.parquet").select(pl.col("entity_id").alias("source1_entity_id"), "country")
        stats = (m.join(s1t, on="source1_entity_id")
                 .with_columns(n=pl.when(pl.col("matched_entity_ids").fill_null("") == "").then(0)
                               .otherwise(pl.col("matched_entity_ids").str.count_matches(",") + 1))
                 .group_by("country").agg(n_s1=pl.len(), mean_matches=pl.col("n").mean(),
                                          empty_share=(pl.col("n") == 0).mean()))
        lines += ["## Test predictions by country (France is unseen in training)", block(stats),
                  "Reference from train ground truth: about 3.5 matches per S1, 5.6% singletons."]
    Path("reports").mkdir(exist_ok=True)
    path = Path("reports") / f"error_analysis_{cfg['run_name']}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it on Kaggle for the best configuration.** Add a driver cell:

```python
subprocess.run([sys.executable, "scripts/error_analysis.py", "--config", CONFIG, *sum([["--set", o] for o in COMB], [])], check=True)
```

Use `COMB` only if M3 was kept. Download `reports/error_analysis_*.md`, commit it, and write the top 3 error patterns into `memory.md`.

- [ ] **Step 3: Run the tuning sweeps.** Pick the rows that match the error patterns.
  - **How to run them:** each sweep is a "Save & Run All" version with `PREV` set to the latest outputs. Re-run only the stages downstream of the changed knob. Every run logs to `experiments.csv`, under both schemes when blocking changes.
  - **Keep rule:** keep a change only if `f05` improves on the s1_random scheme and does not drop by more than 0.002 on the holdout scheme.

| Dominant pattern | Knob (`--set`) | Stages to re-run |
|---|---|---|
| FN `not_in_candidates` above 3% of GT pairs | `blocking.tfidf.top_k_fwd=60`, then `cheap.keep_top=35` | block → … → evaluate |
| FN `scored_below` on multi-match S1s | `features.max_train_s1=1200000`, `gbm.params={num_leaves: 255}` | features (train) → train → evaluate |
| FP concentrated on same-chain names with different numbers | `gbm.params={min_data_in_leaf: 500}`; check that `house_eq`/`num_jaccard` rank in the top 15 importances (`[train] top features`) | train → evaluate |
| Singleton accuracy below 0.9 | compare `f05_expected` against `f05_threshold` in `experiments.csv`; the better rule is already auto-picked, so record the gap | none |

- [ ] **Step 4: France check.**
  1. Re-run `submit` with the final config, then the error-analysis script.
  2. France's `mean_matches` and `empty_share` should fall within the range spanned by US and India, ±25%.
  3. If France is outside that range, re-check the holdout `best_unseen_scale` decision (Task 19). Try one submission with `decide.unseen_country_scale` set to that value, and compare public LB scores.
  4. Record the result in `memory.md`.

- [ ] **Step 5: Final pick and upload.**
  1. Choose the configuration with the best s1_random `f05` that does not regress on the holdout scheme.
  2. Upload its `matching_results.tsv` and record the LB score.
  3. Fill `docs/Documentation.md` section 5 (F0.5, common FPs and FNs from the report).

```bash
git add scripts/error_analysis.py reports configs memory.md docs/Documentation.md notebooks/kaggle_driver.py
git commit -m "chore: M4 error analysis, tuning and France check"
git push
```

---

### Task 25: Final submission package, README, documentation

**Files:**
- Modify: `src/ber/submit.py` (add `build_package`), `src/ber/stages/output.py` (add `stage_package`), `src/ber/pipeline.py`, `README.md`, `docs/Documentation.md`
- Test: `tests/test_package.py`

**Interfaces:**
- Consumes: the final `output/` TSVs; the repo
- Produces:
  - `build_package(repo_root, output_dir, doc_path, team, dest) -> Path`: a zip with `output/{matching_results,candidate_pairs}.tsv`, `code/business_entity_resolution/{src/…, README.md, requirements.txt, …}` and `Documentation_template.md`
  - stage `package`

- [ ] **Step 1: Write the failing test.** Create `tests/test_package.py`:

```python
import zipfile

from ber.submit import build_package


def test_package_layout(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src" / "ber").mkdir(parents=True)
    (repo / "src" / "ber" / "__init__.py").write_text("")
    (repo / "src" / "ber" / "__pycache__").mkdir()
    (repo / "src" / "ber" / "__pycache__" / "x.pyc").write_bytes(b"0")
    for f in ("README.md", "requirements.txt"):
        (repo / f).write_text("x")
    out = tmp_path / "output"
    out.mkdir()
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        (out / f).write_text("h\n")
    doc = tmp_path / "Documentation.md"
    doc.write_text("# doc")
    z = zipfile.ZipFile(build_package(repo, out, doc, "teamx", tmp_path))
    names = set(z.namelist())
    assert {"output/matching_results.tsv", "output/candidate_pairs.tsv", "Documentation_template.md",
            "code/business_entity_resolution/src/ber/__init__.py",
            "code/business_entity_resolution/README.md",
            "code/business_entity_resolution/requirements.txt"} <= names
    assert not any("__pycache__" in n for n in names)
    assert z.filename.endswith("teamx_submission.zip")
```

Run: `.venv/Scripts/python -m pytest tests/test_package.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_package'`

- [ ] **Step 2: Implement.** Append to `src/ber/submit.py` (add `import zipfile` at the top):

```python
CODE_ITEMS = ["src", "configs", "notebooks", "scripts", "tests", "README.md", "requirements.txt",
              "requirements.in", "pyproject.toml"]


def build_package(repo_root: Path, output_dir: Path, doc_path: Path, team: str, dest: Path) -> Path:
    zip_path = Path(dest) / f"{team}_submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(Path(output_dir) / name, f"output/{name}")
        for item in CODE_ITEMS:
            path = Path(repo_root) / item
            if not path.exists():
                continue
            files = [path] if path.is_file() else sorted(
                f for f in path.rglob("*") if f.is_file() and "__pycache__" not in f.parts)
            for f in files:
                z.write(f, f"code/business_entity_resolution/{f.relative_to(repo_root).as_posix()}")
        z.write(doc_path, "Documentation_template.md")
    return zip_path
```

Append to `src/ber/stages/output.py` (import `build_package`):

```python
def stage_package(cfg: dict, split: str) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    out = Path(cfg["paths"]["output_dir"])
    path = build_package(repo_root, out, repo_root / "docs" / "Documentation.md", cfg["submit"]["team_name"], out.parent)
    print(f"[package] wrote {path}")
```

Register `"package": output.stage_package`.

Run: `.venv/Scripts/python -m pytest -v`
Expected: all pass

- [ ] **Step 3: Write the final `README.md`.** It must contain these exact sections, filled in:
  1. **Overview:** the task, the approach in 5 bullets, and the models used with their licenses (`intfloat/multilingual-e5-small` MIT, `xlm-roberta-base` MIT, LightGBM MIT).
  2. **Compliance statement:** "No external data, APIs or geocoding are used; only the provided train/test files."
  3. **Environment:** Python ≥ 3.10; `pip install -e .[nn,ann]` then `pip install -r requirements.txt` (pinned from Kaggle).
  4. **Data:** where to put the organizer dataset (`paths.raw_dir` or `paths.raw_zip`).
  5. **Reproduce end-to-end:** the ordered list of `python -m ber.pipeline --config configs/kaggle.yaml --stage <stage> --split <split>` commands, exactly as in `notebooks/kaggle_driver.py` for the final configuration (sessions A/B, M2 GPU, M3 GPU/CPU), with rough runtimes from `memory.md`.
  6. **Outputs:** `output/matching_results.tsv`, `output/candidate_pairs.tsv` (the exact set the final model scored), and the validator command.
  7. **Repository layout:** the file-structure table from `plan.md`.

- [ ] **Step 4: Finalize `docs/Documentation.md`.** Fill every template section:
  - team/date
  - executive summary
  - problem analysis (EDA numbers)
  - strategy (Hybrid: blocking + GBM + bi-encoder + cross-encoder)
  - blocking (keys, TF-IDF views, embedding k-NN, cheap cut, total candidate pairs on test, recall on validation)
  - matching model (full feature list grouped by name/address/context, models, calibration, one-owner, expected-F0.5 rule)
  - results (validation F0.5 for both schemes, public LB, FP/FN patterns)
  - conclusion
  - appendix A (code layout and entry points)

- [ ] **Step 5: Build and verify the final zip.** Run on Kaggle (driver cell `stage("package", "test")`) or locally after downloading the final `output/`. Then check it:

```bash
.venv/Scripts/python -c "import zipfile,sys; z=zipfile.ZipFile(sys.argv[1]); print('\n'.join(n for n in z.namelist() if n.count('/')<=3))" <path-to>/<team>_submission.zip
.venv/Scripts/python -m ber.vendor.validate_submission --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir <test dir>
```

Expected: the top-level `output/`, `code/business_entity_resolution/` and `Documentation_template.md` are present, and the validator prints `PASS`. The user submits the zip. Record the submission date in `memory.md`.

- [ ] **Step 6: Commit and push.**

```bash
git add src/ber/submit.py src/ber/stages/output.py src/ber/pipeline.py tests/test_package.py README.md docs/Documentation.md memory.md
git commit -m "feat: final submission package; README and methodology documentation"
git push
```

---

## Self-review (spec → tasks)

| Spec section | Task(s) |
|---|---|
| §1 constraints (license, no external data, validator, zip layout) | Global Constraints; 1 (vendored validator); 16–17 (submit gate); 25 (zip) |
| §2 Kaggle, staged hybrid, keep-only-if-both-schemes | 18, 19, 21 (step 8), 23 (step 6), 24 (step 3) |
| §3 architecture, parquet stages, resume across sessions | 1 (`read_path`/`prev_work_dirs`), 2 (CLI), every stage task |
| §4 validation: 80/20 by S1, country holdout, dev slice, experiment log | 3, 5, 17 (evaluate), 19 |
| §5 normalization, mined lexicon, French forms, keys, bidirectional TF-IDF, embed k-NN, cheap cut, recall targets | 6–9, 10, 11, 12, 13, 21 |
| §6 features (name, address, context, cluster support, source, embed, cross) | 13, 14, 21 (`embed_attach`), 23 (`cross_p`) |
| §7 models (LightGBM, e5-small bi-encoder, xlm-r cross-encoder on the band, combiner) | 13, 17, 20–23 |
| §8 isotonic calibration, one owner, expected-F0.5 vs threshold, France scale + monitoring | 15, 17, 19, 24 |
| §9 checks (all S1s present, S2/S3 only, no dupes, matches ⊆ candidates) | 16 (writer + official validator), 17 (`submit` raises on failure) |
| §10 unit tests + integration test | tests in every code task; 17 (synthetic end-to-end) |
| §11 milestones M0–M4 | M0: 1–5; M1: 6–19; M2: 20–21; M3: 22–23; M4: 24–25 |
| §12 out of scope | nothing plans LLM judges, external data or big ensembles |

Type and name consistency was checked across tasks:
- `load/save/load_norm/sample_l/eval_folds/es_fold/label_pairs`
- `CHEAP_COLUMNS/FEATURE_COLUMNS`
- `train_binary/predict/load_booster/train_combiner`
- `choose_expected_f05/choose_threshold/select/one_owner/calibrate/scale_unseen`
- `record_text` (bi-encoder, with "query: ") vs `pair_text` (cross-encoder)
- stage names in `pipeline.STAGES`: `ingest, dev_slice, split, lexicon, normalize, block, cheap_train, cheap_apply, features, train, evaluate, predict, submit, embed_train, embed_encode, embed_attach, cross_train, cross_score, combine_train, combine_predict, package`
