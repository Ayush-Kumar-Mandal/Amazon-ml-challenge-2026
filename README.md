# ber — Business Entity Resolution (Amazon ML Challenge 2026)

## 1. Overview

**Task:** for every S1 (source-1) business record, list every S2/S3 record that refers to the
same real-world business. The match list may be empty. The scoring metric is macro F0.5 per
S1 (`1.25 * TP / (0.25 * |T| + |P|)`; a singleton scores 1.0 only when the prediction is also
empty). Test data adds a third country (France) that never appears in training.

**Approach**, a staged hybrid pipeline (later stages are optional — see §5):

- **Normalize** every record: script detection, transliteration, legal-form/abbreviation
  expansion (using a lexicon mined from the training ground truth), and structured name/address
  parsing.
- **Block** candidates per S1 with a union of exact-key blocking and bidirectional char-trigram
  TF-IDF nearest-neighbour search within each country, then cut to the top-k per S1 with a cheap
  LightGBM scorer.
- **Score** the surviving candidates with a full-feature LightGBM model (name, address, context
  and cluster-support features), calibrate with isotonic regression, and decide matches with a
  one-owner rule plus a per-S1 expected-F0.5 subset rule (compared against a tuned global
  threshold).
- **(Optional, M2)** re-block with a fine-tuned multilingual bi-encoder's nearest neighbours and
  add its cosine similarity as a feature.
- **(Optional, M3)** rerank uncertain-probability pairs with a fine-tuned cross-encoder and
  combine its score with the stage-1 features in an out-of-fold LightGBM combiner.

**Models used, all MIT-licensed and under the 8B-parameter cap:**

| Model | License | Role |
|---|---|---|
| LightGBM | MIT | cheap-cut scorer, full-feature matcher, out-of-fold combiner |
| `intfloat/multilingual-e5-small` | MIT | bi-encoder for embedding-based blocking (M2, optional) |
| `xlm-roberta-base` | MIT | cross-encoder reranker on uncertain pairs (M3, optional) |

## 2. Compliance statement

No external data, APIs or geocoding are used; only the provided train/test files.

## 3. Environment

- Python >= 3.10
- Install the package and its optional extras (bi-encoder/cross-encoder + FAISS blocking), then
  the exact pinned versions used on Kaggle:

```bash
pip install -e .[nn,ann]
pip install -r requirements.txt
```

`requirements.txt` is frozen from the Kaggle session with `pip freeze` (see
`docs/KAGGLE_RUNBOOK.md`); it is not committed until that run has happened.

## 4. Data

Put the organizer dataset where `configs/*.yaml` → `paths.raw_dir` points, or set
`paths.raw_zip` to the path of the organizer's zip file (`ber.io.open_raw` reads a raw directory
or a zip transparently). Locally this is `configs/dev.yaml` → `paths.raw_zip`, the organizer zip
in the repo root (`6ab10eb3b23ba_student_resource.zip`, gitignored); `C:/Users/AYUSH/ber_data/`
holds only the *derived* `work_dir`/`scratch_dir`/`output_dir`, never the raw dataset itself. On
Kaggle the raw dataset is the private dataset `ber-raw`, mounted at
`/kaggle/input/ber-raw/student_resource/dataset` (see `configs/kaggle.yaml`).

## 5. Reproduce end-to-end

Every stage is `python -m ber.pipeline --config <config> --stage <stage> --split <split> [--set k=v]`.
Stages read the previous stage's parquet files from `paths.work_dir`, falling back to
`paths.prev_work_dirs` — this is how a new Kaggle session resumes a prior one's output.

**Runtimes below are estimated from the local dev-slice timings, extrapolated to full scale
(`memory.md` → "Kaggle time extrapolation"). They are marked ESTIMATED and must be replaced
with the real Kaggle run's numbers once available.**

M2 and M3 are **optional pipeline stages**: per the project's stage rule, each is kept only if
it improves macro F0.5 on **both** validation schemes (`s1_random` and `country_holdout`,
i.e. `configs/kaggle.yaml` and `configs/kaggle_holdout.yaml`) over the previous best. If a
stage is not kept, stop after the session before it and run `evaluate`/`predict`/`submit` on
the M1 (or M2) prediction file instead.

### Session A — CPU (ESTIMATED ~2.7h)

```bash
python -m ber.pipeline --config configs/kaggle.yaml --stage ingest    --split train
python -m ber.pipeline --config configs/kaggle.yaml --stage ingest    --split test
python -m ber.pipeline --config configs/kaggle.yaml --stage split     --split train
python -m ber.pipeline --config configs/kaggle.yaml --stage lexicon   --split train
python -m ber.pipeline --config configs/kaggle.yaml --stage normalize --split train
python -m ber.pipeline --config configs/kaggle.yaml --stage normalize --split test
python -m ber.pipeline --config configs/kaggle.yaml --stage block     --split train
python -m ber.pipeline --config configs/kaggle.yaml --stage block     --split test
python -m ber.pipeline --config configs/kaggle.yaml --stage cheap_train --split train
python -m ber.pipeline --config configs/kaggle.yaml --stage cheap_apply --split train
python -m ber.pipeline --config configs/kaggle.yaml --stage cheap_apply --split test
```

### Session B — CPU, resumes Session A's `work_dir` (ESTIMATED ~3.1h)

```bash
PREV='--set paths.prev_work_dirs=[<session-A-work-dir>]'
python -m ber.pipeline --config configs/kaggle.yaml --stage features --split train $PREV
python -m ber.pipeline --config configs/kaggle.yaml --stage features --split test  $PREV
python -m ber.pipeline --config configs/kaggle.yaml --stage train    --split train $PREV
python -m ber.pipeline --config configs/kaggle.yaml --stage evaluate --split train $PREV
python -m ber.pipeline --config configs/kaggle.yaml --stage predict  --split test  $PREV
python -m ber.pipeline --config configs/kaggle.yaml --stage submit   --split test  $PREV
```

This completes M1 (the required baseline). `evaluate` prints the local F0.5; `submit` writes
`output/matching_results.tsv` / `output/candidate_pairs.tsv` and runs the validator.

### M2 — bi-encoder blocking (OPTIONAL; keep only if it clears the stage rule)

GPU session (`GPU T4 x2`), resumes Sessions A+B's `work_dir`s — **no local runtime baseline**
(needs a GPU, unavailable locally; time it on the real Kaggle run):

```bash
EMB='--set blocking.embed.enabled=true'
python -m ber.pipeline --config configs/kaggle.yaml --stage embed_train  --split train $PREV
python -m ber.pipeline --config configs/kaggle.yaml --stage embed_encode --split train $PREV $EMB
python -m ber.pipeline --config configs/kaggle.yaml --stage block        --split train $PREV $EMB
python -m ber.pipeline --config configs/kaggle.yaml --stage embed_attach --split train $PREV
python -m ber.pipeline --config configs/kaggle.yaml --stage embed_encode --split test  $PREV $EMB
python -m ber.pipeline --config configs/kaggle.yaml --stage block        --split test  $PREV $EMB
python -m ber.pipeline --config configs/kaggle.yaml --stage embed_attach --split test  $PREV
```

CPU session, re-runs the cheap-cut through submit with `run_name=kaggle_m2` — **no local
runtime baseline** (row counts are unchanged from M1, so it should be close to the M1
`cheap_train`/`cheap_apply`/`features`/`train`/`evaluate` total, ~2h, but this is not measured):

```bash
python -m ber.pipeline --config configs/kaggle.yaml --stage cheap_train --split train $PREV
python -m ber.pipeline --config configs/kaggle.yaml --stage cheap_apply --split train $PREV
python -m ber.pipeline --config configs/kaggle.yaml --stage cheap_apply --split test  $PREV
python -m ber.pipeline --config configs/kaggle.yaml --stage features    --split train $PREV
python -m ber.pipeline --config configs/kaggle.yaml --stage features    --split test  $PREV
python -m ber.pipeline --config configs/kaggle.yaml --stage train       --split train $PREV --set run_name=kaggle_m2
python -m ber.pipeline --config configs/kaggle.yaml --stage evaluate    --split train $PREV --set run_name=kaggle_m2
python -m ber.pipeline --config configs/kaggle.yaml --stage predict     --split test  $PREV
python -m ber.pipeline --config configs/kaggle.yaml --stage submit      --split test  $PREV
```

### M3 — cross-encoder + combiner (OPTIONAL; keep only if it clears the stage rule)

GPU session (`GPU T4 x2`, ESTIMATED ~2-3h per the plan's own estimate for the cross-encoder
fine-tune — not re-derived from the dev slice since there is no local GPU):

```bash
python -m ber.pipeline --config configs/kaggle.yaml --stage cross_train --split train $PREV
python -m ber.pipeline --config configs/kaggle.yaml --stage cross_score --split train $PREV
python -m ber.pipeline --config configs/kaggle.yaml --stage cross_score --split test  $PREV
```

CPU session — no local runtime baseline:

```bash
COMB="--set evaluate.pred_file=valid_pred_combined.parquet --set submit.pred_file=pred_combined.parquet"
python -m ber.pipeline --config configs/kaggle.yaml --stage combine_train   --split train $PREV
python -m ber.pipeline --config configs/kaggle.yaml --stage evaluate        --split train $PREV --set run_name=kaggle_m3 $COMB
python -m ber.pipeline --config configs/kaggle.yaml --stage combine_predict --split test  $PREV
python -m ber.pipeline --config configs/kaggle.yaml --stage submit          --split test  $PREV $COMB
```

### Final packaging

```bash
python -m ber.pipeline --config configs/kaggle.yaml --stage package --split test $PREV
```

writes `<team_name>_submission.zip` next to `output/` (see §6).

Full step-by-step Kaggle instructions (dataset upload, sessions, keep/drop decisions, error
analysis, France checks) are in `docs/KAGGLE_RUNBOOK.md`.

## 6. Outputs

- `output/matching_results.tsv` — header `source1_entity_id\tmatched_entity_ids`, one row per
  test S1, comma-joined S2/S3 IDs (empty string for no match). **This is the only scored file.**
- `output/candidate_pairs.tsv` — header `source1_entity_id\tcandidate_entity_ids`, the exact
  candidate set the final model scored (matches are always a subset of it).

Both are produced by whichever `submit` invocation matches the kept configuration (M1, M2 or
M3 — see §5). Validate before uploading:

```bash
python -m ber.vendor.validate_submission --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir <test dir>
```

Expected: `PASS` printed, exit code 0.

## 7. Repository layout

| Path | Responsibility |
|---|---|
| `pyproject.toml`, `requirements.in`, `requirements.txt` | package `ber` (src layout); unpinned inputs; pinned freeze from Kaggle |
| `.gitignore` | keep data, zip, PDF, venv and work dirs out of git |
| `configs/base.yaml` | every tunable, with defaults |
| `configs/dev.yaml` | local paths and small sample sizes |
| `configs/kaggle.yaml` | Kaggle paths |
| `configs/kaggle_holdout.yaml` | the US->India scheme |
| `src/ber/config.py` | YAML loading (`extends`, env vars), `--set` overrides, `read_path`/`write_path`, split-mapping tables |
| `src/ber/io.py` | raw TSV (zip or directory) -> `s1.parquet`, `right.parquet`, `gt.parquet` |
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
| `docs/KAGGLE_RUNBOOK.md` | step-by-step checklist for every deferred, human-only Kaggle/portal step |
| `plan.md`, `memory.md` | this plan; the living project logbook |

**Artifacts** in `work_dir/<split>/` (split in `train`, `test`, `dev`):
- data: `s1.parquet`, `right.parquet`, `gt.parquet`, `folds.parquet`, `lexicon.json`, `s1_norm.parquet`, `right_norm.parquet`
- candidates: `cands_keys.parquet`, `cands_embed.parquet`, `cands_all.parquet`, `cands_final.parquet`
- features and predictions: `features/part-*.parquet`, `valid_pred.parquet`, `fit_pred.parquet`, `pred.parquet`
- models and decisions: `cheap_model.txt`, `gbm_model.txt`, `calibrator.joblib`, `decision.json`
