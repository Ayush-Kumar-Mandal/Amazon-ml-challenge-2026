# Project Memory: Business Entity Resolution (Amazon ML Challenge 2026)

> This is the project's living logbook. **Update it at the end of every task in `plan.md`:** Status, Results, Decisions and Pitfalls. Put the newest entries first within each section. Keep facts short and dated (YYYY-MM-DD).

## Quick facts
- **Task:** for every S1 record, list all S2/S3 records that are the same real business. The match list may be empty.
- **Metric:** macro F0.5 per S1, computed as 1.25·TP / (0.25·|T| + |P|). A singleton scores 1.0 only if the prediction is empty.

| Split | S1 | S2 | S3 | Countries |
|---|---|---|---|---|
| train | 2,206,821 | 5,034,616 | 5,285,603 | US 60%, India 40% |
| test | 1,732,544 | 4,887,273 | 5,082,316 | India 47%, US 38%, **France 15% (test only)** |

- **Ground truth:** 7,638,365 pairs, about 3.5 per S1. **5.6% of S1 are singletons.** Match counts go from 0 to 10+.
- **Output files:**
  - `matching_results.tsv` (the only scored file)
  - `candidate_pairs.tsv` (the exact set the final model scored; matches must be a subset of it)
  - both are TSV, one row per test S1, comma-joined S2/S3 IDs
- **Rules:**
  - The model must be MIT or Apache-2.0 licensed with at most 8B parameters.
  - **No external data, APIs or geocoding** (disqualification).
  - Final ranking uses the **private** leaderboard.
- **Final zip:** `output/` (both TSVs), `code/business_entity_resolution/` (src/, README, requirements.txt), and `Documentation_template.md` filled in.

## Environment
- **Local:**
  - Windows 11, 15.3 GB RAM, 8 cores, no CUDA GPU, about 9.4 GB free on C: (checked 2026-09-25)
  - Python 3.14 is the default; the project venv uses **3.10** (`py -3.10 -m venv .venv`, confirmed Python 3.10.11)
  - 2026-09-25 (T1): `pip install -e ".[dev,ann]"` succeeded with no errors, including `sparse_dot_topn` (1.2.0) and `faiss-cpu` (1.15.1), both of which have prebuilt Windows/cp310 wheels — no install workarounds needed.
- **Local data:** `C:/Users/AYUSH/ber_data/`, outside OneDrive. Never put data in the synced project folder.
- **Kaggle:** about 29 GB RAM, 4 CPU cores, T4×2 or P100, 12-hour sessions, 30 GPU-hours per week. Raw data is the private dataset `ber-raw`.
- **Code sync:** private GitHub repo → Kaggle, using `git clone` with the Kaggle Secret `GITHUB_TOKEN`.
  - Repo URL: https://github.com/Ayush-Kumar-Mandal/Amazon-ml-challenge-2026. It is **PUBLIC**: the user chose on 2026-09-25 to keep it public, accepting that other teams can see the code. Because it's public, Kaggle can clone it without a token.
  - Git flow: `main` holds the plan, spec and logbook. Implementation happens on `impl/m0-m1` and is merged after the final review.
- **Kaggle paths:** _(confirm in Task 18: `raw_dir`, `df -h` for /kaggle/working and /kaggle/tmp)_

## Key decisions (and why)
- 2026-09-25: **Approach:** a staged hybrid.
  - M1 is classic: blocking, then LightGBM, then the decision rule.
  - M2 adds a multilingual-e5-small bi-encoder.
  - M3 adds an xlm-roberta-base cross-encoder and a combiner.
  - A stage is kept only if F0.5 improves on **both** the s1_random and country_holdout (US→India) schemes. Why: this aims for the top of the leaderboard while protecting France, which has no labels.
- 2026-09-25: **Validation:** an 80/20 split by S1, with blocking over all train S2/S3 records. The US→India holdout stands in for France.
- 2026-09-25: **Decision rule:** isotonic calibration → one owner per S2/S3 → expected-F0.5 subset per S1, compared automatically against a tuned global threshold.
- 2026-09-25: **Compute:** Kaggle; the code sync method is a private GitHub repo (the user's choice). `memory.md` is this project logbook (the user's choice).

## Status
- Current milestone: **M0, T2 done.** Plan: `plan.md`. Spec: `docs/superpowers/specs/2026-09-25-business-entity-resolution-design.md`.
- M0:
  - [x] T1 scaffold/config/validator/GitHub
  - [x] T2 ingest + CLI
  - [x] T3 metrics
  - [ ] T4 EDA
  - [ ] T5 folds + dev slice
- M1:
  - [ ] T6 name text
  - [ ] T7 address
  - [ ] T8 lexicon
  - [ ] T9 normalize stage
  - [ ] T10 keys
  - [ ] T11 TF-IDF
  - [ ] T12 merge/block
  - [ ] T13 cheap cut
  - [ ] T14 full features
  - [ ] T15 decide
  - [ ] T16 submit writer
  - [ ] T17 model stages + e2e test
  - [ ] T18 Kaggle run + first LB
  - [ ] T19 holdout baseline
- M2:
  - [ ] T20 bi-encoder
  - [ ] T21 embedding k-NN + M2 run
- M3:
  - [ ] T22 cross-encoder
  - [ ] T23 combiner + M3 run
- M4:
  - [ ] T24 error analysis + tuning + France
  - [ ] T25 package + README + docs

## Results
| Date | Run | Scheme | recall_all | recall_final | cands/S1 | F0.5 (valid) | F0.5 India (holdout) | Public LB | Notes |
|---|---|---|---|---|---|---|---|---|---|

## EDA findings (Task 4)
- Cross-country match share: _(fill in)_ → `blocking.same_country` = _(fill in)_
- Right records with more than one S1 owner: _(fill in)_ → `decide.one_owner` = _(fill in)_
- Singleton rate by country; non-ASCII rates; 5- and 6-digit number rates: _(fill in)_
- Noise examples seen in the sample rows:
  - Hindi-script names
  - legal suffix at the front ("Pvt. EFS Print Ventures Ltd.", "LLC Moncada …")
  - typos ("Tetlecommunication")
  - website as the name ("wilfordhancock.com")
  - junk prefixes ("-- Holloway …")
  - reordered address parts
  - Kannada state names
  - missing addresses

## Stage timings (for planning Kaggle sessions)
| Stage | dev (local) | train (Kaggle) | test (Kaggle) |
|---|---|---|---|

## Pitfalls and gotchas
- **TSV reads:** always use `separator="\t"` and `quote_char=None`, with every column read as a string. Names contain quotes and commas.
- **TSV writes:** use `quote_style="never"`, otherwise empty strings may be written as `""`.
- **Stale artifacts:** `read_path` falls back to `prev_work_dirs`. Skipping a stage in a new run silently reuses old artifacts, so always run downstream stages in order.
- **Token leak:** never let the GitHub token persist in Kaggle outputs. The driver resets the remote URL after cloning.
- **Leakage:** the country_holdout scheme must mine the lexicon and train every model (cheap, GBM, bi-encoder, cross-encoder, combiner) on US `fit` rows only.
- **Validator:** `validate_submission` must print PASS before any upload.
- **Disk:** the local C: drive has little free space. Keep only the dev slice and the train parquet locally.

## Next steps
1. Start Task 1 (scaffold) from `plan.md`.

