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
- 2026-09-25 (T8): **`lexicon.min_share` raised from 0.6 to 0.75 in `base.yaml`.** On the real dev fit fold, at 0.6 an estimated 52% of the 257 mined `addr` entries were junk (single-letter transliteration targets, e.g. `podder -> p`, or generic locality-name attractors, e.g. `flat -> kolkata`, `benchmark -> bengal`) — bad entries dominated the addr map. Raising `min_share` to 0.75 cut addr entries to 211 (removed 46, including `united -> limited`, `creative -> private`, `innovative -> private`, `arihant -> private`, `vrait -> private` from `name`, and most single-letter-target addr junk), at the cost of `name` shrinking from 140 to 135 and addr junk only dropping to roughly 48% (the residual cluster's *share* is genuinely high, not just noisy, because the dev slice has only 5 localities so e.g. Kolkata-suburb tokens co-occur with "bengal"/"howrah" almost every time — min_share alone can't distinguish that from a real translation). Kept at 0.75 per the brief's rule since it measurably improves precision without an alternative lever available in this task.
- 2026-09-25 (T4): **`blocking.same_country` stays `true`.** The M0 EDA measured a cross-country GT match share of exactly 0.0 across all 7,638,365 train pairs, so `base.yaml` was not changed.
- 2026-09-25 (T4): **`decide.one_owner` stays `true`.** The M0 EDA found 0 right records (S2/S3) with more than one S1 owner in train GT — 0%, well under the 0.1% threshold that would have flipped this to `false` — so `base.yaml` was not changed.
- 2026-09-25: **Approach:** a staged hybrid.
  - M1 is classic: blocking, then LightGBM, then the decision rule.
  - M2 adds a multilingual-e5-small bi-encoder.
  - M3 adds an xlm-roberta-base cross-encoder and a combiner.
  - A stage is kept only if F0.5 improves on **both** the s1_random and country_holdout (US→India) schemes. Why: this aims for the top of the leaderboard while protecting France, which has no labels.
- 2026-09-25: **Validation:** an 80/20 split by S1, with blocking over all train S2/S3 records. The US→India holdout stands in for France.
- 2026-09-25: **Decision rule:** isotonic calibration → one owner per S2/S3 → expected-F0.5 subset per S1, compared automatically against a tuned global threshold.
- 2026-09-25: **Compute:** Kaggle; the code sync method is a private GitHub repo (the user's choice). `memory.md` is this project logbook (the user's choice).

## Status
- Current milestone: **M1, T8 done.** Plan: `plan.md`. Spec: `docs/superpowers/specs/2026-09-25-business-entity-resolution-design.md`.
- M0:
  - [x] T1 scaffold/config/validator/GitHub
  - [x] T2 ingest + CLI
  - [x] T3 metrics
  - [x] T4 EDA
  - [x] T5 folds + dev slice
- M1:
  - [x] T6 name text
  - [x] T7 address
  - [x] T8 lexicon
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
- Ran on the real train ingest (`configs/dev.yaml`), 2026-09-25: s1=2,206,821, right=10,320,219, gt pairs=7,638,365 — no "ids not found" warning.
- **Cross-country match share: 0.0** (all 7,638,365 GT pairs are within-country) → `blocking.same_country` stays `true` (no change to `base.yaml`).
- **Right records with more than one S1 owner: 0** (0% of matched right records, well under the 0.1% threshold) → `decide.one_owner` stays `true` (no change to `base.yaml`).
- 74.0% of right records (S2+S3) match some S1 in train; the rest are pure distractors.
- Singleton rate (S1 with 0 GT matches): **US 5.58%, India 5.59%** (close to the global 5.6%). Mean matches/S1: US 3.459, India 3.465; p99 = 8 for both.
- Non-ASCII rates: S1 names/addresses are ~0% non-ASCII in both countries (S1 India addr 0.06%). S2/S3 are much noisier — India business_name non-ASCII 27.9% (S2) / 18.5% (S3); India business_address non-ASCII 23.7% (S2) / 22.5% (S3) (Devanagari/Telugu/Odia script mixed into otherwise-Latin fields). US S2/S3 non-ASCII name ~6.7-6.8%, address ~0.002%.
- Empty-address rate: 0% for S1 (both countries); S2/S3 empty_addr ~2.9-3.7% across sources/countries. empty_name is 0% everywhere.
- 5-digit number in address (US ZIP-like): S1 US 10.82%, S2 US 10.13%, S3 US 10.17%; India 5-digit rate is much lower (~0.3-1.1%, mostly PIN-adjacent noise, not systematic). 6-digit number (India PIN-like): **essentially 0% in all India rows** (S1/S2/S3), while US 6-digit rate is ~0.1-1.4% (likely incidental, e.g. phone/account digits) — India PIN codes are not reliably present as a clean 6-digit token in `business_address`.
- Noise examples seen in the sample rows (seed=0, 40 sampled GT pairs via `scripts/eda_m0.py`):
  - Devanagari (Hindi) script names/addresses: "बॉम्बे इन्वेस्टमेंट्स प्राइवेट लिमिटेड" for "Bombay Investments Private Limited"; "इनोवेटिव गैलेक्सी प्रोडक्ट्स प्राइवेट लिमिटेड" for "Innovative Galaxy Products Private Limited"
  - Partial-script substitution: "ఓం Finance Pvt Ltd" for "Om Finance Pvt Ltd" (only the first word transliterated to Telugu)
  - Regional-script state names embedded in an otherwise-English address: "ఓడ़ిశా"/"ଓଡ଼ିଶା" for "Orissa", "తెలంగాణ" for "Telangana"
  - Accent-character noise substituted into ASCII names: "Créative" for "Creative", "Stáffing" for "Staffing", "Límited" for "Limited", "Áll" for "All"
  - Legal suffix moved to the front: "LLC Stout Stáffing" vs "Stout Staffing LLC"; "LLC 5uperior Analytics" vs "Superior Analytics LLC"
  - Digit/letter typo substitution: "5uperior" for "Superior"; "Ines" for "Inks"; "Clrinic" for "Clinic"
  - Website used as the business name: "johnsonweber.com", "#dhariniindia", "TEAMSTERSLOCALUNION.COM"
  - Literal "null"/"N/A" tokens embedded inside address strings: "BENGALURU, null, Karnataka"; "MINDEN DRIVE, NULL, INDIANAPOLIS, IN"; "5 MORNING VIEW CT, N/A"
  - House/plot numbers masked with "#": "R.NO.##1 SR.NO.207" vs "R.No.1 Sr.No.207"; "###203/A" vs "No. 203/A"
  - Heavily reordered address token order between the S1 and the matched right record (city/street/state order scrambled, e.g. "Indianapolis, 5842 Minden Drive, IN" vs "MINDEN DRIVE, NULL, INDIANAPOLIS, IN")

## Stage timings (for planning Kaggle sessions)
| Stage | dev (local) | train (Kaggle) | test (Kaggle) |
|---|---|---|---|
| ingest --split train (local, full train, `configs/dev.yaml`) | 20s | | |
| dev_slice (localities: phoenix, cleveland, tyler, kolkata, bhopal) | 2s | | |
| split --split dev (s1_random, valid_frac=0.2) | <1s | | |
| lexicon --split dev (fit fold, ~170k GT pairs, min_share=0.75) | 66s | | |

## Task 5: folds + dev slice (2026-09-25)
- Dev slice built from `configs/dev.yaml` `dev.localities: [phoenix, cleveland, tyler, kolkata, bhopal]` against the real local train parquet: **s1=60,664, right=278,368, gt=211,276**. This is within the brief's "roughly 10k-60k, drop a locality if >80k" guidance (60,664 is slightly above 60k but well under the 80k drop threshold, so `configs/dev.yaml` was left unchanged).
- `split --stage split --split dev` (default `s1_random`, `valid_frac=0.2`, `seed=42` from `base.yaml`) produced fold counts: **fit=48,546, valid=12,118** (19.97% valid share).
- Deviation from the brief's verbatim code: `stage_split`'s `print(folds.group_by("fold").len())` raised `UnicodeEncodeError` on the Windows cp1252 console (polars' box-drawing table glyphs aren't encodable there). Fixed by replacing it with a plain ASCII summary line (`print("[split] " + ", ".join(...))`) that reports the same counts — smallest change that keeps the brief's intent (visibility into fold sizes) without depending on `PYTHONIOENCODING`.

## Task 8: lexicon mining (2026-09-25)
- Ran `lexicon --split dev` on the real dev fit fold (`configs/dev.yaml`): 48,546 fit S1 rows, GT `fit`-fold pairs joined to `s1`/`right` text (~170k matched pairs, well under `sample_pairs: 500000` so no downsampling). Took 66s locally. Final `C:/Users/AYUSH/ber_data/work/dev/lexicon.json` (at the raised `min_share=0.75`): **name=135 entries, addr=211 entries**.
- **10 good entries** (romanization/typo fixes and standard abbreviations, all semantically correct):
  - name: `praivet -> private`, `kansaltin -> consulting`, `helathakeyar -> healthcare`, `teknolonji -> technology`, `intaranesanal -> international`, `lajistikas -> logistics`, `sonlyusans -> solutions`
  - addr: `blvd -> boulevard`, `klkata -> kolkata` (typo fix), `mh -> maharashtra` (state abbreviation)
- **10 bad entries** (semantically wrong, survived even at `min_share=0.75`):
  - name: `man -> maa`, `hai -> high`, `vest -> best`, `vig -> big`, `aiti -> it`, `life -> limited`
  - addr: `flat -> kolkata`, `bagan -> bengal`, `benchmark -> bengal`, `podder -> p`
- Root cause of the addr noise: the dev slice has only 5 localities (`phoenix, cleveland, tyler, kolkata, bhopal`), so many rare English tokens that appear only in Kolkata-area addresses get Jaro-Winkler-aligned to the ubiquitous nearby word ("bengal", "howrah", "kolkata") even though there's no real translation/abbreviation relationship — their co-occurrence *share* is genuinely high (not just noisy), so raising `min_share` further would start cutting good entries before it clears this cluster. This is expected to shrink on the full train fit fold, which has far more localities and countries diluting any single place-name attractor.
- Applied the brief's remedy: raised `lexicon.min_share` 0.6 -> 0.75 in `base.yaml` (see Key decisions) since bad entries were an outright majority (52%) of the addr map at 0.6.

## Pitfalls and gotchas
- **Windows console encoding (R10, T8):** the cp1252 Windows console crashes (`UnicodeEncodeError`) when a stage prints polars tables or non-Latin text (e.g. `lexicon.json` entries with Devanagari-derived tokens, or a wide polars `group_by` table). Fixed once, centrally, at the top of `main()` in `pipeline.py`: reconfigure `sys.stdout`/`sys.stderr` to `encoding="utf-8", errors="replace"` when the stream supports `.reconfigure`. This supersedes the narrower per-stage `print` workaround from T5 (`stage_split`'s ASCII-only summary line) — that workaround is now redundant but harmless, so it was left as-is.
- **Address placeholder tokens (R8, T7):** literal "null"/"N/A"/"NA"/"none"/"nil" show up embedded in real addresses (see T4 EDA examples). `normalize_address` drops them after `basic_clean` via `_drop_placeholders`. Gotcha: `basic_clean` turns "N/A" into the two separate tokens "n","a" (slash -> space), and "n" alone is a real address abbreviation (`ADDR_ABBREV["n"] == "north"`), so the filter must match the adjacent pair `("n","a")` specifically, before `map_tokens` runs — matching "n" or "a" individually would wrongly eat "N Main St" and standalone "a" tokens (e.g. "Block A").
- **TSV reads:** always use `separator="\t"` and `quote_char=None`, with every column read as a string. Names contain quotes and commas.
- **TSV writes:** use `quote_style="never"`, otherwise empty strings may be written as `""`.
- **Stale artifacts:** `read_path` falls back to `prev_work_dirs`. Skipping a stage in a new run silently reuses old artifacts, so always run downstream stages in order.
- **Token leak:** never let the GitHub token persist in Kaggle outputs. The driver resets the remote URL after cloning.
- **Leakage:** the country_holdout scheme must mine the lexicon and train every model (cheap, GBM, bi-encoder, cross-encoder, combiner) on US `fit` rows only.
- **Validator:** `validate_submission` must print PASS before any upload.
- **Disk:** the local C: drive has little free space. Keep only the dev slice and the train parquet locally.

## Next steps
1. Start Task 9 (normalize stage) from `plan.md`.

