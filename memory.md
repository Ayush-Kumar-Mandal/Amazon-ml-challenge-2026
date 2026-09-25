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
- 2026-09-25 (T11): `sparse_dot_topn` 1.2.0's `sp_matmul_topn(A, B, top_n, threshold=None, sort=False, density=None, n_threads=None, idx_dtype=None)` signature matches the task-11 brief's call exactly (`top_n=`, `threshold=`, `sort=`, `n_threads=` all present) — no adaptation needed, ran and passed locally rather than skipping.

## Key decisions (and why)
- 2026-09-25 (T8 fix round 1): **`lexicon.min_share` reverted from 0.75 back to 0.6 in `base.yaml`, after fixing `mine_lexicon`'s real bug.** Root cause of the T8 addr noise wasn't `min_share` at all: `detect_script` was called on the *whole* raw string, so a single stray non-Latin character (e.g. one Kannada word in an otherwise-English address) classified the entire string as non-Latin, dumping its plain-English tokens ("east", "chennai", ...) into the transliteration source pool, where they got Jaro-Winkler-aligned to whatever Latin word happened to be nearby (`east -> west`, `chennai -> phoenix`, 23 tokens -> `bengal`, single-letter targets like `podder -> p`). Fixed in `src/ber/lexicon.py`: (1) per-token script classification — the transliteration source pool is now built only from raw tokens (split on whitespace/punctuation, preserving combining marks so Indic scripts don't fragment) that are themselves non-Latin, with the abbreviation branch left untouched; (2) `_align` now requires `len(a) >= 3` and `len(best) >= 3`, killing the single-letter-target artifacts outright. Re-mining the dev fit fold at the old `min_share=0.6` post-fix gave **addr=65 entries with zero remaining generic-attractor/single-letter junk** (down from 257, ~52% junk) and **name=134** (down from 140, having also dropped several leaked-Latin-token artifacts: `creative -> private`, `innovative -> private`, `united -> limited`, `life -> limited`, `ma -> maa`, `aiti -> it`). Re-tested at 0.75: it only removed 2 more (still-)bad `name` entries (`arihant`, `vrait`) while also removing 2 good `addr` entries (`in -> indiana`, `ks -> kansas`) — no net benefit once the actual bug is fixed — so `min_share` went back to the brief's original default, 0.6.
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
- Current milestone: **M1, T15 done.** Plan: `plan.md`. Spec: `docs/superpowers/specs/2026-09-25-business-entity-resolution-design.md`.
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
  - [x] T9 normalize stage
  - [x] T10 keys
  - [x] T11 TF-IDF
  - [x] T12 merge/block
  - [x] T13 cheap cut
  - [x] T14 full features
  - [x] T15 decide
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
| 2026-09-25 | block --split dev (`configs/dev.yaml`, s1_random valid fold) | s1_random | 0.9871 | | 98.4 | | | | recall by source: from_keys 0.8179, tfidf_name 0.8516, tfidf_name_addr 0.9768 (union 0.9871 clears the 0.97 gate; no `--set` adjustment needed) |
| 2026-09-25 | cheap_train + cheap_apply --split dev (`configs/dev.yaml`, s1_random valid fold, `cheap.keep_top=25`) | s1_random | 0.9871 | 0.9854 | 24.9 | | | | GBM cheap-cut clears the 0.965 gate; cands_all 6,000,665 -> cands_final 1,511,179 pairs |
| 2026-09-25 | features --split dev (`configs/dev.yaml`, s1_random, fit+valid folds = full dev S1) | s1_random | | | | | | | 1,511,179 pairs written to `features/part-0000.parquet` (46 feature columns + `y`); y positives=208,159 (13.8%), negatives=1,303,020 |

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
| lexicon --split dev (fit fold, ~170k GT pairs, min_share=0.6, post-fix) | 80s | | |
| normalize --split dev (s1=60,664 + right=278,368, `n_jobs=6`) | 21s (~16,200 rows/s) | | |
| block --split dev (keys + tfidf_name + tfidf_name_addr, `n_threads=8`, `max_df=0.05`) | 135s total (measured per-view separately: keys 1.8s; tfidf_name 28.2s, India 2,227,445 + US 935,696 pairs; tfidf_name_addr 82.0s, India 2,116,441 + US 717,276 pairs) | | |
| cheap_train --split dev (300 rounds requested, no early stop, ~6M cands x fit sample) | 107s | | |
| cheap_apply --split dev (~6,000,665 cands, `chunk_rows=5,000,000`, top-25 cut) | 44s | | |
| features --split dev (1,511,179 cands, one chunk since `chunk_rows=2,000,000`, `vectorizer_fit_rows=2,000,000`) | 44s (~34,300 rows/s) | | |

## Task 5: folds + dev slice (2026-09-25)
- Dev slice built from `configs/dev.yaml` `dev.localities: [phoenix, cleveland, tyler, kolkata, bhopal]` against the real local train parquet: **s1=60,664, right=278,368, gt=211,276**. This is within the brief's "roughly 10k-60k, drop a locality if >80k" guidance (60,664 is slightly above 60k but well under the 80k drop threshold, so `configs/dev.yaml` was left unchanged).
- `split --stage split --split dev` (default `s1_random`, `valid_frac=0.2`, `seed=42` from `base.yaml`) produced fold counts: **fit=48,546, valid=12,118** (19.97% valid share).
- Deviation from the brief's verbatim code: `stage_split`'s `print(folds.group_by("fold").len())` raised `UnicodeEncodeError` on the Windows cp1252 console (polars' box-drawing table glyphs aren't encodable there). Fixed by replacing it with a plain ASCII summary line (`print("[split] " + ", ".join(...))`) that reports the same counts — smallest change that keeps the brief's intent (visibility into fold sizes) without depending on `PYTHONIOENCODING`.

## Task 8: lexicon mining (2026-09-25, updated in fix round 1)
- Original run: `lexicon --split dev` on the real dev fit fold (`configs/dev.yaml`): 48,546 fit S1 rows, GT `fit`-fold pairs joined to `s1`/`right` text (~170k matched pairs, well under `sample_pairs: 500000` so no downsampling). At the brief's default `min_share=0.6`, addr came out at 257 entries, an estimated 52% junk — root-caused to a real bug in `mine_lexicon`, not to `min_share` (see Key decisions).
- **Fix round 1 (code bug, not a tuning issue):** `detect_script` was being called on the whole raw name/address string. A single stray non-Latin character (e.g. one Kannada word riding along an otherwise-English address) made the *entire* string classify as non-Latin, so its plain-English tokens ("east", "chennai", ...) leaked into the transliteration source pool and got Jaro-Winkler-aligned to whatever nearby Latin word happened to score >= 0.7 — producing `east -> west`, `chennai -> phoenix`, 23 tokens -> `bengal`, many -> `howrah`, 19 -> `madhya`, and single-letter targets (`podder -> p`, `alif -> l`, `crescent -> c`). Fixed in `src/ber/lexicon.py`:
  1. Per-token script classification: the transliteration source pool is now built only from *raw* tokens (split on whitespace/punctuation via `_raw_tokens`, which deliberately avoids `\w` because Python's `\w` excludes combining marks — using it fragmented Devanagari/Kannada conjuncts into single characters) that are themselves non-Latin (`detect_script(token) != "latin"`); the other side counts as the Latin target pool only when none of its raw tokens are non-Latin. The abbreviation branch was left unchanged, as directed.
  2. `_align` now requires `len(a) >= 3` and `len(best) >= 3`, removing single-letter-target artifacts outright.
  3. Added `tests/test_mixed_script_side_does_not_leak_latin_tokens` (a Kannada+English mixed address pair, `Plot 5 East Behala ಕರ್ನಾಟಕ` / `Plot 5 West Behala Karnataka`, x10): asserts `"east" not in lex["addr"]` and `lex["addr"]["karnatak"] == "karnataka"`. All 4 prior tests still pass (5/5 total).
- Re-mined the dev fit fold post-fix. At `min_share=0.6`: **name=134, addr=65** (addr junk is gone entirely; `name` lost the leaked-token artifacts `creative -> private`, `innovative -> private`, `united -> limited`, `life -> limited`, `ma -> maa`, `aiti -> it`). At `min_share=0.75`: name=132, addr=63 — only removed 2 more bad `name` entries (`arihant`, `vrait`) while also removing 2 *good* `addr` entries (`in -> indiana`, `ks -> kansas`), i.e. no net benefit now that the real bug is fixed. Per the evidence, `min_share` was reverted to **0.6** (the brief's original default) in `base.yaml`. Final `C:/Users/AYUSH/ber_data/work/dev/lexicon.json`: **name=134 entries, addr=65 entries**, mined in 80s (well inside the 10-minute budget).
- **10 good entries** (post-fix, all semantically correct):
  - name: `praivet -> private`, `kanasaltin -> consulting`, `helathakeyar -> healthcare`, `teknolonji -> technology`, `intaranesanal -> international`, `krietiv -> creative`, `inovetiv -> innovative`
  - addr: `blvd -> boulevard`, `karnatak -> karnataka` (the regression-test case, genuinely mined from real data too), `mh -> maharashtra`
- **Bad entries remaining post-fix:** only 6 (down from an estimated 60+ before the fix), all in `name`, all short-token Jaro-Winkler mis-alignments of a genuinely non-Latin source token (not the leaked-token bug) — an inherent, much smaller residual noise level in the alignment heuristic itself: `man -> maa`, `hai -> high`, `vest -> best`, `vig -> big`, `arihant -> private`, `vrait -> private`. `addr` has zero remaining bad entries out of 65 (all standard abbreviations, US/Indian state abbreviations, or genuine city-name typo fixes, e.g. `ceveland/cleeland/cleveand/... -> cleveland`, `phenix/phoeix/... -> phoenix`).

## Task 9: normalize stage (2026-09-25)
- `normalize --split dev` on the real dev slice (`configs/dev.yaml`, `normalize.n_jobs: 6`): s1=60,664 rows, right=278,368 rows, both written in **21s total (~16,200 rows/s)** using a spawn-context `multiprocessing.Pool`. Spawn-context Pool worked without issue on Windows/py3.10 — no fallback needed, matching the brief's expectation.
- Deviation from the brief's verbatim code: `pl.concat([df, norm], how="horizontal")` raises `DeprecationWarning` under the installed polars 1.44.2 (the plain `"horizontal"` default is being tightened to require equal heights; `"horizontal_extend"` is the direct replacement that keeps the old behavior). Fixed by using `how="horizontal_extend"` in `normalize_frame` — smallest change, same concat semantics, no other code affected.
- Eyeballed 15 sampled `right_norm.parquet` rows (seed=1, phoenix/kolkata-heavy dev localities): `name_core`/`legal_form` splitting looks correct even when legal terms sit mid-string ("Dynamic Limited Private Marketing" -> core `dynamic marketing`, legal `ltd pvt`; "Om Ltd Center" -> core `om center`, legal `ltd`). No systematic parsing failures found beyond the already-documented (T4/T7) postcode sparsity: across all of `right_norm.parquet`, only **2.35%** of rows have a non-empty `postcode` (s1: 0.90%), while `house_no` is populated for **92.3%** of right rows — consistent with the M0 EDA finding that clean 5-digit US ZIPs and 6-digit India PINs are rare in `business_address`, not a normalize bug.
- `name_key` (2-token metaphone) and `street_tok` fields were not separately audited beyond the unit test; they follow directly from `text.py` (Tasks 6-7), which already has its own test coverage.

## Task 12: merge candidates + block stage (2026-09-25)
- `block --split dev` on the real dev slice (`configs/dev.yaml`, `blocking.tfidf.n_threads: 8`, `max_df: 0.05`): s1=60,664, right=278,368. Union of exact-key blocking (`record_keys`/`join_keys`) and two TF-IDF char-trigram views (`name`, `name_addr`) produced **6,000,665 candidate pairs, 98.9 per S1**, written to `cands_all.parquet`. `blocking.embed.enabled` is `false` in `base.yaml`/`dev.yaml`, so `cands_embed.parquet` is not loaded/merged (deferred to T20/T21).
- **Overall recall on the valid fold: 0.9871** (clears the 0.97 gate on the first attempt — no `--set` adjustment needed). Recall by source (each measured by filtering `cands_all.parquet` to that source's rows before joining to `gt.parquet`, so sources overlap and don't sum to the union): `from_keys` 0.8179, `tfidf_name` 0.8516, `tfidf_name_addr` 0.9768.
- Candidates per S1 (valid fold): **98.4**.
- Elapsed time per view (measured by timing `record_keys`+`join_keys` and each `tfidf_candidates` call separately, same dev inputs/config as the real stage run): **keys 1.8s** (2,129,681 pairs before top-k/threshold, i.e. pre-merge); **tfidf_name 28.2s** (India 2,227,445 + US 935,696 pairs); **tfidf_name_addr 82.0s** (India 2,116,441 + US 717,276 pairs). Total `block` stage wall time on dev: **135s** (`[pipeline] block --split dev done in 135s`), consistent with the sum of the three views plus merge/write overhead.
- No deviation from the brief's verbatim code for `merge.py` or `stage_block`; `pipeline.py` updated exactly as instructed (`from ber.stages import candidates, data`, `"block": candidates.stage_block`).

## Task 13: cheap features, LightGBM helpers, cheap-cut stages (2026-09-25)
- Implemented `src/ber/features.py` (`CHEAP_COLUMNS`, `attach_sides`, `sim`, `tri`, `cheap_features`), `src/ber/models/gbm.py` (`DEFAULT_PARAMS`, `to_matrix`, `train_binary`, `predict`, `load_booster`), added `common.es_fold`, and `stage_cheap_train`/`stage_cheap_apply` in `stages/candidates.py`, registered in `pipeline.STAGES`. Confirmed the installed `rapidfuzz` (3.14.5) has `process.cpdist`, so the brief's `sim()` needed no rewrite.
- **Deviation from the brief's verbatim code (real bug, not style):** `train_binary`'s `lgb.early_stopping(early_stopping_rounds, verbose=True)` call, combined with `DEFAULT_PARAMS["metric"] = ["binary_logloss", "auc"]`, uses LightGBM's default `first_metric_only=False`. That makes early stopping trigger as soon as *either* tracked metric stalls, not just the primary one. Reproduced directly: on `tests/test_gbm.py`'s synthetic data, training stopped at iteration 8 (AUC peaked there, even though logloss kept improving), leaving predicted probabilities clustered in [0.46, 0.56] and accuracy at 0.918 (test requires > 0.95). Fixed by adding `first_metric_only=True` to the `lgb.early_stopping(...)` call — smallest change that keeps the brief's intent (log both `binary_logloss` and `auc`, but early-stop on the primary metric, `binary_logloss`, which is first in the list). With the fix, training ran the full 200 rounds in the unit test and reached 99.8% accuracy; `test_gbm.py::test_train_predict_roundtrip` and `test_features_cheap.py::test_cheap_features_values_and_order` both pass with `-W error::DeprecationWarning`. `attach_sides` used `how="horizontal_extend"` per R11 (matches the pattern already used in `normalize_frame`).
- Real dev run (`configs/dev.yaml`, s1_random scheme, so `es_fold(cfg) == "valid"`): `cheap_train --split dev` trained on the `fit` fold (48,546 S1, no `max_train_s1` cap needed at dev scale) against a `valid`-fold ES sample (capped at 50,000 S1), ran the full 300 boost rounds without early stopping (`min_data_in_leaf=100, num_leaves=63, learning_rate=0.1` from `base.yaml`'s `cheap.params`), final `valid_0` binary_logloss=0.019496, auc=0.997481, in **107s**. `cheap_apply --split dev` scored and cut all 6,000,665 candidate pairs in one chunk (`chunk_rows=5,000,000` means 2 chunks) to top-25-per-S1 in **44s**: **6,000,665 -> 1,511,179 pairs**.
- **Recall on the valid fold: all=0.9871 (unchanged from T12's blocking recall, as expected since scoring doesn't drop any pair before the top-k cut), final=0.9854** after the top-25-per-S1 cut — clears the 0.965 gate with room to spare. **Candidates per S1: 24.9** (target ~25, matches `cheap.keep_top=25`).

## Task 14: context + full pair features, `features` stage (2026-09-25)
- Appended `CONTEXT_COLUMNS`, `NAME_COLUMNS`, `ADDR_COLUMNS`, `FEATURE_COLUMNS`, `context_features`, `fit_vectorizers`, `_rowdot`, `_jaccard`, `full_features` to `src/ber/features.py`, verbatim from the brief. Created `src/ber/stages/model.py` (`load_features`, `stage_features`), registered `"features": model.stage_features` in `pipeline.STAGES` (`from ber.stages import candidates, data, model`). Added `tests/__init__.py` (empty) so `from tests.test_features_cheap import CANDS, LEFT, RIGHT` resolves, and `tests/test_features_full.py` verbatim from the brief.
- **No deviations from the brief's verbatim code were needed.** All the polars 1.44.2 calls the controller flagged for double-checking worked as written with zero warnings under `-W error::DeprecationWarning`: `list.set_union`/`list.set_intersection` (in `_jaccard`), `str.extract_all` (numeric-token extraction for `name_num_eq`), `pl.min_horizontal`/`pl.max_horizontal` (in `len_ratio`), `Series.gather` (attaching the "other" candidate's name/addr in `full_features`), and `pl.len().over(...)` (in `context_features`). `TfidfVectorizer`/`Levenshtein` imports and the empty-vocabulary fallback (`min_df=2` -> `min_df=1` on tiny fixture data) also worked without changes.
- TDD: RED first (`pytest tests/test_features_full.py -v` failed with `ImportError: cannot import name 'FEATURE_COLUMNS' from 'ber.features'`, exactly as the brief predicted), then GREEN (`pytest tests/test_features_full.py tests/test_features_cheap.py -v -W error::DeprecationWarning` -> 3 passed). Full suite after adding `tests/__init__.py`: `pytest -q -W error::DeprecationWarning` -> **58 passed**, no warnings — the new `tests/__init__.py` package marker did not break collection of any existing test module.
- Real dev run: `features --split dev` (`configs/dev.yaml`; `features.max_train_s1=100000` from `dev.yaml`, `features.chunk_rows=2,000,000` and `features.vectorizer_fit_rows=2,000,000` from `base.yaml` defaults, neither overridden by `dev.yaml`). Dev's `fit` (48,546) + `valid` (12,118) folds already cover the full 60,664-row dev S1 set (no separate `test` fold at dev scale), so the fold-based `l_idx` filter kept all of `cands_final.parquet`: **1,511,179 pairs unchanged**, written as a single chunk (`features/part-0000.parquet`, since 1,511,179 < `chunk_rows`). Output schema is exactly `["l_idx", "r_idx", *FEATURE_COLUMNS, "y"]` (46 feature columns + `y`), matching the test's schema assertion. `y` label balance: 208,159 positive (13.8%), 1,303,020 negative — consistent with T13's post-cut recall (0.9854) and candidate count. Spot-checked `null_count()`: zero nulls in every feature column except `num_jaccard` (9,139 nulls, exactly the rows where neither side has a numeric address token, per `_jaccard`'s designed `None`-on-empty-union behavior); `sim_other_name`/`sim_other_addr` use float `NaN` (not polars null) for candidates with no "other" rank, matching the test's `np.isnan` assertion.
- **Elapsed time: 44s** for 1,511,179 pairs (`[pipeline] features --split dev done in 44s`) = **~34,300 rows/second**. This is the number to extrapolate Kaggle `train`/`test` timing from (T18).

## Task 15: decision rules (2026-09-25)
- Created `src/ber/decide.py` verbatim from the brief: `THRESHOLDS`/`SCALES` constants, `fit_calibrator`/`calibrate` (isotonic regression), `crossfit_calibrate` (half-split cross-fit calibration keyed on `l_idx`), `one_owner` (best-claim-per-`r_idx` dedup), `choose_threshold`, `choose_expected_f05` (plug-in expected-F0.5 top-k rule per S1, comparing against the "predict nothing" score), `tune_threshold` (grid search against `macro_f05_frame` from Task 3), `scale_unseen` (multiplicative probability scaling for S1s in unseen countries), and `select` (threshold vs. expected-F0.5 dispatch). Pure module — no stage registration, no real-data run, per the controller's task scope.
- **No deviations from the brief's verbatim code were needed**, including the `[0.5, 0.6]` exact float-equality assert in `test_scale_unseen_only_touches_unseen_countries` — it compared exactly with no floating-point noise, so `pytest.approx` was not needed.
- TDD: RED first (`pytest tests/test_decide.py -v` failed with `ModuleNotFoundError: No module named 'ber.decide'`, as predicted), then GREEN (`pytest tests/test_decide.py -v -W error::DeprecationWarning` -> 5 passed, no warnings). Full suite: `pytest -q -W error::DeprecationWarning` -> **63 passed**, no warnings.

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
1. Start Task 15 (decide) from `plan.md`.

