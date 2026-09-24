# Business Entity Resolution: Design Spec

- **Date:** 2026-09-25
- **Challenge:** Amazon ML Challenge 2026
- **Status:** approved design, awaiting spec review

## 1. Problem summary
- **Input:** For every test Source 1 (S1) record, find all Source 2 (S2) and Source 3 (S3) records that refer to the same real-world business.
- **Record fields:** `entity_id`, `business_name`, `business_address`, `country`.
- **Metric:** macro F0.5, computed per S1 record and then averaged. Singletons score 1.0 only when we predict an empty list.

| Split | S1 | S2 | S3 | Countries |
|---|---|---|---|---|
| train | 2.21M | 5.03M | 5.29M | US, India |
| test | 1.73M | 4.89M | 5.08M | US, India, **France (test only)** |

- **Training labels:** 7.64M matches, about 3.5 per S1 record. 5.6% of S1 records are singletons. The largest groups have 10 or more matches.
- **Hard constraints:**
  - The model must be MIT or Apache-2.0 licensed with at most 8B parameters.
  - No external data, APIs or geocoding.
  - Both output TSVs must pass `utils/validate_submission.py`.
  - The final zip layout is fixed by the problem statement.

## 2. Environment and goal
- **Compute:** Kaggle notebooks (about 29 GB RAM, 4 CPU cores, T4 or P100 GPUs, 12-hour sessions, 30 GPU-hours per week).
- **Development:** code is written locally and tested on a dev slice.
- **Goal:** a top-of-leaderboard push over several weeks.
- **Approach:** a staged hybrid:
  1. Classic pipeline
  2. Fine-tuned multilingual bi-encoder
  3. Cross-encoder reranker
- **Rule for each stage:** keep it only if it improves held-out F0.5 on **both** validation schemes (§4).

## 3. Architecture
The stages form a pipeline. Each one reads the parquet files from the previous stage and writes its own, so a run can resume across Kaggle sessions (save `/kaggle/working` outputs as a Kaggle dataset version).

```
code/business_entity_resolution/
  src/ber/
    io.py              TSV → parquet (explicit sep="\t", all columns read as strings, empty strings kept)
    normalize.py       name/address normalization, script conversion, legal-form and component extraction
    blocking/
      keys.py          exact-key blocking
      tfidf_ann.py     char-3-gram TF-IDF top-k search, both directions, in chunks
      embed_ann.py     FAISS search over bi-encoder embeddings (Stage 2)
      merge.py         union, deduplicate, cheap-ranker cut → final candidate set
    features.py        pair features (§6)
    models/
      gbm.py           LightGBM training and inference (cheap ranker + main model + combining model)
      biencoder.py     multilingual-e5-small fine-tuning and encoding
      crossencoder.py  xlm-roberta-base pair classifier
    decide.py          calibration, one-owner rule, expected-F0.5 subset choice
    metrics.py         macro F0.5, blocking recall, reduction ratio
    submit.py          write TSVs, run validator, build the submission zip
    pipeline.py        CLI: python -m ber.pipeline --stage <name> --split <train|test> --config <yaml>
  configs/             base.yaml, dev.yaml, kaggle.yaml
  tests/
  notebooks/kaggle_driver.ipynb
  README.md, requirements.txt (pinned)
```

**Libraries:**
- Data: polars, pyarrow
- String similarity: rapidfuzz, jellyfish
- Vectors and search: scikit-learn, sparse_dot_topn, faiss
- Models: lightgbm, sentence-transformers, transformers, torch
- Script conversion: indic-transliteration, unidecode

## 4. Validation
- **Main split:** an 80/20 split by S1 record, with a fixed seed. Blocking searches the **full** train S2 and S3 pools, so the held-out records face realistic distractors. Models train on the 80% records' candidate pairs and are evaluated on the 20%.
- **Unseen-country check:** train on US-only S1 records and evaluate on India. This stands in for France, which has no labels. It is reported for every experiment.
- **Dev slice:** a small set of cities or states chosen so that their true matches stay together. It is used for fast local iteration and is not a basis for decisions.
- **Experiment log:** every run appends its config hash, blocking recall, candidates per S1, main-split F0.5 and US→India F0.5 to `experiments.csv`.

## 5. Normalization and blocking

### Normalization
Raw fields are kept, and normalized versions are added as new columns.

- **Name:**
  - NFKC, lowercase, strip accents, `&`→`and`, remove punctuation and junk prefixes.
  - Extract the legal form into a separate canonical field (inc, llc, corp, ltd, pvt ltd, llp, sarl, sas, sa, eurl, …).
  - Detect website names (`x.com`).
  - The core name is what remains.
- **Address:**
  - Expand abbreviations (st, rd, ave, blvd/bd, r.→rue, …).
  - Extract components with patterns:
    - postal code: 5 digits for US and France, 6-digit PIN for India
    - house number
    - city
    - state
  - State names are mapped across forms: US codes to names, and Indian native-script state names to Latin.
- **Script:**
  - Devanagari and Kannada are converted to Latin with a library.
  - This is refined by a **token dictionary mined from the training ground truth**: we align words of matched non-Latin and Latin names. Abbreviation dictionaries are also mined from co-occurring words in matched pairs.
  - France gets a small general list of French legal forms and street words. There are no other country-specific rules.

### Blocking
Candidates from all sources are combined into one set.

- **Exact keys** (within country, if M0 confirms that matches never cross countries; very large blocks are skipped):
  - postal code + name token
  - city + name prefix
  - sound-alike name key + state
  - house number + street token + city
- **TF-IDF search:**
  - char 3-gram TF-IDF on core name, and separately on name+address
  - top-k cosine, **in both directions**: S1→S2/S3 and S2/S3→S1
- **Embedding search (Stage 2):** FAISS on GPU over fine-tuned bi-encoder embeddings, top 30 per record.

### Cheap cut
- A cheap LightGBM on about 10 fast features keeps the top ~25 candidates per S1 record.
- The kept set is the final candidate set. It is written to `candidate_pairs.tsv` and is exactly what the main model scores.

### Targets (main split)
- Blocking recall of 98% or more after the merge, and 97% or more after the cheap cut.
- About 25 candidates per S1 record on average.

## 6. Features
Features are computed per (S1, candidate) pair.

**Country is not a feature.** Script type is used instead.

- **Name:**
  - Jaro-Winkler, Levenshtein, and token set/sort/partial ratios, on both the core name and the raw name
  - char and word TF-IDF cosine
  - legal form: agree, conflict or missing
  - first word equal
  - numbers in the name match
  - length ratio
- **Address:**
  - postal code: equal or missing
  - house number: equal, conflict or missing
  - street-word Jaccard
  - city similarity
  - state equal
  - address TF-IDF cosine
  - missing-field flags
- **Context:**
  - the candidate's rank and score gap within its S1 record's list
  - how many S1 records claim this candidate, and whether the two are each other's best match
  - how many candidates the S1 record has
  - **cluster support**: similarity to the other high-scoring candidates of the same S1 record
- **Source:** S2 or S3.
- **Model outputs** (later stages): bi-encoder cosine (Stage 2) and cross-encoder probability (Stage 3).

## 7. Models
- **Stage 1: LightGBM**
  - Binary classifier on final candidate pairs; label = the pair appears in the ground truth.
  - Early stopping on the 20% split.
- **Stage 2: bi-encoder**
  - `intfloat/multilingual-e5-small` (MIT, 118M), input text `"name | address"`.
  - Fine-tuned with in-batch negatives plus hard negatives from blocking, for 1–2 epochs on about 2M pairs (roughly 1–2 hours on a T4).
  - Used for embedding search and as a feature.
- **Stage 3: cross-encoder**
  - `xlm-roberta-base` (MIT), fine-tuned on pairs.
  - Runs only on pairs whose Stage-1/2 probability falls in the uncertain band (about 10–15% of pairs).
  - A final combining LightGBM uses its output together with all other features.

## 8. Decision rule (`decide.py`)
1. **Calibrate:** isotonic calibration of pair probabilities on the validation split.
2. **One owner:** each S2/S3 record is assigned to at most one S1 record, the one with the highest probability. This works because S1 is deduplicated.
3. **Choose the match set:** for each S1 record, sort its remaining candidates by probability and choose the top-k (k ≥ 0) that maximizes **expected F0.5**.
   - The empty set scores 1.0 only if every candidate is a non-match, so its expected score is ∏(1−pᵢ).
   - This is compared against a single global threshold, and the better one on validation is kept.
4. **France check:**
   - Measure how the optimal threshold or decision shifts in the US→India experiment, and apply that adjustment to unseen countries if it helps India.
   - Monitor France's predicted match rate and set-size distribution on test against US and India.

## 9. Error handling and checks
Checks run after each stage and stop the run on failure:
- All test S1 IDs are present (France included), with exactly one row each.
- Candidate and match lists contain only S2/S3 IDs from the correct split.
- No duplicate IDs.
- Matches ⊆ candidates.
- Row counts are unchanged.

`submit.py` refuses to package unless `validate_submission.py` returns PASS.

## 10. Testing
- **pytest unit tests:**
  - normalization, with fixtures taken from real sample rows (Hindi and Kannada text, reordered addresses, legal forms in odd positions)
  - macro F0.5, including the PDF worked example = 0.714 and singleton scoring
  - expected-F0.5 subset choice, on hand-made probability sets
  - one-owner rule
  - TSV writer: synthetic output must pass the official validator
- **Integration test:** the full pipeline on the dev slice, finishing in a few minutes locally.

## 11. Milestones
- **M0: data and validation.**
  - Load to parquet.
  - Check whether matches ever cross countries.
  - Build the splits and dev slice.
  - Implement metrics and the experiment log.
- **M1: first submission.**
  - Normalization, key and TF-IDF blocking, cheap cut, features, LightGBM, decision rule.
  - Leaderboard upload, to compare the local score with the public leaderboard.
- **M2:** bi-encoder, used for blocking and as a feature.
- **M3:** cross-encoder on the uncertain band, plus the combining LightGBM.
- **M4:**
  - Error analysis by country and noise type.
  - Feature additions.
  - Final tuning of the decision rule and France check.
  - Package, with the documentation written gradually from M1 onward.

## 12. Out of scope
- LLM judges.
- Any external data.
- Large GBM ensembles, unless M4 error analysis shows they're needed.
