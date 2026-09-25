# Kaggle runbook — deferred human steps

Everything in `src/`, `configs/`, `scripts/`, `tests/` and `notebooks/kaggle_driver.py` is done
and unit-tested. What's left needs **your** Kaggle account, GPU quota and the challenge portal —
the SDD implementers could not do it. This is the checklist, in the order to run it, with the
exact cells/commands, what to write into `memory.md`, and the keep/drop rules.

Conventions used below:
- **Cell** refers to a `# %%`-marked section of `notebooks/kaggle_driver.py`, pasted into Kaggle
  as one notebook cell per marker.
- `PREV` is the list variable in the driver's `# %% config` cell — set it before "Save & Run All"
  each time a session needs a prior session's `work_dir` (Kaggle exposes it at
  `/kaggle/input/<version-slug>/work` once you attach that version's output as an input).
- Every stage writes to `paths.experiments_csv` (`/kaggle/working/experiments.csv` in
  `configs/kaggle.yaml`) — this is your source of truth for `f05` numbers; copy rows from it
  into `memory.md`, don't retype numbers from the printed JSON.

---

## Task 18, steps 3–7: first Kaggle run, first leaderboard score (M1)

### Step 3: one-time Kaggle setup

1. **Dataset.** Kaggle -> Datasets -> New Dataset. Upload the organizer zip
   (`6ab10eb3b23ba_student_resource.zip`, repo root, gitignored). Name it `ber-raw`. Keep it
   **Private** (it's the competition's own data, not your code).
2. **GITHUB_TOKEN (optional).** The repo is public
   (`github.com/Ayush-Kumar-Mandal/Amazon-ml-challenge-2026`), so the driver's setup cell clones
   it without a token (`try/except` around `UserSecretsClient().get_secret("GITHUB_TOKEN")` falls
   back to a tokenless clone URL). You do **not** need to add this secret. If you want to raise
   GitHub's anonymous API rate limit anyway, create a fine-grained token with read-only
   "Contents" access and add it under the notebook's Add-ons -> Secrets as `GITHUB_TOKEN`; the
   driver picks it up automatically and never persists it in saved output.
3. **Notebook.** Create a new Kaggle notebook. Paste `notebooks/kaggle_driver.py` in, one cell
   per `# %%` marker. Attach `ber-raw` as an input dataset.
4. **Run the `# %% setup` cell** and check its output:
   - `ls /kaggle/input` should show `ber-raw`.
   - Find where `test_source1.tsv` actually lands under it, and compare against
     `configs/kaggle.yaml -> paths.raw_dir`
     (`/kaggle/input/ber-raw/student_resource/dataset`). If they don't match, edit
     `configs/kaggle.yaml` locally, commit, push to `impl/m0-m1`, then re-run the setup cell
     (it re-clones only if `/kaggle/working/ber` doesn't already exist — delete it first, or
     start a fresh notebook version, to pick up the fix).
   - Record the `df -h /kaggle/working /kaggle/tmp` output in `memory.md` -> Environment,
     replacing the placeholder line under "Kaggle paths".

### Step 4: session A

Run cell `# %% session A (CPU notebook): data, lexicon, normalization, blocking, cheap cut` (and
the `# %% config` cell above it, with `PREV = []`) as **Save Version -> Save & Run All**, version
name **`ber-m1-a`**. When it finishes, copy the per-stage wall-clock lines
(`[pipeline] <stage> --split <split> done in ...s`) from the log into `memory.md` -> Stage
timings, replacing the `(est.)` column for each stage that just ran on the full scale.

**Also do the two Task 8/9 re-inspection items now, while the full-train artifacts exist:**

- **Full-train lexicon re-inspection (Task 8 minor-deferred item).** Download
  `work/train/lexicon.json` from the `ber-m1-a` output (or `cat` it in a scratch cell). The dev
  run found 6 residual bad `name` entries, one of them harmful because it turns a core business
  name into a legal-form token: `arihant -> private`. Check whether `arihant -> private` (or any
  other name-token -> legal-form-token mapping) is still present at full scale — it's harmful
  because every business actually named "Arihant ..." would have that token rewritten to
  "private", corrupting `name_core` for every downstream stage. If it's still there:
  - it's evidence the alignment heuristic's short-token noise (documented in `memory.md` Task 8)
    doesn't disappear with more data, so record the full-train `name`/`addr` lexicon sizes and
    the harmful-entry count in `memory.md` -> Task 18 section;
  - do **not** hand-patch `lexicon.json` (it's a generated artifact); if the entry is clearly
    hurting F0.5 in the Task 24 error analysis later, that's the point to reconsider
    `lexicon.min_share` or add a short manual exclude-list to `mine_lexicon`, not now.
- **`stage_normalize` multi-slice path (Task 9 minor-deferred item).** The dev run's
  `normalize` stage was small enough that `normalize.n_jobs` slicing may not have exercised more
  than one slice. At full scale it will. Check the `[normalize]` log lines for both `train` and
  `test`: confirm the row counts printed for `s1`/`right` match the ingest counts (2,206,821 /
  10,320,219 for train; 1,732,544 / 9,969,589 for test — see `memory.md` -> Quick facts) with no
  rows silently dropped across slice boundaries. Record the confirmation (or the discrepancy) in
  `memory.md`.

### Step 5: session B

1. Start a new notebook version with `ber-m1-a`'s output attached as an input dataset.
2. In the `# %% config` cell, set `PREV = ["/kaggle/input/<ber-m1-a-slug>/work"]` (check the
   exact slug with `ls /kaggle/input`).
3. Run cell `# %% session B (CPU notebook, PREV = session A output): features, model, evaluation, submission`
   as **Save & Run All**, version name **`ber-m1-b`**.

Expected: `evaluate` prints JSON with `f05`; `submit` prints
`PASS — no blocking issues found. Safe to submit.`. Record `f05`, the per-country breakdown, and
the `train`/`evaluate`/`predict`/`submit` stage timings in `memory.md` -> Results and -> Stage
timings.

### Step 6: leaderboard

1. Download `output/matching_results.tsv` from the `ber-m1-b` version's output.
2. Submit it in the challenge Portal.
3. Record the public LB score next to the local `f05` in `memory.md` -> Results. The gap between
   them is how well local validation calibrates to the real leaderboard — write it down even if
   it's small.

### Step 7: freeze requirements, start Documentation.md

1. Run cell `# %% pinned requirements (run once per milestone; download and commit as requirements.txt)`
   (in the `ber-m1-b` session, or any session with the full dependency set installed). Download
   `/kaggle/working/requirements.txt` and put it at the repo root.
2. Create `docs/Documentation.md` by copying `docs/Documentation_template_original.md`.
3. Fill in section 2.1 (EDA facts — pull straight from `memory.md` -> EDA findings), section 3
   (blocking keys, TF-IDF views, candidate counts, recall — from `memory.md` Task 12/13) and
   section 4 (feature list, LightGBM, decision rule — from `memory.md` Task 14/15/18), using the
   numbers already measured. Leave sections 1, 5 and 6 and Appendix A for Task 25 step 4, once
   the final (possibly M2/M3) configuration is chosen.

### Commit

```bash
git add notebooks/kaggle_driver.py docs/Documentation.md requirements.txt configs/kaggle.yaml memory.md
git commit -m "chore: Kaggle driver, pinned requirements, M1 results"
git push
```

---

## Task 19: country-holdout baseline (US -> India), the France stand-in

This is the run every later milestone (M2, M3, tuning) must beat on **both** schemes before
being kept — do it right after M1, before starting M2.

### Step 1: run the holdout sessions

- In the driver's `# %% config` cell, set `CONFIG = "configs/kaggle_holdout.yaml"` and
  `PREV = ["/kaggle/input/<ber-m1-a-slug>/work"]`.
- Run these stages **in this exact order**, all with `--split train` (i.e. call `stage(name, "train")`
  for each): `split`, `lexicon`, `normalize`, `block`, `cheap_train`, `cheap_apply`, `features`,
  `train`, `evaluate`.
- **Skip `ingest`** — `s1.parquet`/`right.parquet`/`gt.parquet` are read straight from `PREV`
  (M1's ingest output; the raw data doesn't change between schemes).
- **Never skip any of the other stages.** `read_path` silently falls back to `PREV` when a
  stage's own output is missing, so skipping `lexicon` (say) would silently reuse M1's
  `s1_random`-scheme lexicon instead of one mined from US-only training data — a leakage bug that
  produces no error, just a wrong (too-optimistic) `f05`.
- Split across two "Save & Run All" versions (`ber-m1-holdout-a`, `ber-m1-holdout-b`) if the
  12-hour session limit is a concern; there's no different session-A/B stage split required here,
  just wherever you choose to cut.

### Step 2: record results and apply the France rule

1. Copy the holdout row of `/kaggle/working/experiments.csv` into `memory.md` -> Results.
2. **Read `f05` and `f05_scale_1.0` from that row — there is no `f05_India` column.** (The
   holdout scheme's `evaluate` stage reports overall `f05` at the best unseen-country scale
   alongside `f05_scale_<value>` columns for each candidate scale in `decide.py`'s `SCALES`; it
   does **not** break the holdout result out by country the way the `s1_random` scheme's
   `evaluate` reports `f05_India`/`f05_US` — see `memory.md` Task 17 pitfall notes.)
3. If `f05_scale_<best>` minus `f05_scale_1.0` is >= 0.005, set
   `decide.unseen_country_scale: <best_unseen_scale>` in `configs/kaggle.yaml` (not the holdout
   config — this value is meant to protect the real unseen country, France, in the actual test
   run). Otherwise leave it at `1.0`.
4. Write the decision and the exact numbers behind it under `memory.md` -> Key decisions.

### Commit

```bash
git add memory.md configs/kaggle.yaml
git commit -m "chore: country-holdout baseline and unseen-country scale decision"
git push
```

---

## Task 21, steps 7–8: M2 bi-encoder run (optional stage)

The driver already has the M2 cells (`# %% M2 GPU smoke test on the dev slice ...`,
`# %% M2 GPU session ...`, `# %% M2 CPU session ...`) — no notebook edits needed here.

### Step 7: run M2 on Kaggle, both schemes are required

1. **Smoke test** — run `# %% M2 GPU smoke test on the dev slice (PREV = ber-m1-a output)` on a
   GPU notebook (`PREV = ["/kaggle/input/<ber-m1-a-slug>/work"]`). It must finish without errors
   before you spend GPU-hours on the full run.
2. **GPU session** — `# %% M2 GPU session (PREV = [ber-m1-a work, ber-m1-b work])`, version
   `ber-m2-a`, accelerator "GPU T4 x2", `PREV` = both M1 work dirs.
3. **CPU session** — `# %% M2 CPU session (PREV = M2 GPU output + M1 outputs)`, version
   `ber-m2-b`, `PREV` = the M2 GPU output plus the M1 work dirs.
4. **Repeat steps 1–3 for `configs/kaggle_holdout.yaml`** (`ber-m2-holdout-a`/`-b`, `PREV` = the
   Task 19 holdout M1 outputs). There, `embed_train` sees only US `fit` S1 records, so India
   stays genuinely unseen — this is what makes the holdout scheme a valid France stand-in.

### Step 8: keep/drop rule

1. Compare `f05` and `recall_cands_all` for the `kaggle_m2` run against the M1 run, in **both**
   `experiments.csv` rows (`s1_random` from `configs/kaggle.yaml`, `country_holdout` from
   `configs/kaggle_holdout.yaml`).
2. **Keep M2 only if `f05` improved in both schemes.**
   - If kept: set `blocking.embed.enabled: true` in **both** `configs/kaggle.yaml` and
     `configs/kaggle_holdout.yaml`; upload the M2 `matching_results.tsv` to the leaderboard;
     record the public LB score in `memory.md`.
   - If not kept: leave `blocking.embed.enabled: false`, and write down the actual numbers (not
     just "didn't help") under `memory.md` -> Key decisions, so Task 24's error analysis isn't
     repeated against a stage that was already ruled out.
3. Update `docs/Documentation.md` sections 3 (blocking) and 4 (matching model) with the M2 numbers
   either way — a documented negative result is still a result.

### Commit

```bash
git add notebooks/kaggle_driver.py configs memory.md docs/Documentation.md
git commit -m "chore: M2 bi-encoder results"
git push
```

---

## Task 23, step 6: M3 cross-encoder + combiner run (optional stage)

Same pattern as M2. The driver already has `# %% M3 GPU smoke test (dev) ...`,
`# %% M3 GPU session (PREV = latest train/test outputs) ...`, `# %% M3 CPU session ...`.

1. **Smoke test** on a GPU notebook, PREV pointing at the dev pipeline's own output (the smoke
   cell runs `features`/`train` on `dev` first, then `cross_train`/`cross_score` — it's
   self-contained given the dev slice already exists from M1/M2).
2. **GPU session** `ber-m3-a` (GPU T4 x2, roughly 2–3 hours per the plan's own estimate for the
   cross-encoder fine-tune — there is no local GPU to re-derive this from, so time it for real
   here and record the actual number).
3. **CPU session** `ber-m3-b` (combiner train/evaluate/predict/submit).
4. **Repeat 1–3 for `configs/kaggle_holdout.yaml`.** There, `cross_train` learns only from US
   `fit` pairs, and `combine_train` trains its out-of-fold folds on `valid_seen` rows only —
   confirm this by checking the `GroupKFold` row count printed by `stage_combine_train` against
   the holdout scheme's `valid_seen` fold size in `folds.parquet`.
5. **Keep M3 only if `f05` improves over the best previously-kept stage (M1, or M2 if it was
   kept) in both schemes.**
   - If kept: point `evaluate.pred_file`/`submit.pred_file` at the combined files
     (`valid_pred_combined.parquet`/`pred_combined.parquet`) in `configs/kaggle.yaml`; upload to
     the leaderboard; record the score.
   - Also record the cross-encoder's band share (the fraction of pairs with
     `cross.band_lo <= p <= cross.band_hi`, target 10–15%). If it's off, adjust
     `cross.band_lo`/`cross.band_hi` and re-run `cross_score` -> `combine_train` -> `evaluate`
     before deciding keep/drop.
6. Update `memory.md` and `docs/Documentation.md` section 4 either way.

### Commit

```bash
git add notebooks/kaggle_driver.py configs memory.md docs/Documentation.md
git commit -m "chore: M3 cross-encoder + combiner results"
git push
```

---

## Task 24, steps 2–5: error analysis, tuning, France check, final pick

### Step 2: run the error analysis on the best (kept) configuration

Add a one-off cell to the driver session that has the final kept configuration's outputs
(M1, M2 or M3, whichever was kept):

```python
COMB = ["evaluate.pred_file=valid_pred_combined.parquet", "submit.pred_file=pred_combined.parquet"] if M3_KEPT else []
subprocess.run([sys.executable, "scripts/error_analysis.py", "--config", CONFIG, "--split", "train",
                *sum([["--set", o] for o in COMB], [])], check=True)
```

(`--split train` is explicit here because `scripts/error_analysis.py` takes a `--split` flag,
default `"train"`, added so the script could be smoke-tested locally on `dev` — see `memory.md`
Task 24 dev-smoke entry. On Kaggle you always want `train`, which is also the default.) This
writes `reports/error_analysis_<run_name>.md` inside the Kaggle session's own
`/kaggle/working/ber/reports/` directory.

**Reports stay LOCAL to you — never commit them (Ruling R13).** The repo is public, and the
report's "sample false positives"/"sample false negatives" sections embed near-verbatim
competition record text (names/addresses), which would redistribute organizer data. `reports/`
is already in `.gitignore` — do not `git add -f` it, and there is no step below that stages it.
Download the `.md` file from the Kaggle output pane to your own machine and read it there. Then
write only a **pattern-level** summary into `memory.md` (no verbatim name/address text) — follow
the exact style already used for the dev-smoke run in `memory.md` -> Task 24 (three bullet
points: dominant FN-blocking-miss pattern, dominant FN-scored-below pattern, dominant FP pattern,
each with the group-by counts but described by pattern, not quoted).

### Step 3: tuning sweeps

Pick the rows in the report that match these patterns and run the corresponding sweep. Each
sweep is its own "Save & Run All" version with `PREV` set to the latest kept outputs, re-running
only the stages downstream of the changed knob. Every run logs a new row to `experiments.csv`
(both schemes, if the changed knob touches blocking).

**Keep rule:** keep a change only if `f05` improves on `s1_random` **and** does not drop by more
than 0.002 on `country_holdout`.

| Dominant pattern | Knob (`--set`) | Stages to re-run |
|---|---|---|
| FN `not_in_candidates` above 3% of GT pairs | `blocking.tfidf.top_k_fwd=60`, then `cheap.keep_top=35` | `block` -> ... -> `evaluate` |
| FN `scored_below` on multi-match S1s | `features.max_train_s1=1200000`, `gbm.params={num_leaves: 255}` | `features` (train) -> `train` -> `evaluate` |
| FP concentrated on same-chain names with different numbers | `gbm.params={min_data_in_leaf: 500}`; check that `house_eq`/`num_jaccard` rank in the top 15 importances (`[train] top features` log line) | `train` -> `evaluate` |
| Singleton accuracy below 0.9 | compare `f05_expected` against `f05_threshold` in `experiments.csv` — the better rule is already auto-picked (`decide.select`); just record the gap | none |

### Step 4: France check

1. Re-run `submit` with the final (post-tuning) config, then re-run the error-analysis cell from
   step 2.
2. In the report's "Test predictions by country" table, France's `mean_matches` and
   `empty_share` should fall within the range spanned by US and India, +/- 25%. (Reference:
   train ground truth averages about 3.5 matches per S1 and a 5.6% singleton rate — see
   `memory.md` -> Quick facts.)
3. If France is outside that range, revisit the Task 19 `best_unseen_scale` decision: submit one
   version with `decide.unseen_country_scale` set to that value (even if Task 19's >= 0.005 gate
   didn't fire) and compare the two public LB scores.
4. Record the outcome (in range / out of range / rescale tried) in `memory.md`.

### Step 5: final pick and upload

1. Choose the configuration with the best `s1_random` `f05` that does not regress on
   `country_holdout` (i.e. whatever kept-stage + kept-tuning combination survived steps 8/6/3
   above).
2. Upload its `matching_results.tsv`, record the LB score in `memory.md`.
3. Fill `docs/Documentation.md` section 5 (F0.5 for both schemes, public LB, common FP/FN
   patterns — pattern-level language, same R13 constraint as step 2 above).

### Commit

```bash
git add scripts/error_analysis.py configs memory.md docs/Documentation.md notebooks/kaggle_driver.py
git commit -m "chore: M4 error analysis, tuning and France check"
git push
```

(`reports/` is deliberately not in this `git add` list — see the R13 note in step 2.)

---

## Task 25, steps 4–5: fill Documentation.md, build the final zip, submit

These are the two steps this dispatch explicitly left undone (controller ruling R14), now that
you have real Kaggle results from the steps above.

### Step 4: finalize `docs/Documentation.md`

Fill every remaining template section (team/date, executive summary, problem analysis are
already partly filled from Task 18 step 7):
- **Team/date, executive summary:** 2–3 sentences, plus the submission date.
- **Problem analysis:** the EDA numbers already in `memory.md` -> EDA findings.
- **Strategy:** "Hybrid: blocking + GBM" and, if kept, "+ bi-encoder" / "+ cross-encoder".
- **Blocking:** keys, TF-IDF views, embedding k-NN (if kept), the cheap cut, total candidate
  pairs on test, recall on validation — all already measured, in `memory.md` Tasks 12/13/21.
- **Matching model:** the full feature list grouped by name/address/context (`FEATURE_COLUMNS`
  in `src/ber/features.py`), the models used, calibration, the one-owner rule, the
  expected-F0.5 decision rule.
- **Results:** validation F0.5 for both schemes, the public LB score(s), FP/FN patterns from the
  Task 24 report (pattern-level, not verbatim).
- **Conclusion.**
- **Appendix A:** code layout and entry points — reuse the `README.md` §7 table and §5 command
  list rather than re-deriving them.

Also, **update `README.md` §5** at this point: replace each "(ESTIMATED ...)" runtime with the
real measured wall-clock time from the Kaggle session logs recorded in `memory.md` -> Stage
timings, and drop the "ESTIMATED" qualifier once real numbers are in.

### Step 5: build and verify the final zip

On Kaggle (in the session holding the final kept configuration's `work_dir` and `output/`), add:

```python
stage("package", "test")
```

or, after downloading that session's `output/` directory locally and checking out the matching
commit:

```bash
.venv/Scripts/python -m ber.pipeline --config configs/kaggle.yaml --stage package --split test --set paths.prev_work_dirs=[<final-session-work-dir>]
```

Then verify the zip:

```bash
.venv/Scripts/python -c "import zipfile,sys; z=zipfile.ZipFile(sys.argv[1]); print('\n'.join(n for n in z.namelist() if n.count('/')<=3))" <path-to>/<team>_submission.zip
.venv/Scripts/python -m ber.vendor.validate_submission --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir <test dir>
```

Expected: the top-level listing shows `output/`, `code/business_entity_resolution/` and
`Documentation_template.md`; the validator prints
`PASS — no blocking issues found. Safe to submit.`. Submit the zip through the portal. Record the
submission date in `memory.md`.

### Commit

```bash
git add docs/Documentation.md README.md memory.md
git commit -m "docs: fill Documentation.md with final Kaggle results, update README runtimes"
git push
```

(`src/ber/submit.py`, `src/ber/stages/output.py`, `src/ber/pipeline.py`, `tests/test_package.py`
and the rest of `README.md` were already committed in the task-25 code dispatch — this final
commit only carries the Kaggle-results fill-in.)

---

## Before the final merge

The controller ledger (`.superpowers/sdd/plan/progress.md`) tracks small deferred items from
code review that were accepted as low-cost/low-risk rather than fixed immediately. None of them
block a submission, but they're worth a look before merging `impl/m0-m1` into `main`. Grouped by
area:

**Config / pipeline plumbing**
- `config.load_config` has no cycle detection on the `extends:` chain (Task 1).
- `docs/Documentation_template_original.md` wasn't listed in Task 1's brief `Files:` (authoring
  nit only).

**Metrics (`src/ber/metrics.py`)**
- `macro_f05_frame` does not `.unique()` the ground truth (asymmetric vs. how it treats
  predictions) (Task 3).
- `ev = eval_l.alias(...).to_frame().unique()` is repeated 3x instead of factored out (Task 3).
- `macro_f05_frame` on an empty `eval_l` returns `None`, unguarded (Task 3).

**Split / dev slice (`src/ber/split.py`)**
- `make_dev_slice`'s locality filter is an unanchored substring match (Task 5).
- `stage_dev_slice` ignores its `split` argument by design; the code has no comment saying so
  (Task 5).

**Text / lexicon / normalize (`src/ber/text.py`, `lexicon.py`, `normalize.py`)**
- `map_tokens` uses `seed.get(t) or lex.get(t, t)` — a falsy mapped value (e.g. an empty string)
  would fall through to the next lookup instead of being used (Task 6).
- No test for a lone `"n"` token surviving the address placeholder filter as `"north"` (Task 7).
- The placeholder-token filter also strips the genuine locality tokens `"Nil"`/`"NA"` — an
  accepted tradeoff under Ruling R8 (Task 7).
- `stage_lexicon` reads `cfg["lexicon"]` twice (Task 8).
- 6 residual noisy `name`-lexicon entries survive on the dev slice (`man->maa`, `hai->high`,
  `vest->best`, `vig->big`, `arihant->private`, `vrait->private`); `arihant->private` is
  potentially harmful — **this is the item the Task 18 section above asks you to re-check on the
  full-train lexicon.**
- `stage_normalize`'s multi-slice path was untested on dev (single slice covered it) —
  **this is the second item the Task 18 section above asks you to verify on the full train run.**

**Blocking (`src/ber/blocking/*.py`)**
- `_keys_for_slice` is a per-row Python loop — flagged as a throughput watch item for the
  full-scale Kaggle run (Task 10). Worth checking the `block` stage's actual wall time against
  the extrapolation once you have it.
- `sorted(country set)` would raise `TypeError` if a `None` country ever appeared (the convention
  is `""`, never `None`, so currently unreachable) (Task 11).
- The `same_country=False` code path is untested (Task 11).
- `merge_candidates`' `frames[:1]` fallback would `IndexError` on an empty frame list — currently
  unreachable since at least one blocking view always runs (Task 12).

**Features / models (`src/ber/features.py`, `src/ber/models/gbm.py`)**
- `train_binary`'s `first_metric_only=True` fix (Task 13) relies on `binary_logloss` being first
  in `params["metric"]` — this dependency isn't documented in a code comment.
- `memory.md`/an earlier report say "46 feature columns"; the real count is 44 features plus
  `l_idx`/`r_idx` (Task 14) — a documentation-only discrepancy, already corrected in this
  README's Repository-layout section.
- `sim_other_*` features compute a self-comparison for rows with no "other" candidate, then
  discard it — wasted work, not a correctness bug (Task 14).
- `tests/test_gbm.py` has a mid-file duplicate import of `ber.models.gbm` (plan-mandated style)
  (Task 23).

**Decision rules (`src/ber/decide.py`)**
- `choose_expected_f05`'s tie-break via `sort_by("ef").last()` isn't guaranteed stable by the
  polars API, though it's score-neutral either way (Task 15).
- `decide.select` treats any method string other than `"expected_f05"` as "threshold", with no
  validation of the value (Task 15).
- No isolated unit test for `fit_calibrator`/`calibrate` (only exercised indirectly) (Task 15).

**Submission (`src/ber/submit.py`)**
- `ensure_test_dir`'s zip-fallback branch (no `raw_dir` configured) has no direct unit test —
  only the `raw_dir` branch is covered, via the Task 17 end-to-end test (Task 16).
- **Resolved by Task 25:** the Task 16 item "`submit.py` docstring mentions building the zip
  before T25 adds it" no longer applies — Task 25 added `build_package` to `src/ber/submit.py`,
  so the module docstring's "...build the final zip" is now accurate, not forward-looking.

**End-to-end tests (`tests/test_end_to_end.py`)**
- The synthetic e2e fixture is perfectly separable (AUC = 1.0), so its quality asserts can't
  catch a calibration or decision-rule regression; adding same-name distractors would strengthen
  it (Task 17).
- The e2e test only covers the `s1_random` scheme; parametrizing it over `country_holdout` would
  catch holdout-specific regressions earlier (Task 17).
- No focused regression test for the crossfit-calibration seed salt (`np.random.default_rng([seed, 1])`)
  fixed in Task 17 — a comment noting the salt must stay non-zero would help (Task 17).
- `s1_random` early-stops the GBM on the valid fold, a slight optimistic bias — this is
  plan-mandated, not a bug (Task 17).
- `evaluate`'s one-owner rule only sees valid-fold rows, while `submit` applies it to all of
  test S1 — a scale difference to keep in mind when comparing `evaluate`'s printed metrics
  against the real submission (Task 17).
- An unused `numpy` import in `src/ber/stages/model.py` (Task 17).
- `stage_predict` re-scans the source `LazyFrame` once per output chunk instead of once overall
  (Task 17).

**Embeddings (`src/ber/models/biencoder.py`, `src/ber/blocking/embed_ann.py`, `src/ber/stages/nn.py`)**
- No counter for how often `make_training_triplets` falls back to a random negative instead of a
  hard mined one (Task 20).
- `encode_to_memmap` hardcodes `device="cuda"` with no clear error message if no GPU is present
  (Task 20).
- `stage_embed_attach` calls `np.load` without `mmap_mode` — fits in Kaggle's ~29 GB RAM at the
  competition's scale, but `mmap_mode="r"` would be cheaper (Task 21).

**Logbook conventions**
- The `[~]` partial-status marker in `memory.md` -> Status is a convention introduced mid-project
  (Task 18), not one `plan.md` itself defines — harmless, but worth a one-line note if `main`
  ever needs the convention explained to someone new.

None of the above block a submission. Fix opportunistically, or leave them — they're all
"accepted as-is" in the controller ledger, not open defects.
