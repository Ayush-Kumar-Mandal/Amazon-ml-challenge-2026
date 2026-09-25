# %% [markdown]
# # ber: Kaggle driver
# Inputs to attach: dataset `ber-raw` (the organizer zip; Kaggle auto-extracts it) and, when resuming,
# the output of an earlier committed version of this notebook (set PREV below).
# Secrets: GITHUB_TOKEN is optional. The repo is public, so cloning works without it; if a
# fine-grained token (read-only "Contents") is added as a Kaggle Secret named GITHUB_TOKEN, it is
# used to raise the GitHub API rate limit for the clone, but nothing here requires it.

# %% setup
import os
import subprocess
import sys

from kaggle_secrets import UserSecretsClient

GH_REPO = "Ayush-Kumar-Mandal/Amazon-ml-challenge-2026"
BRANCH = "impl/m0-m1"
CODE = "/kaggle/working/ber"
try:
    token = UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception:
    token = None  # no GITHUB_TOKEN secret configured; the repo is public, so clone without one
if not os.path.exists(CODE):
    url = f"https://{token}@github.com/{GH_REPO}.git" if token else f"https://github.com/{GH_REPO}.git"
    subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, url, CODE], check=True)
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

# %% pinned requirements (run once per milestone; download and commit as requirements.txt)
subprocess.run(
    f"{sys.executable} -m pip freeze | grep -iE '^(polars|pyarrow|numpy|scipy|scikit-learn|joblib|rapidfuzz|"
    "jellyfish|unidecode|indic.transliteration|sparse.dot.topn|lightgbm|pyyaml|faiss.*|torch|transformers|"
    "sentence.transformers|datasets|accelerate)==' > /kaggle/working/requirements.txt", shell=True, check=True)
