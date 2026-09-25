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

# %% pinned requirements (run once per milestone; download and commit as requirements.txt)
subprocess.run(
    f"{sys.executable} -m pip freeze | grep -iE '^(polars|pyarrow|numpy|scipy|scikit-learn|joblib|rapidfuzz|"
    "jellyfish|unidecode|indic.transliteration|sparse.dot.topn|lightgbm|pyyaml|faiss.*|torch|transformers|"
    "sentence.transformers|datasets|accelerate)==' > /kaggle/working/requirements.txt", shell=True, check=True)
