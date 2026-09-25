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
    cfg["blocking"]["tfidf"].update(min_df=1, max_df_frac=1.0, max_df_abs=10**9, threshold=0.1,
                                    n_threads=1, chunk_size=50)
    cfg["cheap"].update(max_train_s1=None, params={"num_leaves": 15, "learning_rate": 0.1, "min_data_in_leaf": 5})
    cfg["features"].update(max_train_s1=None, vectorizer_fit_rows=100_000)
    cfg["gbm"].update(num_boost_round=300, early_stopping_rounds=30, max_es_s1=None,
                      params={"num_leaves": 15, "min_data_in_leaf": 5})
    return cfg
