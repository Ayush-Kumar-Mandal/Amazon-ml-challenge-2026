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
