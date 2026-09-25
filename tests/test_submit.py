import polars as pl

from ber.submit import run_validator, write_id_lists

U = pl.UInt32
S1 = pl.DataFrame({"idx": [0, 1, 2], "entity_id": ["S1-a", "S1-b", "S1-c"]}, schema_overrides={"idx": U})
RIGHT = pl.DataFrame({"idx": [0, 1, 2], "entity_id": ["S2-x", "S3-y", "S2-z"]}, schema_overrides={"idx": U})
PAIRS = pl.DataFrame({"l_idx": [0, 0, 0, 1], "r_idx": [1, 0, 1, 2]}, schema={"l_idx": U, "r_idx": U})
SRC1 = "entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-a\ta\tb\tUS\nS1-b\ta\tb\tUS\nS1-c\ta\tb\tFrance\n"


def test_writer_format(tmp_path):
    m = tmp_path / "matching_results.tsv"
    write_id_lists(m, "matched_entity_ids", S1, RIGHT, PAIRS)
    lines = m.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "source1_entity_id\tmatched_entity_ids"
    assert sorted(lines[1:]) == ["S1-a\tS2-x,S3-y", "S1-b\tS2-z", "S1-c\t"]


def test_validator_pass_and_fail(tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    (test_dir / "test_source1.tsv").write_text(SRC1, encoding="utf-8")
    m, c = tmp_path / "matching_results.tsv", tmp_path / "candidate_pairs.tsv"
    write_id_lists(m, "matched_entity_ids", S1, RIGHT, PAIRS)
    write_id_lists(c, "candidate_entity_ids", S1, RIGHT, PAIRS)
    assert run_validator(m, c, test_dir) == 0
    write_id_lists(m, "matched_entity_ids", S1.filter(pl.col("idx") < 2), RIGHT, PAIRS)  # drops S1-c
    assert run_validator(m, c, test_dir) == 1
