import zipfile
from pathlib import Path

import polars as pl

from ber.io import ingest, read_tsv

S1 = "entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-1\tAcme \"Best\" Inc\t1 Main St, Tyler, TX\tUS\nS1-2\tSolo Co\t\tUS\n"
S2 = "entity_id\tbusiness_name\tbusiness_address\tcountry\nS2-1\tACME BEST INC\t1 MAIN ST, TYLER\tUS\n"
S3 = "entity_id\tbusiness_name\tbusiness_address\tcountry\nS3-1\tAcme Best\tTyler, TX\tUS\nS3-2\tOther\tX\tUS\n"
GT = "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S3-1\nS1-2\t\n"


def _zip(tmp_path: Path) -> Path:
    path = tmp_path / "raw.zip"
    with zipfile.ZipFile(path, "w") as z:
        base = "student_resource/dataset/train/"
        z.writestr(base + "train_source1.tsv", S1)
        z.writestr(base + "train_source2.tsv", S2)
        z.writestr(base + "train_source3.tsv", S3)
        z.writestr(base + "train_ground_truth.tsv", GT)
    return path


def test_read_tsv_keeps_quotes_and_empty_strings():
    df = read_tsv(S1.encode())
    assert df["business_name"][0] == 'Acme "Best" Inc'
    assert df["business_address"][1] == ""


def test_ingest_from_zip(tmp_path: Path):
    cfg = {"paths": {"raw_zip": str(_zip(tmp_path)), "raw_dir": "", "work_dir": str(tmp_path / "w")}}
    ingest(cfg, "train")
    s1 = pl.read_parquet(tmp_path / "w/train/s1.parquet")
    right = pl.read_parquet(tmp_path / "w/train/right.parquet")
    gt = pl.read_parquet(tmp_path / "w/train/gt.parquet")
    assert s1["idx"].to_list() == [0, 1] and s1["idx"].dtype == pl.UInt32
    assert right["source"].to_list() == ["S2", "S3", "S3"]
    assert right["entity_id"].to_list() == ["S2-1", "S3-1", "S3-2"]
    assert sorted(zip(gt["l_idx"].to_list(), gt["r_idx"].to_list())) == [(0, 0), (0, 1)]
