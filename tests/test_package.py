import zipfile

from ber.submit import build_package


def test_package_layout(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src" / "ber").mkdir(parents=True)
    (repo / "src" / "ber" / "__init__.py").write_text("")
    (repo / "src" / "ber" / "__pycache__").mkdir()
    (repo / "src" / "ber" / "__pycache__" / "x.pyc").write_bytes(b"0")
    for f in ("README.md", "requirements.txt"):
        (repo / f).write_text("x")
    out = tmp_path / "output"
    out.mkdir()
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        (out / f).write_text("h\n")
    doc = tmp_path / "Documentation.md"
    doc.write_text("# doc")
    z = zipfile.ZipFile(build_package(repo, out, doc, "teamx", tmp_path))
    names = set(z.namelist())
    assert {"output/matching_results.tsv", "output/candidate_pairs.tsv", "Documentation_template.md",
            "code/business_entity_resolution/src/ber/__init__.py",
            "code/business_entity_resolution/README.md",
            "code/business_entity_resolution/requirements.txt"} <= names
    assert not any("__pycache__" in n for n in names)
    assert z.filename.endswith("teamx_submission.zip")
