import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    "litpcba_external", Path(__file__).parents[2] / "benchmarks/litpcba_external.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_preserves_sid_split_and_rejects_malformed():
    rows = MODULE.parse_smi("CCO 123\nCCC 456\n", "PPARG", "active_T.smi")
    assert [r["source_sid"] for r in rows] == ["123", "456"]
    assert rows[0]["source_partition"] == "T"
    assert rows[0]["y_active"] == 1
    with pytest.raises(ValueError, match="line 1"):
        MODULE.parse_smi("CCO\n", "PPARG", "active_T.smi")


def test_conflicts_duplicates_and_cohorts():
    frame = pd.DataFrame({
        "assay": ["a"] * 5, "identity": ["x", "x", "y", "y", "z"],
        "y_active": [1, 1, 1, 0, 0], "source_sid": ["1", "2", "3", "4", "5"],
        "source_partition": ["T", "V", "T", "V", "T"],
        "original_smiles": ["C", "C", "CC", "CC", "CCC"],
        "murcko_scaffold": ["s", "s", "t", "t", ""], "eligible": [True] * 5,
    })
    clean, conflicts = MODULE.collapse_records(frame)
    assert clean.identity.tolist() == ["x", "z"]
    assert clean.iloc[0].source_sids == '["1", "2"]'
    assert set(conflicts.identity) == {"y"}
    flagged = MODULE.cohort_flags(clean, {"x"}, {""}, {"x"})
    assert flagged.primary_identity_novel.tolist() == [False, True]
    assert not flagged.secondary_scaffold_novel.any()
    assert flagged.overlap_b09_fitted.tolist() == [True, False]


def test_extract_rejects_traversal_and_ignores_links(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("../PPARG/active_T.smi")
        member.size = 6
        tar.addfile(member, io.BytesIO(b"CCO 1\n"))
    with pytest.raises(ValueError, match="Unsafe"):
        MODULE.extract_smiles(archive, tmp_path / "out")
    assert not (tmp_path / "PPARG").exists()


def test_extract_regular_sources_only_and_reject_duplicate(tmp_path):
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("root/PPARG/active_T.smi")
        member.size = 6
        tar.addfile(member, io.BytesIO(b"CCO 1\n"))
        link = tarfile.TarInfo("root/PPARG/active_V.smi")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../outside"
        tar.addfile(link)
    paths = MODULE.extract_smiles(archive, tmp_path / "out")
    assert len(paths) == 1
    assert paths[0].read_text() == "CCO 1\n"
    with tarfile.open(archive, "w:gz") as tar:
        for _ in range(2):
            member = tarfile.TarInfo("root/PPARG/active_T.smi")
            member.size = 6
            tar.addfile(member, io.BytesIO(b"CCO 1\n"))
    with pytest.raises(ValueError, match="Duplicate archive"):
        MODULE.extract_smiles(archive, tmp_path / "out")


def test_download_is_hashed_and_cached(tmp_path, monkeypatch):
    calls = []

    def request(url, timeout):
        calls.append((url, timeout))
        return SimpleNamespace(content=b"evidence", raise_for_status=lambda: None)

    monkeypatch.setattr(MODULE.requests, "get", request)
    path = tmp_path / "source"
    first = MODULE.download("https://source.invalid", path)
    assert first == MODULE.download("https://source.invalid", path)
    assert len(calls) == 1
    assert first["sha256"] == hashlib.sha256(b"evidence").hexdigest()
    assert first["bytes"] == 8


@pytest.fixture
def miniature_study(tmp_path, monkeypatch):
    raw = tmp_path / "data/lit-pcba"
    raw.mkdir(parents=True)
    archive = raw / "LIT-PCBA_AVE_unbiased.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for assay in MODULE.ASSAYS:
            for name in sorted(MODULE.FILES):
                content = b"CCO 123\n" if name.startswith("active_") else b"CCC 456\n"
                member = tarfile.TarInfo(f"root/{assay}/{name}")
                member.size = len(content)
                tar.addfile(member, io.BytesIO(content))
    for assay, (_, aid) in MODULE.ASSAYS.items():
        (raw / f"{assay}_AID{aid}_description.json").write_text("{}")

    class InlinePool:
        def __init__(self, max_workers):
            assert max_workers == 1

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, function, chunks):
            return map(function, chunks)

    monkeypatch.setattr(MODULE, "ProcessPoolExecutor", InlinePool)
    exposed = hashlib.sha256(b"CCO").hexdigest()
    monkeypatch.setattr(MODULE, "exposure_pool", lambda study: ({exposed}, {""}, {exposed}, {}))
    monkeypatch.setattr(MODULE.requests, "get", lambda *a, **k: pytest.fail("network forbidden"))
    return tmp_path


def test_prepare_manifest_hashes_cohorts_and_verified_cache(miniature_study, monkeypatch):
    study = miniature_study
    MODULE.prepare(study, 1)
    output = study / "artifacts/s2-decision-external-v1/data"
    manifest_path = output / "preparation-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    dataset = pd.read_parquet(output / "dataset.parquet")
    assert len(dataset) == 4
    assert dataset.primary_identity_novel.sum() == 2
    assert not dataset.secondary_scaffold_novel.any()
    assert manifest["unique_source_smiles"] == 2
    assert manifest["invalid_records"] == manifest["conflict_records"] == 0
    for key, name in [("dataset_sha256", "dataset.parquet"),
                      ("chemistry_sha256", "chemistry.json"),
                      ("source_manifest_sha256", "source-manifest.json")]:
        assert manifest[key] == MODULE.file_hash(output / name)
    with pytest.raises(ValueError, match="Completed preparation"):
        MODULE.prepare(study, 1)
    manifest_path.unlink()
    monkeypatch.setattr(MODULE, "feature_chunk", lambda _: pytest.fail("valid cache recomputed"))
    MODULE.prepare(study, 1)
    pd.testing.assert_frame_equal(dataset, pd.read_parquet(output / "dataset.parquet"))


def test_prepare_rejects_missing_sources_and_untrusted_cache(miniature_study, monkeypatch):
    study = miniature_study
    extract = MODULE.extract_smiles
    monkeypatch.setattr(MODULE, "extract_smiles", lambda *a: extract(*a)[:-1])
    with pytest.raises(ValueError, match="Incomplete assay"):
        MODULE.prepare(study, 1)
    monkeypatch.setattr(MODULE, "extract_smiles", extract)
    MODULE.prepare(study, 1)
    output = study / "artifacts/s2-decision-external-v1/data"
    (output / "preparation-manifest.json").unlink()
    cache_manifest = output / "features-manifest.json"
    original = json.loads(cache_manifest.read_text())
    cache_manifest.unlink()
    with pytest.raises(ValueError, match="missing manifest"):
        MODULE.prepare(study, 1)
    MODULE.write_json(cache_manifest, {**original, "chemistry_hash": "different"})
    with pytest.raises(ValueError, match="chemistry, source, code, or content mismatch"):
        MODULE.prepare(study, 1)
    cache = output / "features.parquet"
    features = pd.read_parquet(cache).assign(original_smiles=["CCCC", "CCCCC"])
    features.to_parquet(cache, index=False)
    MODULE.write_json(cache_manifest, {**original, "features_sha256": MODULE.file_hash(cache)})
    with pytest.raises(ValueError, match="cache source mismatch"):
        MODULE.prepare(study, 1)


def test_exposure_union_recovers_scaffolds_and_fails_closed(tmp_path):
    data = tmp_path / "artifacts/s2-decision-multitask-v1/data"
    data.mkdir(parents=True)
    pd.DataFrame({"identity": ["fit", "discarded"], "murcko_scaffold": ["A", "B"]}).to_parquet(
        data / "features-before-identity-collapse.parquet"
    )
    pd.DataFrame({"identity": ["fit"]}).to_parquet(data / "molecules.parquet")
    historical = tmp_path / "artifacts/data-capacity-v1"
    bundle = historical / "datasets/task"
    bundle.mkdir(parents=True)
    (bundle / "records.jsonl").write_text(json.dumps({"identity": "dataset", "murcko_scaffold": "C"}) + "\n")
    historical_json = historical / "excluded-previous-identities.json"
    blind_identity = hashlib.sha256(b"CCO").hexdigest()
    MODULE.write_json(historical_json, {"identities": ["recover", blind_identity]})
    pd.DataFrame({"identity": ["recover"]}).to_csv(historical / "seen.csv", index=False)
    pd.DataFrame({"other": [1]}).to_csv(historical / "ignored.csv", index=False)
    recovery = tmp_path / "artifacts/z-recovery"
    recovery.mkdir()
    (recovery / "records.jsonl").write_text(json.dumps({"identity": "recover", "murcko_scaffold": "D"}) + "\n")
    article = tmp_path / "artigo_SMILES2Select/03_dados/benchmark_independente_v1"
    annotations = article / "prepared_001"
    annotations.mkdir(parents=True)
    blind = article / "blind.csv"
    pd.DataFrame({"smiles": ["not-a-smiles", "CCO", "CC"]}).to_csv(blind, index=False)
    MODULE.write_json(article / "historical_exposure_snapshot.json", {
        "sources": [{"path": str(blind), "kind": "blind_csv"}]
    })
    pd.DataFrame({"original_smiles": ["CCC"]}).to_csv(annotations / "source_annotations.csv", index=False)
    ids, scaffolds, fitted, manifest = MODULE.exposure_pool(tmp_path)
    assert ids == {"fit", "discarded", "dataset", "recover", blind_identity}
    assert scaffolds == {"A", "B", "C", "D", ""}
    assert fitted == {"fit"}
    assert manifest["historical_unique"] == 3
    MODULE.write_json(historical_json, {"identities": ["unrecoverable"]})
    with pytest.raises(ValueError, match="Historical scaffolds unresolved"):
        MODULE.exposure_pool(tmp_path)


def test_parser_rejects_unexpected_source_and_skips_blank_lines():
    with pytest.raises(ValueError, match="Unexpected"):
        MODULE.parse_smi("CC 1", "unknown", "active_T.smi")
    rows = MODULE.parse_smi("\nCCC 10\n\n", "PPARG", "inactive_V.smi")
    assert rows[0]["source_line"] == 2
    assert rows[0]["y_active"] == 0
