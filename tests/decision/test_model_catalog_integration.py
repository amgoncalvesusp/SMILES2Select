"""Local model installation keeps untrusted packages outside the application tree."""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from s2s_decision.artifacts import file_hash, write_bundle, write_json
from s2s_decision.features import chemistry_manifest
from s2s_decision.model_catalog import (
    bundled_model_root,
    import_model_package,
    list_bundled_models,
    list_catalog_models,
    list_user_models,
    user_model_root,
)
from s2s_decision.preprocessing import Preprocessor
from s2s_decision.schema import SCHEMA_VERSION, FeatureSet


def model_package(path):
    path.mkdir()
    (path / "model.onnx").write_bytes(b"discovery checks bytes; inference is separate")
    chemistry = chemistry_manifest(2048)
    write_bundle(FeatureSet(pd.DataFrame({"record_id": [1]}), chemistry), path / "references")
    refs = json.loads((path / "references" / "manifest.json").read_text(encoding="utf-8"))
    write_json(
        path / "manifest.json",
        {
            "schema": SCHEMA_VERSION,
            "format_version": 1,
            "runtime_ready": True,
            "fingerprint_bits": 2048,
            "preprocessing": Preprocessor((0.0,) * 27, (0.0,) * 27, (1.0,) * 27, ()).to_dict(),
            "regression_mean": 0.0,
            "regression_scale": 1.0,
            "regression_supported": False,
            "onnx_sha256": file_hash(path / "model.onnx"),
            "reference_records_sha256": refs["records_sha256"],
            "provenance": {**chemistry, "target_id": "T1", "endpoint": "IC50", "threshold": 6},
            "calibrator": {"slope": 1.0, "intercept": 0.0, "status": "fitted"},
        },
    )
    (path / "MODEL_CARD.md").write_text("Experimental T1 IC50 model.\n", encoding="utf-8")
    return path


def test_import_valid_package_is_atomic_and_idempotent(tmp_path):
    source = model_package(tmp_path / "source")
    root = tmp_path / "user-models"

    installed = import_model_package(source, root=root)

    assert installed.parent == root
    assert {p.relative_to(installed).as_posix() for p in installed.rglob("*") if p.is_file()} == {
        "manifest.json",
        "model.onnx",
        "MODEL_CARD.md",
        "references/manifest.json",
        "references/records.jsonl",
    }
    assert import_model_package(source, root=root) == installed
    assert [item["compatible"] for item in list_user_models(root)] == [True]
    assert sorted(path.name for path in root.iterdir()) == [installed.name]


def test_rejects_unexpected_files_without_publishing(tmp_path):
    source = model_package(tmp_path / "source")
    (source / "run.py").write_text("raise AssertionError('never execute')", encoding="utf-8")
    root = tmp_path / "user-models"

    with pytest.raises(ValueError, match="unexpected|allowed"):
        import_model_package(source, root=root)

    assert not root.exists() or not list(root.iterdir())


def test_rejects_modified_model_checksum_without_publishing(tmp_path):
    source = model_package(tmp_path / "source")
    (source / "model.onnx").write_bytes(b"tampered")
    root = tmp_path / "user-models"

    with pytest.raises(ValueError, match="checksum"):
        import_model_package(source, root=root)

    assert not root.exists() or not list(root.iterdir())


def test_rejects_modified_reference_checksum_without_publishing(tmp_path):
    source = model_package(tmp_path / "source")
    (source / "references" / "records.jsonl").write_text("changed\n", encoding="utf-8")
    root = tmp_path / "user-models"

    with pytest.raises(ValueError, match="checksum"):
        import_model_package(source, root=root)

    assert not root.exists() or not list(root.iterdir())


def test_rejects_incomplete_package_without_publishing(tmp_path):
    source = model_package(tmp_path / "source")
    (source / "references" / "manifest.json").unlink()
    root = tmp_path / "user-models"

    with pytest.raises(ValueError, match="missing required"):
        import_model_package(source, root=root)

    assert not root.exists() or not list(root.iterdir())


def test_import_rejects_source_changed_during_copy(tmp_path, monkeypatch):
    source = model_package(tmp_path / "source")
    root = tmp_path / "user-models"
    real_copy = shutil.copyfile

    def changed_copy(src, dst):
        if src.name == "model.onnx":
            src.write_bytes(b"mutated during copy")
        return real_copy(src, dst)

    monkeypatch.setattr("s2s_decision.model_catalog.shutil.copyfile", changed_copy)
    with pytest.raises(ValueError, match="changed during import"):
        import_model_package(source, root=root)

    assert not list(root.iterdir())


def test_existing_install_cannot_hide_tampered_bytes(tmp_path):
    source = model_package(tmp_path / "source")
    root = tmp_path / "user-models"
    installed = import_model_package(source, root=root)
    (installed / "model.onnx").write_bytes(b"tampered")

    assert not list_user_models(root)[0]["compatible"]
    with pytest.raises(ValueError, match="conflicts"):
        import_model_package(source, root=root)


def test_concurrent_identical_install_reuses_verified_package(tmp_path, monkeypatch):
    source = model_package(tmp_path / "source")
    root = tmp_path / "user-models"
    real_rename = Path.rename

    def raced_rename(staged, destination):
        if staged.name == "package":
            shutil.copytree(staged, destination)
            raise FileExistsError(destination)
        return real_rename(staged, destination)

    monkeypatch.setattr(Path, "rename", raced_rename)

    installed = import_model_package(source, root=root)

    assert installed.is_dir()
    assert list_user_models(root)[0]["compatible"]


def test_catalog_shows_incompatible_packages_and_user_data_location(tmp_path):
    source = model_package(tmp_path / "source")
    root = tmp_path / "user-models"
    root.mkdir()
    shutil.copytree(source, root / "existing")
    (root / "existing" / "model.onnx").write_bytes(b"tampered")

    listed = list_user_models(root, target="OTHER")

    assert len(listed) == 1
    assert listed[0]["reason"]
    assert user_model_root().name == "models"
    assert user_model_root().parent.name == "SMILES2Select"


def test_rejects_symlinked_package_member(tmp_path):
    source = model_package(tmp_path / "source")
    target = tmp_path / "outside.txt"
    target.write_text("outside", encoding="utf-8")
    (source / "MODEL_CARD.md").unlink()
    try:
        (source / "MODEL_CARD.md").symlink_to(target)
    except OSError:
        pytest.skip("Symlink creation unavailable on this host")

    with pytest.raises(ValueError, match="link|regular"):
        import_model_package(source, root=tmp_path / "user-models")


def test_bundled_models_match_frozen_validation_choices_and_train_only_refs():
    expected = {
        "Q72547_WT_IC50": "d2c526617dce41e68823c20497497150a8410da2639a5501191d57315ef59dec",
        "P0DMS8_WT_Ki": "494d4203fdc6103b60fe177aeaa9f5e9185e8c6570c5c4e0450bdf610a215227",
        "Q07869_WT_EC50": "fc5b7a1aa7382645e0e4fb52abd394b5778ff5cf95193b69d8ac0a1f0ad96912",
    }
    bundled = list_bundled_models()

    assert {item["name"] for item in bundled} == set(expected)
    assert all(item["compatible"] and item["origin"] == "bundled" for item in bundled)
    assert all(item["estimator"] != "tiny" for item in bundled)
    for name, digest in expected.items():
        package = bundled_model_root() / name
        assert file_hash(package / "manifest.json") == digest
        assert (package / "MODEL_CARD.md").is_file()
        with (package / "references" / "records.jsonl").open(encoding="utf-8") as stream:
            assert {json.loads(line)["split"] for line in stream} == {"train"}
    assert "10.5281/zenodo.13987985" in (
        bundled_model_root() / "LICENSE_CC_BY_SA_4.0.md"
    ).read_text(encoding="utf-8")
    assert (bundled_model_root() / "LICENSE_CC_BY_SA_4.0.txt").is_file()


def test_catalog_keeps_bundled_and_user_models_distinct(tmp_path):
    root = tmp_path / "user-models"
    assert len(list_catalog_models(root)) == 3
    import_model_package(model_package(tmp_path / "source"), root=root)

    items = list_catalog_models(root, target="T1")

    assert len(items) == 4
    assert [item["origin"] for item in items].count("user") == 1
    assert [item["compatible"] for item in items].count(True) == 1
    assert items[-1]["target_id"] == "T1"


def test_missing_bundled_assets_leave_user_catalog_available(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "s2s_decision.model_catalog.bundled_model_root", lambda: tmp_path / "absent"
    )
    assert list_bundled_models() == []
    assert list_catalog_models(tmp_path / "empty") == []


def test_catalog_marks_models_unavailable_without_optional_inference(tmp_path, monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    items = list_catalog_models(tmp_path / "empty")
    assert len(items) == 3
    assert all(not item["compatible"] for item in items)
    assert all("inference" in item["reason"] for item in items)
