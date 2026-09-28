"""Local model installation keeps untrusted packages outside the application tree."""

import json
import shutil
from pathlib import Path

import numpy as np
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


FROZEN_MODELS = {
    "Q72547_WT_IC50__logistic_scalar_morgan": (
        "13a922cae2670fa9eb41c7cf86a0c23b255b2bf0920f3fa708591a5b275c85c3",
        "37a584265554675fc01f259de4842954695d8c640900e60df28a64d1f984285a",
    ),
    "Q72547_WT_IC50": (
        "d2c526617dce41e68823c20497497150a8410da2639a5501191d57315ef59dec",
        "6647342a920b7b7410697ca71ffe1dd1460238449c94875dbc189f9a13847b1c",
    ),
    "Q72547_WT_IC50__gradient_boosting_scalar_morgan": (
        "44af0da893d9a84070ad218f754f91adb73f5195a2928d1c2885f0329e23f06d",
        "b59b5e0873d9778737d65683d6e2d59983323090d8e69b61cb2753ea113f9b66",
    ),
    "Q72547_WT_IC50__tiny_scalar_morgan": (
        "39f41f08ea457486df3669e768f3fd6a7cfc2c5a32322eea4c7540470ad85031",
        "3f204aee8559744f468257580144f361761331ae45b4b1c407dfcb4827a241ac",
    ),
    "P0DMS8_WT_Ki": (
        "494d4203fdc6103b60fe177aeaa9f5e9185e8c6570c5c4e0450bdf610a215227",
        "f1334c10f3dbed984784ec7b3e1490c8c7d2fa46d48f9ba5fb1bab250e3bedee",
    ),
    "P0DMS8_WT_Ki__gradient_boosting_scalar": (
        "bb68b68912d6dc14839a83a89d774fa356ffccd9affa343a25587e7eaf83e148",
        "1dca37a1b9f9e5d7a12d8f769cefee236fd2f55f62adadc67cfce283d5aa12a3",
    ),
    "P0DMS8_WT_Ki__gradient_boosting_scalar_morgan": (
        "898308f99deb91874b2cfaecfe5e6ddb94ef7df7fd71385f86bb82103ea7bda2",
        "d25939d7da1fdc056e67e63a36461af5663236a3d514a97817a2cf81a0f61906",
    ),
    "P0DMS8_WT_Ki__tiny_scalar_morgan": (
        "27e7f37e79926f34f6a4b92dc8cb63db254709fbe84ada210912688cc6b8dc89",
        "11a302b321157aa054d5ccdfbf7e5c52510a5af44b50a60dee3a1f84c01f1db5",
    ),
    "Q07869_WT_EC50": (
        "fc5b7a1aa7382645e0e4fb52abd394b5778ff5cf95193b69d8ac0a1f0ad96912",
        "0267bd1f9b1e71660c91f3d9f527bda526417b9258afc5f58b0a219109870a3d",
    ),
    "Q07869_WT_EC50__gradient_boosting_scalar": (
        "32416060dbe408186a07216aff375c0fe1e1baf427d23ab05069110dac776443",
        "f5f9869ef0584ac14ca965c46b5aacef50efd25dbcb2d4b5e19c181aa24b2c27",
    ),
    "Q07869_WT_EC50__gradient_boosting_scalar_morgan": (
        "bc6d04116a3cd751bc394a34232016329393aae5b3851002f9d2fc5e20194efa",
        "a6283d5a487d66cf6a457560255f22217b23eb1b9508306a860dc280ce09a487",
    ),
    "Q07869_WT_EC50__tiny_scalar_morgan": (
        "4e139abc0b401b727ac9a7c84bb7594d89c5942ae08e2fc36aecc1a7e681e3d6",
        "6b7ec93307a41b2ecaa342647abd0d4dffaab33cb07c593765d3be2d029f0e8f",
    ),
}


def test_bundled_models_match_frozen_candidates_and_train_only_refs():
    expected = FROZEN_MODELS
    bundled = list_bundled_models()

    assert {item["name"] for item in bundled} == set(expected)
    assert all(item["compatible"] and item["origin"] == "bundled" for item in bundled)
    assert sum(item["estimator"] == "tiny" for item in bundled) == 3
    assert {item["name"] for item in bundled if item["validation_selected"]} == {
        "Q72547_WT_IC50", "P0DMS8_WT_Ki", "Q07869_WT_EC50",
    }
    for name, (manifest_digest, onnx_digest) in expected.items():
        package = bundled_model_root() / name
        assert file_hash(package / "manifest.json") == manifest_digest
        assert file_hash(package / "model.onnx") == onnx_digest
        assert (package / "MODEL_CARD.md").is_file()
        with (package / "references" / "records.jsonl").open(encoding="utf-8") as stream:
            assert {json.loads(line)["split"] for line in stream} == {"train"}
    assert "10.5281/zenodo.13987985" in (
        bundled_model_root() / "LICENSE_CC_BY_SA_4.0.md"
    ).read_text(encoding="utf-8")
    assert (bundled_model_root() / "LICENSE_CC_BY_SA_4.0.txt").is_file()


def test_catalog_keeps_bundled_and_user_models_distinct(tmp_path):
    root = tmp_path / "user-models"
    assert len(list_catalog_models(root)) == 12
    import_model_package(model_package(tmp_path / "source"), root=root)

    items = list_catalog_models(root, target="T1")

    assert len(items) == 13
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
    assert len(items) == 12
    assert all(not item["compatible"] for item in items)
    assert all("inference" in item["reason"] for item in items)


@pytest.mark.parametrize("model_name", tuple(FROZEN_MODELS))
def test_all_bundled_candidates_produce_real_calibrated_probabilities(model_name):
    from s2s_decision.artifacts import read_bundle
    from s2s_decision.context import build_context
    from s2s_decision.features import featurize
    from s2s_decision.inference import predict

    frame = featurize(
        pd.DataFrame(
            {
                "record_id": [19, 7, 31],
                "original_smiles": ["CCO", "CCN", "CC(=O)O"],
            }
        )
    ).records
    package = bundled_model_root() / model_name
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    frame = frame.join(build_context(frame, read_bundle(package / "references").records))

    scores = predict(frame, package, batch_size=2)

    assert scores.record_id.tolist() == [19, 7, 31]
    assert scores.activity_probability.between(0, 1).all()
    assert np.isfinite(scores.activity_probability).all()
    np.testing.assert_array_equal(scores.priority_score, scores.activity_probability)
    assert scores.calibration_status.eq(manifest["calibrator"]["status"]).all()
    if manifest["regression_supported"]:
        assert np.isfinite(scores.predicted_pactivity).all()
    else:
        assert scores.predicted_pactivity.isna().all()
