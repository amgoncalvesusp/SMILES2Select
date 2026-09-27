import json

import pandas as pd
import pytest

from s2s_decision.artifacts import file_hash, read_bundle, write_bundle, write_json
from s2s_decision.decision import export_decision, list_models, preview_decision
from s2s_decision.features import chemistry_manifest
from s2s_decision.preprocessing import Preprocessor
from s2s_decision.schema import SCHEMA_VERSION, FeatureSet


def library(tmp_path, annotated=True):
    path = tmp_path / "library.csv"
    data = {
        "record_id": [11, 12, 13],
        "ID": ["001", "=1+1", "NA"],
        "SMILES": ["CCO", "c1ccccc1", "CC(=O)O"],
        "Pinned": ["yes", "no", "no"],
    }
    if annotated:
        data["selected"] = [True, True, False]
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def test_preview_preserves_pins_source_and_requires_explicit_export(tmp_path):
    source = library(tmp_path)
    digest = file_hash(source)
    preview = tmp_path / "preview"
    summary = preview_decision(source, preview, n=1)
    assert summary["original_ids"] == [11, 12]
    assert summary["proposed_ids"] == [11]
    assert summary["removed_ids"] == [12]
    assert summary["ranking_mode"] == "chemical_qed"
    assert not (preview / "final.csv").exists()
    assert file_hash(source) == digest
    proposed = read_bundle(preview / "proposed")
    assert not proposed.records.loc[proposed.records.record_id.eq(13), "eligible"].item()
    assert "activity_probability" not in proposed.records
    exported = tmp_path / "adopted"
    export_decision(preview, exported)
    assert file_hash(exported / "comparison.json") == file_hash(preview / "comparison.json")
    assert file_hash(exported / "comparison.csv") == file_hash(preview / "comparison.csv")
    assert pd.read_csv(exported / "final.csv", dtype=str).molecule_id.tolist() == ["001"]
    assert read_bundle(exported).manifest["decision_status"] == "adopted"
    with pytest.raises(ValueError, match="already exists"):
        preview_decision(source, preview, n=1)
    with pytest.raises(ValueError, match="already exists"):
        export_decision(preview, exported)


def test_unknown_original_and_formula_safe_csv(tmp_path):
    preview = tmp_path / "preview"
    summary = preview_decision(library(tmp_path, False), preview, n=3)
    assert summary["original_ids"] is None
    assert summary["added_ids"] is None
    table = pd.read_csv(preview / "comparison.csv", keep_default_na=False)
    assert set(table.change) == {"unknown_original"}
    assert "'=1+1" in table.molecule_id.tolist()
    export_decision(preview, tmp_path / "adopted")
    assert "=1+1" in read_bundle(tmp_path / "adopted").records.molecule_id.tolist()
    assert "'=1+1" in (tmp_path / "adopted" / "final.csv").read_text(encoding="utf-8")


def test_export_rejects_modified_preview(tmp_path):
    preview = tmp_path / "preview"
    preview_decision(library(tmp_path), preview, n=1)
    path = preview / "proposed" / "manifest.json"
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["constraints"]["target_count"] = 2
    write_json(path, changed)
    with pytest.raises(ValueError, match="changed|checksum"):
        export_decision(preview, tmp_path / "adopted")
    assert not (tmp_path / "adopted").exists()


def model_package(path):
    path.mkdir()
    (path / "model.onnx").write_bytes(b"not executed in discovery")
    chemistry = chemistry_manifest(2048)
    write_bundle(FeatureSet(pd.DataFrame({"record_id": [1]}), chemistry), path / "references")
    references = read_bundle(path / "references")
    manifest = {
        "schema": SCHEMA_VERSION,
        "format_version": 1,
        "runtime_ready": True,
        "fingerprint_bits": 2048,
        "preprocessing": Preprocessor((0.0,) * 27, (0.0,) * 27, (1.0,) * 27, ()).to_dict(),
        "regression_mean": 0.0,
        "regression_scale": 1.0,
        "regression_supported": False,
        "onnx_sha256": file_hash(path / "model.onnx"),
        "reference_records_sha256": references.manifest["records_sha256"],
        "provenance": {**chemistry, "target_id": "T1", "endpoint": "IC50", "threshold": 6},
        "calibrator": {
            "slope": 1.0,
            "intercept": 0.0,
            "status": "fitted_held_out_not_prospectively_validated",
        },
    }
    write_json(path / "manifest.json", manifest)
    return manifest


def test_discovery_visible_invalid_and_bounded_task_mismatch(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    model_package(root / "good")
    model_package(root / "broken")
    (root / "broken" / "model.onnx").write_bytes(b"changed")
    nested = root / "experiments"
    nested.mkdir()
    model_package(nested / "not_discovered")
    found = list_models(root)
    assert len(found) == 2
    by_name = {item["name"]: item for item in found}
    assert by_name["good"]["compatible"]
    assert by_name["good"]["train_reference_count"] == 1
    assert not by_name["broken"]["compatible"]
    assert "checksum" in by_name["broken"]["reason"].lower()
    assert not list_models(root / "good", target="OTHER")[0]["compatible"]
    assert len(list_models(root / "good")) == 1


def test_preview_rejects_incompatible_model_without_partial_directory(tmp_path):
    model = tmp_path / "model"
    manifest = model_package(model)
    write_json(model / "manifest.json", {**manifest, "schema": "wrong"})
    with pytest.raises(ValueError, match="schema"):
        preview_decision(library(tmp_path), tmp_path / "preview", model, n=1)
    assert not (tmp_path / "preview").exists()


def test_excluded_pin_conflict_is_not_silently_overridden(tmp_path):
    with pytest.raises(ValueError, match="both pinned and excluded"):
        preview_decision(library(tmp_path), tmp_path / "preview", n=1, exclude=(11,))
    assert not (tmp_path / "preview").exists()


def test_prepared_bundle_task_conflict_and_final_basket_original(tmp_path):
    initial = tmp_path / "initial"
    preview_decision(library(tmp_path, False), initial, n=2)
    features = read_bundle(initial / "prepared")
    prepared = tmp_path / "prepared"
    write_bundle(
        FeatureSet(
            features.records.assign(target_id="OTHER"),
            {**features.manifest, "input_scope": "final_basket"},
        ),
        prepared,
    )
    model = tmp_path / "model"
    model_package(model)
    item = list_models(model, prepared)[0]
    assert not item["compatible"]
    assert "target_id" in item["reason"]
    summary = preview_decision(prepared, tmp_path / "chemical", n=2)
    assert summary["original_ids"] == [11, 12, 13]


def test_supervised_preview_keeps_support_and_comparison(tmp_path, monkeypatch):
    import s2s_decision.decision as decision

    model = tmp_path / "model"
    model_package(model)

    def fake_score(source, model_dir, destination):
        features = read_bundle(source)
        records = features.records.assign(
            priority_score=[0.1, 0.8, 0.2],
            reference_count=[8, 9, 9],
            reference_similarity_max=[0.2, 0.9, 0.1],
            reference_distance_status="measured_similarity_not_confidence",
        )
        write_bundle(
            FeatureSet(records, {**features.manifest, "calibration_status": "test_status"}),
            destination,
        )

    monkeypatch.setattr(decision, "score_candidates", fake_score)
    summary = preview_decision(library(tmp_path, False), tmp_path / "preview", model, n=2)
    assert summary["proposed_ids"] == [11, 12]
    assert summary["ranking_mode"] == "experimental_model"
    assert summary["calibration_status"] == "test_status"
    table = pd.read_csv(tmp_path / "preview" / "comparison.csv")
    assert table.reference_count.tolist() == [8, 9, 9]
    assert table.reference_similarity_max.tolist() == [0.2, 0.9, 0.1]


@pytest.mark.parametrize(
    "patch",
    [
        {"estimator": "unknown"},
        {"input_layout": "unknown"},
        {"calibrator": {"slope": -1, "intercept": 0, "status": "invalid"}},
        {"preprocessing": {}},
    ],
)
def test_discovery_rejects_invalid_runtime_contract(tmp_path, patch):
    path = tmp_path / "model"
    manifest = model_package(path)
    write_json(path / "manifest.json", {**manifest, **patch})
    result = list_models(path)[0]
    assert not result["compatible"]
    assert result["reason"]


def test_adopted_bundle_reimport_keeps_latest_basket_and_all_evidence(tmp_path):
    preview = tmp_path / "preview"
    preview_decision(library(tmp_path, False), preview, n=1, pins=(12,), exclude=(13,))
    exported = tmp_path / "adopted"
    export_decision(preview, exported)
    bundle = read_bundle(exported)
    assert len(bundle.records) == 3
    assert bundle.records.pinned.tolist() == [True, True, False]
    assert bundle.records.eligible.tolist() == [True, True, False]
    assert bundle.manifest["eligible_count"] == 2
    csv = pd.read_csv(exported / "final.csv", dtype=str)
    assert csv.pinned.tolist() == ["True", "True"]
    assert csv.loc[csv.record_id.eq("12"), "source__Pinned"].item() == "no"
    audit = json.loads((exported / "selection.json").read_text(encoding="utf-8"))
    assert audit["records_sha256"] == file_hash(exported / "records.jsonl")
    summary = preview_decision(exported, tmp_path / "reopened", n=1)
    assert summary["original_ids"] == [11, 12]
    assert summary["proposed_ids"] == [11, 12]


def test_chemical_rescoring_does_not_present_stale_activity_support(tmp_path):
    initial = tmp_path / "initial"
    preview_decision(library(tmp_path, False), initial, n=2)
    features = read_bundle(initial / "prepared")
    source = tmp_path / "scored_input"
    write_bundle(
        FeatureSet(
            features.records.assign(
                activity_probability=0.99,
                reference_similarity_max=0.95,
                similarity_active_max=0.95,
                reference_count=100,
                priority_percentile=99,
            ),
            {**features.manifest, "calibration_status": "fitted", "model_sha256": "old"},
        ),
        source,
    )
    summary = preview_decision(source, tmp_path / "chemical", n=2)
    assert summary["calibration_status"] == "not_applicable_chemical_ranking"
    scored = read_bundle(tmp_path / "chemical" / "scored")
    assert "model_sha256" not in scored.manifest
    assert "activity_probability" not in scored.records
    assert "reference_count" not in scored.records
    table = pd.read_csv(tmp_path / "chemical" / "comparison.csv")
    assert table.reference_similarity_max.isna().all()
