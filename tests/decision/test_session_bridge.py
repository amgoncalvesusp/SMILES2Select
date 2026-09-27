from concurrent.futures import CancelledError
from types import SimpleNamespace

import pandas as pd
import pytest

from s2s_decision.artifacts import read_bundle, write_bundle
from s2s_decision.decision import preview_session_decision, verify_preview
from s2s_decision.features import chemistry_manifest
from s2s_decision.schema import FeatureSet
from s2s_decision.session_adapter import snapshot_from_workspace
from smiles2select.selection_intelligence.basket import SelectionBasket
from smiles2select.selection_intelligence.constrained_selection import SelectionConstraints
from smiles2select.selection_intelligence.states import (
    ChemicalStatus,
    MoleculeState,
    SelectionStatus,
)


def workspace(smiles=("CCO", "c1ccccc1", "CC(=O)O")):
    ids = (11, 12, 13)
    frame = pd.DataFrame(
        {
            "original_smiles": smiles,
            "canonical_smiles": smiles,
            "valid": True,
            "evaluable": True,
            "duplicate_of": pd.NA,
            "molecule_id": ("same", "same", "other"),
        },
        index=pd.Index(ids, name="record_id"),
    )
    result = SimpleNamespace(
        descriptors=frame,
        alerts=pd.DataFrame(),
        scores=pd.DataFrame(),
        config=SimpleNamespace(
            exclude_reference_duplicates=False, selection_strategy="balanced"
        ),
        profiles=(),
        zone_allocation=None,
    )
    candidates = frame.assign(murcko_scaffold=("", "c1ccccc1", ""), cluster_id=(1, 2, 3))
    basket = SelectionBasket(
        [
            MoleculeState(11, ChemicalStatus.AUTO_PASS, SelectionStatus.FINAL_SELECTED),
            MoleculeState(12, ChemicalStatus.AUTO_PASS),
            MoleculeState(13, ChemicalStatus.AUTO_PASS),
        ]
    )
    return result, candidates, basket


def fake_model_preview(snapshot, tmp_path, monkeypatch):
    from s2s_decision import decision

    model = tmp_path / "model"
    model.mkdir()
    (model / "manifest.json").write_text('{"fingerprint_bits": 2048}', encoding="utf-8")
    monkeypatch.setattr(decision, "_inspect_model", lambda *_args: {"compatible": True})

    def score(source, _model, destination):
        features = read_bundle(source)
        scored = FeatureSet(
            features.records.assign(priority_score=features.records.qed, calibration_status="fixture"),
            {**features.manifest, "calibration_status": "fixture"},
        )
        write_bundle(scored, destination)

    monkeypatch.setattr(decision, "score_candidates", score)
    return preview_session_decision(snapshot, tmp_path / "preview", model)


def test_snapshot_keeps_full_pre_count_pool_and_isolated_rows(tmp_path, monkeypatch):
    result, candidates, basket = workspace()
    snapshot = snapshot_from_workspace(
        result, candidates, basket, SelectionConstraints(target_count=1), session_id="run-1"
    )
    result.descriptors.loc[12, "original_smiles"] = "invalid"
    features = snapshot.prepared()
    assert features.records.record_id.tolist() == [11, 12, 13]
    assert features.records.eligible.tolist() == [True, True, True]
    assert features.records.loc[features.records.record_id.eq(12), "original_smiles"].item() == "c1ccccc1"
    assert snapshot.original_ids == (11,)
    summary = fake_model_preview(snapshot, tmp_path, monkeypatch)
    assert summary["original_ids"] == [11]
    assert summary["eligible_count"] == 3
    assert summary["source"]["session_revision"] == snapshot.revision
    assert verify_preview(tmp_path / "preview")["proposed_ids"] == summary["proposed_ids"]
    sealed = tmp_path / "preview" / "comparison.csv"
    sealed.write_text(sealed.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        verify_preview(tmp_path / "preview")


def test_cancelled_model_preview_never_publishes_partial_files(tmp_path, monkeypatch):
    from s2s_decision import decision

    result, candidates, basket = workspace()
    snapshot = snapshot_from_workspace(
        result, candidates, basket, SelectionConstraints(target_count=1),
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "manifest.json").write_text('{"fingerprint_bits": 2048}', encoding="utf-8")
    monkeypatch.setattr(decision, "_inspect_model", lambda *_args: {"compatible": True})
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return calls >= 2

    with pytest.raises(CancelledError, match="cancelled"):
        preview_session_decision(snapshot, tmp_path / "preview", model, cancelled=cancelled)
    assert not (tmp_path / "preview").exists()
    assert not list(tmp_path.glob(".s2s-preview-*"))


def test_session_preview_uses_native_scaffold_and_pin_policy(tmp_path, monkeypatch):
    result, candidates, basket = workspace()
    basket.pin([11])
    candidates.loc[[11, 12], "murcko_scaffold"] = "session-core"
    snapshot = snapshot_from_workspace(
        result, candidates, basket,
        SelectionConstraints(target_count=2, max_per_scaffold=1),
    )
    proposed = fake_model_preview(snapshot, tmp_path, monkeypatch)
    assert set(proposed["proposed_ids"]) == {11, 13}
    assert proposed["pinned_ids"] == [11]


def test_snapshot_preserves_pinned_chemical_exception_and_manual_exclusion():
    result, candidates, basket = workspace()
    basket = SelectionBasket(
        [
            MoleculeState(11, ChemicalStatus.AUTO_PASS),
            MoleculeState(12, ChemicalStatus.AUTO_PASS, SelectionStatus.MANUALLY_EXCLUDED),
            MoleculeState(
                13, ChemicalStatus.AUTO_FAIL, SelectionStatus.FINAL_SELECTED,
                pinned=True, note="documented rescue",
            ),
        ]
    )
    snapshot = snapshot_from_workspace(
        result, candidates, basket, SelectionConstraints(target_count=1)
    )
    rows = snapshot.prepared().records.set_index("record_id")
    assert rows.eligible.to_dict() == {11: True, 12: False, 13: True}
    assert rows.loc[13, "chemical_status"] == "AUTO_FAIL"
    assert rows.loc[13, "selection_note"] == "documented rescue"
    assert snapshot.pinned_ids == (13,)
    assert snapshot.excluded_ids == (12,)


def test_snapshot_joins_native_state_by_record_id_after_reordering():
    result, candidates, basket = workspace()
    result.descriptors = result.descriptors.loc[[13, 11, 12]]
    candidates = candidates.loc[[12, 13, 11]]
    snapshot = snapshot_from_workspace(
        result, candidates, basket, SelectionConstraints(target_count=2)
    )
    rows = snapshot.prepared().records.set_index("record_id")
    assert rows.loc[11, "is_final"]
    assert rows.loc[12, "session_scaffold"] == "c1ccccc1"
    assert rows.loc[13, "original_smiles"] == "CC(=O)O"


def test_snapshot_detects_mutation_after_capture():
    result, candidates, basket = workspace()
    snapshot = snapshot_from_workspace(
        result, candidates, basket, SelectionConstraints(target_count=1)
    )
    snapshot._records.loc[0, "original_smiles"] = "CO"
    with pytest.raises(ValueError, match="rows changed"):
        snapshot.prepared()


def test_session_preview_requires_explicit_model(tmp_path):
    result, candidates, basket = workspace()
    snapshot = snapshot_from_workspace(
        result, candidates, basket, SelectionConstraints(target_count=1)
    )
    with pytest.raises(ValueError, match="Choose a compatible model"):
        preview_session_decision(snapshot, tmp_path / "preview")
    assert not (tmp_path / "preview").exists()


def test_stereo_collision_blocks_supervised_preview(tmp_path, monkeypatch):
    result, candidates, basket = workspace(("C[C@H](O)F", "C[C@@H](O)F", "CCO"))
    snapshot = snapshot_from_workspace(
        result, candidates, basket, SelectionConstraints(target_count=1)
    )
    features = snapshot.prepared()
    assert features.records.loc[0, "identity"] == features.records.loc[1, "identity"]
    from s2s_decision import decision

    model = tmp_path / "model"
    model.mkdir()
    (model / "manifest.json").write_text('{"fingerprint_bits": 2048}', encoding="utf-8")
    monkeypatch.setattr(decision, "_inspect_model", lambda *_args: {"compatible": True})
    with pytest.raises(ValueError, match="11.*12"):
        preview_session_decision(snapshot, tmp_path / "preview", model)
    assert not (tmp_path / "preview").exists()


def test_chemistry_contract_separates_app_release(monkeypatch):
    import s2s_decision.features as features

    first = chemistry_manifest(1024)
    monkeypatch.setattr(features, "APP_VERSION", "future-release")
    second = chemistry_manifest(1024)
    assert first["chemistry_contract_version"] == 1
    assert first["chemistry_hash"] == second["chemistry_hash"]
    assert first["smiles2select_version"] != second["smiles2select_version"]
    assert first["legacy_chemistry_hash"] != second["legacy_chemistry_hash"]
    with pytest.raises(ValueError, match="hash is missing"):
        features.validate_chemistry_compatibility({"chemistry_contract_version": 1}, {})
    with pytest.raises(ValueError, match="conflicts"):
        features.validate_chemistry_compatibility(
            {**first, "standardization": {"remove_stereo": False}}, first
        )


def test_legacy_release_needs_full_recipe_and_feature_parity(monkeypatch):
    import s2s_decision.features as features

    old = chemistry_manifest(1024)
    legacy_keys = (
        "feature_schema", "smiles2select_version", "rdkit_version", "standardization",
        "fingerprint", "fingerprint_bits", "property_names", "catalogs", "descriptors",
        "profiles", "contrib_resources",
    )
    task = {key: old[key] for key in legacy_keys}
    task["chemistry_hash"] = old["legacy_chemistry_hash"]
    with pytest.raises(ValueError, match="checksum mismatch"):
        features.validate_chemistry_compatibility(
            {**task, "standardization": {"remove_stereo": False}}, task
        )
    with pytest.raises(ValueError, match="recipe is incomplete"):
        features.validate_chemistry_compatibility(
            {"chemistry_hash": old["legacy_chemistry_hash"]}, task
        )
    with pytest.raises(ValueError, match="candidate chemistry recipe checksum mismatch"):
        features.validate_chemistry_compatibility(
            task, {**task, "standardization": {"remove_stereo": False}}
        )
    references = features.featurize(pd.DataFrame({"record_id": [7], "original_smiles": ["CCO"]}), 1024)
    monkeypatch.setattr(features, "APP_VERSION", "future-release")
    current = chemistry_manifest(1024)
    with pytest.raises(ValueError, match="parity"):
        features.validate_chemistry_compatibility(task, current)
    features.validate_chemistry_compatibility(task, current, references.records)
    changed = references.records.assign(qed=0.0)
    with pytest.raises(ValueError, match="parity failed: qed"):
        features.validate_chemistry_compatibility(task, current, changed)
