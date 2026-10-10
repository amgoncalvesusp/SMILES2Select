from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from s2s_decision.artifacts import read_bundle
from s2s_decision.tolerance_decision import (
    preview_tolerance_decision,
    tolerance_package_hash,
    tolerance_snapshot_from_workspace,
    verify_tolerance_decision,
)
from smiles2select.decision.policies import DecisionPolicy
from smiles2select.profiles.loader import builtin_registry
from smiles2select.selection_intelligence.basket import SelectionBasket
from smiles2select.selection_intelligence.constrained_selection import SelectionConstraints
from smiles2select.selection_intelligence.states import (
    ChemicalStatus,
    MoleculeState,
    SelectionStatus,
)


def workspace():
    ids = pd.Index([11, 12, 13], name="record_id")
    records = pd.DataFrame({
        "original_smiles": ["CCO", "c1ccccc1", "CC(=O)O"],
        "canonical_smiles": ["CCO", "c1ccccc1", "CC(=O)O"],
        "valid": True, "evaluable": True, "duplicate_of": pd.NA,
    }, index=ids)
    result = SimpleNamespace(
        descriptors=records, alerts=pd.DataFrame(), scores=pd.DataFrame(index=ids),
        evaluation=SimpleNamespace(status=pd.DataFrame({"lipinski__passed": [True, False, True]}, index=ids)),
        config=SimpleNamespace(policy=DecisionPolicy(roles={"lipinski": "mandatory"}),
                               exclude_reference_duplicates=False, selection_strategy="balanced"),
        profiles=(builtin_registry().get("lipinski"),), zone_allocation=None,
    )
    candidates = records.assign(murcko_scaffold=["", "c1ccccc1", ""], cluster_id=[1, 2, 3])
    basket = SelectionBasket([
        MoleculeState(11, ChemicalStatus.AUTO_PASS, SelectionStatus.FINAL_SELECTED),
        MoleculeState(12, ChemicalStatus.AUTO_FAIL),
        MoleculeState(13, ChemicalStatus.AUTO_PASS, SelectionStatus.MANUALLY_EXCLUDED),
    ])
    return result, candidates, basket


def test_snapshot_keeps_baseline_and_does_not_relax_manual_exclusion():
    result, candidates, basket = workspace()
    snapshot = tolerance_snapshot_from_workspace(result, candidates, basket,
        SelectionConstraints(2), revisited_profiles=("lipinski",))
    rows = snapshot.prepared().records.set_index("record_id")
    assert rows.eligible.to_dict() == {11: True, 12: True, 13: False}
    assert rows.baseline_eligible.to_dict() == {11: True, 12: False, 13: False}
    strict = tolerance_snapshot_from_workspace(result, candidates, basket, SelectionConstraints(2))
    assert strict.revision != snapshot.revision
    assert strict.prepared().records.eligible.tolist() == [True, False, False]


def test_exclusion_role_cannot_be_revisited():
    result, candidates, basket = workspace()
    result.config.policy = replace(result.config.policy, roles={"lipinski": "exclusion"})
    snapshot = tolerance_snapshot_from_workspace(result, candidates, basket,
        SelectionConstraints(2), revisited_profiles=("lipinski",))
    assert not snapshot.prepared().records.set_index("record_id").loc[12, "eligible"]


def test_partial_rule_reconsideration_keeps_unchecked_mandatory_rules():
    result, candidates, basket = workspace()
    strict = result.profiles[0].with_pass_policy({"type": "all_rules"})
    result.profiles = (strict,)
    result.descriptors = result.descriptors.assign(mol_wt=[100., 600., 100.],
        rdkit_wlogp=[1., 7., 1.], hbd_lipinski=1, hba_lipinski=1)
    snapshot = tolerance_snapshot_from_workspace(result, candidates, basket,
        SelectionConstraints(2), revisited_profiles=("lipinski",),
        reconsidered_criteria=("rule:lipinski:lip_mw_max",))
    assert not snapshot.prepared().records.set_index("record_id").loc[12, "eligible"]


def test_selective_alert_reconsideration_keeps_other_entries():
    from rdkit import Chem

    from s2s_decision.tolerance_features import match_entries
    from smiles2select.alerts.engine import ALERT_COLUMNS
    from smiles2select.alerts.policies import AlertPolicy

    result, candidates, basket = workspace()
    text = "O=[N+]([O-])c1ccccc1"
    hits = match_entries(Chem.MolFromSmiles(text), "brenk")
    assert len(hits) >= 2
    result.descriptors.loc[12, ["original_smiles", "canonical_smiles"]] = text
    candidates.loc[12, ["original_smiles", "canonical_smiles"]] = text
    result.config.policy = replace(result.config.policy, alert_policy=AlertPolicy(actions={"brenk": "exclude"}))
    result.alerts = pd.DataFrame([{"record_id": 12, "catalog_id": "brenk",
        "alert_name": hit["alert_name"], "alert_description": hit["alert_description"],
        "occurrence_count": 1, "action": "exclude"} for hit in hits], columns=ALERT_COLUMNS)
    snapshot = tolerance_snapshot_from_workspace(result, candidates, basket, SelectionConstraints(2),
        revisited_profiles=("lipinski",), revisited_alert_entries=(hits[0]["criterion_id"],))
    assert not snapshot.prepared().records.set_index("record_id").loc[12, "eligible"]
    full = tolerance_snapshot_from_workspace(result, candidates, basket, SelectionConstraints(2),
        revisited_profiles=("lipinski",), revisited_alert_entries=tuple(hit["criterion_id"] for hit in hits))
    assert full.prepared().records.set_index("record_id").loc[12, "eligible"]


def fake_package(monkeypatch, tmp_path):
    from s2s_decision import tolerance_decision as module
    package = tmp_path / "package.json"
    package.write_text('{"purpose":"fixture"}', encoding="utf-8")
    monkeypatch.setattr(module, "read_tolerance_package", lambda _path: {"mode": "general", "status": "experimental"})

    def predict(_profile, records, **_kwargs):
        return records.assign(tolerance_state="supported_relaxation", tolerance_eligible=True,
            tolerance_nonconformity=0.1, tolerance_evidence="fixture", tolerance_fallback="none")

    monkeypatch.setattr(module, "predict_tolerance", predict)
    return package


def test_preview_preserves_hard_exclusions_and_replays_changes(monkeypatch, tmp_path):
    package = fake_package(monkeypatch, tmp_path)
    result, candidates, basket = workspace()
    snapshot = tolerance_snapshot_from_workspace(result, candidates, basket,
        SelectionConstraints(2), revisited_profiles=("lipinski",))
    root = tmp_path / "preview"
    summary = preview_tolerance_decision(snapshot, root, package)
    assert summary["ranking_mode"] == "chemical_tolerance"
    assert set(summary["proposed_ids"]) == {11, 12}
    assert basket.final_ids() == (11,)
    rows = read_bundle(root / "proposed").records.set_index("record_id")
    assert not rows.loc[13, "eligible"]
    assert rows.loc[13, "tolerance_state"] == "excluded"
    verify_tolerance_decision(snapshot, root, package)
    package.write_text('{"purpose":"changed"}', encoding="utf-8")
    with pytest.raises(ValueError, match="package|model"):
        verify_tolerance_decision(snapshot, root, package)


def test_cancelled_preview_never_publishes_files(monkeypatch, tmp_path):
    from concurrent.futures import CancelledError
    package = fake_package(monkeypatch, tmp_path)
    result, candidates, basket = workspace()
    snapshot = tolerance_snapshot_from_workspace(result, candidates, basket,
        SelectionConstraints(2), revisited_profiles=("lipinski",))
    with pytest.raises(CancelledError):
        preview_tolerance_decision(snapshot, tmp_path / "preview", package, cancelled=lambda: True)
    assert not (tmp_path / "preview").exists()


def test_replay_rejects_changed_context_and_record_tampering(monkeypatch, tmp_path):
    package = fake_package(monkeypatch, tmp_path)
    result, candidates, basket = workspace()
    snapshot = tolerance_snapshot_from_workspace(result, candidates, basket,
        SelectionConstraints(2), revisited_profiles=("lipinski",))
    root = tmp_path / "preview"
    preview_tolerance_decision(snapshot, root, package, {"target_id": "T1"})
    with pytest.raises(ValueError, match="context|snapshot"):
        verify_tolerance_decision(snapshot, root, package, {"target_id": "T2"})
    (root / "proposed" / "records.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        verify_tolerance_decision(snapshot, root, package, {"target_id": "T1"})


def test_same_rule_id_with_different_threshold_cannot_transfer_tolerance(tmp_path):
    from s2s_decision.features import featurize
    from s2s_decision.tolerance_profile import fit_tolerance, write_tolerance_package

    result, candidates, basket = workspace()
    training = featurize(pd.DataFrame({"original_smiles": ["CCO"], "pactivity": [7.],
                                      "relation": ["="]})).records
    profile = fit_tolerance(training)
    package = tmp_path / "model"
    write_tolerance_package(profile, package)
    native = result.profiles[0]
    rules = tuple(replace(rule, threshold=300) if rule.id == "lip_mw_max" else rule
                  for rule in native.rules)
    result.profiles = (replace(native, rules=rules),)
    snapshot = tolerance_snapshot_from_workspace(result, candidates, basket, SelectionConstraints(2),
        revisited_profiles=("lipinski",))
    with pytest.raises(ValueError, match="rule differs"):
        preview_tolerance_decision(snapshot, tmp_path / "preview", package)
    assert basket.final_ids() == (11,)
    assert not (tmp_path / "preview").exists()


def test_package_signature_rejects_unlisted_members_before_hashing(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "manifest.json").write_text("{}", encoding="utf-8")
    (package / "profile.json").write_text("{}", encoding="utf-8")
    (package / "unexpected.bin").write_bytes(b"fixture")
    with pytest.raises(ValueError, match="unsupported files"):
        tolerance_package_hash(package)


def test_preview_cannot_change_criteria_captured_for_hard_exclusions(monkeypatch, tmp_path):
    package = fake_package(monkeypatch, tmp_path)
    result, candidates, basket = workspace()
    snapshot = tolerance_snapshot_from_workspace(result, candidates, basket, SelectionConstraints(2),
        revisited_profiles=("lipinski",))
    with pytest.raises(ValueError, match="captured criterion"):
        preview_tolerance_decision(snapshot, tmp_path / "preview", package,
            reconsidered_criteria=("rule:lipinski:lip_mw_max",))
    assert basket.final_ids() == (11,)


@pytest.mark.parametrize("entry, message", [
    ("brenk:old-display-name", "versioned entry"),
    ("alert:v2:unavailable:0:old", "unavailable native"),
    ("alert:v2:brenk:0:old", "installed catalog"),
])
def test_alert_scope_rejects_unresolvable_identities(entry, message):
    result, candidates, basket = workspace()
    with pytest.raises(ValueError, match=message):
        tolerance_snapshot_from_workspace(result, candidates, basket, SelectionConstraints(2),
                                         revisited_alert_entries=(entry,))
    assert basket.final_ids() == (11,)


def test_partial_consensus_cannot_bypass_coupled_policy():
    result, candidates, basket = workspace()
    result.config.policy = replace(result.config.policy, roles={"lipinski": "consensus"},
                                   consensus_min_pass=1)
    with pytest.raises(ValueError, match="coupled consensus"):
        tolerance_snapshot_from_workspace(result, candidates, basket, SelectionConstraints(2),
            revisited_profiles=("lipinski",),
            reconsidered_criteria=("rule:lipinski:lip_mw_max",))


def test_known_smarts_criterion_remains_locked():
    from smiles2select.rules.engine import Rule

    result, candidates, basket = workspace()
    smarts = Rule("target_motif", "lipinski", "target_motif", "contains_substructure",
                  smarts="[NX3]")
    result.profiles = (replace(result.profiles[0], rules=(*result.profiles[0].rules, smarts)),)
    with pytest.raises(ValueError, match="Locked SMARTS"):
        tolerance_snapshot_from_workspace(result, candidates, basket, SelectionConstraints(2),
            revisited_profiles=("lipinski",),
            reconsidered_criteria=("rule:lipinski:target_motif",))


def test_existing_preview_is_preserved(monkeypatch, tmp_path):
    package = fake_package(monkeypatch, tmp_path)
    result, candidates, basket = workspace()
    snapshot = tolerance_snapshot_from_workspace(result, candidates, basket, SelectionConstraints(2))
    destination = tmp_path / "preview"
    destination.mkdir()
    sentinel = destination / "user-notes.txt"
    sentinel.write_text("Preserve this result", encoding="utf-8")
    with pytest.raises(ValueError, match="output already exists"):
        preview_tolerance_decision(snapshot, destination, package)
    assert sentinel.read_text(encoding="utf-8") == "Preserve this result"


def test_late_cancellation_removes_staged_preview(monkeypatch, tmp_path):
    from concurrent.futures import CancelledError

    package = fake_package(monkeypatch, tmp_path)
    result, candidates, basket = workspace()
    snapshot = tolerance_snapshot_from_workspace(result, candidates, basket, SelectionConstraints(2),
                                                revisited_profiles=("lipinski",))
    checks = []

    def cancel_when_staged():
        checks.append(True)
        return len(checks) == 4

    with pytest.raises(CancelledError):
        preview_tolerance_decision(snapshot, tmp_path / "preview", package,
                                   cancelled=cancel_when_staged)
    assert len(checks) == 4
    assert not (tmp_path / "preview").exists()
    assert not list(tmp_path.glob(".s2s-tolerance-*"))
    assert basket.final_ids() == (11,)


def test_package_change_during_scoring_rolls_back(monkeypatch, tmp_path):
    from s2s_decision import tolerance_decision as module

    package = fake_package(monkeypatch, tmp_path)
    original_predict = module.predict_tolerance

    def changing_predict(*args, **kwargs):
        package.write_text('{"purpose":"mutated during scoring"}', encoding="utf-8")
        return original_predict(*args, **kwargs)

    monkeypatch.setattr(module, "predict_tolerance", changing_predict)
    result, candidates, basket = workspace()
    snapshot = tolerance_snapshot_from_workspace(result, candidates, basket, SelectionConstraints(2),
                                                revisited_profiles=("lipinski",))
    with pytest.raises(ValueError, match="changed during preview"):
        preview_tolerance_decision(snapshot, tmp_path / "preview", package)
    assert not (tmp_path / "preview").exists()
    assert basket.final_ids() == (11,)


@pytest.mark.parametrize("field, value", [("tolerance_eligible", False),
                                          ("tolerance_nonconformity", 0.9)])
def test_replay_rejects_changed_inference_with_same_package(monkeypatch, tmp_path, field, value):
    from s2s_decision import tolerance_decision as module

    package = fake_package(monkeypatch, tmp_path)
    result, candidates, basket = workspace()
    snapshot = tolerance_snapshot_from_workspace(result, candidates, basket, SelectionConstraints(2),
                                                revisited_profiles=("lipinski",))
    root = tmp_path / "preview"
    preview_tolerance_decision(snapshot, root, package)
    original_predict = module.predict_tolerance

    def changed_predict(*args, **kwargs):
        return original_predict(*args, **kwargs).assign(**{field: value})

    monkeypatch.setattr(module, "predict_tolerance", changed_predict)
    with pytest.raises(ValueError, match="replay differs"):
        verify_tolerance_decision(snapshot, root, package)
    assert basket.final_ids() == (11,)


def test_valid_directory_package_hash_binds_each_allowed_file(tmp_path):
    package = tmp_path / "model"
    package.mkdir()
    (package / "manifest.json").write_text("{}", encoding="utf-8")
    profile = package / "profile.json"
    profile.write_text('{"purpose":"fixture"}', encoding="utf-8")
    signature = tolerance_package_hash(package)
    assert tolerance_package_hash(package) == signature
    profile.write_text('{"purpose":"different fixture"}', encoding="utf-8")
    assert tolerance_package_hash(package) != signature
    with pytest.raises(ValueError, match="does not exist"):
        tolerance_package_hash(tmp_path / "missing")
