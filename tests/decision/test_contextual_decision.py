"""Contextual proposals stay sealed and separate from the current basket."""

from concurrent.futures import CancelledError
from dataclasses import replace

import pytest

from s2s_decision.artifacts import read_bundle, write_json
from s2s_decision.contextual_decision import load_policy_package, preview_contextual_decision
from s2s_decision.decision import verify_preview
from s2s_decision.session_adapter import snapshot_from_workspace
from smiles2select.selection_intelligence.constrained_selection import SelectionConstraints
from tests.decision.test_contextual_model import bundle, molecules
from tests.decision.test_session_bridge import workspace


def package(tmp_path):
    payload = bundle(molecules())
    payload["context"] = {
        "target": "T", "species": "Homo sapiens", "endpoint": "IC50", "stage": "hit_finding",
        "assay_context": "biochemical", "source_version": "test-data",
        "model_version": "test-1", "chemistry_version": payload["chemistry"]["chemistry_hash"],
        "activity_threshold": 1000., "activity_unit": "nM", "activity_relation": "<=",
    }
    payload["calibration"].update(positive_n=15, negative_n=15)
    payload["support"] = []
    path = tmp_path / "contextual.json"
    write_json(path, payload)
    return path


def risk_package(tmp_path, chemistry_version):
    payload = {
        "schema": "s2-decision-risk-benchmark/1", "risk_scope": "hepg2_atp_viability_16h",
        "predictor": {"weights": [0.] * 2048, "intercept": 0.},
        "calibrator": {"method": "monotone_sigmoid", "slope": 1., "intercept": 0.},
        "calibration_class_support": [15, 15], "conformal_thresholds": [.6, .6],
        "conformal_alpha": .1, "data_source": "measured fixture", "data_sha256": "fixture",
        "source_version": "fixture-1", "chemistry_version": chemistry_version,
    }
    path = tmp_path / "risk.json"
    write_json(path, payload)
    return path


def test_contextual_real_score_preview_and_seal(tmp_path):
    result, candidates, basket = workspace()
    basket.exclude([12])
    snapshot = snapshot_from_workspace(
        result, candidates, basket, SelectionConstraints(target_count=2), candidate_scope="all_valid",
    )
    path = package(tmp_path)
    _bundle, context = load_policy_package(path)
    summary = preview_contextual_decision(snapshot, tmp_path / "preview", path, context)
    assert summary["ranking_mode"] == "contextual_policy"
    assert summary["context"]["target"] == "T"
    assert 12 not in summary["proposed_ids"]
    assert basket.final_ids() == (11,)
    verified = verify_preview(tmp_path / "preview")
    assert verified["proposed_ids"] == summary["proposed_ids"]
    rows = read_bundle(tmp_path / "preview" / "proposed").records
    assert "policy_explanation" in rows
    assert "activity_prediction_set" in rows
    assert "risk_prediction_status" in rows
    assert not rows.loc[rows.record_id.eq(12), "eligible"].item()
    with pytest.raises(ValueError, match="already exists"):
        preview_contextual_decision(snapshot, tmp_path / "preview", path, context)


def test_contextual_changed_context_and_smarts_are_recorded(tmp_path):
    result, candidates, basket = workspace()
    snapshot = snapshot_from_workspace(
        result, candidates, basket, SelectionConstraints(target_count=3), candidate_scope="all_valid",
    )
    path = package(tmp_path)
    payload, context = load_policy_package(path)
    payload["support"] = [{"feature_id": "rule__lip_mw_max__violation", "support_n": 20,
        "positive_n": 10, "negative_n": 10, "scaffolds": 5, "documents": 3, "source": "fixture"}]
    write_json(path, payload)
    context = replace(context, stage="lead")
    summary = preview_contextual_decision(
        snapshot, tmp_path / "preview", path, context,
        overrides={"required_smarts": ("c1ccccc1",)},
    )
    assert summary["proposed_ids"] == [12]
    assert summary["context"]["stage"] == "lead"
    assert summary["policy_settings"]["required_smarts"] == ("c1ccccc1",)
    assert summary["warnings"]


def test_contextual_cancel_and_incomplete_context_never_publish(tmp_path):
    result, candidates, basket = workspace()
    snapshot = snapshot_from_workspace(result, candidates, basket, SelectionConstraints(target_count=1))
    path = package(tmp_path)
    payload, context = load_policy_package(path)
    with pytest.raises(CancelledError):
        preview_contextual_decision(snapshot, tmp_path / "preview", path, context, cancelled=lambda: True)
    assert not (tmp_path / "preview").exists()
    write_json(path, {**payload, "context": {}})
    with pytest.raises(ValueError, match="context"):
        load_policy_package(path)


def test_measured_risk_is_separate_and_explicit_cutoff_keeps_unknown_domain(tmp_path):
    result, candidates, basket = workspace()
    snapshot = snapshot_from_workspace(result, candidates, basket, SelectionConstraints(2))
    path = package(tmp_path)
    _bundle, context = load_policy_package(path)
    risk = risk_package(tmp_path, context.chemistry_version)
    summary = preview_contextual_decision(snapshot, tmp_path / "preview", path, context,
        risk_path=risk, overrides={"risk_exclude_at": .4})
    assert summary["proposed_count"] == 0
    assert summary["risk_model"]["endpoint"] == "hepg2_atp_viability_16h"
    rows = read_bundle(tmp_path / "preview" / "proposed").records
    assert rows.risk_score.eq(.5).all()
    assert rows.risk_in_domain.isna().all()
    assert all("risk_domain_unknown" in warnings for warnings in rows.policy_warnings)


def test_information_queue_has_own_budget_and_preserves_main_basket(tmp_path):
    result, candidates, basket = workspace()
    basket.exclude([12])
    snapshot = snapshot_from_workspace(result, candidates, basket, SelectionConstraints(1))
    path = package(tmp_path)
    _bundle, context = load_policy_package(path)
    summary = preview_contextual_decision(snapshot, tmp_path / "preview", path, context, review_count=2)
    rows = read_bundle(tmp_path / "preview" / "proposed").records
    assert rows.is_final.sum() == 1
    assert rows.in_information_queue.sum() == 2
    assert not rows.loc[rows.record_id.eq(12), "in_information_queue"].item()
    assert summary["information_queue"]["requested_count"] == 2


def test_invalid_structure_remains_unscored_and_cannot_be_selected(tmp_path):
    result, candidates, basket = workspace(("CCO", "c1ccccc1", "invalid?"))
    result.descriptors.loc[13, ["valid", "evaluable"]] = False
    snapshot = snapshot_from_workspace(result, candidates.drop(index=13), basket,
        SelectionConstraints(3), candidate_scope="all_valid")
    path = package(tmp_path)
    _bundle, context = load_policy_package(path)
    summary = preview_contextual_decision(snapshot, tmp_path / "preview", path, context)
    rows = read_bundle(tmp_path / "preview" / "proposed").records.set_index("record_id")
    assert summary["proposed_count"] == 2
    assert not rows.loc[13, "valid"] and not rows.loc[13, "eligible"]
    assert rows.loc[13, "activity_score"] is None or rows.activity_score.isna().loc[13]


def test_risk_package_changed_during_loading_is_rejected(tmp_path, monkeypatch):
    from s2s_decision import contextual_decision as bridge

    result, candidates, basket = workspace()
    snapshot = snapshot_from_workspace(result, candidates, basket, SelectionConstraints(2))
    path = package(tmp_path)
    _bundle, context = load_policy_package(path)
    risk = risk_package(tmp_path, context.chemistry_version)
    original_loader = bridge.load_risk_bundle

    def change_after_read(path):
        loaded = original_loader(path)
        write_json(path, {**loaded, "predictor": {**loaded["predictor"], "intercept": 2.}})
        return loaded

    monkeypatch.setattr(bridge, "load_risk_bundle", change_after_read)
    with pytest.raises(ValueError, match="changed during loading"):
        preview_contextual_decision(snapshot, tmp_path / "preview", path, context, risk_path=risk)
    assert not (tmp_path / "preview").exists()
