"""Minimum-core constraints survive native/model decisions, replay and sessions."""

from dataclasses import replace

import pytest

from smiles2select.gui.workspace import criteria_state
from smiles2select.gui.workspace.scenario_dialog import ScenarioDialog
from smiles2select.gui.workspace.workspace_window import WorkspaceWindow
from smiles2select.selection_intelligence import recipes
from smiles2select.selection_intelligence.scenario_io import load_study, save_study
from smiles2select.selection_intelligence.scenarios import evaluate_scenario
from smiles2select.storage.session_store import _decode, _encode
from tests import test_workspace
from tests.test_workspace_model_decision import _proposal

qapp = test_workspace.qapp
result = test_workspace.result
window = test_workspace.window


def test_minimum_core_selection_export_and_session_roundtrip(window, tmp_path):
    window.target_count.setValue(3)
    window.min_scaffolds.setValue(2)
    assert window.has_pending_criteria()
    window._auto_select()
    assert not window.has_pending_criteria()
    artifacts = window.build_artifacts()
    assert artifacts.constraints.min_scaffolds == 2
    assert artifacts.outcome.scaffolds_covered >= 2
    assert recipes.from_dict(artifacts.recipe.as_dict()).min_scaffolds == 2
    assert "Minimum molecular cores: 2" in window.criteria_summary.text()
    restored = WorkspaceWindow.from_session(window.save_session(tmp_path / "minimum.s2s.sqlite"))
    try:
        assert restored.min_scaffolds.value() == 2
        assert restored.build_artifacts().constraints.min_scaffolds == 2
        restored._undo()
        assert restored.min_scaffolds.value() == 0
        restored._redo()
        assert restored.min_scaffolds.value() == 2
        assert not restored.has_pending_criteria()
    finally:
        restored.close()


def test_infeasible_minimum_warns_without_inventing_cores(window):
    window.target_count.setValue(1)
    window.min_scaffolds.setValue(2)
    window._auto_select()
    assert len(window.basket.final_ids()) == 1
    assert "requested minimum" in window.warnings.text().lower()
    assert window.build_artifacts().outcome.scaffolds_covered == 1


def test_scenario_minimum_is_editable_and_reversible(window, tmp_path):
    dialog = ScenarioDialog(window)
    try:
        spec = replace(dialog.capture_spec(), constraints=replace(
            window.constraints(), target_count=3, min_scaffolds=2))
        snapshot = evaluate_scenario(window.result, window.candidates, spec)
        study = tmp_path / "minimum.scenarios.json"
        save_study(study, (snapshot,))
        snapshot, = load_study(study, window.result, window.candidates)
        assert snapshot.spec.constraints.min_scaffolds == 2
        dialog.accept_snapshot(snapshot)
        dialog._load_slot()
        dialog.adopt()
        assert window.constraints() == spec.constraints
        assert not window.has_pending_criteria()
        window._undo()
        assert window.min_scaffolds.value() == 0
        window._redo()
        assert window.min_scaffolds.value() == 2
    finally:
        dialog.close()


def test_model_preview_detects_minimum_changes_and_persists_adopted_constraint(window, tmp_path):
    window.min_scaffolds.setValue(1)
    preview = _proposal(window, tmp_path / "proposal")
    assert preview.snapshot.constraints.min_scaffolds == 1
    window.model_panel.accept_preview(preview)
    before = window.basket.final_ids()
    window.min_scaffolds.setValue(2)
    window.model_panel.adopt()
    assert window.basket.final_ids() == before
    assert "changed" in window.model_panel.report.toPlainText().lower()
    window.min_scaffolds.setValue(1)
    window.model_panel.adopt()
    assert window.build_artifacts().constraints.min_scaffolds == 1
    restored = WorkspaceWindow.from_session(window.save_session(tmp_path / "model.s2s.sqlite"))
    try:
        assert restored.min_scaffolds.value() == 1
        assert restored.build_artifacts().constraints.min_scaffolds == 1
    finally:
        restored.close()


def test_340_criteria_compatibility_accepts_only_known_missing_field(window):
    payload = _encode(criteria_state.capture(window))
    legacy = {**payload, "fields": {key: value for key, value in payload["fields"].items()
                                  if key != "minimum_scaffolds"}}
    assert _decode(legacy).minimum_scaffolds == 0
    broken = {**legacy, "fields": {key: value for key, value in legacy["fields"].items()
                                  if key != "count"}}
    with pytest.raises(ValueError, match="fields changed"):
        _decode(broken)
