"""Real Qt preview/adoption/undo and portable study export checks."""

import json
import time
from dataclasses import replace

import pandas as pd
from PySide6.QtWidgets import QApplication

from smiles2select.export.selection_export import export
from smiles2select.gui.workspace.scenario_dialog import ScenarioDialog
from smiles2select.selection_intelligence.scenario_io import load_study
from smiles2select.selection_intelligence.scenarios import evaluate_scenario
from tests import test_workspace

qapp = test_workspace.qapp
result = test_workspace.result
window = test_workspace.window


def wait_job(dialog):
    for _ in range(2000):
        QApplication.processEvents()
        time.sleep(0.01)
        if not dialog.is_busy:
            return
    raise AssertionError("GUI computation did not finish")


def preview(dialog, count=2):
    dialog.count.setValue(count)
    dialog.preview()
    wait_job(dialog)
    assert dialog.slot.currentText() in dialog.snapshots, dialog.report.toPlainText()


def test_preview_duplicate_comparison_never_mutates_selection(window, tmp_path):
    window.target_count.setValue(3)
    window._auto_select()
    before = window.basket.final_ids()
    dialog = ScenarioDialog(window)
    preview(dialog, 2)
    dialog.duplicate_a()
    preview(dialog, 1)
    dialog.compare()
    wait_job(dialog)
    assert "Stable core" in dialog.report.toPlainText()
    assert window.basket.final_ids() == before
    path = tmp_path / "stability.csv"
    dialog.export_stability(path)
    wait_job(dialog)
    table = pd.read_csv(path)
    assert len(table) == len(window.candidates)
    assert set(table.selection_frequency).issubset({0, 0.5, 1})


def test_scenario_adoption_is_one_undoable_action_and_export_is_frozen(window, tmp_path):
    window.target_count.setValue(3)
    window._auto_select()
    original = window.basket.final_ids()
    dialog = ScenarioDialog(window)
    preview(dialog, 1)
    snapshot = dialog.snapshots["A"]
    old_actions = len(window.basket.log.applied)
    dialog.adopt()
    assert len(window.basket.log.applied) == old_actions + 1
    assert tuple(window.basket.final_ids()) == tuple(snapshot.outcome.selected_ids)
    window.target_count.setValue(77)
    artifacts = window.build_artifacts()
    assert artifacts.recipe.target_count == 1
    assert artifacts.recipe.name == "scenario:A"
    assert artifacts.recipe.input_hash == snapshot.data_fingerprint
    workbook, recipe = export(artifacts, tmp_path / "selected.xlsx")
    assert len(pd.read_excel(workbook, sheet_name="FINAL_SELECTED")) == 1
    payload = json.loads(recipe.read_text(encoding="utf-8"))
    assert payload["final_selection"]["target_count"] == 1
    window._undo()
    assert window.basket.final_ids() == original
    assert window._active_snapshot() is None
    window._redo()
    assert window._active_snapshot() is snapshot


def test_saved_study_replays_frozen_thresholds(window, tmp_path):
    dialog = ScenarioDialog(window)
    rule = dialog.rule.currentData()
    dialog.low.setValue(450)
    dialog._set_threshold()
    preview(dialog)
    path = tmp_path / "study.json"
    dialog.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["scenarios"][0]["spec"]["thresholds"][rule.id] == 450
    replay = load_study(path, window.result, window.candidates)
    assert replay[0].outcome.selected_ids == dialog.snapshots["A"].outcome.selected_ids


def test_adoption_refuses_stale_manual_choices(window):
    dialog = ScenarioDialog(window)
    preview(dialog, 1)
    other = next(s.record_id for s in window.basket.states()
                 if s.chemical_status.passed and s.record_id not in dialog.snapshots["A"].outcome.selected_ids)
    window.basket.add_to_final([other])
    window.basket.pin([other])
    before = window.basket.final_ids()
    dialog.adopt()
    assert "Manual decisions changed" in dialog.report.toPlainText()
    assert window.basket.final_ids() == before


def test_worker_failure_recovers_controls_and_allows_close(window, monkeypatch):
    dialog = ScenarioDialog(window)
    monkeypatch.setattr("smiles2select.gui.workspace.scenario_dialog.evaluate_scenario",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("Invalid threshold")))
    dialog.preview()
    wait_job(dialog)
    assert "Invalid threshold" in dialog.report.toPlainText()
    assert dialog.preview_button.isEnabled()
    assert not window.basket.final_ids()
    assert dialog.close()


def test_new_preview_invalidates_previous_stability(window):
    dialog = ScenarioDialog(window)
    preview(dialog)
    dialog.duplicate_a()
    dialog.compare()
    wait_job(dialog)
    assert dialog._stability is not None
    dialog.duplicate_a()
    assert dialog._stability is None


def test_draft_edits_require_a_new_preview_before_adoption(window):
    dialog = ScenarioDialog(window)
    preview(dialog)
    dialog.count.setValue(1)
    dialog.adopt()
    assert "Draft criteria changed" in dialog.report.toPlainText()
    assert not window.basket.final_ids()


def test_threshold_relaxation_requires_written_original_policy_override(window):
    dialog = ScenarioDialog(window)
    spec = replace(dialog.capture_spec(), thresholds={rule.id: 10000 for profile in window.result.profiles
                                                     for rule in profile.rules},
                   constraints=replace(dialog.capture_spec().constraints, target_count=6))
    snapshot = evaluate_scenario(window.result, window.candidates, spec)
    assert any(not window.basket.state(rid).chemical_status.passed for rid in snapshot.outcome.selected_ids)
    dialog.accept_snapshot(snapshot)
    dialog._load_slot()
    dialog.adopt()
    assert "Write the scientific justification" in dialog.report.toPlainText()
    dialog.justification.setText("Exploratory expansion with relaxed molecular property criteria")
    dialog.adopt()
    assert set(window.basket.final_ids()) == set(snapshot.outcome.selected_ids)


def test_undo_then_auto_select_discards_replaced_scenario_provenance(window):
    dialog = ScenarioDialog(window)
    preview(dialog, 1)
    dialog.adopt()
    assert window._active_snapshot() is not None
    window._undo()
    window._auto_select()
    assert window._active_snapshot() is None
    assert window.build_artifacts().recipe.name == "workspace"
