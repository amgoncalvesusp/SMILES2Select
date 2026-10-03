"""Real portable inference through the existing Qt preview/adopt/undo workflow."""

import json
import time

import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from s2s_decision.artifacts import read_bundle
from smiles2select.export import selection_export
from smiles2select.gui.workspace.workspace_window import WorkspaceWindow
from smiles2select.selection_intelligence import recipes
from tests import test_workspace
from tests.decision.test_contextual_decision import package, risk_package

qapp = test_workspace.qapp
result = test_workspace.result
window = test_workspace.window


def wait_preview(window):
    deadline = time.monotonic() + 60
    while window.is_busy and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(.01)
    QApplication.processEvents()
    assert not window.is_busy
    controls = window.model_panel.contextual
    assert controls._preview is not None, controls.report.toPlainText()


def ready(window, tmp_path):
    controls = window.model_panel.contextual
    controls.load_package(package(tmp_path))
    window.target_count.setValue(6)
    controls.preview()
    wait_preview(window)
    return controls


def test_contextual_real_inference_recovery_adoption_undo_export(window, tmp_path, monkeypatch):
    before = window.basket.final_ids()
    assert 6 not in before
    controls = ready(window, tmp_path)
    assert window.basket.final_ids() == before
    assert controls.target.text() == "T"
    assert controls.review.rowCount() == 6
    proposed = read_bundle(controls._preview.root / "proposed").records
    assert proposed.is_final.sum() == 6
    monkeypatch.setattr(QMessageBox, "question", lambda *_a: QMessageBox.Yes)
    controls.adopt()
    assert len(window.basket.final_ids()) == 6, controls.report.toPlainText()
    artifacts = window.build_artifacts()
    assert artifacts.recipe.strategy == "contextual_policy"
    assert artifacts.recipe.provenance["context"]["target"] == "T"
    assert "rule_evidence" in artifacts.model_scores
    workbook, recipe = selection_export.export(artifacts, tmp_path / "contextual-report.xlsx")
    assert workbook.exists()
    payload = json.loads(recipe.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "5.0"
    assert payload["final_selection"]["strategy"] == "contextual_policy"
    assert recipes.load(recipe).model["context"]["target"] == "T"
    controls.inspect_row(0, 0)
    assert "Policy explanation" in window.inspector.details.toPlainText()
    window._show_molecule(6)
    assert "Activity" in window.inspector.details.toPlainText()
    assert "Risk" in window.inspector.details.toPlainText()
    window._undo()
    assert window.basket.final_ids() == before
    assert window.build_artifacts().model_scores is None
    window._redo()
    assert window.build_artifacts().recipe.strategy == "contextual_policy"
    path = tmp_path / "workspace.sqlite"
    window.save_session(path)
    restored = WorkspaceWindow.from_session(path)
    try:
        assert "rule_evidence" in restored.build_artifacts().model_scores
        restored._undo()
        assert restored.basket.final_ids() == before
    finally:
        restored.close()


def test_contextual_stale_controls_and_tampered_package_block_adoption(window, tmp_path, monkeypatch):
    controls = ready(window, tmp_path)
    before = window.basket.final_ids()
    monkeypatch.setattr(QMessageBox, "question", lambda *_a: QMessageBox.Yes)
    window.target_count.setValue(2)
    controls.adopt()
    assert window.basket.final_ids() == before
    assert "changed" in controls.report.toPlainText().lower()
    controls.stage.setCurrentText("lead")
    assert controls._preview is None
    assert not controls.adopt_button.isEnabled()


def test_contextual_manual_exclusion_and_smarts_remain_binding(window, tmp_path, monkeypatch):
    window.basket.exclude([2])
    controls = window.model_panel.contextual
    controls.load_package(package(tmp_path))
    controls.required_smarts.setPlainText("c1ccccc1")
    window.target_count.setValue(6)
    controls.preview()
    wait_preview(window)
    rows = read_bundle(controls._preview.root / "proposed").records
    selected = set(rows.loc[rows.is_final, "record_id"])
    assert 2 not in selected and 6 not in selected
    monkeypatch.setattr(QMessageBox, "question", lambda *_a: QMessageBox.Yes)
    controls.adopt()
    assert set(window.basket.final_ids()) == selected


def test_optional_risk_and_review_queue_visible_exported_and_tamper_checked(window, tmp_path, monkeypatch):
    controls = window.model_panel.contextual
    controls.load_package(package(tmp_path))
    risk = risk_package(tmp_path, controls.context().chemistry_version)
    controls.load_risk(risk)
    controls.review_count.setValue(2)
    controls.risk_weight.setValue(.2)
    controls.risk_exclude.setChecked(True)
    controls.risk_cutoff.setValue(.8)
    controls.preview()
    wait_preview(window)
    assert controls.review.rowCount() == 2
    assert "hepg2_atp_viability_16h" in controls.risk_label.text()
    monkeypatch.setattr(QMessageBox, "question", lambda *_a: QMessageBox.Yes)
    before = window.basket.final_ids()
    risk.write_text(risk.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    controls.adopt()
    assert window.basket.final_ids() == before
    assert "differs" in controls.report.toPlainText()
    controls.clear_risk()
    assert controls._preview is None
    assert controls.risk_weight.value() == 0
    assert not controls.risk_exclude.isChecked()


def test_bad_smarts_fails_without_changing_basket(window, tmp_path):
    controls = window.model_panel.contextual
    controls.load_package(package(tmp_path))
    controls.required_smarts.setPlainText("C1CC")
    before = window.basket.final_ids()
    controls.preview()
    deadline = time.monotonic() + 15
    while window.is_busy and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(.01)
    QApplication.processEvents()
    assert controls._preview is None
    assert "SMARTS" in controls.report.toPlainText()
    assert window.basket.final_ids() == before


def test_individual_action_edit_invalidates_preview_and_duplicate_fails(window, tmp_path):
    controls = ready(window, tmp_path)
    controls.add_action(kind="rule", identifier="lip_mw_max", action="penalize", penalty=.2)
    assert controls._preview is None
    assert controls.overrides()["rule_actions"][0].origin == "user"
    controls.add_action(kind="rule", identifier="lip_mw_max")
    controls.preview()
    assert "duplicate" in controls.report.toPlainText().lower()
    controls.actions.selectRow(1)
    controls.remove_action()
    assert len(controls.overrides()["rule_actions"]) == 1
    controls.add_action(kind="alert")
    assert controls.actions.cellWidget(1, 1).currentText().startswith(("pains:", "brenk:"))
    controls.add_action(kind="profile", identifier="lipinski", action="exclude", max_violations=2)
    assert controls.overrides()["profiles"][0].max_violations == 2


def test_contextual_recipe_rejects_missing_context_or_wrong_strategy():
    model = {"model_sha256": "fixture", "preview_sha256": "fixture", "policy_settings": {},
             "context": dict.fromkeys(("target", "species", "endpoint", "stage", "assay_context", "source_version", "model_version", "chemistry_version"), "fixture")}
    payload = recipes.SelectionRecipe(strategy="contextual_policy", model=model).as_dict()
    assert recipes.from_dict(payload).strategy == "contextual_policy"
    with pytest.raises(recipes.RecipeError, match="contextual recipe"):
        recipes.from_dict({**payload, "model": {**model, "context": {}}})
    with pytest.raises(recipes.RecipeError, match="contextual schema"):
        recipes.from_dict({**payload, "final_selection": {"strategy": "balanced"}})


def test_lazy_scaffolds_and_risk_context_survive_adopt_save_restore(window, tmp_path, monkeypatch):
    window.candidates["murcko_scaffold"] = pd.NA
    controls = window.model_panel.contextual
    controls.load_package(package(tmp_path))
    controls.load_risk(risk_package(tmp_path, controls.context().chemistry_version))
    window.target_count.setValue(6)
    controls.preview()
    wait_preview(window)
    monkeypatch.setattr(QMessageBox, "question", lambda *_a: QMessageBox.Yes)
    controls.adopt()
    artifacts = window.build_artifacts()
    assert artifacts.scaffolds.notna().all()
    assert sum(artifacts.outcome.scaffold_usage.values()) == 6
    path = tmp_path / "lazy-workspace.sqlite"
    window.save_session(path)
    restored = WorkspaceWindow.from_session(path)
    try:
        assert sum(restored.build_artifacts().outcome.scaffold_usage.values()) == 6
        restored._show_molecule(1)
        text = restored.inspector.details.toPlainText()
        assert "hepg2_atp_viability_16h" in text
        assert "Risk prediction set" in text and "Risk evidence source" in text
        restored._undo()
        assert restored.build_artifacts().model_scores is None
    finally:
        restored.close()
