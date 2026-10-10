"""Actual Qt tolerance proposal, persistence and stale-state behavior."""

import json
import time
from concurrent.futures import CancelledError
from dataclasses import replace
from threading import Event

import pandas as pd
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from s2s_decision.artifacts import read_bundle
from s2s_decision.tolerance_profile import fit_tolerance, write_tolerance_package
from smiles2select.export import selection_export
from smiles2select.gui.workspace.workspace_window import WorkspaceWindow
from tests import test_workspace

qapp = test_workspace.qapp
result = test_workspace.result
window = test_workspace.window


def package(tmp_path):
    frame = pd.DataFrame({"model_smiles": [smiles for _, smiles in test_workspace.LIBRARY],
                          "pactivity": [7.] * 6})
    profile = fit_tolerance(frame, mode="general", recipe="A",
                            support_settings={"min_identities": 1, "min_scaffolds": 1,
                                              "min_documents": 0})
    return write_tolerance_package(profile, tmp_path / "tolerance-model")


def wait_preview(window, controls):
    wait_job(window)
    assert controls._preview is not None, controls.report.toPlainText()


def wait_job(window):
    deadline = time.monotonic() + 60
    while window.is_busy and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(.01)
    QApplication.processEvents()
    assert not window.is_busy


def test_tolerance_real_preview_adopt_undo_export_reopen(window, tmp_path, monkeypatch):
    before = window.basket.final_ids()
    controls = window.model_panel.tolerance
    controls.load_package(package(tmp_path))
    window.target_count.setValue(3)
    controls.preview()
    wait_preview(window, controls)
    assert window.basket.final_ids() == before
    rows = read_bundle(controls._preview.root / "proposed").records
    assert "tolerance_state" in rows
    assert "activity_score" not in rows
    import smiles2select.gui.workspace.tolerance_controls as controls_module

    def no_gui_bundle_reload(*_args, **_kwargs):
        raise AssertionError("Molecule inspection must use worker-loaded evidence")

    monkeypatch.setattr(controls_module, "read_bundle", no_gui_bundle_reload)
    controls.inspect_row(0, 0)
    assert "Tolerance" in window.inspector.details.toPlainText()
    monkeypatch.setattr(QMessageBox, "question", lambda *_a: QMessageBox.Yes)
    controls.adopt()
    wait_job(window)
    assert len(window.basket.final_ids()) == 3, controls.report.toPlainText()
    artifacts = window.build_artifacts()
    assert artifacts.recipe.strategy == "chemical_tolerance"
    assert "tolerance_evidence" in artifacts.model_scores
    assert artifacts.recipe.model["tolerance_profile"]["policy"]["status"] == "uncalibrated"
    assert artifacts.recipe.model["tolerance_profile"]["support_settings"]["min_identities"] == 1
    assert artifacts.recipe.model["profile_sha256"]
    workbook, recipe = selection_export.export(artifacts, tmp_path / "tolerance.xlsx")
    assert workbook.is_file()
    assert json.loads(recipe.read_text(encoding="utf-8"))["schema_version"] == "6.0"
    window._show_molecule(1)
    assert "Tolerance" in window.inspector.details.toPlainText()
    window._undo()
    assert window.basket.final_ids() == before
    assert window.build_artifacts().model_scores is None
    window._redo()
    path = tmp_path / "session.sqlite"
    window.save_session(path)
    controls._package.rename(tmp_path / "package-unavailable")
    restored = WorkspaceWindow.from_session(path)
    try:
        assert restored.build_artifacts().recipe.strategy == "chemical_tolerance"
        restored._show_molecule(1)
        assert "Tolerance" in restored.inspector.details.toPlainText()
        restored._undo()
        assert restored.basket.final_ids() == before
    finally:
        restored.close()


def test_workspace_configured_cache_is_reused_during_adoption(window, tmp_path, monkeypatch):
    from s2s_decision import tolerance_features

    cache_path = tmp_path / "workspace-chemistry.sqlite"
    monkeypatch.setattr(window, "result", replace(window.result,
        config=replace(window.result.config, cache_path=cache_path)))
    controls = window.model_panel.tolerance
    controls.load_package(package(tmp_path))
    window.target_count.setValue(3)
    controls.preview()
    wait_preview(window, controls)
    assert cache_path.is_file()
    proposed = tuple(controls._preview.summary["proposed_ids"])

    def no_native_recomputation(*_args):
        raise AssertionError("Configured workspace cache must be reused during replay")

    monkeypatch.setattr(tolerance_features, "_criterion_annotations", no_native_recomputation)
    monkeypatch.setattr(QMessageBox, "question", lambda *_a: QMessageBox.Yes)
    controls.adopt()
    wait_job(window)
    assert set(window.basket.final_ids()) == set(proposed), controls.report.toPlainText()


def test_tolerance_changed_count_and_tampered_package_block_adoption(window, tmp_path, monkeypatch):
    controls = window.model_panel.tolerance
    path = package(tmp_path)
    controls.load_package(path)
    controls.preview()
    wait_preview(window, controls)
    before = window.basket.final_ids()
    monkeypatch.setattr(QMessageBox, "question", lambda *_a: QMessageBox.Yes)
    window.target_count.setValue(2)
    controls.adopt()
    wait_job(window)
    assert window.basket.final_ids() == before
    assert "changed" in controls.report.toPlainText().lower()
    controls.preview()
    wait_preview(window, controls)
    profile = path / "profile.json"
    profile.write_text(profile.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    controls.adopt()
    wait_job(window)
    assert window.basket.final_ids() == before
    assert "changed" in controls.report.toPlainText().lower()


def test_tolerance_context_change_invalidates_preview(window, tmp_path):
    controls = window.model_panel.tolerance
    controls.load_package(package(tmp_path))
    controls.preview()
    wait_preview(window, controls)
    controls.workflow.setCurrentText("Target-conditioned")
    assert controls._preview is None
    assert not controls.adopt_button.isEnabled()


def test_tolerance_cancelled_job_does_not_leave_proposal(window, tmp_path, monkeypatch):
    from s2s_decision import tolerance_decision

    controls = window.model_panel.tolerance
    controls.load_package(package(tmp_path))
    before = window.basket.final_ids()

    def cancelled_preview(*_args, cancelled=None, **_kwargs):
        deadline = time.monotonic() + 5
        while not cancelled() and time.monotonic() < deadline:
            time.sleep(.01)
        raise CancelledError("Tolerance preview cancelled")

    monkeypatch.setattr(tolerance_decision, "preview_tolerance_decision", cancelled_preview)
    controls.preview()
    window._cancel_job()
    deadline = time.monotonic() + 10
    while window.is_busy and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(.01)
    QApplication.processEvents()
    assert not window.is_busy
    assert controls._preview is None
    assert "cancelled" in controls.report.toPlainText().lower()
    assert window.basket.final_ids() == before


def test_tolerance_adoption_is_cancelable_before_basket_changes(window, tmp_path, monkeypatch):
    from s2s_decision import tolerance_decision
    from s2s_decision.decision import verify_preview

    controls = window.model_panel.tolerance
    controls.load_package(package(tmp_path))
    controls.preview()
    wait_preview(window, controls)
    before = window.basket.final_ids()
    summary = verify_preview(controls._preview.root)
    proposed = read_bundle(controls._preview.root / "proposed")
    started, release = Event(), Event()

    def slow_verify(*_args, **_kwargs):
        started.set()
        assert release.wait(10)
        return summary, proposed

    monkeypatch.setattr(tolerance_decision, "verify_tolerance_decision", slow_verify)
    controls.adopt()
    deadline = time.monotonic() + 10
    while not started.is_set() and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(.01)
    assert started.is_set()
    assert window.is_busy
    assert window.basket.final_ids() == before
    window._cancel_job()
    release.set()
    wait_job(window)
    assert "cancelled" in controls.report.toPlainText().lower()
    assert window.basket.final_ids() == before


def test_tolerance_gui_passes_exact_selected_rule_scope(window, tmp_path, monkeypatch):
    from s2s_decision import tolerance_decision

    controls = window.model_panel.tolerance
    controls.load_package(package(tmp_path))
    item = next(controls.criteria.item(row, 0) for row in range(controls.criteria.rowCount())
                if controls.criteria.item(row, 0).data(Qt.UserRole)[0] == "rule:lipinski:lip_mw_max")
    item.setCheckState(Qt.Checked)
    actual_snapshot = tolerance_decision.tolerance_snapshot_from_workspace
    calls = []

    def snapshot(*args, **kwargs):
        calls.append(kwargs)
        return actual_snapshot(*args, **kwargs)

    monkeypatch.setattr(tolerance_decision, "tolerance_snapshot_from_workspace", snapshot)
    monkeypatch.setattr(QMessageBox, "question", lambda *_a: QMessageBox.Yes)
    controls.preview()
    wait_preview(window, controls)
    controls.adopt()
    wait_job(window)
    assert len(calls) == 2
    assert all(call["reconsidered_criteria"] == ("rule:lipinski:lip_mw_max",) for call in calls)
    assert all(call["revisited_profiles"] == ("lipinski",) for call in calls)


def test_calibrated_criteria_button_rejects_unavailable_scope(window):
    controls = window.model_panel.tolerance
    controls._profile = {"policy": {"scope": {"criteria": ["rule:lipinski:lip_mw_max"]}}}
    controls.use_calibrated_criteria()
    assert controls.reconsidered_criteria() == ("rule:lipinski:lip_mw_max",)
    controls._profile = {"policy": {"scope": {"criteria": ["rule:missing:criterion"]}}}
    controls.use_calibrated_criteria()
    assert "unavailable or locked" in controls.report.toPlainText()
    assert controls.reconsidered_criteria() == ("rule:lipinski:lip_mw_max",)


def test_package_import_context_and_custom_training_handoff(window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    controls = window.model_panel.tolerance
    path = package(tmp_path)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_a: (str(path / "manifest.json"), ""))
    controls.browse_package()
    assert controls._package == path.resolve()
    assert "Training:" in controls.package_label.text()
    assert "Support gates:" in controls.package_label.text()
    controls.workflow.setCurrentText("Target-conditioned")
    with pytest.raises(ValueError, match="requires"):
        controls.context()
    controls.target.setText("P22303")
    controls.species.setText("Homo sapiens")
    controls.endpoint.setText("IC50")
    assert controls.context() == {"target_id": "P22303", "species": "Homo sapiens", "endpoint": "IC50"}
    item = next(controls.criteria.item(row, 0) for row in range(controls.criteria.rowCount())
                if controls.criteria.item(row, 0).data(Qt.UserRole)[0] == "rule:lipinski:lip_mw_max")
    item.setCheckState(Qt.Checked)
    controls.train_custom()
    try:
        assert controls._studio.action.currentText() == "tolerance-train"
        assert controls._studio.fields["target"].text() == "P22303"
        assert controls._studio.fields["endpoint"].currentText() == "IC50"
        assert controls._studio.fields["reconsider-criteria"].text() == "rule:lipinski:lip_mw_max"
        assert not controls.training_busy
    finally:
        controls._studio.close()
    controls._profile = {**controls._profile, "context": controls.context()}
    controls.workflow.setCurrentText("General")
    with pytest.raises(ValueError, match="task-specific"):
        controls.context()
    controls.preview()
    assert "task-specific" in controls.report.toPlainText()
    bad = tmp_path / "invalid.json"
    bad.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_a: (str(bad), ""))
    controls.browse_package()
    assert "Cannot load tolerance package" in controls.report.toPlainText()


def test_included_profile_load_is_explicit_and_does_not_change_basket(window, tmp_path, monkeypatch):
    import smiles2select.gui.workspace.tolerance_controls as controls_module

    base = package(tmp_path)
    directory = tmp_path / "bundled_tolerance_models"
    directory.mkdir()
    directory.joinpath("general_A.json").write_bytes(base.joinpath("profile.json").read_bytes())
    monkeypatch.setattr(controls_module, "files", lambda _name: tmp_path)
    controls = controls_module.ToleranceControls(window, window)
    before = window.basket.final_ids()
    assert controls._package is None
    assert not controls.load_included_button.isEnabled()
    controls.included_profiles.setCurrentIndex(1)
    assert controls.load_included_button.isEnabled()
    controls.load_included()
    assert controls._package == directory.joinpath("general_A.json").resolve()
    assert controls._preview is None
    assert window.basket.final_ids() == before
