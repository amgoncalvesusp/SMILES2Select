"""Real Qt jobs must apply the requested count before either GUI export."""

import time
from dataclasses import replace

import pandas as pd
import pytest
from PySide6.QtCore import QThread

from smiles2select.gui.workspace.workspace_window import WorkspaceWindow
from tests import test_workspace

qapp = test_workspace.qapp
result = test_workspace.result
window = test_workspace.window


def wait_for_jobs(window, qapp):
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.005)
        if not window.is_busy:
            qapp.processEvents()
            if not window.is_busy:
                return
    raise AssertionError(f"Job did not finish: {window.warnings.text()}")


def expanded_result(result, copies=1001):
    def expand(frame):
        frame = pd.concat([frame] * copies, ignore_index=True)
        frame.index = pd.RangeIndex(1, len(frame) + 1, name="record_id")
        return frame

    descriptors = expand(result.descriptors).drop(columns=["murcko_scaffold"], errors="ignore")
    descriptors = descriptors.assign(molecule_id=[f"REC{i:07}" for i in descriptors.index])
    return replace(
        result, descriptors=descriptors, records=expand(result.records), scores=expand(result.scores),
        evaluation=replace(result.evaluation, status=expand(result.evaluation.status)),
        decision=replace(result.decision, decisions=expand(result.decision.decisions)),
    )


def test_real_large_scaffold_job_applies_2000_and_exports_exact_count(qapp, result, tmp_path, monkeypatch):
    hub = WorkspaceWindow(expanded_result(result))
    try:
        assert len(hub.basket.final_ids()) > 2000
        assert hub.candidates.murcko_scaffold.isna().all()
        hub.target_count.setValue(2000)
        hub.strategy.setCurrentText("scaffold_coverage")
        hub.select_button.click()
        wait_for_jobs(hub, qapp)
        assert len(hub.basket.final_ids()) == 2000, hub.warnings.text()
        assert hub.candidates.murcko_scaffold.notna().all()
        assert len(hub.map_view.selected_scatter.points()) == 2000
        assert "Building" not in hub.warnings.text()
        path = tmp_path / "final.csv"
        monkeypatch.setattr(
            "smiles2select.gui.workspace.workspace_window.QFileDialog.getSaveFileName",
            lambda *a: (str(path), "CSV (*.csv)"),
        )
        monkeypatch.setattr(
            "smiles2select.gui.workspace.workspace_window.QMessageBox.information", lambda *a: None,
        )
        hub._export_docking()
        assert len(pd.read_csv(path)) == 2000
    finally:
        wait_for_jobs(hub, qapp)
        hub.close()


@pytest.mark.parametrize("method", ["_export", "_export_docking"])
def test_pending_count_blocks_export_before_file_dialog(window, monkeypatch, method):
    window.target_count.setValue(2)
    monkeypatch.setattr(
        "smiles2select.gui.workspace.workspace_window.QFileDialog.getSaveFileName",
        lambda *a: (_ for _ in ()).throw(AssertionError("Stale selection must not be exported")),
    )
    getattr(window, method)()
    assert "Create selection" in window.warnings.text()


def test_job_completion_failure_is_visible_and_controls_recover(window, qapp):
    original = window.basket.final_ids()
    def broken(_value):
        assert QThread.currentThread() == qapp.thread()
        raise ValueError("Completion could not apply the selection")

    window._start_job(lambda: 1, broken, "Working...")
    wait_for_jobs(window, qapp)
    assert "Completion could not apply" in window.warnings.text()
    assert window.controls_scroll.isEnabled()
    assert window.basket.final_ids() == original


def test_successful_map_does_not_leave_building_message(window, qapp):
    window._request_map()
    wait_for_jobs(window, qapp)
    assert "Building" not in window.warnings.text()


def test_undo_restores_applied_controls_and_export_readiness(window):
    window.target_count.setValue(2)
    window._auto_select()
    window.target_count.setValue(3)
    window._auto_select()
    window._undo()
    assert window.target_count.value() == 2
    assert not window.has_pending_criteria()
    assert window.basket_panel.docking_button.isEnabled()
    assert "3 / 3" not in window.warnings.text()
    assert "3 / 3" not in window._selection_message


def test_lazy_scenario_export_keeps_calculated_scaffolds(window):
    from smiles2select.gui.workspace.scenario_dialog import ScenarioDialog
    from smiles2select.selection_intelligence.scenarios import evaluate_scenario

    window.candidates = window.candidates.assign(murcko_scaffold=pd.NA)
    window.strategy.setCurrentText("scaffold_coverage")
    dialog = ScenarioDialog(window)
    snapshot = evaluate_scenario(window.result, window.candidates, dialog.capture_spec())
    dialog.accept_snapshot(snapshot)
    dialog.adopt()
    artifacts = window.build_artifacts()
    assert artifacts.outcome.scaffold_usage == snapshot.outcome.scaffold_usage
    assert artifacts.scaffolds.reindex(artifacts.outcome.selected_ids).notna().all()
    assert window.candidates.murcko_scaffold.isna().all()
    dialog.close()
