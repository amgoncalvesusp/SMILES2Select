"""Advanced Decision tools belong to the same desktop application."""

import pandas as pd
import shiboken6
from PySide6.QtCore import QCoreApplication, QEvent

from tests import test_gui, test_workspace

qapp = test_gui.qapp
window = test_gui.window
result = test_workspace.result


def test_advanced_tools_open_as_owned_child(window):
    window._open_advanced_tools()
    studio = window._advanced_tools
    assert studio is not None
    assert studio.parent() is window
    assert "SMILES2Select" in studio.windowTitle()
    assert studio.action.findText("train") >= 0 or studio.mode.findText("Training") >= 0
    studio.close()


def test_saved_workspace_opens_from_main_window_and_workspace_signal(
    window, result, tmp_path, monkeypatch,
):
    from smiles2select.gui.workspace.workspace_window import WorkspaceWindow

    original = WorkspaceWindow(result, parent=window)
    path = original.save_session(tmp_path / "session.s2s.sqlite")
    original.close()
    monkeypatch.setattr(
        "smiles2select.gui.main_window.QFileDialog.getOpenFileName",
        lambda *_args: (str(path), ""),
    )
    window.open_session_button.click()
    restored = window._workspace_window
    assert restored is not None
    assert restored.basket.final_ids() == original.basket.final_ids()
    restored.session_open_requested.emit(str(path))
    assert window._workspace_window is not restored
    assert window._workspace_window.basket.final_ids() == original.basket.final_ids()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not shiboken6.isValid(restored)
    window._workspace_window.close()


def test_session_preserves_prepared_candidates_for_export(qapp, result, tmp_path):
    from smiles2select.gui.workspace.workspace_window import WorkspaceWindow

    original = WorkspaceWindow(result)
    original.candidates = original.candidates.assign(murcko_scaffold="persisted-core")
    before = original.build_artifacts()
    saved = original.save_session(tmp_path / "prepared.s2s.sqlite")
    restored = WorkspaceWindow.from_session(saved)
    try:
        pd.testing.assert_frame_equal(restored.candidates, original.candidates)
        assert restored.build_artifacts().outcome.scaffold_usage == before.outcome.scaffold_usage
    finally:
        original.close()
        restored.close()


def test_large_selection_scaffolds_survive_session_reopen(qapp, tmp_path):
    from benchmarks.benchmark_gui_selection import make_result, wait_for_jobs
    from smiles2select.export.selection_export import final_selected_sheet
    from smiles2select.gui.workspace.workspace_window import WorkspaceWindow

    result = make_result(tmp_path, total=5001, evaluable=5001, original=100)
    original = WorkspaceWindow(result)
    try:
        assert original.candidates.murcko_scaffold.isna().all()
        original.target_count.setValue(12)
        original.strategy.setCurrentText("scaffold_coverage")
        original.select_button.click()
        wait_for_jobs(original, qapp)
        assert original.candidates.murcko_scaffold.notna().all()
        before = original.build_artifacts()
        saved = original.save_session(tmp_path / "large-selection.s2s.sqlite")
        restored = WorkspaceWindow.from_session(saved)
        try:
            after = restored.build_artifacts()
            assert after.outcome.scaffold_usage == before.outcome.scaffold_usage
            pd.testing.assert_frame_equal(final_selected_sheet(after), final_selected_sheet(before))
        finally:
            restored.close()
    finally:
        original.close()
