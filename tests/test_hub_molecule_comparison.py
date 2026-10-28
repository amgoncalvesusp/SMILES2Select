"""Molecule comparison reads live records without changing scientific decisions."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace

import pandas as pd
import pytest
from PySide6.QtCore import QRect, Qt
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QPushButton, QTableWidgetItem

from smiles2select.gui.workspace import active_criteria, criteria_state
from smiles2select.gui.workspace.panels import InspectorPanel
from smiles2select.gui.workspace.selection_provenance import AppliedSelection
from smiles2select.selection_intelligence.constrained_selection import SelectionOutcome
from tests import test_workspace
from tests.test_workspace_model_decision import _proposal

qapp = test_workspace.qapp
result = test_workspace.result
window = test_workspace.window


def _open(window, record_id=None):
    if record_id is not None:
        window._show_molecule(record_id)
    window._open_molecule_comparison()
    return window._comparison_dialog


def _column(model, label):
    return next(column for column in range(model.columnCount())
                if model.headerData(column, Qt.Horizontal) == label)


def _recipe(window):
    return {key: value for key, value in window.build_artifacts().recipe.as_dict().items()
            if key not in {"created_at", "updated_at"}}


def test_read_only_inspector_has_no_decision_buttons_and_displays_literal_text(qapp):
    panel = InspectorPanel(read_only=True)
    frame = pd.DataFrame(
        {"molecule_id": ['<b>β 🧪</b>'], "canonical_smiles": ["CCO"]},
        index=pd.Index([101], name="record_id"),
    )
    panel.show_molecule(101, frame, {"Evidence": '<img src="bad"> literal'})
    assert panel.findChildren(QPushButton) == []
    assert panel.title.textFormat() == Qt.PlainText
    assert panel.title.text() == '<b>β 🧪</b>'
    assert '<img src="bad"> literal' in panel.details.toPlainText()
    assert not panel.structure.pixmap().isNull()
    panel.close()


def test_clear_removes_stale_record_structure_and_signal_target(qapp):
    panel = InspectorPanel()
    requested = QSignalSpy(panel.pin_requested)
    frame = pd.DataFrame({"molecule_id": ["A"], "canonical_smiles": ["CCO"]}, index=[7])
    panel.show_molecule(7, frame)
    pin = next(button for button in panel.findChildren(QPushButton) if button.text() == "Pin")
    pin.click()
    assert requested.count() == 1 and requested.at(0) == [7]
    panel.clear()
    assert panel.record_id is None
    assert panel.title.text() == "No molecule selected"
    assert panel.details.toPlainText() == ""
    assert panel.structure.pixmap().isNull()
    assert panel._structure_smiles is None
    pin.click()
    assert requested.count() == 1
    panel.clear("Select another molecule")
    assert panel.title.text() == "Select another molecule"
    panel.close()


def test_open_before_inspection_explains_empty_panes_without_choosing_a_record(window):
    assert window.inspector.record_id is None
    states, log = tuple(window.basket.states()), deepcopy(window.basket.log.history())
    window._open_molecule_comparison()
    dialog = window._comparison_dialog
    assert isinstance(dialog, QDialog) and dialog.parent() is window
    assert dialog.isVisible() and not dialog.isModal()
    assert dialog.reference_id is None and dialog.current_id is None
    for panel in (dialog.reference_panel, dialog.current_panel):
        assert panel.record_id is None
        assert panel.title.text().strip()
        assert panel.details.toPlainText() == ""
    assert not dialog.keep_reference_button.isEnabled()
    assert window.inspector.record_id is None
    assert tuple(window.basket.states()) == states and window.basket.log.history() == log


def test_comparator_requires_explicit_reference_and_follows_current_record(window, qapp, monkeypatch):
    first, second = map(int, window.candidates.index[:2])
    extras = window._molecule_extras
    monkeypatch.setattr(window, "_molecule_extras", lambda identifier: {
        **extras(identifier), "Long evidence": "\n".join(f"Evidence {i}" for i in range(100)),
    })
    window._show_molecule(first)
    window._open_molecule_comparison()
    dialog = window._comparison_dialog
    assert isinstance(dialog, QDialog)
    assert dialog.parent() is window
    assert not dialog.isModal()
    assert dialog.reference_id is None
    assert dialog.reference_panel.record_id is None
    assert dialog.current_id == first
    assert dialog.current_panel.record_id == first
    assert dialog.reference_panel.findChildren(QPushButton) == []
    assert dialog.current_panel.findChildren(QPushButton) == []
    pins = window.basket.pinned_ids()
    dialog.keep_reference_button.click()
    assert dialog.reference_id == first
    assert dialog.reference_panel.record_id == first
    assert dialog.current_id == first
    assert dialog.current_panel.record_id is None
    assert window.basket.pinned_ids() == pins
    qapp.processEvents()
    scrollbar = dialog.reference_panel.details.verticalScrollBar()
    assert scrollbar.maximum() > 0
    scrollbar.setValue(scrollbar.maximum())
    previous_scroll = scrollbar.value()
    window._show_molecule(second)
    qapp.processEvents()
    assert dialog.current_id == second
    assert dialog.current_panel.record_id == second
    assert dialog.reference_panel.record_id == first
    assert scrollbar.value() == previous_scroll


def test_comparator_is_one_owned_modeless_dialog_retaining_reference_on_reopen(window, qapp):
    record_id = int(window.candidates.index[0])
    window._show_molecule(record_id)
    window._open_molecule_comparison()
    dialog = window._comparison_dialog
    dialog.keep_reference_button.click()
    dialog.close()
    qapp.processEvents()
    assert not dialog.isVisible()
    window._open_molecule_comparison()
    qapp.processEvents()
    assert window._comparison_dialog is dialog
    assert dialog.isVisible()
    assert dialog.reference_id == record_id
    assert dialog.windowModality() == Qt.NonModal
    window.close()
    qapp.processEvents()
    assert not dialog.isVisible()


def test_comparator_follows_sorted_duplicate_id_rows_map_and_pareto_without_changing_A(
    window, qapp,
):
    first, second = map(int, window.candidates.index[:2])
    frame = window.candidates.copy(deep=True)
    frame.loc[[first, second], "molecule_id"] = "duplicate"
    window.candidates = frame
    window.view_selector.setCurrentText("Table")
    window.resize(1280, 800)
    window.show()
    dialog = _open(window, first)
    dialog.keep_reference_button.click()
    model = window.table_view.model()
    model.sort(_column(model, "MW"), Qt.DescendingOrder)
    qapp.processEvents()
    index = model.index_for_record(second)
    QTest.mouseClick(window.table_view.viewport(), Qt.LeftButton,
                     pos=window.table_view.visualRect(index).center())
    assert dialog.current_id == second
    assert dialog.current_panel.record_id == second
    assert dialog.reference_panel.record_id == first
    assert dialog.reference_panel.title.text() == dialog.current_panel.title.text() == "duplicate"
    assert f"record_id: {first}" in dialog.reference_panel.details.toPlainText()
    assert f"record_id: {second}" in dialog.current_panel.details.toPlainText()
    coordinates = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]},
                               index=[first, second])
    for view in (window.map_view, window.pareto_view):
        view.set_points(coordinates)
        point = next(point for point in view.scatter.points() if point.data() == first)
        view._on_click(view.scatter, [point])
        assert dialog.current_id == first and dialog.current_panel.record_id is None
        point = next(point for point in view.scatter.points() if point.data() == second)
        view._on_click(view.scatter, [point])
        assert dialog.current_id == second and dialog.current_panel.record_id == second
        assert dialog.reference_id == first


def test_same_missing_and_empty_current_records_never_leave_stale_panels(window):
    first, second = map(int, window.candidates.index[:2])
    dialog = _open(window, first)
    dialog.keep_reference_button.click()
    assert dialog.current_id == first
    assert dialog.current_panel.record_id is None
    assert dialog.current_panel.title.text().strip()
    assert dialog.current_panel.details.toPlainText() == ""
    assert dialog.current_panel.structure.pixmap().isNull()
    dialog.set_current_id(second)
    assert dialog.current_panel.record_id == second
    window.candidates = window.candidates.drop(index=second)
    dialog.refresh()
    assert dialog.current_panel.record_id is None
    assert dialog.current_panel.details.toPlainText() == ""
    assert dialog.current_panel.structure.pixmap().isNull()
    assert not dialog.keep_reference_button.isEnabled()
    dialog.set_current_id(None)
    assert dialog.current_id is None and dialog.current_panel.record_id is None
    window.candidates = window.candidates.drop(index=first)
    dialog.refresh()
    assert dialog.reference_id is None
    assert dialog.reference_panel.record_id is None
    assert dialog.reference_panel.details.toPlainText() == ""
    assert dialog.reference_panel.structure.pixmap().isNull()


def test_refresh_reads_replacement_candidate_frame_for_both_records(window):
    first, second = map(int, window.candidates.index[:2])
    dialog = _open(window, first)
    dialog.keep_reference_button.click()
    window._show_molecule(second)
    updated = window.candidates.copy(deep=True)
    updated.loc[first, ["molecule_id", "canonical_smiles", "mol_wt"]] = ["Updated A", "CO", 32.04]
    updated.loc[second, ["molecule_id", "canonical_smiles", "mol_wt"]] = ["Updated B", "CC", 30.07]
    window.candidates = updated.iloc[::-1]
    window.refresh()
    assert dialog.reference_id == first and dialog.current_id == second
    for panel, label, smiles, mw in (
        (dialog.reference_panel, "Updated A", "CO", "MW: 32.040"),
        (dialog.current_panel, "Updated B", "CC", "MW: 30.070"),
    ):
        assert panel.title.text() == label
        assert panel._structure_smiles == smiles
        assert mw in panel.details.toPlainText()


def test_membership_and_manual_decisions_refresh_both_panels_through_undo_redo(window):
    first, second = map(int, window.basket.final_ids()[:2])
    dialog = _open(window, first)
    dialog.keep_reference_button.click()
    window._show_molecule(second)
    for record_id, panel in ((first, dialog.reference_panel), (second, dialog.current_panel)):
        assert "Included in docking export: Yes" in panel.details.toPlainText()
        window._decide(record_id, "exclude")
        assert "Included in docking export: No" in panel.details.toPlainText()
        window._undo()
        assert "Included in docking export: Yes" in panel.details.toPlainText()
        window._redo()
        assert "Included in docking export: No" in panel.details.toPlainText()


def test_adoption_changes_applied_banner_and_membership_but_pending_controls_do_not(
    window, qapp, tmp_path,
):
    first, second = map(int, window.candidates.index[:2])
    dialog = _open(window, first)
    dialog.keep_reference_button.click()
    window._show_molecule(second)
    original = active_criteria.ranking_source(window)
    assert original in dialog.status.text()
    window.method_selector.setCurrentIndex(window.method_selector.findData("model"))
    preview = _proposal(window, tmp_path / "comparison-proposal")
    window.model_panel.accept_preview(preview)
    assert original in dialog.status.text()
    window.model_panel.adopt()
    applied = active_criteria.ranking_source(window)
    assert "Experimental model ranking" in applied and applied != original
    assert applied in dialog.status.text()
    for record_id, panel in ((first, dialog.reference_panel), (second, dialog.current_panel)):
        expected = "Yes" if record_id in window.basket.final_ids() else "No"
        assert f"Included in docking export: {expected}" in panel.details.toPlainText()
    window.first_objective.setCurrentText(window.second_objective.currentText())
    window.model_panel.models.setCurrentIndex(0)
    qapp.processEvents()
    dialog.refresh()
    assert applied in dialog.status.text()
    window._undo()
    assert original in dialog.status.text()
    window._redo()
    assert applied in dialog.status.text()


@pytest.mark.parametrize("kind", ["tolerance", "contextual"])
def test_preview_review_follows_current_record_but_comparison_keeps_applied_evidence(
    window, monkeypatch, kind,
):
    first, second = map(int, window.candidates.index[:2])
    context = {"target": "Applied target", "endpoint": "IC50"}
    applied = AppliedSelection.capture(
        constraints=window.constraints(), objectives=window.objectives(), pareto=None,
        strategy="chemical_tolerance" if kind == "tolerance" else "contextual_policy",
        input_hash="comparison-input", provenance={"context": context},
    )
    rows = pd.DataFrame(
        {"tolerance_state": ["accepted", "review"],
         "tolerance_evidence": ["Applied A evidence", "Applied B evidence"],
         "policy_explanation": ["Applied A evidence", "Applied B evidence"]},
        index=pd.Index([first, second], name="record_id"),
    )
    window._apply_selection(SelectionOutcome(selected_ids=(first,)), window.constraints(), applied,
                            model_scores=rows)
    dialog = _open(window, first)
    dialog.keep_reference_button.click()
    controls = getattr(window.model_panel, kind)
    controls.review.setRowCount(1)
    controls.review.setItem(0, 0, QTableWidgetItem(str(second)))
    preview_rows = pd.DataFrame(
        {"tolerance_state": ["review"], "tolerance_evidence": ["Preview-only evidence"],
         "policy_explanation": ["Preview-only evidence"]},
        index=pd.Index([second], name="record_id"),
    )
    summary = {"context": {"target": "Preview target"}, "policy_settings": {}}
    controls._preview = SimpleNamespace(
        root=Path("unused-preview"), records=preview_rows, summary=summary,
        temporary=TemporaryDirectory(),
    )
    if kind == "contextual":
        monkeypatch.setattr(
            "smiles2select.gui.workspace.contextual_controls.read_bundle",
            lambda _path: SimpleNamespace(records=preview_rows.reset_index()),
        )
        monkeypatch.setattr(
            "smiles2select.gui.workspace.contextual_controls.verify_preview", lambda _path: summary,
        )
    controls.inspect_row(0, 0)
    assert dialog.current_id == second and dialog.current_panel.record_id == second
    assert "Preview-only evidence" in window.inspector.details.toPlainText()
    assert "Applied A evidence" in dialog.reference_panel.details.toPlainText()
    assert "Applied B evidence" in dialog.current_panel.details.toPlainText()
    for panel in (dialog.reference_panel, dialog.current_panel):
        assert "Preview-only evidence" not in panel.details.toPlainText()
        assert "Preview target" not in panel.details.toPlainText()
        assert "Pareto rank:" not in panel.details.toPlainText()
        assert "Map point:" not in panel.details.toPlainText()
    assert active_criteria.ranking_source(window) in dialog.status.text()


def test_comparison_only_operations_preserve_full_log_pins_recipe_and_candidate_data(window):
    first, second = map(int, window.candidates.index[:2])
    window._decide(first, "pin")
    window._decide(second, "exclude")
    window._undo()
    before = {
        "states": tuple(window.basket.states()), "log": deepcopy(window.basket.log.history()),
        "final": window.basket.final_ids(), "pins": window.basket.pinned_ids(),
        "criteria": criteria_state.capture(window),
        "recipe": _recipe(window),
    }
    frame = window.candidates.copy(deep=True)
    dialog = _open(window, first)
    dialog.keep_reference_button.click()
    window._show_molecule(second)
    dialog.refresh()
    dialog.keep_reference_button.click()
    dialog.close()
    window._open_molecule_comparison()
    assert tuple(window.basket.states()) == before["states"]
    assert window.basket.log.history() == before["log"]
    assert window.basket.log.can_redo
    assert window.basket.final_ids() == before["final"]
    assert window.basket.pinned_ids() == before["pins"]
    assert criteria_state.capture(window) == before["criteria"]
    assert _recipe(window) == before["recipe"]
    pd.testing.assert_frame_equal(window.candidates, frame)


def test_busy_job_blocks_open_and_keep_but_allows_dialog_close_and_reenables_keep(window, qapp):
    first, second = map(int, window.candidates.index[:2])
    dialog = _open(window, first)
    dialog.keep_reference_button.click()
    window._show_molecule(second)
    released = Event()
    window._start_job(lambda: released.wait(3), lambda _value: None, "Testing comparison busy state")
    try:
        assert window.is_busy
        assert not window.molecule_compare_button.isEnabled()
        assert not dialog.keep_reference_button.isEnabled()
        dialog.keep_reference_button.click()
        assert dialog.reference_id == first
        dialog.close()
        assert not dialog.isVisible()
        window._open_molecule_comparison()
        assert not dialog.isVisible()
    finally:
        released.set()
        for _ in range(200):
            qapp.processEvents()
            if not window.is_busy:
                break
            QTest.qWait(10)
    assert not window.is_busy
    window._open_molecule_comparison()
    assert dialog.keep_reference_button.isEnabled()
    dialog.keep_reference_button.click()
    assert dialog.reference_id == second


def test_comparison_button_and_two_readable_panels_fit_notebook_window(window, qapp):
    window.resize(1280, 800)
    window.show()
    qapp.processEvents()
    button = window.molecule_compare_button
    assert button is not window.compare_button
    assert window.selection_actions.isAncestorOf(button)
    assert button.isVisible() and button.isEnabled()
    assert window.width() <= 1280 and window.height() <= 800
    assert window.rect().contains(button.rect().translated(button.mapTo(window, button.rect().topLeft())))
    first, second = map(int, window.candidates.index[:2])
    button.click()
    dialog = window._comparison_dialog
    dialog.set_current_id(first)
    dialog.keep_reference_button.click()
    window._show_molecule(second)
    qapp.processEvents()
    assert dialog.width() <= 1280 and dialog.height() <= 800
    assert dialog.reference_panel.mapTo(dialog, dialog.reference_panel.rect().topLeft()).x() < (
        dialog.current_panel.mapTo(dialog, dialog.current_panel.rect().topLeft()).x())
    for widget in (dialog.reference_panel, dialog.current_panel, dialog.keep_reference_button,
                   dialog.status):
        assert widget.isVisible()
        assert dialog.rect().contains(widget.rect().translated(widget.mapTo(dialog, widget.rect().topLeft())))
    assert dialog.reference_panel.details.viewport().height() >= 100
    assert dialog.current_panel.details.viewport().height() >= 100


def test_unbroken_long_ids_preserve_literal_identity_without_expanding_comparison(
    window, qapp, monkeypatch,
):
    bounds = QRect(0, 0, 1280, 800)
    monkeypatch.setattr(window, "screen", lambda: SimpleNamespace(availableGeometry=lambda: bounds))
    first, second, third = map(int, window.candidates.index[:3])
    names = {first: "MOL" + "x" * 100, second: "MOL" + "y" * 200,
             third: "MOL" + "z" * 100}
    frame = window.candidates.copy(deep=True)
    for record_id, name in names.items():
        frame.loc[record_id, "molecule_id"] = name
    window.candidates = frame
    window.resize(1280, 800)
    window.show()
    dialog = _open(window, first)
    close = dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Close)

    def fits():
        qapp.processEvents()
        assert bounds.contains(dialog.frameGeometry())
        for widget in (dialog.reference_panel, dialog.current_panel, close):
            assert widget.isVisible()
            assert dialog.rect().contains(
                widget.rect().translated(widget.mapTo(dialog, widget.rect().topLeft())))

    assert dialog.current_panel.title.textFormat() == Qt.PlainText
    assert dialog.current_panel.title.text() == names[first]
    assert f"ID: {names[first]}" in dialog.current_panel.details.toPlainText()
    fits()
    dialog.keep_reference_button.click()
    assert dialog.reference_panel.title.text() == names[first]
    fits()
    for record_id in (second, third):
        window._show_molecule(record_id)
        assert dialog.current_panel.title.text() == names[record_id]
        assert f"ID: {names[record_id]}" in dialog.current_panel.details.toPlainText()
        assert dialog.reference_panel.title.text() == names[first]
        fits()
