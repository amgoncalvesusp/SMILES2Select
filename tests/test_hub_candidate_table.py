"""Candidate rows, plots and the inspector share record IDs, never row positions."""

from __future__ import annotations

from importlib import import_module
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pandas as pd
import pytest
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QTableView, QTableWidget, QTableWidgetItem

from smiles2select.gui.workspace import criteria_state
from smiles2select.gui.workspace.views import ChemicalSpaceView, ParetoView
from smiles2select.gui.workspace.workspace_window import WorkspaceWindow
from smiles2select.selection_intelligence.basket import SelectionBasket
from smiles2select.selection_intelligence.states import (
    ChemicalStatus,
    MoleculeState,
    SelectionStatus,
)
from tests import test_workspace
from tests.test_workspace_model_decision import _proposal

qapp = test_workspace.qapp
result = test_workspace.result
window = test_workspace.window


@pytest.fixture
def table_types(qapp):
    module = import_module("smiles2select.gui.workspace.candidate_table")
    return module.CandidateTableModel, module.CandidateTableView


@pytest.fixture
def candidates():
    return pd.DataFrame(
        {"molecule_id": ["duplicate", "duplicate", '<b>β 🧪</b>', ""],
         "mol_wt": [1000.0, 9.5, 11.0, pd.NA], "qed": [0.2, 0.05, 0.2, pd.NA],
         "canonical_smiles": ["CCO", "CC", "CO", None]},
        index=pd.Index([101, 7, 9003, 42], name="record_id"),
    )


@pytest.fixture
def basket(candidates):
    value = SelectionBasket(MoleculeState(int(i), ChemicalStatus.AUTO_PASS)
                            for i in candidates.index)
    value.add_to_final([7])
    return value


def _column(model, label):
    for column in range(model.columnCount()):
        if model.headerData(column, Qt.Horizontal, Qt.DisplayRole) == label:
            return column
    pytest.fail(f"Missing candidate column: {label}")


def _cell(model, record_id, label):
    index = model.index_for_record(record_id)
    assert index.isValid(), f"Record {record_id} missing from the candidate table"
    return model.data(index.siblingAtColumn(_column(model, label)), Qt.DisplayRole)


def _ordered_ids(model):
    return [model.record_id(model.index(row, 0)) for row in range(model.rowCount())]


def _table_mode(window, qapp):
    assert window.view_selector.findText("Table") >= 0
    window.view_selector.setCurrentText("Table")
    qapp.processEvents()
    assert window.views.currentWidget() is window.table_view


def _ring_ids(view):
    return [point.data() for point in view.inspected_scatter.points()]


def test_model_identity_is_record_id_despite_duplicate_external_ids(table_types, candidates, basket):
    model = table_types[0]()
    model.refresh(candidates, basket)
    assert isinstance(model, QAbstractTableModel)
    assert model.rowCount() == 4
    assert _ordered_ids(model) == [101, 7, 9003, 42]
    assert _cell(model, 101, "Molecule ID") == _cell(model, 7, "Molecule ID") == "duplicate"
    assert _cell(model, 101, "Record ID") != _cell(model, 7, "Record ID")
    assert model.record_id(QModelIndex()) is None
    assert not model.index_for_record(999999).isValid()


def test_numeric_sort_uses_numbers_stable_ties_and_missing_last(table_types, candidates, basket):
    model = table_types[0]()
    model.refresh(candidates, basket)
    model.sort(_column(model, "MW"), Qt.AscendingOrder)
    assert _ordered_ids(model) == [7, 9003, 101, 42]
    model.sort(_column(model, "QED"), Qt.AscendingOrder)
    assert _ordered_ids(model) == [7, 101, 9003, 42]
    model.sort(_column(model, "QED"), Qt.DescendingOrder)
    assert _ordered_ids(model) == [101, 9003, 7, 42]
    for record_id in candidates.index:
        assert model.record_id(model.index_for_record(record_id)) == record_id


def test_values_are_literal_read_only_and_missing_values_are_blank(table_types, candidates, basket):
    model = table_types[0]()
    model.refresh(candidates, basket)
    assert _cell(model, 9003, "Molecule ID") == '<b>β 🧪</b>'
    assert _cell(model, 42, "MW") == ""
    assert _cell(model, 42, "SMILES") == ""
    index = model.index_for_record(9003)
    assert not model.flags(index) & Qt.ItemIsEditable
    assert not model.setData(index, "replaced", Qt.EditRole)
    assert _cell(model, 9003, "Molecule ID") == '<b>β 🧪</b>'
    assert model.data(QModelIndex(), Qt.DisplayRole) is None


def test_view_retains_inspected_record_through_sort_and_frame_replacement(
    table_types, candidates, basket,
):
    view = table_types[1]()
    view.refresh(candidates, basket)
    requested = QSignalSpy(view.molecule_requested)
    view.set_inspected_id(101)
    model = view.model()
    model.sort(_column(model, "MW"), Qt.AscendingOrder)
    assert model.record_id(view.currentIndex()) == 101
    view.refresh(candidates.iloc[::-1], basket)
    assert model.record_id(view.currentIndex()) == 101
    view.refresh(candidates.drop(index=101), basket)
    assert not view.currentIndex().isValid()
    assert requested.count() == 0
    view.close()


def test_large_table_exposes_all_203390_rows_without_copying_frame_or_cell_widgets(
    table_types, monkeypatch,
):
    frame = pd.DataFrame({"molecule_id": ["same"] * 203390, "qed": 0.5},
                         index=pd.RangeIndex(10, 406790, 2, name="record_id"))
    basket = SelectionBasket([MoleculeState(10), MoleculeState(406788)])

    def forbidden_copy(*args, **kwargs):
        pytest.fail("The candidate table must reuse the frame instead of copying 203390 rows")

    monkeypatch.setattr(frame, "copy", forbidden_copy)
    view = table_types[1]()
    view.refresh(frame, basket)
    model = view.model()
    assert isinstance(view, QTableView) and not isinstance(view, QTableWidget)
    assert isinstance(model, QAbstractTableModel)
    assert model.rowCount() == len(frame) == 203390
    assert model.record_id(model.index(203389, 0)) == 406788
    assert model.index_for_record(406788).row() == 203389
    assert view.findChildren(QTableWidget) == []
    assert _cell(model, 406788, "Molecule ID") == "same"
    view.close()


def test_table_mouse_and_keyboard_inspect_by_id_without_changing_decisions(window, qapp):
    _table_mode(window, qapp)
    window.resize(1280, 800)
    window.show()
    model = window.table_view.model()
    model.sort(_column(model, "MW"), Qt.DescendingOrder)
    qapp.processEvents()
    selected = window.basket.final_ids()
    actions = tuple(window.basket.log.applied)
    criteria = criteria_state.capture(window)
    first = model.index(0, 0)
    QTest.mouseClick(window.table_view.viewport(), Qt.LeftButton,
                     pos=window.table_view.visualRect(first).center())
    qapp.processEvents()
    first_id = model.record_id(first)
    assert window.inspector.record_id == first_id
    assert _ring_ids(window.map_view) == [first_id]
    window.table_view.setFocus()
    QTest.keyClick(window.table_view, Qt.Key_Down)
    qapp.processEvents()
    second_id = model.record_id(model.index(1, 0))
    assert window.inspector.record_id == second_id
    assert _ring_ids(window.map_view) == [second_id]
    assert window.basket.final_ids() == selected
    assert tuple(window.basket.log.applied) == actions
    assert criteria_state.capture(window) == criteria
    assert not window.has_pending_criteria()


def test_map_click_synchronizes_sorted_table_without_emitting_another_inspection(window, qapp):
    model = window.table_view.model()
    model.sort(_column(model, "MW"), Qt.DescendingOrder)
    record_id = int(window.candidates.index[1])
    point = next(point for point in window.map_view.scatter.points() if point.data() == record_id)
    table_requests = QSignalSpy(window.table_view.molecule_requested)
    map_requests = QSignalSpy(window.map_view.point_clicked)
    actions = tuple(window.basket.log.applied)
    window.map_view._on_click(window.map_view.scatter, [point])
    qapp.processEvents()
    assert map_requests.count() == 1
    assert table_requests.count() == 0
    assert window.inspector.record_id == record_id
    assert model.record_id(window.table_view.currentIndex()) == record_id
    assert _ring_ids(window.map_view) == [record_id]
    assert tuple(window.basket.log.applied) == actions


def test_final_membership_cells_follow_manual_decisions_undo_and_redo(window, qapp):
    _table_mode(window, qapp)
    model = window.table_view.model()
    record_id = window.basket.final_ids()[0]
    window._show_molecule(record_id)
    assert _cell(model, record_id, "Final for export") == "Yes"
    window._decide(record_id, "exclude")
    assert _cell(model, record_id, "Final for export") == "No"
    assert _cell(model, record_id, "Selection status") == SelectionStatus.MANUALLY_EXCLUDED.value
    window._undo()
    assert _cell(model, record_id, "Final for export") == "Yes"
    window._redo()
    assert _cell(model, record_id, "Final for export") == "No"
    assert model.record_id(window.table_view.currentIndex()) == record_id


def test_model_adoption_refreshes_table_membership_without_losing_inspected_id(
    window, qapp, tmp_path,
):
    _table_mode(window, qapp)
    window.show()
    window.method_selector.setCurrentIndex(window.method_selector.findData("model"))
    qapp.processEvents()
    record_id = int(window.candidates.index[0])
    window._show_molecule(record_id)
    preview = _proposal(window, tmp_path / "proposal")
    window.model_panel.accept_preview(preview)
    assert window.model_panel.adopt_button.isVisible()
    window.model_panel.adopt_button.click()
    model = window.table_view.model()
    for identifier in window.candidates.index:
        expected = "Yes" if identifier in window.basket.final_ids() else "No"
        assert _cell(model, identifier, "Final for export") == expected
    assert model.record_id(window.table_view.currentIndex()) == record_id
    assert window.build_artifacts().recipe.strategy == "experimental_model"


def test_session_reopen_rebinds_table_to_restored_candidates_and_basket(window, qapp, tmp_path):
    _table_mode(window, qapp)
    window.target_count.setValue(2)
    window._auto_select()
    path = tmp_path / "table-session.s2s"
    window.save_session(path)
    restored = WorkspaceWindow.from_session(path)
    try:
        _table_mode(restored, qapp)
        model = restored.table_view.model()
        assert model is not window.table_view.model()
        assert model.rowCount() == len(restored.candidates)
        assert set(_ordered_ids(model)) == set(restored.candidates.index)
        for record_id in restored.candidates.index:
            expected = "Yes" if record_id in restored.basket.final_ids() else "No"
            assert _cell(model, record_id, "Final for export") == expected
        record_id = int(restored.candidates.index[-1])
        restored.table_view.set_inspected_id(record_id)
        assert model.record_id(restored.table_view.currentIndex()) == record_id
    finally:
        restored.close()


def test_table_refresh_never_computes_or_draws_hidden_map_or_pareto(window, qapp, monkeypatch):
    def hidden_view_work(*args, **kwargs):
        pytest.fail("Table refresh must not compute or draw a hidden scatter view")

    monkeypatch.setattr(window, "_map_projection", hidden_view_work)
    monkeypatch.setattr(window.map_view, "set_points", hidden_view_work)
    monkeypatch.setattr(window.pareto_view, "show_objectives", hidden_view_work)
    monkeypatch.setattr(window.pareto_view, "set_points", hidden_view_work)
    _table_mode(window, qapp)
    window.refresh()
    assert window.table_view.model().rowCount() == len(window.candidates)


def test_inspection_ring_is_independent_of_final_squares_and_does_not_invent_coordinates(qapp):
    coordinates = pd.DataFrame({"x": [1.0, 3.0], "y": [2.0, 4.0]}, index=[7, 101])
    original = coordinates.copy(deep=True)
    for view_type in (ChemicalSpaceView, ParetoView):
        view = view_type()
        view.set_points(coordinates, selected_ids=[7])
        view.set_inspected_id(101)
        assert _ring_ids(view) == [101]
        assert [(p.pos().x(), p.pos().y()) for p in view.inspected_scatter.points()] == [(3.0, 4.0)]
        assert [p.data() for p in view.selected_scatter.points()] == [7]
        assert view.inspected_scatter.opts["symbol"] == "o"
        brush = view.inspected_scatter.opts["brush"]
        assert brush.style() == Qt.NoBrush or brush.color().alpha() == 0
        view.selection_only.setChecked(True)
        assert not view.inspected_scatter.isVisible() or not _ring_ids(view)
        assert view.inspection_notice.text().strip()
        view.set_inspected_id(7)
        assert view.inspected_scatter.isVisible()
        assert _ring_ids(view) == [7]
        view.set_inspected_id(9003)
        assert _ring_ids(view) == []
        assert "9003" in view.inspection_notice.text()
        assert [p.data() for p in view.selected_scatter.points()] == [7]
        pd.testing.assert_frame_equal(coordinates, original)
        view.set_inspected_id(None)
        assert _ring_ids(view) == []
        view.close()


def test_table_and_essential_actions_fit_notebook_window(window, qapp):
    _table_mode(window, qapp)
    window.resize(1280, 800)
    window.show()
    qapp.processEvents()
    assert window.width() <= 1280 and window.height() <= 800
    for widget in (window.table_view, window.inspector, window.target_count,
                   window.basket_panel.undo_button, window.basket_panel.docking_button):
        assert widget.isVisible()
        assert window.rect().contains(widget.rect().translated(widget.mapTo(window, widget.rect().topLeft())))
    first = window.table_view.model().index(0, 0)
    assert window.table_view.visualRect(first).width() > 0
    assert window.table_view.viewport().rect().contains(window.table_view.visualRect(first))


def test_inspection_sync_preserves_clicked_column_and_horizontal_scroll(
    table_types, candidates, basket, qapp,
):
    view = table_types[1]()
    view.refresh(candidates, basket)
    view.molecule_requested.connect(view.set_inspected_id)
    view.resize(420, 180)
    view.show()
    qapp.processEvents()
    column = _column(view.model(), "SMILES")
    scrollbar = view.horizontalScrollBar()
    assert scrollbar.maximum() > 0
    scrollbar.setValue(scrollbar.maximum())
    qapp.processEvents()
    original_scroll = scrollbar.value()
    clicked = view.model().index(2, column)
    assert view.viewport().rect().contains(view.visualRect(clicked).center())
    QTest.mouseClick(view.viewport(), Qt.LeftButton, pos=view.visualRect(clicked).center())
    qapp.processEvents()
    assert view.model().record_id(view.currentIndex()) == 9003
    assert view.currentIndex().column() == column
    assert scrollbar.value() == original_scroll
    view.set_inspected_id(7)
    assert view.model().record_id(view.currentIndex()) == 7
    assert view.currentIndex().column() == column
    assert scrollbar.value() == original_scroll
    view.refresh(candidates.iloc[::-1], basket)
    qapp.processEvents()
    assert view.model().record_id(view.currentIndex()) == 7
    assert view.currentIndex().column() == column
    assert scrollbar.value() == original_scroll
    view.close()


def test_tolerance_review_inspection_synchronizes_table_and_both_available_rings(window, qapp):
    previous, record_id = map(int, window.candidates.index[:2])
    coordinates = pd.DataFrame({"x": [1.0, 3.0], "y": [2.0, 4.0]},
                               index=[previous, record_id])
    for view in (window.map_view, window.pareto_view):
        view.set_points(coordinates, selected_ids=window.basket.final_ids())
    window._show_molecule(previous)
    actions = tuple(window.basket.log.applied)
    selected = window.basket.final_ids()
    controls = window.model_panel.tolerance
    controls.review.setRowCount(1)
    item = QTableWidgetItem(str(record_id))
    item.setData(Qt.UserRole, record_id)
    controls.review.setItem(0, 0, item)
    controls._preview = SimpleNamespace(
        records=pd.DataFrame({"tolerance_state": ["review"],
                              "tolerance_evidence": ["Observed active-series support"]},
                             index=[record_id]),
        summary={"context": {"target": "test-target"}}, temporary=TemporaryDirectory(),
    )
    controls.inspect_row(0, 0)
    qapp.processEvents()
    assert window.inspector.record_id == record_id
    assert "Observed active-series support" in window.inspector.details.toPlainText()
    assert window.table_view.model().record_id(window.table_view.currentIndex()) == record_id
    assert _ring_ids(window.map_view) == [record_id]
    assert _ring_ids(window.pareto_view) == [record_id]
    assert tuple(window.basket.log.applied) == actions
    assert window.basket.final_ids() == selected
