"""A desktop resource check fails when an exercised Hub interaction is broken."""

from __future__ import annotations

import pytest

from smiles2select.gui import desktop_self_check
from smiles2select.gui.workspace import candidate_table, molecule_comparison
from smiles2select.gui.workspace.workspace_window import WorkspaceWindow
from tests import test_workspace

qapp = test_workspace.qapp
result = test_workspace.result


@pytest.fixture
def window(qapp, result):
    existing_widgets = set(qapp.topLevelWidgets())
    workspace = WorkspaceWindow(result)
    workspace.resize(1400, 900)
    workspace.show()
    qapp.processEvents()
    try:
        yield workspace
    finally:
        desktop_self_check._dispose_widget(workspace, existing_widgets)


def test_hub_check_inspects_real_records_and_distinct_rendered_structures(window):
    report = desktop_self_check._check_hub_gui(window)
    first, second = map(int, window.candidates.index[:2])
    comparison = report["comparison"]
    assert report["table"]["record_ids"] == list(window.candidates.index)
    assert report["table"]["inspected_id"] == second == window.inspector.record_id
    assert comparison["reference_id"] == first
    assert comparison["current_id"] == second
    assert comparison["reference_smiles"] == window.candidates.at[first, "canonical_smiles"]
    assert comparison["current_smiles"] == window.candidates.at[second, "canonical_smiles"]
    assert comparison["reference_smiles"] != comparison["current_smiles"]
    dialog = window._comparison_dialog
    assert dialog.reference_panel.structure.pixmap().toImage() != (
        dialog.current_panel.structure.pixmap().toImage())
    assert report["final_ids_before"] == report["final_ids_after"] == list(window.basket.final_ids())


def test_disconnected_table_inspection_fails(window):
    window.table_view.selectionModel().currentRowChanged.disconnect(window.table_view._inspect_row)
    with pytest.raises(ValueError, match="[Tt]able.*inspect"):
        desktop_self_check._check_hub_gui(window)


def test_wrong_table_record_identity_fails(window, monkeypatch):
    monkeypatch.setattr(window.table_view, "sortByColumn", lambda *_: None)
    monkeypatch.setattr(candidate_table.CandidateTableModel, "record_id", lambda *_: 999999)
    with pytest.raises(ValueError, match="[Tt]able.*identity"):
        desktop_self_check._check_hub_gui(window)


def test_table_sort_that_does_nothing_fails(window, monkeypatch):
    monkeypatch.setattr(window.table_view, "sortByColumn", lambda *_: None)
    with pytest.raises(ValueError, match="[Tt]able.*(identity|order)"):
        desktop_self_check._check_hub_gui(window)


@pytest.mark.parametrize("broken", ["reference", "modal", "editable", "structure"])
def test_broken_comparison_cannot_report_success(window, monkeypatch, broken):
    original = molecule_comparison.MoleculeComparisonDialog.__init__

    def altered(dialog, workspace):
        original(dialog, workspace)
        if broken == "reference":
            dialog.keep_reference_button.clicked.disconnect()
        elif broken == "modal":
            dialog.setModal(True)
        elif broken == "editable":
            dialog.current_panel.details.setReadOnly(False)
        else:
            original_show = dialog.current_panel.show_molecule

            def without_structure(*args, **kwargs):
                original_show(*args, **kwargs)
                dialog.current_panel.structure.clear()

            dialog.current_panel.show_molecule = without_structure

    monkeypatch.setattr(molecule_comparison.MoleculeComparisonDialog, "__init__", altered)
    with pytest.raises(ValueError, match="[Cc]omparison"):
        desktop_self_check._check_hub_gui(window)


@pytest.mark.parametrize("change", ["basket", "log", "cursor", "criteria"])
def test_comparison_side_effects_fail(window, monkeypatch, change):
    first = int(window.candidates.index[0])
    window.basket.pin([first])
    window.basket.unpin([first])
    original = molecule_comparison.MoleculeComparisonDialog._keep_reference

    def changed(dialog):
        original(dialog)
        if change == "basket":
            dialog.workspace.basket.pin([dialog.current_id])
        elif change == "log":
            dialog.workspace.basket.log.applied[0].previous_state[999999] = {"note": "changed"}
        elif change == "cursor":
            dialog.workspace.basket.log.undo()
        else:
            dialog.workspace.target_count.setValue(dialog.workspace.target_count.value() + 1)

    monkeypatch.setattr(molecule_comparison.MoleculeComparisonDialog, "_keep_reference", changed)
    with pytest.raises(ValueError, match="[Hh]ub.*changed"):
        desktop_self_check._check_hub_gui(window)
