"""The plotted, inspected and exported final library is the same decision."""

from tests import test_workspace

qapp = test_workspace.qapp
result = test_workspace.result
window = test_workspace.window


def test_opening_hub_preserves_original_selected_molecules(window, result):
    assert window.basket.final_ids() == tuple(sorted(result.decision.selected_ids()))
    assert "Original pipeline" in window.selection_summary.text()
    from smiles2select.export.selection_export import selection_summary_sheet

    summary = selection_summary_sheet(window.build_artifacts())
    assert summary.loc[summary["item"] == "strategy", "value"].iloc[0] == result.config.selection_strategy


def test_inspector_explicitly_identifies_export_membership(window):
    record_id = window.basket.final_ids()[0]
    window._show_molecule(record_id)
    assert "Included in docking export: Yes" in window.inspector.details.toPlainText()
    window._decide(record_id, "exclude")
    assert "Included in docking export: No" in window.inspector.details.toPlainText()


def test_changed_target_does_not_claim_selection_was_recomputed(window):
    window.target_count.setValue(2)
    window._auto_select()
    window.target_count.setValue(4)
    assert "2 final" in window.selection_summary.text()
    assert "not applied" in window.selection_summary.text()


def test_map_cache_is_extended_when_final_selection_changes(window):
    key = (window.projection_selector.currentData(), False)
    projection, refs = window._map_projection()
    from dataclasses import replace

    remaining = projection.projection.coordinates.iloc[:1]
    window._projection_results[key] = (
        replace(projection, projection=replace(projection.projection, coordinates=remaining)), refs
    )
    refreshed, _ = window._map_projection()
    assert set(window.basket.final_ids()) <= set(refreshed.projection.coordinates.index)


def test_empty_docking_export_is_explained_without_opening_dialog(window, monkeypatch):
    window.basket.replace_final([])
    monkeypatch.setattr(
        "smiles2select.gui.workspace.workspace_window.QFileDialog.getSaveFileName",
        lambda *a: (_ for _ in ()).throw(AssertionError("Do not ask for an empty export")),
    )
    window._export_docking()
    assert "No final molecules" in window.warnings.text()


def test_export_freezes_applied_criteria_and_undo_restores_them(window):
    window.target_count.setValue(2)
    window._auto_select()
    first = window.build_artifacts()
    window.target_count.setValue(4)
    window.per_scaffold.setValue(1)
    window.first_objective.setCurrentText("tpsa")
    edited = window.build_artifacts()
    assert edited.constraints == first.constraints
    assert edited.recipe.objectives == first.recipe.objectives
    import pandas as pd

    pd.testing.assert_frame_equal(edited.pareto.table, first.pareto.table)
    window._auto_select()
    window._undo()
    restored = window.build_artifacts()
    assert restored.constraints == first.constraints
    assert restored.recipe.objectives == first.recipe.objectives
    assert restored.recipe.final_selected_ids == first.recipe.final_selected_ids


def test_gui_docking_action_exports_current_basket(window, tmp_path, monkeypatch):
    import json

    import pandas as pd

    window.target_count.setValue(2)
    window._auto_select()
    path = tmp_path / "current.csv"
    monkeypatch.setattr(
        "smiles2select.gui.workspace.workspace_window.QFileDialog.getSaveFileName",
        lambda *a: (str(path), "CSV (*.csv)"),
    )
    monkeypatch.setattr(
        "smiles2select.gui.workspace.workspace_window.QMessageBox.information", lambda *a: None,
    )
    window._export_docking()
    table = pd.read_csv(path)
    manifest = json.loads(path.with_suffix(".docking.json").read_text(encoding="utf-8"))
    assert len(table) == 2
    assert manifest["recipe"]["final_selection"]["selected_ids"] == list(window.basket.final_ids())


def test_background_selection_refreshes_missing_finals_after_job_finishes(window, qapp, monkeypatch):
    import pandas as pd

    window.candidates = pd.concat([window.candidates] * 1000, ignore_index=True)
    window._map_requested = True
    requested = []
    monkeypatch.setattr(window, "_request_map", lambda: requested.append(True))
    monkeypatch.setattr(window, "refresh", lambda: None)
    window._job = object()
    from smiles2select.selection_intelligence.constrained_selection import SelectionOutcome

    window._apply_selection(SelectionOutcome(selected_ids=(1,)), window.constraints())
    assert window._pending_map_refresh
    window._job = None
    window._finish_job()
    qapp.processEvents()
    assert requested == [True]


def test_large_selection_uses_objectives_and_export_declares_approximation(window, monkeypatch):
    from dataclasses import replace

    import pandas as pd

    from smiles2select.selection_intelligence.basket import SelectionBasket
    from smiles2select.selection_intelligence.states import MoleculeState

    frame = pd.concat([window.candidates.iloc[:1]] * 2100, ignore_index=True)
    window.candidates = frame.assign(qed=[float(i) / 2100 for i in range(2100)], mol_wt=200.)
    window.result = replace(window.result, descriptors=window.candidates)
    window.basket = SelectionBasket(MoleculeState(i) for i in range(2100))
    window.pareto = None
    monkeypatch.setattr(window, "refresh", lambda: None)
    monkeypatch.setattr(window, "_start_job", lambda function, completed, message: completed(function()))
    window.target_count.setValue(2)
    window._auto_select()
    assert window.basket.final_ids() == (2098, 2099)
    assert window.build_artifacts().recipe.provenance["ranking_method"] == "weighted_percentile"


def test_invalid_objectives_do_not_change_applied_selection_or_recipe(window):
    window.target_count.setValue(2)
    window._auto_select()
    before = window.build_artifacts()
    window.second_objective.setCurrentText(window.first_objective.currentText())
    window._auto_select()
    assert "repeated" in window.warnings.text()
    assert window.basket.final_ids() == before.outcome.selected_ids
    assert window.build_artifacts().recipe.objectives == before.recipe.objectives
