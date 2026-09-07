"""Selection workspace regressions, exercised with real Qt widgets."""

import pytest

from smiles2select.export.selection_export import final_selected_sheet
from smiles2select.selection_intelligence.constrained_selection import Strategy
from tests import test_workspace

qapp = test_workspace.qapp
result = test_workspace.result
window = test_workspace.window


def test_auto_select_does_not_override_failed_chemistry(window):
    rejected = [s.record_id for s in window.basket.states() if not s.chemical_status.passed]
    assert rejected
    window.basket.pin(rejected)
    window.target_count.setValue(len(window.candidates))
    window._auto_select()
    assert not set(rejected).intersection(window.basket.final_ids())


def test_auto_select_replaces_previous_selection_and_undo_restores_it(window):
    window.target_count.setValue(3)
    window._auto_select()
    previous = window.basket.final_ids()
    previous_reasons = window.build_artifacts().outcome.reasons
    assert len(previous) == 3
    window.target_count.setValue(1)
    window._auto_select()
    assert len(window.basket.final_ids()) == 1
    window._undo()
    assert window.basket.final_ids() == previous
    assert window.build_artifacts().outcome.reasons == previous_reasons
    window._redo()
    assert len(window.basket.final_ids()) == 1


@pytest.mark.parametrize("action", ["exclude", "undo", "redo", "add"])
def test_export_reflects_current_basket_after_auto_select(window, action):
    window.target_count.setValue(2)
    window._auto_select()
    if action == "exclude":
        window._decide(window.basket.final_ids()[0], "exclude")
    elif action in {"undo", "redo"}:
        window._undo()
        if action == "redo":
            window._redo()
    else:
        record_id = next(
            s.record_id for s in window.basket.states()
            if s.chemical_status.passed and not s.is_selected
        )
        window._decide(record_id, "final")
    artifacts = window.build_artifacts()
    assert set(artifacts.outcome.selected_ids) == set(window.basket.final_ids())
    assert set(final_selected_sheet(artifacts)["record_id"]) == set(window.basket.final_ids())
    selected = window.candidates.reindex(window.basket.final_ids())
    assert artifacts.outcome.scaffold_usage == selected["murcko_scaffold"].value_counts().to_dict()
    assert artifacts.outcome.cluster_usage == selected["cluster_id"].value_counts().to_dict()


def test_auto_select_preserves_justified_pinned_override(window, monkeypatch):
    record_id = next(s.record_id for s in window.basket.states() if not s.chemical_status.passed)
    monkeypatch.setattr(
        "smiles2select.gui.workspace.workspace_window.QInputDialog.getText",
        lambda *args: ("Reference control for this assay", True),
    )
    window._decide(record_id, "final")
    window._decide(record_id, "pin")
    before = window.basket.state(record_id)
    window.target_count.setValue(1)
    window._auto_select()
    assert window.basket.final_ids() == (record_id,)
    assert window.basket.state(record_id) == before
    assert window.build_artifacts().outcome.reasons[record_id] == [
        "Reference control for this assay"
    ]


def test_inspector_tracks_decision_and_undo(window):
    record_id = int(window.candidates.index[0])
    window._show_molecule(record_id)
    window._decide(record_id, "exclude")
    assert "Selection status: MANUALLY_EXCLUDED" in window.inspector.details.toPlainText()
    window._undo()
    assert "Selection status: UNDECIDED" in window.inspector.details.toPlainText()


def test_invalid_pareto_objectives_clear_previous_points(window):
    window.view_selector.setCurrentText("Pareto")
    assert len(window.pareto_view.scatter.points())
    window.second_objective.setCurrentText(window.first_objective.currentText())
    assert not len(window.pareto_view.scatter.points())


def test_export_preserves_strategy_used_for_selection(window):
    window.strategy.setCurrentText("balanced")
    window.target_count.setValue(2)
    window._auto_select()
    window.strategy.setCurrentText("diversity_first")
    artifacts = window.build_artifacts()
    assert artifacts.outcome.strategy is Strategy.BALANCED
    assert artifacts.recipe.strategy == "balanced"


def test_export_undo_restores_strategy_used_for_selection(window):
    window.strategy.setCurrentText("balanced")
    window.target_count.setValue(2)
    window._auto_select()
    window.strategy.setCurrentText("diversity_first")
    window._auto_select()
    window._undo()
    artifacts = window.build_artifacts()
    assert artifacts.outcome.strategy is Strategy.BALANCED
    assert artifacts.recipe.strategy == "balanced"
