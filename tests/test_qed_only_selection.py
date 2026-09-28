"""QED-only remains a score baseline, independent of property and model ranks."""

from dataclasses import replace

import pandas as pd
import pytest

from smiles2select.selection_intelligence.constrained_selection import (
    SelectionConstraints,
    Strategy,
    select,
)
from smiles2select.selection_intelligence.objectives import Direction, Objective
from smiles2select.selection_intelligence.scenarios import ScenarioSpec, _rank, evaluate_scenario
from tests import test_scenarios, test_workspace

cached = test_scenarios.cached
qapp = test_workspace.qapp
result = test_workspace.result
window = test_workspace.window


def test_qed_only_ignores_other_scores_and_keeps_pins_and_quotas():
    candidates = pd.DataFrame({
        "qed": [.8, .9, .9, .2, .7],
        "selection_priority": [-10., 0., 1., 2., 3.],
        "pareto_rank": [1, 10, 10, 1, 1],
        "murcko_scaffold": ["A", "A", "B", "C", "D"],
        "cluster_id": [1, 1, 2, 3, 4],
    }, index=[5, 3, 2, 9, 7])
    outcome = select(candidates, SelectionConstraints(
        target_count=3, max_per_scaffold=1, max_per_cluster=1, min_scaffolds=3,
    ), Strategy.QED_ONLY, pinned_ids=(9,))
    assert outcome.selected_ids == (9, 2, 3)
    assert all("Pareto" not in reason for reasons in outcome.reasons.values() for reason in reasons)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -0.1, 1.1, "bad"])
def test_qed_only_rejects_invalid_scores_without_silent_fallback(value):
    candidates = pd.DataFrame({"qed": [.8, value]}, index=[1, 2])
    with pytest.raises(ValueError, match="QED"):
        select(candidates, SelectionConstraints(target_count=2), Strategy.QED_ONLY)
    assert select(candidates, strategy=Strategy.QED_ONLY, excluded_ids=(2,)).selected_ids == (1,)


def test_qed_only_requires_cached_qed():
    with pytest.raises(ValueError, match="QED"):
        select(pd.DataFrame({"mol_wt": [100]}, index=[1]), strategy=Strategy.QED_ONLY)


def test_qed_only_rejects_nullable_missing_values():
    candidates = pd.DataFrame({"qed": pd.Series([.8, pd.NA], dtype="Float64")})
    with pytest.raises(ValueError, match="QED"):
        select(candidates, strategy=Strategy.QED_ONLY)


def test_qed_only_large_scenario_rank_never_uses_percentile_objectives():
    pool = pd.DataFrame({"qed": [i / 2100 for i in range(2100)], "mol_wt": range(2100)})
    spec = ScenarioSpec("QED", objectives=(Objective("mol_wt", Direction.MINIMIZE),),
                        strategy=Strategy.QED_ONLY, constraints=SelectionConstraints(target_count=2))
    ranked, method, warnings = _rank(pool, spec)
    assert spec.objectives == ()
    assert method == "qed_descending"
    assert warnings == ()
    assert select(ranked, spec.constraints, spec.strategy).selected_ids == (2099, 2098)


def test_qed_only_scenario_replays_with_ineligible_invalid_qed(cached, tmp_path):
    from smiles2select.selection_intelligence.scenario_io import load_study, save_study

    result, candidates = cached
    candidates.loc[3, "qed"] = float("nan")  # Chemical rules reject this row.
    spec = ScenarioSpec("QED", strategy=Strategy.QED_ONLY,
                        constraints=SelectionConstraints(target_count=1))
    snapshot = evaluate_scenario(result, candidates, spec)
    assert snapshot.outcome.selected_ids == (2,)
    path = tmp_path / "qed.json"
    save_study(path, (snapshot,))
    restored = load_study(path, result, candidates)[0]
    assert restored.outcome.selected_ids == (2,)
    assert restored.provenance["ranking_method"] == "qed_descending"


def test_qed_workspace_ignores_invalid_objectives_and_roundtrips_session(window, tmp_path):
    from smiles2select.gui.workspace.workspace_window import WorkspaceWindow

    window.strategy.setCurrentText("qed_only")
    window.target_count.setValue(2)
    window.second_objective.setCurrentText(window.first_objective.currentText())
    window._auto_select()
    applied = window.build_artifacts()
    eligible = [state.record_id for state in window.basket.states() if state.chemical_status.passed]
    expected = window.candidates.loc[eligible].sort_values("qed", ascending=False).index[:2]
    assert set(window.basket.final_ids()) == set(expected)
    assert applied.recipe.strategy == "qed_only"
    assert applied.recipe.objectives == ()
    assert applied.pareto is None
    assert applied.recipe.provenance["ranking_method"] == "qed_descending"
    assert applied.model_scores is None
    restored = WorkspaceWindow.from_session(window.save_session(tmp_path / "qed.s2s.sqlite"))
    try:
        reopened = restored.build_artifacts()
        assert reopened.recipe.objectives == ()
        assert reopened.recipe.strategy == applied.recipe.strategy
        assert reopened.recipe.final_selected_ids == applied.recipe.final_selected_ids
        assert reopened.recipe.provenance == applied.recipe.provenance
        assert restored.strategy.currentData() == "qed_only"
    finally:
        restored.close()


def test_large_qed_workspace_has_no_property_rank_takeover(window, monkeypatch):
    from smiles2select.selection_intelligence.basket import SelectionBasket
    from smiles2select.selection_intelligence.states import MoleculeState

    frame = pd.concat([window.candidates.iloc[:1]] * 2100, ignore_index=True)
    window.candidates = frame.assign(qed=[i / 2100 for i in range(2100)], mol_wt=range(2100))
    window.result = replace(window.result, descriptors=window.candidates)
    window.basket = SelectionBasket(MoleculeState(i) for i in range(2100))
    window.pareto = None
    monkeypatch.setattr(window, "refresh", lambda: None)
    monkeypatch.setattr(window, "_start_job", lambda run, complete, message: complete(run()))
    window.strategy.setCurrentText("qed_only")
    window.target_count.setValue(2)
    window._auto_select()
    assert window.basket.final_ids() == (2098, 2099)
    assert window.build_artifacts().recipe.provenance["ranking_method"] == "qed_descending"


def test_qed_workspace_rejects_missing_scores_before_mutation(window):
    window.strategy.setCurrentText("qed_only")
    window.candidates = window.candidates.assign(qed=float("nan"))
    before = window.basket.final_ids()
    window._auto_select()
    assert window.basket.final_ids() == before
    assert "QED" in window.warnings.text()
    assert "finite" in window.warnings.text()


def test_qed_scenario_capture_and_adoption_ignore_inactive_objectives(window):
    from smiles2select.gui.workspace import criteria_state
    from smiles2select.gui.workspace.scenario_dialog import ScenarioDialog

    window.second_objective.setCurrentText(window.first_objective.currentText())
    dialog = ScenarioDialog(window)
    try:
        dialog.strategy.setCurrentText("qed_only")
        spec = dialog.capture_spec()
        assert spec.objectives == ()
        adopted = criteria_state.from_scenario(window, spec)
        criteria_state.apply(window, adopted)
        assert window.strategy.currentData() == "qed_only"
        assert not window.objective_editors[0].isEnabled()
        assert "ignored" in window.criteria_summary.text()
    finally:
        dialog.close()


def test_qed_selection_replaces_model_provenance_and_undo_restores_model(window):
    from smiles2select.gui.workspace.selection_provenance import AppliedSelection
    from smiles2select.selection_intelligence.constrained_selection import SelectionOutcome

    ids = tuple(window.candidates.index[:2])
    model = AppliedSelection.capture(
        constraints=window.constraints(), objectives=window.objectives(), pareto=None,
        strategy="experimental_model", input_hash="model-input",
        provenance={"source": "experimental_model", "ranking_method": "onnx_score",
                    "model_sha256": "a" * 64, "target": "example", "endpoint": "IC50"},
    )
    window._apply_selection(
        SelectionOutcome(selected_ids=ids), window.constraints(), model,
        model_scores=pd.DataFrame({"priority_score": [.8, .7]}, index=ids),
    )
    window.strategy.setCurrentText("qed_only")
    window._auto_select()
    artifacts = window.build_artifacts()
    assert artifacts.recipe.strategy == "qed_only"
    assert artifacts.recipe.model == {}
    assert artifacts.model_scores is None
    assert "model_sha256" not in artifacts.recipe.provenance
    window._undo()
    assert window.build_artifacts().recipe.strategy == "experimental_model"
    assert window.build_artifacts().model_scores is not None
