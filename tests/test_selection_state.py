"""Phase 1: selection states, basket, history, undo/redo, recipes and storage."""

from __future__ import annotations

import pytest

from smiles2select.selection_intelligence import recipes
from smiles2select.selection_intelligence.action_log import ActionLog, ActionType, SelectionAction
from smiles2select.selection_intelligence.basket import JustificationRequired, SelectionBasket
from smiles2select.selection_intelligence.states import (
    ChemicalStatus,
    MoleculeState,
    SelectionOrigin,
    SelectionStatus,
    chemical_status_from_run,
)
from smiles2select.storage.selection_store import SelectionStore

pytestmark = pytest.mark.unit


def build_basket() -> SelectionBasket:
    return SelectionBasket(
        [
            MoleculeState(1, ChemicalStatus.AUTO_PASS),
            MoleculeState(2, ChemicalStatus.BORDERLINE),
            MoleculeState(3, ChemicalStatus.AUTO_FAIL),
            MoleculeState(4, ChemicalStatus.WARNING),
        ],
        target_count=3,
    )


# --- states ------------------------------------------------------------------


def test_chemical_and_selection_status_are_independent():
    """The whole point of the layer: a decision never rewrites the chemistry."""
    state = MoleculeState(1, ChemicalStatus.AUTO_FAIL)
    decided = state.with_selection(
        SelectionStatus.FINAL_SELECTED, SelectionOrigin.RESCUED, "análogo aprovado"
    )
    assert decided.chemical_status is ChemicalStatus.AUTO_FAIL
    assert decided.is_selected
    assert decided.contradicts_chemistry


def test_warning_and_borderline_count_as_passing():
    assert ChemicalStatus.WARNING.passed
    assert ChemicalStatus.BORDERLINE.passed
    assert not ChemicalStatus.AUTO_FAIL.passed
    assert not ChemicalStatus.INVALID.passed


@pytest.mark.parametrize(
    ("valid", "passed", "borderline", "alerted", "expected"),
    [
        (False, False, False, False, ChemicalStatus.INVALID),
        (True, False, False, False, ChemicalStatus.AUTO_FAIL),
        (True, True, True, True, ChemicalStatus.BORDERLINE),
        (True, True, False, True, ChemicalStatus.WARNING),
        (True, True, False, False, ChemicalStatus.AUTO_PASS),
    ],
)
def test_chemical_status_mapping(valid, passed, borderline, alerted, expected):
    assert (
        chemical_status_from_run(valid=valid, passed=passed, borderline=borderline, alerted=alerted)
        is expected
    )


def test_pinning_survives_a_status_change():
    state = MoleculeState(1).with_pin(True)
    assert state.with_selection(SelectionStatus.SHORTLISTED, SelectionOrigin.MANUAL).pinned


# --- basket ------------------------------------------------------------------


def test_basket_counters():
    basket = build_basket()
    basket.add_to_final([1, 2], origin=SelectionOrigin.PARETO)
    basket.pin([1])
    basket.exclude([4])

    counters = basket.counters()
    assert counters.candidates == 4
    assert counters.final_selected == 2
    assert counters.pinned == 1
    assert counters.manually_excluded == 1
    assert counters.remaining == 1


def test_selecting_a_failed_molecule_requires_a_reason():
    basket = build_basket()
    with pytest.raises(JustificationRequired, match="justificativa"):
        basket.add_to_final([3], origin=SelectionOrigin.MANUAL)

    basket.add_to_final([3], origin=SelectionOrigin.RESCUED, reason="análogo aprovado MOL001")
    assert basket.state(3).is_selected
    assert basket.overrides()[0].record_id == 3


def test_borderline_molecule_needs_no_justification():
    basket = build_basket()
    basket.add_to_final([2], origin=SelectionOrigin.PARETO)
    assert basket.state(2).is_selected
    assert not basket.state(2).contradicts_chemistry


def test_unknown_ids_are_ignored_not_created():
    basket = build_basket()
    action = basket.add_to_final([1, 999])
    assert action.record_ids == (1,)
    assert 999 not in basket


def test_preview_reports_what_a_bulk_action_would_do():
    basket = build_basket()
    basket.add_to_final([1])
    basket.exclude([4])
    preview = basket.preview([1, 2, 4, 999])
    assert preview["solicitadas"] == 4
    assert preview["desconhecidas"] == 1
    assert preview["já selecionadas"] == 1
    assert preview["excluídas manualmente"] == 1


# --- history -----------------------------------------------------------------


def test_undo_and_redo_restore_exact_states():
    basket = build_basket()
    basket.add_to_final([1, 2], origin=SelectionOrigin.PARETO)
    assert basket.final_ids() == (1, 2)

    basket.undo()
    assert basket.final_ids() == ()
    basket.redo()
    assert basket.final_ids() == (1, 2)


def test_undo_walks_back_several_actions():
    basket = build_basket()
    basket.add_to_final([1])
    basket.add_to_final([2])
    basket.undo()
    basket.undo()
    assert basket.final_ids() == ()
    assert not basket.log.can_undo


def test_a_new_action_drops_the_redo_tail():
    basket = build_basket()
    basket.add_to_final([1])
    basket.undo()
    basket.add_to_final([2])
    assert not basket.log.can_redo
    assert basket.final_ids() == (2,)


def test_undo_on_an_empty_history_returns_none():
    assert SelectionBasket().undo() is None


def test_actions_record_source_and_reason():
    basket = build_basket()
    action = basket.add_to_final(
        [1],
        origin=SelectionOrigin.LASSO_SELECTION,
        source="CHEMICAL_SPACE_LASSO",
        reason="região diversa",
    )
    assert action.action_type is ActionType.ADD_TO_FINAL
    assert "CHEMICAL_SPACE_LASSO" in action.describe()
    assert action.timestamp.endswith("+00:00")


def test_history_marks_which_events_are_applied():
    log = ActionLog()
    log.record(SelectionAction(ActionType.PIN, (1,)))
    log.record(SelectionAction(ActionType.EXCLUDE, (2,)))
    log.undo()
    assert [event["applied"] for event in log.history()] == [True, False]


# --- recipes -----------------------------------------------------------------


def test_recipe_round_trips(tmp_path):
    recipe = recipes.SelectionRecipe(
        name="projeto",
        input_hash="abc123",
        objectives=({"field": "qed", "direction": "maximize"},),
        original_thresholds={"lip_mw_max": 500.0},
        applied_thresholds={"lip_mw_max": 450.0},
        target_count=500,
        max_per_scaffold=5,
        pinned_ids=(1, 2),
    )
    path = recipes.save(recipe, tmp_path / "projeto.selection.json")
    restored = recipes.load(path)
    assert restored.target_count == 500
    assert restored.pinned_ids == (1, 2)
    assert restored.changed_thresholds == {"lip_mw_max": (500.0, 450.0)}


def test_recipe_rejects_another_schema():
    with pytest.raises(recipes.RecipeError, match="schema"):
        recipes.from_dict({"schema_version": "9.9"})


def test_recipe_comparison_lists_differences():
    first = recipes.SelectionRecipe(applied_thresholds={"lip_mw_max": 500.0}, target_count=100)
    second = recipes.SelectionRecipe(applied_thresholds={"lip_mw_max": 450.0}, target_count=200)
    differences = recipes.compare(first, second)
    assert any("lip_mw_max" in text for text in differences)
    assert any("quantidade final" in text for text in differences)


def test_recipe_default_path():
    assert recipes.default_path("meu_projeto").name == "meu_projeto.selection.json"


# --- storage -----------------------------------------------------------------


@pytest.mark.integration
def test_session_survives_a_round_trip(tmp_path):
    basket = build_basket()
    basket.add_to_final([1, 2], origin=SelectionOrigin.PARETO, source="pareto")
    basket.pin([1])
    basket.exclude([4], reason="fora do escopo")

    database = tmp_path / "run.sqlite"
    with SelectionStore(database) as store:
        assert store.save_basket(basket) == 4

    with SelectionStore(database) as store:
        restored = store.load_basket(target_count=3)

    assert restored.final_ids() == (1, 2)
    assert restored.pinned_ids() == (1,)
    assert restored.excluded_ids() == (4,)
    assert restored.state(3).chemical_status is ChemicalStatus.AUTO_FAIL
    assert len(restored.log) == len(basket.log)


@pytest.mark.integration
def test_restored_session_can_still_undo(tmp_path):
    basket = build_basket()
    basket.add_to_final([1])
    database = tmp_path / "run.sqlite"
    with SelectionStore(database) as store:
        store.save_basket(basket)
    with SelectionStore(database) as store:
        restored = store.load_basket()

    assert restored.log.can_undo
    restored.undo()
    assert restored.final_ids() == ()


@pytest.mark.integration
def test_store_keeps_recipes(tmp_path):
    with SelectionStore(tmp_path / "run.sqlite") as store:
        store.save_recipe("r1", recipes.SelectionRecipe(name="primeira", target_count=10))
        store.save_recipe("r2", recipes.SelectionRecipe(name="segunda", target_count=20))
        assert set(store.recipe_ids()) == {"r1", "r2"}
        assert store.load_recipe("r1").target_count == 10
        with pytest.raises(KeyError):
            store.load_recipe("missing")


@pytest.mark.integration
def test_store_writes_the_analysis_tables(tmp_path):
    with SelectionStore(tmp_path / "run.sqlite") as store:
        store.save_pareto("r1", [{"record_id": 1, "pareto_rank": 1, "distance_to_ideal": 0.2}])
        store.save_margins(
            [
                {
                    "record_id": 1,
                    "rule_id": "lip_mw_max",
                    "normalized_margin": 0.4,
                    "margin_status": "ROBUST_PASS",
                }
            ]
        )
        store.save_coordinates("pca", [{"record_id": 1, "x": 0.1, "y": -0.3, "cluster_id": 7}])
        assert store.count("pareto_results") == 1
        assert store.count("rule_margins") == 1
        assert store.count("chemical_space_coordinates") == 1
