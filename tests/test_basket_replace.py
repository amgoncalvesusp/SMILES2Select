"""Automatic re-selection replaces the basket atomically and preserves human choices."""

import pytest

from smiles2select.selection_intelligence.action_log import ActionType
from smiles2select.selection_intelligence.basket import JustificationRequired, SelectionBasket
from smiles2select.selection_intelligence.states import (
    ChemicalStatus,
    MoleculeState,
    SelectionOrigin,
    SelectionStatus,
)


def test_replace_final_is_one_undoable_action_and_preserves_manual_provenance():
    retained = MoleculeState(
        2, selection_status=SelectionStatus.FINAL_SELECTED,
        origin=SelectionOrigin.MANUAL, note="human choice",
    )
    pinned = MoleculeState(
        3, chemical_status=ChemicalStatus.AUTO_FAIL,
        selection_status=SelectionStatus.FINAL_SELECTED,
        origin=SelectionOrigin.RESCUED, pinned=True, note="validated exception",
    )
    basket = SelectionBasket([
        MoleculeState(1, selection_status=SelectionStatus.FINAL_SELECTED),
        retained, pinned, MoleculeState(4),
        MoleculeState(5, selection_status=SelectionStatus.MANUALLY_EXCLUDED),
    ])
    before = basket.states()

    action = basket.replace_final([2, 3, 4, 4, 999], source="weighted_score")

    assert basket.final_ids() == (2, 3, 4)
    assert basket.state(2) == retained
    assert basket.state(3) == pinned
    assert basket.state(4).origin is SelectionOrigin.AUTOMATIC
    assert basket.state(5) == before[4]
    assert action.action_type is ActionType.AUTOMATIC_SELECTION
    assert action.source == "weighted_score"
    assert set(action.previous_state) == {1, 4}
    assert len(basket.log.applied) == 1
    after = basket.states()
    assert basket.undo() == action
    assert basket.states() == before
    assert basket.redo() == action
    assert basket.states() == after


def test_empty_replacement_keeps_pinned_final():
    pinned = MoleculeState(1, selection_status=SelectionStatus.FINAL_SELECTED, pinned=True)
    basket = SelectionBasket([
        pinned, MoleculeState(2, selection_status=SelectionStatus.FINAL_SELECTED),
    ])
    basket.replace_final([])
    assert basket.final_ids() == (1,)
    assert basket.state(1) == pinned


@pytest.mark.parametrize("method", ["add_to_final", "replace_final"])
def test_failed_selection_changes_neither_state_nor_history(method):
    basket = SelectionBasket([
        MoleculeState(1), MoleculeState(2, chemical_status=ChemicalStatus.AUTO_FAIL),
        MoleculeState(3, selection_status=SelectionStatus.FINAL_SELECTED),
    ])
    before = basket.states()
    with pytest.raises(JustificationRequired):
        getattr(basket, method)([1, 2])
    assert basket.states() == before
    assert not basket.log.applied


def test_replace_final_can_promote_pinned_shortlist_without_unpinning():
    basket = SelectionBasket([
        MoleculeState(1, selection_status=SelectionStatus.SHORTLISTED, pinned=True),
    ])
    basket.replace_final([1])
    assert basket.final_ids() == (1,)
    assert basket.pinned_ids() == (1,)
