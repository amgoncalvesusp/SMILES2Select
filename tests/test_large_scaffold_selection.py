"""Regression for full-library molecular-core selection above the GUI map limit."""

import pandas as pd
import pytest

from smiles2select.selection_intelligence.constrained_selection import (
    SelectionConstraints,
    Strategy,
    select,
)


def test_scaffold_strategy_covers_cores_before_taking_more_analogues():
    candidates = pd.DataFrame(
        {"murcko_scaffold": ["A", "A", "B", "B", "C", "C"],
         "selection_priority": [-9, -8, -7, -6, -5, -4]},
        index=[1, 2, 3, 4, 5, 6],
    )
    outcome = select(candidates, SelectionConstraints(target_count=3),
                     Strategy.SCAFFOLD_COVERAGE)
    assert outcome.selected_ids == (1, 3, 5)
    assert outcome.scaffolds_covered == 3
    shuffled = select(candidates.sample(frac=1, random_state=17),
                      SelectionConstraints(target_count=3), Strategy.SCAFFOLD_COVERAGE)
    assert shuffled.selected_ids == outcome.selected_ids


def test_scaffold_rarity_precedes_weighted_objective_score_and_preserves_pins():
    candidates = pd.DataFrame(
        {"murcko_scaffold": ["A", "A", "B", "C"],
         "selection_priority": [-9, -8, -7, -6]},
        index=[1, 2, 3, 4],
    )
    assert select(candidates, SelectionConstraints(target_count=2),
                  Strategy.SCAFFOLD_COVERAGE).selected_ids == (3, 4)
    pinned = select(candidates, SelectionConstraints(target_count=3, max_per_scaffold=1),
                    Strategy.SCAFFOLD_COVERAGE, pinned_ids=[2])
    assert pinned.selected_ids == (2, 3, 4)


def test_large_missing_scaffold_library_selects_exactly_2000_without_mutation():
    from smiles2select.selection_intelligence.preparation import ensure_selection_scaffolds

    size = 199243  # User-sized candidate count; three known chemistry fixtures, not unique chemistry.
    candidates = pd.DataFrame(
        {"canonical_smiles": (["c1ccccc1", "c1ccncc1", "CCO"] * (size // 3 + 1))[:size],
         "murcko_scaffold": pd.NA},
        index=pd.RangeIndex(1, size + 1, name="record_id"),
    )
    constraints = SelectionConstraints(target_count=2000)
    prepared = ensure_selection_scaffolds(candidates, constraints, Strategy.SCAFFOLD_COVERAGE)
    assert candidates.murcko_scaffold.isna().all()
    assert prepared.murcko_scaffold.notna().all()
    assert prepared.loc[3, "murcko_scaffold"] == ""  # Acyclic is a real known group.
    outcome = select(prepared, constraints, Strategy.SCAFFOLD_COVERAGE)
    assert outcome.count == 2000
    assert outcome.selected_ids[:3] == (2, 3, 1)  # First core has one extra member, hence lower rarity.
    assert outcome.scaffolds_covered == 3
    assert prepared.index.equals(candidates.index)


def test_scaffolds_are_only_computed_on_demand_and_cached_values_are_preserved():
    from smiles2select.selection_intelligence.preparation import ensure_selection_scaffolds

    candidates = pd.DataFrame({"canonical_smiles": ["invalid", "CCO"],
                               "murcko_scaffold": ["existing", pd.NA]}, index=[1, 2])
    ordinary = ensure_selection_scaffolds(candidates, SelectionConstraints(), Strategy.BALANCED)
    assert ordinary is candidates
    prepared = ensure_selection_scaffolds(candidates, SelectionConstraints(max_per_scaffold=1),
                                         Strategy.BALANCED)
    assert prepared.murcko_scaffold.tolist() == ["existing", ""]


def test_invalid_smiles_is_not_silently_treated_as_acyclic():
    from smiles2select.selection_intelligence.preparation import ensure_selection_scaffolds

    candidates = pd.DataFrame({"canonical_smiles": ["invalid"], "murcko_scaffold": [pd.NA]},
                              index=[7])
    with pytest.raises(ValueError, match="record 7"):
        ensure_selection_scaffolds(candidates, SelectionConstraints(), Strategy.SCAFFOLD_COVERAGE)


def test_scaffold_preparation_reports_progress_and_handles_absent_column():
    from smiles2select.selection_intelligence.preparation import ensure_selection_scaffolds

    messages = []
    candidates = pd.DataFrame({"canonical_smiles": ["CCO", "c1ccccc1"]}, index=[4, 2])
    prepared = ensure_selection_scaffolds(candidates, SelectionConstraints(min_scaffolds=1),
                                         Strategy.BALANCED, progress=messages.append)
    assert prepared.murcko_scaffold.to_dict() == {4: "", 2: "c1ccccc1"}
    assert "murcko_scaffold" not in candidates
    assert "Calculating" in messages[0]
    assert "ready" in messages[-1]
