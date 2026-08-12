"""Phase 2: objectives, dominance and Pareto ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smiles2select.selection_intelligence import pareto
from smiles2select.selection_intelligence.objectives import (
    MAX_USEFUL_OBJECTIVES,
    Direction,
    Objective,
    ObjectiveError,
    ObjectiveSet,
    objectives_from_dicts,
)
from smiles2select.selection_intelligence.pareto import (
    crowding_distance,
    distance_to_ideal,
    non_dominated_sort,
    normalise,
)
from smiles2select.selection_intelligence.pareto_ranking import (
    ParetoRanker,
    order_within_front,
    rank_candidates,
)

pytestmark = pytest.mark.unit


def frame(**columns) -> pd.DataFrame:
    data = pd.DataFrame(columns)
    data.index = pd.RangeIndex(1, len(data) + 1, name="record_id")
    return data


# --- objectives --------------------------------------------------------------


def test_maximize_keeps_the_value_and_minimize_flips_it():
    values = pd.Series([1.0, 5.0])
    assert Objective("qed").desirability(values).tolist() == [1.0, 5.0]
    assert Objective("mol_wt", Direction.MINIMIZE).desirability(values).tolist() == [-1.0, -5.0]


def test_target_range_scores_the_distance_outside_the_range():
    objective = Objective("tpsa", Direction.TARGET_RANGE, target_low=60, target_high=110)
    scores = objective.desirability(pd.Series([40.0, 60.0, 85.0, 110.0, 130.0]))
    assert scores.tolist() == [-20.0, 0.0, 0.0, 0.0, -20.0]


def test_target_value_scores_the_absolute_distance():
    objective = Objective("mol_wt", Direction.TARGET_VALUE, target_value=350)
    assert objective.desirability(pd.Series([300.0, 350.0, 400.0])).tolist() == [-50.0, 0.0, -50.0]


def test_missing_values_get_the_worst_score():
    """A molecule with an unknown objective must never dominate a known one."""
    scores = Objective("qed").desirability(pd.Series([0.9, np.nan, 0.2]))
    assert scores.iloc[1] < 0.2
    assert scores.notna().all()


def test_original_values_are_never_modified():
    values = pd.Series([1.0, 5.0])
    Objective("mol_wt", Direction.MINIMIZE).desirability(values)
    assert values.tolist() == [1.0, 5.0]


def test_target_range_needs_both_bounds():
    with pytest.raises(ObjectiveError, match="target_range"):
        Objective("tpsa", Direction.TARGET_RANGE, target_low=60)


def test_repeated_objective_fields_are_rejected():
    with pytest.raises(ObjectiveError, match="repeated"):
        ObjectiveSet([Objective("qed"), Objective("qed", Direction.MINIMIZE)])


def test_objective_set_validates_columns():
    objectives = ObjectiveSet([Objective("qed"), Objective("missing_column")])
    assert any("missing_column" in problem for problem in objectives.validate(["qed"]))


def test_disabled_objectives_are_ignored():
    objectives = ObjectiveSet([Objective("qed"), Objective("mol_wt", enabled=False)])
    assert objectives.fields() == ("qed",)
    assert len(objectives) == 1


def test_too_many_objectives_is_reported():
    fields = [f"field_{index}" for index in range(MAX_USEFUL_OBJECTIVES + 1)]
    objectives = ObjectiveSet(Objective(name) for name in fields)
    assert any("maximum supported" in problem for problem in objectives.validate(fields))


def test_objectives_round_trip_through_dicts():
    original = ObjectiveSet(
        [
            Objective("qed"),
            Objective("tpsa", Direction.TARGET_RANGE, target_low=60, target_high=110),
        ]
    )
    assert objectives_from_dicts(original.as_dicts()).as_dicts() == original.as_dicts()


def test_constant_objective_is_flagged():
    warnings = ObjectiveSet([Objective("qed")]).diagnostics(frame(qed=[0.5, 0.5, 0.5]))
    assert any("nearly constant" in warning for warning in warnings)


def test_redundant_objectives_are_flagged():
    objectives = ObjectiveSet([Objective("a"), Objective("b")])
    warnings = objectives.diagnostics(frame(a=[1.0, 2.0, 3.0, 4.0], b=[2.0, 4.0, 6.0, 8.0]))
    assert any("are redundant" in warning for warning in warnings)


# --- dominance ---------------------------------------------------------------


def test_known_two_objective_fronts():
    """A: best on both. B and C: trade-offs. D: dominated by everything."""
    matrix = np.array([[3.0, 3.0], [1.0, 3.0], [3.0, 1.0], [0.5, 0.5]])
    result = non_dominated_sort(matrix)
    assert result.ranks.tolist() == [1, 2, 2, 3]
    assert result.front_sizes == {1: 1, 2: 2, 3: 1}


def test_identical_rows_share_a_front():
    """Ties do not dominate each other."""
    matrix = np.array([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    assert non_dominated_sort(matrix).ranks.tolist() == [1, 1, 2]


def test_domination_counts_are_reported():
    matrix = np.array([[2.0, 2.0], [1.0, 1.0], [0.0, 0.0]])
    result = non_dominated_sort(matrix)
    assert result.domination_count.tolist() == [0, 1, 2]
    assert result.dominated_count.tolist() == [2, 1, 0]


def test_every_row_is_non_dominated_when_all_trade_off():
    matrix = np.array([[3.0, 1.0], [2.0, 2.0], [1.0, 3.0]])
    assert non_dominated_sort(matrix).ranks.tolist() == [1, 1, 1]


def test_empty_matrix_is_handled():
    assert non_dominated_sort(np.zeros((0, 2))).ranks.size == 0


def test_blocked_pass_matches_the_unblocked_one():
    """The block loop must not change the answer for sets larger than a block."""
    generator = np.random.default_rng(11)
    matrix = generator.normal(size=(120, 3))
    expected = non_dominated_sort(matrix).ranks.tolist()

    original_block = pareto._BLOCK
    try:
        pareto._BLOCK = 16
        blocked = non_dominated_sort(matrix).ranks.tolist()
    finally:
        pareto._BLOCK = original_block
    assert blocked == expected


def test_normalisation_handles_a_constant_objective():
    scaled = normalise(np.array([[1.0, 5.0], [3.0, 5.0]]))
    assert scaled[:, 0].tolist() == [0.0, 1.0]
    assert scaled[:, 1].tolist() == [1.0, 1.0]


def test_distance_to_ideal_is_zero_for_the_best_row():
    distances = distance_to_ideal(np.array([[1.0, 1.0], [0.0, 0.0]]))
    assert distances[0] == pytest.approx(0.0)
    assert distances[1] > distances[0]


def test_crowding_marks_the_extremes_as_isolated():
    matrix = np.array([[0.0, 3.0], [1.0, 2.0], [3.0, 0.0]])
    distances = crowding_distance(matrix, np.array([1, 1, 1]))
    assert np.isinf(distances[0])
    assert np.isinf(distances[2])
    assert np.isfinite(distances[1])


# --- ranking -----------------------------------------------------------------


def test_ranking_combines_maximisation_and_minimisation():
    candidates = frame(qed=[0.9, 0.5, 0.2], mol_wt=[300.0, 400.0, 500.0])
    objectives = ObjectiveSet([Objective("qed"), Objective("mol_wt", Direction.MINIMIZE)])
    result = rank_candidates(candidates, objectives)
    assert result.table.loc[1, "pareto_rank"] == 1
    assert result.table.loc[3, "pareto_rank"] == 3
    assert list(result.front(1)) == [1]


def test_ranking_uses_the_desirability_of_a_target_range():
    candidates = frame(tpsa=[85.0, 200.0], qed=[0.5, 0.5])
    objectives = ObjectiveSet(
        [
            Objective("tpsa", Direction.TARGET_RANGE, target_low=60, target_high=110),
            Objective("qed"),
        ]
    )
    result = rank_candidates(candidates, objectives)
    assert result.table.loc[1, "pareto_rank"] == 1
    assert result.table.loc[2, "pareto_rank"] == 2


def test_crowded_first_front_produces_a_warning():
    candidates = frame(qed=[0.9, 0.5, 0.1], mol_wt=[300.0, 400.0, 500.0])
    objectives = ObjectiveSet([Objective("qed"), Objective("mol_wt", Direction.MINIMIZE)])
    result = rank_candidates(candidates, objectives)
    assert any("first front" in warning for warning in result.warnings)


def test_rows_for_storage_match_the_pareto_table():
    candidates = frame(qed=[0.9, 0.2], mol_wt=[300.0, 500.0])
    objectives = ObjectiveSet([Objective("qed"), Objective("mol_wt", Direction.MINIMIZE)])
    rows = rank_candidates(candidates, objectives).rows_for_storage()
    assert set(rows[0]) == {
        "record_id",
        "pareto_rank",
        "domination_count",
        "dominated_count",
        "distance_to_ideal",
    }


def test_missing_objective_column_is_rejected():
    with pytest.raises(ObjectiveError, match="missing column"):
        rank_candidates(frame(qed=[0.5]), ObjectiveSet([Objective("not_there")]))


def test_ranker_reuses_the_cached_result():
    candidates = frame(qed=[0.9, 0.2], mol_wt=[300.0, 500.0])
    objectives = ObjectiveSet([Objective("qed"), Objective("mol_wt", Direction.MINIMIZE)])
    ranker = ParetoRanker()
    first = ranker.rank(candidates, objectives)
    assert ranker.rank(candidates, objectives) is first


def test_changing_an_objective_invalidates_the_cache():
    candidates = frame(qed=[0.9, 0.2], mol_wt=[300.0, 500.0])
    ranker = ParetoRanker()
    first = ranker.rank(candidates, ObjectiveSet([Objective("qed")]))
    second = ranker.rank(candidates, ObjectiveSet([Objective("mol_wt", Direction.MINIMIZE)]))
    assert first is not second


def test_changing_the_data_invalidates_the_cache():
    objectives = ObjectiveSet([Objective("qed")])
    ranker = ParetoRanker()
    first = ranker.rank(frame(qed=[0.9, 0.2]), objectives)
    second = ranker.rank(frame(qed=[0.4, 0.2]), objectives)
    assert first is not second


def test_selection_order_starts_with_the_first_front():
    candidates = frame(qed=[0.9, 0.5, 0.2], mol_wt=[300.0, 400.0, 500.0])
    objectives = ObjectiveSet([Objective("qed"), Objective("mol_wt", Direction.MINIMIZE)])
    order = order_within_front(rank_candidates(candidates, objectives), objectives)
    assert order[0] == 1
    assert list(order) == [1, 2, 3]


def test_weights_do_not_change_the_fronts():
    """Weighting must only break ties, never alter dominance."""
    candidates = frame(qed=[0.9, 0.5, 0.2], mol_wt=[300.0, 400.0, 500.0])
    plain = ObjectiveSet([Objective("qed"), Objective("mol_wt", Direction.MINIMIZE)])
    weighted = ObjectiveSet(
        [Objective("qed", weight=10.0), Objective("mol_wt", Direction.MINIMIZE, weight=0.1)]
    )
    assert (
        rank_candidates(candidates, plain).table["pareto_rank"].tolist()
        == rank_candidates(candidates, weighted).table["pareto_rank"].tolist()
    )
