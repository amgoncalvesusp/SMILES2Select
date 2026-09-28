"""Scientific selection contracts: explicit ties, identity and honest limitations."""

import pandas as pd
import pytest

from smiles2select.selection_intelligence.constrained_selection import (
    SelectionConstraints,
    Strategy,
    order_candidates,
    select,
)


@pytest.mark.parametrize("strategy", list(Strategy))
def test_score_ties_use_record_id_independent_of_candidate_row_order(strategy):
    candidates = pd.DataFrame({"qed": [0.8, 0.8, 0.8], "pareto_rank": [1, 1, 1]}, index=[30, 2, 12])
    assert list(order_candidates(candidates, strategy).index) == [2, 12, 30]
    assert select(candidates, SelectionConstraints(target_count=2), strategy).selected_ids == (
        2,
        12,
    )
    assert select(
        candidates.iloc[::-1], SelectionConstraints(target_count=2), strategy
    ).selected_ids == (2, 12)


def test_ties_without_scores_and_pin_order_are_canonical():
    candidates = pd.DataFrame(index=[30, 2, 12])
    assert select(candidates, SelectionConstraints(target_count=2)).selected_ids == (2, 12)
    assert select(
        candidates, SelectionConstraints(target_count=3), pinned_ids=[30, 12]
    ).selected_ids == (12, 30, 2)


@pytest.mark.parametrize(
    "name", ["target_count", "max_per_scaffold", "max_per_cluster", "min_scaffolds"]
)
@pytest.mark.parametrize("value", [True, 1.5, float("nan"), float("inf"), "2"])
def test_count_constraints_require_positive_integers(name, value):
    with pytest.raises(ValueError, match="positive integer"):
        SelectionConstraints(**{name: value})


@pytest.mark.parametrize("index", [[1, 1], [1.2, 1.8], [True, False], ["1", "2"]])
def test_record_identity_must_be_unique_integer(index):
    with pytest.raises(ValueError, match="record_id"):
        select(pd.DataFrame({"qed": [0.9, 0.8]}, index=index))


def test_first_cluster_member_is_not_claimed_to_be_a_centroid():
    candidates = pd.DataFrame({"cluster_id": [4, 4], "qed": [0.8, 0.9]}, index=[1, 2])
    result = select(candidates, SelectionConstraints(target_count=1))
    assert result.selected_ids == (2,)
    assert result.reasons[2] == ["first selected member of cluster 4 (not a centroid)"]


def test_crossed_quotas_repair_greedy_shortfall_when_target_is_feasible():
    candidates = pd.DataFrame(
        {"murcko_scaffold": ["A", "A", "B"], "cluster_id": [1, 2, 1], "qed": [0.9, 0.8, 0.7]},
        index=[1, 2, 3],
    )
    constraints = SelectionConstraints(target_count=2, max_per_scaffold=1, max_per_cluster=1)
    result = select(candidates, constraints)
    assert result.selected_ids == (2, 3)
    assert not result.warnings(constraints)


@pytest.mark.parametrize("shuffle_seed", range(5))
def test_quota_repair_is_deterministic_preserves_pins_and_scaffold_coverage(shuffle_seed):
    candidates = pd.DataFrame(
        {
            "murcko_scaffold": ["A", "A", "B", "C", "C"],
            "cluster_id": [1, 2, 1, 3, 4],
            "qed": [0.9, 0.8, 0.7, 0.6, 0.5],
        },
        index=[1, 2, 3, 4, 5],
    ).sample(frac=1, random_state=shuffle_seed)
    constraints = SelectionConstraints(
        target_count=3, max_per_scaffold=1, max_per_cluster=1, min_scaffolds=3
    )
    result = select(candidates, constraints, pinned_ids=[5])
    assert result.selected_ids == (5, 2, 3)
    assert result.scaffolds_covered == 3
    assert not result.warnings(constraints)
    assert (
        select(candidates, constraints, pinned_ids=[5], explain_rejections=False).selected_ids
        == result.selected_ids
    )


def test_quota_repair_never_displaces_a_pinned_candidate():
    candidates = pd.DataFrame(
        {"murcko_scaffold": ["A", "A", "B"], "cluster_id": [1, 2, 1], "qed": [0.9, 0.8, 0.7]},
        index=[1, 2, 3],
    )
    constraints = SelectionConstraints(target_count=2, max_per_scaffold=1, max_per_cluster=1)
    result = select(candidates, constraints, pinned_ids=[1])
    assert result.selected_ids == (1,)
    assert result.shortfall(constraints) == 1


def test_quota_repair_cardinality_matches_exhaustive_small_libraries():
    from itertools import combinations
    from random import Random

    rng = Random(43)
    for case in range(100):
        size = 7
        scaffolds = [rng.choice([None, "A", "B", "C"]) for _ in range(size)]
        clusters = [rng.choice([None, 1, 2, 3]) for _ in range(size)]
        candidates = pd.DataFrame(
            {
                "murcko_scaffold": scaffolds,
                "cluster_id": clusters,
                "qed": [rng.random() for _ in range(size)],
            },
            index=range(size),
        )
        pins = set(rng.sample(range(size), rng.randrange(3)))
        excluded = {rng.randrange(size)} - pins
        constraints = SelectionConstraints(
            target_count=rng.randint(max(1, len(pins)), size),
            max_per_scaffold=rng.randint(1, 2),
            max_per_cluster=rng.randint(1, 2),
        )
        result = select(
            candidates, constraints, pinned_ids=sorted(pins), excluded_ids=sorted(excluded)
        )
        optimum = 0
        for count in range(len(pins), constraints.target_count + 1):
            for ids in combinations(set(range(size)) - excluded, count):
                if not pins.issubset(ids):
                    continue
                valid = True
                for labels, maximum in [
                    (scaffolds, constraints.max_per_scaffold),
                    (clusters, constraints.max_per_cluster),
                ]:
                    for label in set(labels) - {None}:
                        pinned_usage = sum(labels[rid] == label for rid in pins)
                        if sum(labels[rid] == label for rid in ids) > max(maximum, pinned_usage):
                            valid = False
                if valid:
                    optimum = max(optimum, count)
        assert result.count == optimum, (case, candidates, constraints, pins, result)
        assert pins.issubset(result.selected_ids)
        assert not excluded.intersection(result.selected_ids)
        for labels, maximum in [
            (scaffolds, constraints.max_per_scaffold),
            (clusters, constraints.max_per_cluster),
        ]:
            for label in set(labels) - {None}:
                pinned_usage = sum(labels[rid] == label for rid in pins)
                usage = sum(labels[rid] == label for rid in result.selected_ids)
                assert usage <= max(maximum, pinned_usage)


def test_unsatisfied_scaffold_minimum_discloses_greedy_coverage_limit():
    candidates = pd.DataFrame(
        {"murcko_scaffold": ["A", "A", "B"], "cluster_id": [1, 2, 1], "qed": [0.9, 0.8, 0.7]},
        index=[1, 2, 3],
    )
    constraints = SelectionConstraints(target_count=2, min_scaffolds=2, max_per_cluster=1)
    result = select(candidates, constraints)
    assert result.count == 2
    assert result.scaffolds_covered == 1
    assert "does not prove" in " ".join(result.warnings(constraints))


@pytest.mark.parametrize("strategy", list(Strategy))
def test_explicit_priority_preserves_upstream_order_except_qed_only(strategy):
    candidates = pd.DataFrame(
        {
            "qed": [0.9, 0.8, 0.7],
            "selection_priority": [-0.5, -0.9, -0.9],
            "scaffold_size": [1, 10, 10],
        },
        index=[1, 3, 2],
    )
    expected = (1, 3) if strategy is Strategy.QED_ONLY else (2, 3)
    assert select(candidates, SelectionConstraints(target_count=2), strategy).selected_ids == expected
