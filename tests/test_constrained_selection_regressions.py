import pandas as pd
import pytest

from smiles2select.selection_intelligence.constrained_selection import (
    SelectionConstraints,
    select,
)


def candidates():
    return pd.DataFrame(
        {"qed": [0.9, 0.8, 0.7, 0.6], "murcko_scaffold": ["A", "A", "B", "C"],
         "cluster_id": [1, 1, 2, 3]},
        index=pd.Index([1, 2, 3, 4], name="record_id"),
    )


@pytest.mark.parametrize("pinned", [(), (2,)])
def test_minimum_scaffolds_reserves_places_before_ranked_fill(pinned):
    constraints = SelectionConstraints(target_count=3, min_scaffolds=3)
    result = select(candidates(), constraints, pinned_ids=pinned)
    assert result.count == 3
    assert result.scaffolds_covered == 3
    assert set(pinned) <= set(result.selected_ids)
    assert not result.warnings(constraints)


def test_unreachable_scaffold_minimum_still_fills_available_places():
    constraints = SelectionConstraints(target_count=3, min_scaffolds=4)
    result = select(candidates(), constraints, excluded_ids=[4])
    assert set(result.selected_ids) == {1, 2, 3}
    assert result.warnings(constraints)


def test_scaffold_minimum_respects_cluster_quota_and_exclusions():
    data = candidates().assign(cluster_id=[1, 1, 1, 2])
    constraints = SelectionConstraints(target_count=3, min_scaffolds=3, max_per_cluster=1)
    result = select(data, constraints, excluded_ids=[2])
    assert result.selected_ids == (1, 4)
    assert result.rejections[3] == ["cluster quota exceeded"]


def test_pinned_overrides_report_count_and_quota_violations():
    constraints = SelectionConstraints(target_count=1, max_per_scaffold=1, max_per_cluster=1)
    result = select(candidates(), constraints, pinned_ids=[1, 2])
    assert result.selected_ids == (1, 2)
    warnings = " ".join(result.warnings(constraints)).lower()
    assert "count" in warnings
    assert "scaffold" in warnings
    assert "cluster" in warnings
