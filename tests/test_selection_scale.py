"""Lightweight selection preserves scientific choices without retaining every rejection."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from smiles2select.selection_intelligence.constrained_selection import (
    SelectionConstraints,
    Strategy,
    select,
)


@pytest.fixture
def library():
    return pd.DataFrame(
        {
            "qed": [0.9, 0.9, 0.8, 0.7, 0.6, 0.5, np.nan],
            "pareto_rank": [1, 1, 2, 2, 2, 3, 3],
            "murcko_scaffold": ["A", "A", "B", "C", "D", None, ""],
            "cluster_id": [1, 1, 1, 2, 3, pd.NA, 3],
            "robustness_score": [0.9, 0.3, np.nan, 0.7, 0.2, 0.5, 0.1],
        },
        index=pd.Index(range(10, 17), name="record_id"),
    )


@pytest.mark.parametrize("strategy", list(Strategy))
@pytest.mark.parametrize(
    "constraints,pins,exclusions",
    [
        (SelectionConstraints(target_count=3), (), ()),
        (SelectionConstraints(target_count=3, min_scaffolds=3), (11,), (14,)),
        (SelectionConstraints(target_count=5, max_per_cluster=1), (), ()),
        (SelectionConstraints(target_count=2, max_per_scaffold=1), (10, 11, 10), ()),
        (SelectionConstraints(min_scaffolds=8, max_per_scaffold=1), (99,), (10,)),
        (SelectionConstraints(target_count=2, preserve_pinned=False), (16,), (11,)),
    ],
)
def test_light_selection_preserves_full_selection(library, strategy, constraints, pins, exclusions):
    before = library.copy(deep=True)
    full = select(library, constraints, strategy, pins, exclusions)
    light = select(library, constraints, strategy, pins, exclusions, explain_rejections=False)
    assert light == replace(full, rejections={})
    pd.testing.assert_frame_equal(library, before)


def test_light_selection_uses_no_series_per_candidate(monkeypatch):
    data = pd.DataFrame({"qed": np.arange(100_000) / 100_000})

    def forbidden(*args, **kwargs):
        raise AssertionError("candidate iteration must not allocate pandas Series")

    monkeypatch.setattr(pd.DataFrame, "iterrows", forbidden)
    result = select(data, SelectionConstraints(target_count=3), explain_rejections=False)
    assert result.selected_ids == (99_999, 99_998, 99_997)
    assert result.rejections == {}


def test_light_empty_and_minimal_library():
    constraints = SelectionConstraints(target_count=2)
    assert select(pd.DataFrame(), constraints, explain_rejections=False).count == 0
    result = select(pd.DataFrame(index=[12, 2, 30]), constraints, explain_rejections=False)
    assert result.selected_ids == (12, 2)
    assert result.reasons == {12: ["strategy ordering"], 2: ["strategy ordering"]}


def test_default_retains_rejection_explanations(library):
    result = select(library, SelectionConstraints(target_count=1))
    assert result.selected_ids == (10,)
    assert result.rejections == {rid: ["final count reached"] for rid in range(11, 17)}
