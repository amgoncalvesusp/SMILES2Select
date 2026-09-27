import numpy as np
import pandas as pd
import pytest

from s2s_decision.schema import FeatureSet
from s2s_decision.selection import select_candidates


def candidates():
    return pd.DataFrame(
        {
            "record_id": [3, 2, 1, 4],
            "eligible": [True] * 4,
            "valid": [True] * 4,
            "priority_score": [0.8, 0.9, 0.9, 0.2],
            "murcko_scaffold": ["A", "A", "B", ""],
            "identity": ["a", "b", "c", "d"],
            "molecule_id": ["a", "b", "c", "d"],
            "original_smiles": ["CC", "CCC", "CCCC", "CO"],
            "cluster_id": [0, 0, 1, 1],
            "source__is_final": [False, False, True, False],
        }
    )


def test_score_order_tie_break_provenance_and_no_mutation():
    frame = candidates()
    original = frame.copy(deep=True)
    result = select_candidates(FeatureSet(frame, {"source_hash": "abc"}), 2)
    assert result.manifest["final_ids"] == [1, 2]
    assert result.manifest["input_provenance"] == {"source_hash": "abc"}
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(result.records[original.columns], original)
    assert result.records.is_final.tolist() == [False, True, True, False]


def test_pins_override_count_and_quotas_explicitly():
    result = select_candidates(candidates(), 1, max_per_scaffold=1, pinned_ids=(3, 2))
    assert result.manifest["final_ids"] == [2, 3]
    assert result.manifest["excess"] == 1
    assert result.manifest["status"] == "constraint_conflict"
    assert len(result.manifest["warnings"]) == 2
    assert "pinned molecule" in result.records.loc[0, "selection_reason"]


def test_quotas_coverage_and_shortfall():
    result = select_candidates(candidates(), 4, max_per_scaffold=1, min_scaffolds=3)
    assert result.manifest["final_ids"] == [1, 2, 4]
    assert result.manifest["shortfall"] == 1
    assert result.manifest["scaffold_counts"] == {"B": 1, "A": 1, "": 1}
    limited = select_candidates(candidates(), 3, max_per_cluster=1)
    assert limited.manifest["final_count"] == 2


def test_coverage_reserves_before_filling():
    result = select_candidates(candidates(), 3, min_scaffolds=3)
    assert result.manifest["final_ids"] == [1, 2, 4]


def test_ineligible_and_excluded_remain_with_reasons():
    frame = candidates().assign(eligible=[False, True, True, True])
    result = select_candidates(frame, 4, excluded_ids=(2,))
    assert result.manifest["final_ids"] == [1, 4]
    assert result.records.loc[0, "selection_reason"] == "not eligible"
    assert result.records.loc[1, "selection_reason"] == "excluded by user"


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"pinned_ids": (99,)}, "Unknown pinned"),
        ({"excluded_ids": (99,)}, "Unknown excluded"),
        ({"pinned_ids": (1,), "excluded_ids": (1,)}, "both pinned and excluded"),
        ({"pinned_ids": (True,)}, "integers"),
    ],
)
def test_invalid_pin_and_exclusion(kwargs, match):
    with pytest.raises(ValueError, match=match):
        select_candidates(candidates(), 2, **kwargs)


def test_ineligible_pin_rejected():
    with pytest.raises(ValueError, match="not eligible"):
        select_candidates(candidates().assign(eligible=False), 2, pinned_ids=(1,))


@pytest.mark.parametrize(
    "column,values,match",
    [
        ("record_id", [1, 1, 2, 3], "unique integers"),
        ("record_id", [1.0, 2.0, 3.0, 4.0], "unique integers"),
        ("priority_score", [np.nan, 0.2, 0.3, 0.4], "finite"),
        ("priority_score", [1.1, 0.2, 0.3, 0.4], "finite"),
        ("eligible", ["false"] * 4, "boolean"),
        ("valid", [False] * 4, "invalid"),
        ("identity", ["a", "a", "c", "d"], "Duplicate chemical"),
        ("murcko_scaffold", [None, "A", "B", ""], "scaffold"),
    ],
)
def test_invalid_candidates(column, values, match):
    with pytest.raises(ValueError, match=match):
        select_candidates(candidates().assign(**{column: values}), 2)


def test_cluster_quota_requires_valid_labels():
    with pytest.raises(ValueError, match="cluster_id"):
        select_candidates(candidates().drop(columns="cluster_id"), 2, max_per_cluster=1)


def test_empty_pool_and_invalid_rows_need_no_scores():
    frame = candidates().assign(eligible=False, valid=False, priority_score=np.nan)
    result = select_candidates(frame, 2)
    assert result.manifest["final_ids"] == []
    assert result.manifest["shortfall"] == 2


@pytest.mark.parametrize("n", [None, 0, -1, True, 2.5])
def test_invalid_count(n):
    with pytest.raises(ValueError, match="positive integer"):
        select_candidates(candidates(), n)


def test_object_booleans_remain_valid():
    frame = candidates().astype({"eligible": object, "valid": object})
    assert select_candidates(frame, 1).manifest["final_ids"] == [1]


def test_inherited_low_score_pin_retained_and_unioned_with_explicit_pins():
    frame = candidates().assign(pinned=[False, False, False, True])
    result = select_candidates(frame, 1)
    assert result.manifest["final_ids"] == [4]
    assert result.records.loc[3, "selection_reason"] == "pinned molecule"
    combined = select_candidates(frame, 1, pinned_ids=(2,))
    assert combined.manifest["final_ids"] == [2, 4]
    assert combined.manifest["excess"] == 1
    assert combined.manifest["status"] == "constraint_conflict"
    assert combined.manifest["pinned_ids"] == [2, 4]


def test_inherited_pin_conflicts_with_exclusion():
    frame = candidates().assign(pinned=[False, False, False, True])
    with pytest.raises(ValueError, match="both pinned and excluded"):
        select_candidates(frame, 1, excluded_ids=(4,))


@pytest.mark.parametrize("value", ["false", None, 1])
def test_inherited_pin_requires_boolean(value):
    with pytest.raises(ValueError, match="pinned must contain boolean"):
        select_candidates(candidates().assign(pinned=value), 1)


@pytest.mark.parametrize("valid", [True, False])
def test_inherited_ineligible_pin_rejected(valid):
    frame = candidates().assign(
        pinned=[False, False, False, True],
        eligible=[True, True, True, False],
        valid=[True, True, True, valid],
    )
    with pytest.raises(ValueError, match="not eligible"):
        select_candidates(frame, 1)
