"""Continuous scores: QED handling, consensus and ranking."""

from __future__ import annotations

import pandas as pd
import pytest

from smiles2select.scores import ranking
from smiles2select.scores.consensus import (
    ConsensusWeights,
    alert_penalty,
    consensus_score,
    hard_violation_count,
    profile_pass_fraction,
    score_table,
)
from smiles2select.scores.qed import QedSelection, distribution, passes, percentile, rank

pytestmark = pytest.mark.unit


@pytest.fixture
def status():
    index = pd.RangeIndex(1, 4, name="record_id")
    return pd.DataFrame(
        {
            "lipinski__passed": [True, True, False],
            "lipinski__violations": [0, 0, 2],
            "veber__passed": [True, False, False],
            "veber__violations": [0, 1, 1],
        },
        index=index,
    )


def test_qed_compute_and_rank_modes_never_exclude():
    values = pd.Series([0.1, 0.9], index=[1, 2])
    assert passes(values, QedSelection(mode="compute")).tolist() == [True, True]
    assert passes(values, QedSelection(mode="rank")).tolist() == [True, True]
    assert QedSelection(mode="compute").excludes is False


def test_qed_threshold_and_percentile_modes_exclude():
    values = pd.Series([0.1, 0.5, 0.9], index=[1, 2, 3])
    assert passes(values, QedSelection(mode="threshold", threshold=0.5)).tolist() == [
        False,
        True,
        True,
    ]
    assert passes(values, QedSelection(mode="top_percentile", percentile=50)).tolist() == [
        False,
        True,
        True,
    ]


def test_qed_percentile_outside_range_is_rejected():
    with pytest.raises(ValueError):
        QedSelection(mode="top_percentile", percentile=0)


def test_qed_ranking_puts_the_best_first():
    values = pd.Series([0.3, 0.9, 0.6], index=[1, 2, 3])
    assert rank(values).tolist() == [3, 1, 2]
    assert percentile(values).round(0).tolist() == [33.0, 100.0, 67.0]


def test_qed_distribution_covers_the_unit_interval():
    histogram = distribution(pd.Series([0.05, 0.15, 0.95]), bins=10)
    assert histogram["count"].sum() == 3
    assert histogram["bin_lower"].min() == pytest.approx(0.0)
    assert histogram["bin_upper"].max() == pytest.approx(1.0)


def test_qed_distribution_of_empty_input_is_empty():
    assert distribution(pd.Series(dtype=float)).empty


def test_profile_pass_fraction_and_violation_totals(status):
    fraction = profile_pass_fraction(status, ["lipinski", "veber"])
    assert fraction.tolist() == [1.0, 0.5, 0.0]
    assert hard_violation_count(status, ["lipinski", "veber"]).tolist() == [0, 1, 3]


def test_alert_penalty_saturates():
    counts = pd.Series([0, 1, 3, 10])
    assert alert_penalty(counts, saturate_at=3).tolist() == [0.0, pytest.approx(1 / 3), 1.0, 1.0]


def test_consensus_score_rewards_agreement(status):
    qed = pd.Series([0.9, 0.5, 0.1], index=status.index)
    alerts = pd.Series([0, 0, 2], index=status.index)
    score = consensus_score(status, ["lipinski", "veber"], qed, alerts)
    assert score.is_monotonic_decreasing
    assert score.min() >= 0.0
    assert score.max() <= 1.0


def test_consensus_score_without_qed_renormalises(status):
    score = consensus_score(status, ["lipinski", "veber"], None, None, ConsensusWeights())
    assert score.tolist() == [1.0, 0.5, 0.0]


def test_score_table_contains_every_reported_column(status):
    qed = pd.Series([0.9, 0.5, 0.1], index=status.index)
    table = score_table(
        status, ["lipinski", "veber"], qed, pd.Series([0, 0, 1], index=status.index)
    )
    assert set(table.columns) == {
        "profile_pass_fraction",
        "hard_violation_count",
        "qed",
        "alert_count",
        "consensus_score",
    }


def test_ranking_helpers():
    table = pd.DataFrame({"qed": [0.3, 0.9, 0.6]}, index=[1, 2, 3])
    assert ranking.rank_by(table, "qed").tolist() == [3, 1, 2]
    assert ranking.top_n(table, "qed", 2).index.tolist() == [2, 3]
    assert ranking.top_percentile(table, "qed", 50).index.tolist() == [2, 3]


def test_ranking_by_missing_column_is_reported():
    with pytest.raises(KeyError):
        ranking.rank_by(pd.DataFrame({"qed": [0.1]}), "consensus_score")


def test_top_n_rejects_non_positive_counts():
    with pytest.raises(ValueError):
        ranking.top_n(pd.DataFrame({"qed": [0.1]}), "qed", 0)
