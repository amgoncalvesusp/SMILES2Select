"""Consensus and aggregate scores.

The consensus score is an internal software metric for ranking within one run.
It is not a validated pharmacokinetic model and must never be reported as one.
It mixes quantities with different meanings (rule agreement, a desirability
score, an alert count), so only comparisons inside the same run are meaningful.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ConsensusWeights:
    """Relative contribution of each component to the consensus score."""

    profile_pass_fraction: float = 0.5
    qed: float = 0.35
    alert_penalty: float = 0.15

    def total(self) -> float:
        return self.profile_pass_fraction + self.qed + self.alert_penalty


def profile_pass_fraction(status: pd.DataFrame, profile_ids: Sequence[str]) -> pd.Series:
    """Fraction of the selected profiles each molecule passes (0-1)."""
    if not profile_ids:
        return pd.Series(0.0, index=status.index)
    columns = [f"{profile_id}__passed" for profile_id in profile_ids]
    return status[columns].astype(float).mean(axis=1)


def hard_violation_count(status: pd.DataFrame, profile_ids: Sequence[str]) -> pd.Series:
    """Total hard violations across the selected profiles."""
    if not profile_ids:
        return pd.Series(0, index=status.index, dtype="int64")
    columns = [f"{profile_id}__violations" for profile_id in profile_ids]
    return status[columns].sum(axis=1).astype("int64")


def alert_penalty(alert_counts: pd.Series, *, saturate_at: int = 3) -> pd.Series:
    """Alert count mapped to [0, 1]; ``saturate_at`` alerts give the full penalty."""
    if saturate_at <= 0:
        raise ValueError("saturate_at must be positive")
    return (alert_counts.fillna(0).clip(lower=0, upper=saturate_at) / saturate_at).astype(float)


def consensus_score(
    status: pd.DataFrame,
    profile_ids: Sequence[str],
    qed_values: pd.Series | None = None,
    alert_counts: pd.Series | None = None,
    weights: ConsensusWeights = ConsensusWeights(),
) -> pd.Series:
    """Weighted combination of rule agreement, QED and alert penalty.

    Components that were not computed are dropped and the remaining weights are
    renormalised, so switching QED off shifts its weight to the other terms
    instead of silently scoring every molecule lower.
    """
    components: list[tuple[float, pd.Series]] = [
        (weights.profile_pass_fraction, profile_pass_fraction(status, profile_ids))
    ]
    if qed_values is not None:
        components.append((weights.qed, qed_values.reindex(status.index).fillna(0.0).astype(float)))
    if alert_counts is not None:
        penalty = alert_penalty(alert_counts.reindex(status.index))
        components.append((weights.alert_penalty, 1.0 - penalty))

    total_weight = sum(weight for weight, _ in components)
    if total_weight <= 0:
        return pd.Series(0.0, index=status.index)

    weighted = pd.Series(0.0, index=status.index)
    for weight, series in components:
        weighted = weighted + weight * series
    return (weighted / total_weight).clip(lower=0.0, upper=1.0)


def score_table(
    status: pd.DataFrame,
    profile_ids: Sequence[str],
    qed_values: pd.Series | None = None,
    alert_counts: pd.Series | None = None,
    weights: ConsensusWeights = ConsensusWeights(),
) -> pd.DataFrame:
    """All continuous scores in one frame, ready for storage and export."""
    table = pd.DataFrame(index=status.index)
    table["profile_pass_fraction"] = profile_pass_fraction(status, profile_ids)
    table["hard_violation_count"] = hard_violation_count(status, profile_ids)
    if qed_values is not None:
        table["qed"] = qed_values.reindex(status.index)
    table["alert_count"] = (
        alert_counts.reindex(status.index).fillna(0).astype("int64")
        if alert_counts is not None
        else 0
    )
    table["consensus_score"] = consensus_score(
        status,
        profile_ids,
        qed_values,
        table["alert_count"],
        weights,
    )
    return table
