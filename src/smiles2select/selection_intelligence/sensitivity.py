"""Threshold sensitivity.

Answers the question a slider asks: if this limit moves, who enters, who
leaves, and how much of the library survives?

It works entirely on the pre-computed indexes - no descriptor is recalculated,
which is what allows a simulation to be applied, compared and discarded without
touching the active recipe.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from smiles2select.selection_intelligence.threshold_indexes import ThresholdIndexSet

#: The four ways a molecule can move when thresholds change.
STAYED_IN = "permaneceu selecionada"
ENTERED = "entrou na seleção"
LEFT = "saiu da seleção"
STAYED_OUT = "permaneceu excluída"

_RESTRICTIVE_COLUMNS = [
    "rule_id",
    "profile_id",
    "descriptor",
    "threshold",
    "retained",
    "rejected",
    "retention",
]


@dataclass(frozen=True)
class RetentionSummary:
    """What one threshold change does to a single rule."""

    rule_id: str
    original_threshold: float
    applied_threshold: float
    retained_before: int
    retained_after: int
    total: int

    @property
    def delta(self) -> int:
        return self.retained_after - self.retained_before

    @property
    def retention(self) -> float:
        return self.retained_after / self.total if self.total else 0.0

    def describe(self) -> str:
        if self.delta == 0:
            return f"{self.rule_id}: {self.applied_threshold:g} não altera a retenção"
        direction = "recupera" if self.delta > 0 else "perde"
        return (
            f"{self.rule_id}: {self.original_threshold:g} -> {self.applied_threshold:g} "
            f"{direction} {abs(self.delta)} molécula(s)"
        )


@dataclass(frozen=True)
class SensitivityResult:
    """Effect of a whole set of threshold changes."""

    mask_before: pd.Series
    mask_after: pd.Series
    per_rule: tuple[RetentionSummary, ...] = ()
    applied_thresholds: Mapping[str, float] = field(default_factory=dict)

    @property
    def transitions(self) -> pd.Series:
        """One label per molecule: stayed in, entered, left, stayed out."""
        before = self.mask_before.astype(bool)
        after = self.mask_after.reindex(before.index).fillna(False).astype(bool)
        labels = pd.Series(STAYED_OUT, index=before.index, dtype="object")
        labels[before & after] = STAYED_IN
        labels[~before & after] = ENTERED
        labels[before & ~after] = LEFT
        return labels

    @property
    def counts(self) -> dict[str, int]:
        counted = self.transitions.value_counts()
        return {
            STAYED_IN: int(counted.get(STAYED_IN, 0)),
            ENTERED: int(counted.get(ENTERED, 0)),
            LEFT: int(counted.get(LEFT, 0)),
            STAYED_OUT: int(counted.get(STAYED_OUT, 0)),
        }

    @property
    def retained(self) -> int:
        return int(self.mask_after.sum())

    @property
    def retention(self) -> float:
        total = len(self.mask_after)
        return self.retained / total if total else 0.0

    def entered_ids(self) -> pd.Index:
        return self.transitions.index[self.transitions == ENTERED]

    def left_ids(self) -> pd.Index:
        return self.transitions.index[self.transitions == LEFT]

    def summary_rows(self) -> list[tuple[str, object]]:
        counts = self.counts
        return [
            ("candidatos mantidos", self.retained),
            ("percentual de retenção", f"{self.retention * 100:.1f}%"),
            ("entraram", counts[ENTERED]),
            ("saíram", counts[LEFT]),
            ("permaneceram selecionadas", counts[STAYED_IN]),
        ]


def evaluate_mask(
    indexes: ThresholdIndexSet,
    index: pd.Index,
    thresholds: Mapping[str, float] | None = None,
    rule_ids: Sequence[str] | None = None,
) -> pd.Series:
    """Molecules satisfying every indexed rule at the given thresholds.

    Rules not named in ``thresholds`` keep the value their profile shipped
    with, so a simulation only changes what the user actually moved.
    """
    chosen = list(rule_ids) if rule_ids is not None else list(indexes.rule_ids())
    mask = pd.Series(True, index=index)
    overrides = thresholds or {}
    for rule_id in chosen:
        threshold_index = indexes.get(rule_id)
        threshold = float(overrides.get(rule_id, threshold_index.original_threshold))
        mask &= threshold_index.mask(threshold, index)
    return mask


def simulate(
    indexes: ThresholdIndexSet,
    index: pd.Index,
    thresholds: Mapping[str, float],
    rule_ids: Sequence[str] | None = None,
) -> SensitivityResult:
    """Compare the current thresholds with a proposed set."""
    chosen = list(rule_ids) if rule_ids is not None else list(indexes.rule_ids())
    before = evaluate_mask(indexes, index, None, chosen)
    after = evaluate_mask(indexes, index, thresholds, chosen)

    summaries = []
    for rule_id, value in thresholds.items():
        threshold_index = indexes.get(rule_id)
        summaries.append(
            RetentionSummary(
                rule_id=rule_id,
                original_threshold=threshold_index.original_threshold,
                applied_threshold=float(value),
                retained_before=threshold_index.retention(threshold_index.original_threshold),
                retained_after=threshold_index.retention(float(value)),
                total=threshold_index.count,
            )
        )

    return SensitivityResult(
        mask_before=before,
        mask_after=after,
        per_rule=tuple(summaries),
        applied_thresholds=dict(thresholds),
    )


def retention_table(indexes: ThresholdIndexSet, points: int = 60) -> dict[str, pd.DataFrame]:
    """Retention curve of every sliderable rule, for the studio charts."""
    return {index.rule_id: index.retention_curve(points) for index in indexes}


def most_restrictive(indexes: ThresholdIndexSet, index: pd.Index, top: int = 5) -> pd.DataFrame:
    """Rules that reject the most molecules on their own.

    Reported one rule at a time: a rule that looks harmless inside a profile
    may be the one actually driving the whole selection.
    """
    total = len(index)
    rows = [
        {
            "rule_id": threshold_index.rule_id,
            "profile_id": threshold_index.profile_id,
            "descriptor": threshold_index.descriptor,
            "threshold": threshold_index.original_threshold,
            "retained": threshold_index.retention(threshold_index.original_threshold),
            "rejected": total - threshold_index.retention(threshold_index.original_threshold),
            "retention": (
                threshold_index.retention(threshold_index.original_threshold) / total
                if total
                else 0.0
            ),
        }
        for threshold_index in indexes
    ]
    if not rows:
        return pd.DataFrame(columns=_RESTRICTIVE_COLUMNS)
    return pd.DataFrame(rows).sort_values("retention").head(top).reset_index(drop=True)
