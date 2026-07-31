"""Pre-computed indexes behind the threshold sliders.

Moving a slider must never recompute a descriptor. Every value the rules read
was already calculated once; this module sorts those values per rule so the
question "how many molecules survive at threshold X" becomes a binary search
instead of a new pass over the library.

That is what keeps the interface answering in milliseconds while the user drags
a limit, no matter how large the library is.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from smiles2select.rules.engine import Rule

#: Operators a slider can move. The rest (substructure, membership) have no
#: continuous threshold to drag.
SLIDEABLE_OPERATORS = frozenset({"<", "<=", ">", ">="})

_CURVE_COLUMNS = ["threshold", "retained", "retention"]
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
class ThresholdIndex:
    """Sorted values of one descriptor, for one rule."""

    rule_id: str
    profile_id: str
    descriptor: str
    operator: str
    original_threshold: float
    record_ids: np.ndarray
    sorted_values: np.ndarray
    missing_ids: np.ndarray

    @property
    def count(self) -> int:
        return int(self.record_ids.size + self.missing_ids.size)

    @property
    def is_upper_bound(self) -> bool:
        return self.operator in {"<", "<="}

    def retention(self, threshold: float) -> int:
        """How many molecules satisfy the rule at this threshold.

        Molecules whose descriptor is missing are never counted as retained:
        the rule engine fails them closed, and the slider must agree with it.
        """
        values = self.sorted_values
        if self.operator == "<=":
            return int(np.searchsorted(values, threshold, side="right"))
        if self.operator == "<":
            return int(np.searchsorted(values, threshold, side="left"))
        if self.operator == ">=":
            return int(values.size - np.searchsorted(values, threshold, side="left"))
        return int(values.size - np.searchsorted(values, threshold, side="right"))

    def passing_ids(self, threshold: float) -> np.ndarray:
        """Record ids that satisfy the rule at this threshold."""
        retained = self.retention(threshold)
        if self.is_upper_bound:
            return self.record_ids[:retained]
        return self.record_ids[self.record_ids.size - retained :]

    def mask(self, threshold: float, index: pd.Index) -> pd.Series:
        """Boolean mask aligned to ``index``; missing descriptors are False."""
        passing = pd.Index(self.passing_ids(threshold))
        return pd.Series(index.isin(passing), index=index)

    def retention_curve(self, points: int = 60) -> pd.DataFrame:
        """Molecules retained across the observed range of the descriptor."""
        if self.sorted_values.size == 0:
            return pd.DataFrame(columns=_CURVE_COLUMNS)
        lowest = float(self.sorted_values[0])
        highest = float(self.sorted_values[-1])
        thresholds = (
            np.array([lowest], dtype=float)
            if lowest == highest
            else np.linspace(lowest, highest, points)
        )
        retained = np.array([self.retention(value) for value in thresholds], dtype=float)
        return pd.DataFrame(
            {
                "threshold": thresholds,
                "retained": retained.astype(int),
                "retention": retained / self.count if self.count else 0.0,
            }
        )

    def change_points(self, curve: pd.DataFrame | None = None, top: int = 3) -> pd.DataFrame:
        """Thresholds where retention moves fastest.

        These are where a small change of limit costs or gains many molecules -
        the places worth looking at before settling on a cut-off.
        """
        data = self.retention_curve() if curve is None else curve
        if len(data) < 3:
            return data.assign(slope=pd.Series(dtype=float)).head(0)
        slope = np.abs(np.gradient(data["retained"].to_numpy(dtype=float)))
        return data.assign(slope=slope).sort_values("slope", ascending=False).head(top)

    def recovered_between(self, current: float, proposed: float) -> int:
        """How many molecules the move would add (negative means lost)."""
        return self.retention(proposed) - self.retention(current)


def build_index(
    rule: Rule, descriptors: pd.DataFrame, threshold: float | None = None
) -> ThresholdIndex | None:
    """Sort one descriptor for one rule, or None when the rule has no slider."""
    if rule.operator not in SLIDEABLE_OPERATORS or rule.descriptor not in descriptors.columns:
        return None

    values = pd.to_numeric(descriptors[rule.descriptor], errors="coerce")
    known = values.dropna().sort_values(kind="stable")
    return ThresholdIndex(
        rule_id=rule.id,
        profile_id=rule.profile_id,
        descriptor=rule.descriptor,
        operator=rule.operator,
        original_threshold=float(threshold if threshold is not None else rule.threshold),
        record_ids=known.index.to_numpy(),
        sorted_values=known.to_numpy(dtype=float),
        missing_ids=values.index[values.isna()].to_numpy(),
    )


class ThresholdIndexSet:
    """Every sliderable rule of a run, indexed once."""

    def __init__(self, indexes: Iterable[ThresholdIndex] = ()) -> None:
        self._indexes = {index.rule_id: index for index in indexes}

    def __len__(self) -> int:
        return len(self._indexes)

    def __contains__(self, rule_id: object) -> bool:
        return rule_id in self._indexes

    def __iter__(self):
        return iter(self._indexes.values())

    def get(self, rule_id: str) -> ThresholdIndex:
        try:
            return self._indexes[rule_id]
        except KeyError as exc:
            raise KeyError(
                f"rule '{rule_id}' has no threshold index "
                f"(only {sorted(SLIDEABLE_OPERATORS)} operators can be moved)"
            ) from exc

    def rule_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._indexes))

    def original_thresholds(self) -> dict[str, float]:
        return {rule_id: index.original_threshold for rule_id, index in self._indexes.items()}

    @classmethod
    def from_rules(cls, rules: Sequence[Rule], descriptors: pd.DataFrame) -> "ThresholdIndexSet":
        built = (build_index(rule, descriptors) for rule in rules)
        return cls(index for index in built if index is not None)
