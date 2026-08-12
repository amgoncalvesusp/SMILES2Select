"""Final selection under constraints.

Picks exactly N molecules while respecting scaffold and cluster quotas, keeping
pinned compounds, and preferring good Pareto fronts, spread and robustness.

The order of operations is not cosmetic. Pinned molecules go in first and are
never displaced - a human decision outranks any automatic ranking. Quotas are
enforced while the count is filled, because 500 compounds from three scaffolds
is a worse library than 400 from a hundred.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class Strategy(str, Enum):
    """How candidates are ordered before the quotas are applied."""

    BALANCED = "balanced"
    PARETO_FIRST = "pareto_first"
    DIVERSITY_FIRST = "diversity_first"
    SCAFFOLD_COVERAGE = "scaffold_coverage"
    MANUAL_ASSISTED = "manual_assisted"


#: Sort keys per strategy: (column, ascending). Missing columns are skipped, so
#: a run without Pareto or robustness still selects sensibly.
_STRATEGY_KEYS: dict[Strategy, tuple[tuple[str, bool], ...]] = {
    Strategy.BALANCED: (
        ("pareto_rank", True),
        ("crowding_distance", False),
        ("robustness_score", False),
        ("qed", False),
    ),
    Strategy.PARETO_FIRST: (("pareto_rank", True), ("distance_to_ideal", True), ("qed", False)),
    Strategy.DIVERSITY_FIRST: (
        ("crowding_distance", False),
        ("pareto_rank", True),
        ("robustness_score", False),
    ),
    Strategy.SCAFFOLD_COVERAGE: (
        ("scaffold_size", True),
        ("pareto_rank", True),
        ("robustness_score", False),
    ),
    Strategy.MANUAL_ASSISTED: (("pareto_rank", True), ("qed", False)),
}

LIMIT_REACHED = "final count reached"
SCAFFOLD_QUOTA = "scaffold quota exceeded"
CLUSTER_QUOTA = "cluster quota exceeded"


@dataclass(frozen=True)
class SelectionConstraints:
    """What the final set must satisfy."""

    target_count: int | None = None
    max_per_scaffold: int | None = None
    min_scaffolds: int | None = None
    max_per_cluster: int | None = None
    preserve_pinned: bool = True

    def __post_init__(self) -> None:
        for name in ("target_count", "max_per_scaffold", "max_per_cluster", "min_scaffolds"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be at least 1")

    def summary_rows(self) -> list[tuple[str, object]]:
        return [
            ("final molecule count", self.target_count or "unlimited"),
            ("maximum per scaffold", self.max_per_scaffold or "-"),
            ("minimum scaffolds", self.min_scaffolds or "-"),
            ("maximum per cluster", self.max_per_cluster or "-"),
            ("preserve pinned molecules", "yes" if self.preserve_pinned else "no"),
        ]


@dataclass(frozen=True)
class SelectionOutcome:
    """Who was selected, who was not, and why in both cases."""

    selected_ids: tuple[int, ...]
    reasons: dict[int, list[str]] = field(default_factory=dict)
    rejections: dict[int, list[str]] = field(default_factory=dict)
    scaffold_usage: dict[str, int] = field(default_factory=dict)
    cluster_usage: dict[int, int] = field(default_factory=dict)
    strategy: Strategy = Strategy.BALANCED

    @property
    def count(self) -> int:
        return len(self.selected_ids)

    @property
    def scaffolds_covered(self) -> int:
        return len(self.scaffold_usage)

    @property
    def clusters_covered(self) -> int:
        return len(self.cluster_usage)

    def shortfall(self, constraints: SelectionConstraints) -> int:
        """How many molecules are missing from the requested count."""
        if constraints.target_count is None:
            return 0
        return max(0, constraints.target_count - self.count)

    def warnings(self, constraints: SelectionConstraints) -> list[str]:
        messages: list[str] = []
        missing = self.shortfall(constraints)
        if missing:
            messages.append(
                f"{constraints.target_count} molecules were requested, but the constraints "
                f"allowed only {self.count}. Missing: {missing}."
            )
        if constraints.min_scaffolds and self.scaffolds_covered < constraints.min_scaffolds:
            messages.append(
                f"The selection covers {self.scaffolds_covered} scaffolds; the requested minimum was "
                f"{constraints.min_scaffolds}."
            )
        return messages


def order_candidates(
    candidates: pd.DataFrame, strategy: Strategy = Strategy.BALANCED
) -> pd.DataFrame:
    """Sort the candidate table according to the chosen strategy."""
    keys = [
        (column, ascending)
        for column, ascending in _STRATEGY_KEYS[strategy]
        if column in candidates.columns
    ]
    if not keys:
        return candidates
    return candidates.sort_values(
        by=[column for column, _ in keys],
        ascending=[ascending for _, ascending in keys],
        kind="stable",
    )


def _scaffold_of(row: pd.Series) -> str | None:
    value = row.get("murcko_scaffold")
    return None if value is None or pd.isna(value) else str(value)


def _cluster_of(row: pd.Series) -> int | None:
    value = row.get("cluster_id")
    return None if value is None or pd.isna(value) else int(value)


def _quota_block(
    scaffold: str | None,
    cluster: int | None,
    scaffold_usage: dict[str, int],
    cluster_usage: dict[int, int],
    constraints: SelectionConstraints,
) -> list[str]:
    blocked: list[str] = []
    if (
        constraints.max_per_scaffold is not None
        and scaffold is not None
        and scaffold_usage.get(scaffold, 0) >= constraints.max_per_scaffold
    ):
        blocked.append(SCAFFOLD_QUOTA)
    if (
        constraints.max_per_cluster is not None
        and cluster is not None
        and cluster_usage.get(cluster, 0) >= constraints.max_per_cluster
    ):
        blocked.append(CLUSTER_QUOTA)
    return blocked


def _selection_reasons(
    row: pd.Series,
    scaffold: str | None,
    cluster: int | None,
    scaffold_usage: dict[str, int],
    cluster_usage: dict[int, int],
) -> list[str]:
    reasons: list[str] = []
    rank = row.get("pareto_rank")
    if rank is not None and not pd.isna(rank):
        reasons.append(f"Pareto Front {int(rank)}")
    if cluster is not None and cluster_usage.get(cluster, 0) == 0:
        reasons.append(f"cluster {cluster} representative")
    if scaffold is not None and scaffold_usage.get(scaffold, 0) == 0:
        reasons.append("first member of its scaffold")
    robustness = row.get("robustness_score")
    if robustness is not None and not pd.isna(robustness) and float(robustness) >= 0.5:
        reasons.append("robust approval")
    return reasons or ["strategy ordering"]


def _consume(
    scaffold: str | None,
    cluster: int | None,
    scaffold_usage: dict[str, int],
    cluster_usage: dict[int, int],
) -> None:
    if scaffold is not None:
        scaffold_usage[scaffold] = scaffold_usage.get(scaffold, 0) + 1
    if cluster is not None:
        cluster_usage[cluster] = cluster_usage.get(cluster, 0) + 1


def select(
    candidates: pd.DataFrame,
    constraints: SelectionConstraints = SelectionConstraints(),
    strategy: Strategy = Strategy.BALANCED,
    pinned_ids: Sequence[int] = (),
    excluded_ids: Sequence[int] = (),
) -> SelectionOutcome:
    """Compose the final set.

    ``candidates`` is indexed by ``record_id`` and may carry ``pareto_rank``,
    ``crowding_distance``, ``robustness_score``, ``qed``, ``murcko_scaffold``
    and ``cluster_id``. Whatever is present is used; whatever is missing is
    skipped rather than faked.
    """
    reasons: dict[int, list[str]] = {}
    rejections: dict[int, list[str]] = {}
    scaffold_usage: dict[str, int] = {}
    cluster_usage: dict[int, int] = {}
    selected: list[int] = []

    pool = candidates.loc[~candidates.index.isin(set(excluded_ids))]

    if constraints.preserve_pinned:
        for raw_id in pinned_ids:
            record_id = int(raw_id)
            if record_id not in pool.index or record_id in selected:
                continue
            row = pool.loc[record_id]
            selected.append(record_id)
            reasons[record_id] = ["pinned molecule"]
            _consume(_scaffold_of(row), _cluster_of(row), scaffold_usage, cluster_usage)

    for raw_id, row in order_candidates(pool, strategy).iterrows():
        record_id = int(raw_id)
        if record_id in selected:
            continue
        if constraints.target_count is not None and len(selected) >= constraints.target_count:
            rejections[record_id] = [LIMIT_REACHED]
            continue

        scaffold = _scaffold_of(row)
        cluster = _cluster_of(row)
        blocked_by = _quota_block(scaffold, cluster, scaffold_usage, cluster_usage, constraints)
        if blocked_by:
            rejections[record_id] = blocked_by
            continue

        selected.append(record_id)
        reasons[record_id] = _selection_reasons(
            row, scaffold, cluster, scaffold_usage, cluster_usage
        )
        _consume(scaffold, cluster, scaffold_usage, cluster_usage)

    return SelectionOutcome(
        selected_ids=tuple(selected),
        reasons=reasons,
        rejections=rejections,
        scaffold_usage=scaffold_usage,
        cluster_usage=cluster_usage,
        strategy=strategy,
    )
