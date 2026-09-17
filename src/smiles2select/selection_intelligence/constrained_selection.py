"""Final selection under constraints.

Fills the requested count where feasible under scaffold and cluster quotas, keeping
pinned compounds, and preferring good Pareto fronts, spread and robustness.

The order of operations is not cosmetic. Pinned molecules go in first and are
never displaced - a human decision outranks any automatic ranking. Quotas are
enforced while the count is filled, because 500 compounds from three scaffolds
is a worse library than 400 from a hundred.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

CORE_VERSION = "3.3.1"


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
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        if type(self.preserve_pinned) is not bool:
            raise ValueError("preserve_pinned must be boolean")

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
                f"{constraints.target_count} molecules were requested, but the selection "
                f"reached only {self.count}. Missing: {missing}."
            )
        if constraints.min_scaffolds and self.scaffolds_covered < constraints.min_scaffolds:
            messages.append(
                f"The selection covers {self.scaffolds_covered} scaffolds; the requested minimum was "
                f"{constraints.min_scaffolds}. "
                "Scaffold coverage uses greedy reservation; this does not prove "
                "that the requested minimum is infeasible."
            )
        if constraints.target_count is not None and self.count > constraints.target_count:
            messages.append(
                f"Selected molecules exceed the requested count: {self.count} selected "
                f"for a target of {constraints.target_count}."
            )
        for label, usage, maximum in (
            ("scaffold", self.scaffold_usage, constraints.max_per_scaffold),
            ("cluster", self.cluster_usage, constraints.max_per_cluster),
        ):
            if maximum is not None and any(count > maximum for count in usage.values()):
                messages.append(f"Selected molecules exceed the maximum per {label} ({maximum}).")
        return messages


def order_candidates(
    candidates: pd.DataFrame, strategy: Strategy = Strategy.BALANCED
) -> pd.DataFrame:
    """Lexicographic strategy ranking; ties use ascending record_id, NaNs last."""
    candidates = candidates.sort_index(kind="stable")
    keys = [
        (column, ascending)
        for column, ascending in _STRATEGY_KEYS[strategy]
        if column in candidates.columns
    ]
    if "selection_priority" in candidates:
        keys = [("selection_priority", True)]
        if (
            strategy is Strategy.SCAFFOLD_COVERAGE
            and "murcko_scaffold" in candidates
            and "scaffold_size" in candidates
        ):
            keys = [("scaffold_size", True), *keys]
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
    row: Mapping[str, object] | pd.Series,
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
        reasons.append(f"first selected member of cluster {cluster} (not a centroid)")
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


# Bipartite residual graph: every edge is a distinct candidate record, including
# parallel edges. None denotes an unconstrained missing scaffold/cluster label.
_Node = tuple[str, object]


def _augmenting_path(
    graph: dict[_Node, list[tuple[_Node, int]]],
    chosen: set[int],
    usage: dict[_Node, int],
    constraints: SelectionConstraints,
) -> list[int]:
    """Find the first ranked residual path; never displace a fixed pin."""
    limits = {"scaffold": constraints.max_per_scaffold, "cluster": constraints.max_per_cluster}

    def has_capacity(node: _Node) -> bool:
        maximum = limits[node[0]]
        return node[1] is None or maximum is None or usage.get(node, 0) < maximum

    starts = [node for node in graph if node[0] == "scaffold" and has_capacity(node)]
    previous: dict[_Node, tuple[_Node, int] | None] = dict.fromkeys(starts)
    queue = deque(starts)
    while queue:
        node = queue.popleft()
        if node[0] == "cluster" and has_capacity(node):
            path = []
            while previous[node] is not None:
                node, record_id = previous[node]
                path.append(record_id)
            return path
        for neighbour, record_id in graph[node]:
            forward = node[0] == "scaffold"
            if neighbour in previous or (record_id in chosen) == forward:
                continue
            previous[neighbour] = (node, record_id)
            queue.append(neighbour)
    return []


def _repair_quota_shortfall(
    ordered: pd.DataFrame,
    selected: list[int],
    fixed_pins: set[int],
    constraints: SelectionConstraints,
) -> list[int]:
    """Augment only a shortfall; maximize count under the two upper quotas.

    Existing scaffold coverage is preserved. This is not a weighted optimum or
    a solver for a still-unsatisfied minimum scaffold count at the final N.
    """
    target = constraints.target_count or len(ordered)
    if (
        len(selected) >= target
        or constraints.max_per_scaffold is None
        or constraints.max_per_cluster is None
    ):
        return selected
    labels = ordered.reindex(columns=["murcko_scaffold", "cluster_id"])
    graph: dict[_Node, list[tuple[_Node, int]]] = {}
    endpoints: dict[int, tuple[_Node, _Node]] = {}
    for record_id, raw_scaffold, raw_cluster in labels.itertuples(index=True, name=None):
        scaffold = None if pd.isna(raw_scaffold) else str(raw_scaffold)
        cluster = None if pd.isna(raw_cluster) else int(raw_cluster)
        left, right = ("scaffold", scaffold), ("cluster", cluster)
        endpoints[record_id] = (left, right)
        if record_id not in fixed_pins:
            graph.setdefault(left, []).append((right, record_id))
            graph.setdefault(right, []).append((left, record_id))
    chosen = set(selected)
    while len(chosen) < target:
        usage: dict[_Node, int] = {}
        for record_id in chosen:
            for node in endpoints[record_id]:
                usage[node] = usage.get(node, 0) + 1
        path = _augmenting_path(graph, chosen, usage, constraints)
        if not path:
            break
        chosen = chosen.symmetric_difference(path)
    if chosen == set(selected):
        return selected
    automatic_chosen = chosen - fixed_pins
    return [record_id for record_id in selected if record_id in fixed_pins] + [
        int(record_id) for record_id in ordered.index if record_id in automatic_chosen
    ]


def _explain_repaired_selection(
    ordered: pd.DataFrame,
    selected: list[int],
    original: set[int],
    fixed_pins: set[int],
    constraints: SelectionConstraints,
    explain_rejections: bool,
) -> tuple[dict, dict, dict, dict]:
    reasons: dict[int, list[str]] = {}
    rejections: dict[int, list[str]] = {}
    scaffold_usage: dict[str, int] = {}
    cluster_usage: dict[int, int] = {}
    for record_id in selected:
        row = ordered.loc[record_id]
        scaffold, cluster = _scaffold_of(row), _cluster_of(row)
        reasons[record_id] = (
            ["pinned molecule"]
            if record_id in fixed_pins
            else _selection_reasons(row, scaffold, cluster, scaffold_usage, cluster_usage)
        )
        if record_id not in original:
            reasons[record_id].append("quota repair to reach requested count")
        _consume(scaffold, cluster, scaffold_usage, cluster_usage)
    if explain_rejections:
        for values in ordered.itertuples(index=True, name=None):
            record_id = int(values[0])
            if record_id in reasons:
                continue
            row = dict(zip(ordered.columns, values[1:], strict=True))
            rejections[record_id] = _quota_block(
                _scaffold_of(row), _cluster_of(row), scaffold_usage, cluster_usage, constraints
            ) or [LIMIT_REACHED]
    return reasons, rejections, scaffold_usage, cluster_usage


def select(
    candidates: pd.DataFrame,
    constraints: SelectionConstraints = SelectionConstraints(),
    strategy: Strategy = Strategy.BALANCED,
    pinned_ids: Sequence[int] = (),
    excluded_ids: Sequence[int] = (),
    *,
    explain_rejections: bool = True,
) -> SelectionOutcome:
    """Compose the final set.

    ``candidates`` is indexed by ``record_id`` and may carry ``pareto_rank``,
    ``crowding_distance``, ``robustness_score``, ``qed``, ``murcko_scaffold``
    and ``cluster_id``. Whatever is present is used; whatever is missing is
    skipped rather than faked.

    Strategy keys are compared lexicographically, with missing values last;
    ascending record_id breaks ties. An upstream ``selection_priority`` column
    replaces property-ranking keys (smaller is better). Scaffold coverage prioritizes
    rare cores, then property ranking, reserving one member of each feasible core
    before ranked filling. Exclusions precede sorted fixed pins and greedy scaffold
    reservation. Empty Murcko SMILES form one acyclic group. When crossed upper
    quotas leave a count shortfall, deterministic augmenting paths maximize the
    cardinality up to the target without moving pins or losing covered scaffolds.
    This does not globally optimize ranking quality or minimum scaffold coverage.

    Set ``explain_rejections=False`` for large-library scenarios: selection and
    selected explanations are unchanged, but rejected IDs are not retained and
    scanning stops when the requested count is reached. Sorting still requires
    memory proportional to the candidate count, not pairwise comparisons.
    """
    if not candidates.index.is_unique or (
        len(candidates) and candidates.index.inferred_type != "integer"
    ):
        raise ValueError("candidate record_id values must be unique integers")
    reasons: dict[int, list[str]] = {}
    rejections: dict[int, list[str]] = {}
    scaffold_usage: dict[str, int] = {}
    cluster_usage: dict[int, int] = {}
    selected: list[int] = []

    # Only carry columns used here; SMILES and unrelated descriptors can be large.
    needed = {
        "murcko_scaffold",
        "cluster_id",
        "pareto_rank",
        "robustness_score",
        "selection_priority",
    }
    needed.update(column for column, _ in _STRATEGY_KEYS[strategy])
    pool = candidates.loc[:, [column for column in candidates.columns if column in needed]]
    if len(excluded_ids):
        pool = pool.loc[~pool.index.isin(set(excluded_ids))]
    if strategy is Strategy.SCAFFOLD_COVERAGE and "murcko_scaffold" in pool:
        pool = pool.assign(
            scaffold_size=pool.murcko_scaffold.map(pool.murcko_scaffold.value_counts(dropna=True))
        )

    if constraints.preserve_pinned:
        for raw_id in sorted(set(pinned_ids)):
            record_id = int(raw_id)
            if record_id not in pool.index or record_id in reasons:
                continue
            row = pool.loc[record_id]
            selected.append(record_id)
            reasons[record_id] = ["pinned molecule"]
            _consume(_scaffold_of(row), _cluster_of(row), scaffold_usage, cluster_usage)

    ordered = order_candidates(pool, strategy)
    scaffold_position = (
        ordered.columns.get_loc("murcko_scaffold") + 1 if ("murcko_scaffold" in ordered) else None
    )
    cluster_position = (
        ordered.columns.get_loc("cluster_id") + 1 if ("cluster_id" in ordered) else None
    )
    # Reserve scaffold coverage before ranked filling; pins have precedence.
    coverage_target = constraints.min_scaffolds or 0
    if strategy is Strategy.SCAFFOLD_COVERAGE and "murcko_scaffold" in pool:
        coverage_target = max(coverage_target, pool.murcko_scaffold.nunique(dropna=True))
    for coverage_first in (True, False) if coverage_target else (False,):
        for values in ordered.itertuples(index=True, name=None):
            if coverage_first and len(scaffold_usage) >= coverage_target:
                break
            record_id = int(values[0])
            if record_id in reasons:
                continue
            if constraints.target_count is not None and len(selected) >= constraints.target_count:
                if not explain_rejections:
                    break
                rejections[record_id] = [LIMIT_REACHED]
                continue

            raw_scaffold = values[scaffold_position] if scaffold_position is not None else None
            scaffold = None if raw_scaffold is None or pd.isna(raw_scaffold) else str(raw_scaffold)
            if coverage_first and (scaffold is None or scaffold in scaffold_usage):
                continue
            raw_cluster = values[cluster_position] if cluster_position is not None else None
            cluster = None if raw_cluster is None or pd.isna(raw_cluster) else int(raw_cluster)
            blocked_by = _quota_block(scaffold, cluster, scaffold_usage, cluster_usage, constraints)
            if blocked_by:
                if explain_rejections:
                    rejections[record_id] = blocked_by
                continue

            selected.append(record_id)
            reasons[record_id] = _selection_reasons(
                dict(zip(ordered.columns, values[1:], strict=True)),
                scaffold,
                cluster,
                scaffold_usage,
                cluster_usage,
            )
            _consume(scaffold, cluster, scaffold_usage, cluster_usage)

    fixed_pins = set(pinned_ids).intersection(selected) if constraints.preserve_pinned else set()
    repaired = _repair_quota_shortfall(ordered, selected, fixed_pins, constraints)
    if repaired != selected:
        reasons, rejections, scaffold_usage, cluster_usage = _explain_repaired_selection(
            ordered, repaired, set(selected), fixed_pins, constraints, explain_rejections
        )
        selected = repaired

    return SelectionOutcome(
        selected_ids=tuple(selected),
        reasons=reasons,
        rejections=rejections,
        scaffold_usage=scaffold_usage,
        cluster_usage=cluster_usage,
        strategy=strategy,
    )
