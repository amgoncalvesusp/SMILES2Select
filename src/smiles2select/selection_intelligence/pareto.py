"""Pareto dominance.

Works on a plain "higher is better" matrix, so it knows nothing about
molecules, columns or directions - that translation happened in
:mod:`smiles2select.selection_intelligence.objectives`.

A dominates B when A is at least as good on every objective and strictly better
on at least one. Ties therefore do not dominate each other: two identical
molecules land on the same front, which is the correct answer and not a bug to
be broken arbitrarily.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Above this, the pairwise pass is split into blocks to bound peak memory.
_BLOCK = 2048


@dataclass(frozen=True)
class DominanceResult:
    """Front assignment plus the counts behind it."""

    ranks: np.ndarray
    domination_count: np.ndarray
    dominated_count: np.ndarray

    @property
    def front_sizes(self) -> dict[int, int]:
        values, counts = np.unique(self.ranks, return_counts=True)
        return {int(value): int(count) for value, count in zip(values, counts, strict=True)}

    def front(self, rank: int) -> np.ndarray:
        return np.flatnonzero(self.ranks == rank)


def dominance_counts(matrix: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """For each row: how many rows dominate it, and which rows it dominates.

    Computed in blocks: the full n x n boolean comparison would need n^2 bytes,
    which stops being reasonable well before the library sizes this tool
    targets.
    """
    count = matrix.shape[0]
    dominated_by = np.zeros(count, dtype=np.int64)
    dominates: list[list[int]] = [[] for _ in range(count)]

    for start in range(0, count, _BLOCK):
        stop = min(start + _BLOCK, count)
        block = matrix[start:stop][:, None, :]  # (b, 1, m)
        others = matrix[None, :, :]  # (1, n, m)

        at_least_as_good = np.all(block <= others, axis=2)
        strictly_better = np.any(block < others, axis=2)
        dominated = at_least_as_good & strictly_better  # others dominate block rows

        dominated_by[start:stop] = dominated.sum(axis=1)
        for offset in range(stop - start):
            for dominator in np.flatnonzero(dominated[offset]):
                dominates[int(dominator)].append(start + offset)

    return dominated_by, [np.array(items, dtype=np.int64) for items in dominates]


def non_dominated_sort(matrix: np.ndarray) -> DominanceResult:
    """Assign every row to a Pareto front, 1 being non-dominated.

    Standard fast non-dominated sort: peel off the current front, decrement the
    counters of everything it dominated, repeat.
    """
    count = matrix.shape[0]
    if count == 0:
        empty = np.zeros(0, dtype=np.int64)
        return DominanceResult(empty, empty, empty)

    dominated_by, dominates = dominance_counts(matrix)
    remaining = dominated_by.copy()
    ranks = np.zeros(count, dtype=np.int64)

    current = np.flatnonzero(remaining == 0)
    front_number = 1
    while current.size:
        ranks[current] = front_number
        following: list[int] = []
        for index in current:
            for target in dominates[int(index)]:
                remaining[target] -= 1
                if remaining[target] == 0:
                    following.append(int(target))
        current = np.array(sorted(set(following)), dtype=np.int64)
        front_number += 1

    return DominanceResult(
        ranks=ranks,
        domination_count=dominated_by,
        dominated_count=np.array([len(items) for items in dominates], dtype=np.int64),
    )


def normalise(matrix: np.ndarray) -> np.ndarray:
    """Scale each objective to [0, 1]; a constant objective becomes all ones."""
    lowest = matrix.min(axis=0)
    highest = matrix.max(axis=0)
    spread = highest - lowest
    scaled = np.ones_like(matrix, dtype=float)
    varying = spread > 0
    scaled[:, varying] = (matrix[:, varying] - lowest[varying]) / spread[varying]
    return scaled


def distance_to_ideal(matrix: np.ndarray) -> np.ndarray:
    """Euclidean distance to the ideal point, on normalised objectives.

    The ideal point is the best observed value of every objective at once -
    usually achieved by no molecule. The distance is a tie-breaker inside a
    front, not a ranking that overrides dominance.
    """
    if matrix.size == 0:
        return np.zeros(0, dtype=float)
    scaled = normalise(matrix)
    return np.sqrt(((1.0 - scaled) ** 2).sum(axis=1))


def crowding_distance(matrix: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    """NSGA-II crowding distance, computed inside each front.

    Larger means more isolated. Used to prefer molecules from sparse regions of
    a front, so a front is not represented by a cluster of near-identical
    compounds.
    """
    count, objectives = matrix.shape
    distances = np.zeros(count, dtype=float)
    if count == 0:
        return distances

    scaled = normalise(matrix)
    for rank in np.unique(ranks):
        members = np.flatnonzero(ranks == rank)
        if members.size <= 2:
            distances[members] = np.inf
            continue
        for objective in range(objectives):
            order = members[np.argsort(scaled[members, objective], kind="stable")]
            distances[order[0]] = np.inf
            distances[order[-1]] = np.inf
            values = scaled[order, objective]
            spread = values[-1] - values[0]
            if spread <= 0:
                continue
            distances[order[1:-1]] += (values[2:] - values[:-2]) / spread
    return distances
