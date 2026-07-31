"""Pareto ranking of a candidate table.

Joins the objectives to the dominance code and returns a table indexed like the
input, ready for the ``pareto_results`` table and the Pareto Explorer.

Results are cached by the hash of (objectives, data), so an interaction that
does not touch the objectives never recomputes the ranking.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from smiles2select.selection_intelligence.objectives import ObjectiveSet
from smiles2select.selection_intelligence.pareto import (
    crowding_distance,
    distance_to_ideal,
    non_dominated_sort,
)

RESULT_COLUMNS = (
    "pareto_rank",
    "domination_count",
    "dominated_count",
    "distance_to_ideal",
    "crowding_distance",
)

#: A first front larger than this share of the candidates stops being a
#: priority list and becomes a restatement of the input.
CROWDED_FRONT_FRACTION = 0.25


@dataclass(frozen=True)
class ParetoResult:
    """Ranking plus the warnings a user needs before trusting it."""

    table: pd.DataFrame
    objective_fields: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    cache_key: str = ""
    objective_values: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def front_sizes(self) -> dict[int, int]:
        counts = self.table["pareto_rank"].value_counts().sort_index()
        return {int(rank): int(size) for rank, size in counts.items()}

    def front(self, rank: int = 1) -> pd.Index:
        return self.table.index[self.table["pareto_rank"] == rank]

    def up_to_front(self, max_front: int) -> pd.Index:
        return self.table.index[self.table["pareto_rank"] <= max_front]

    def rows_for_storage(self) -> list[dict[str, object]]:
        """Rows for ``pareto_results``; crowding stays in memory only."""
        return [
            {
                "record_id": int(record_id),
                "pareto_rank": int(row.pareto_rank),
                "domination_count": int(row.domination_count),
                "dominated_count": int(row.dominated_count),
                "distance_to_ideal": float(row.distance_to_ideal),
            }
            for record_id, row in self.table.iterrows()
        ]


def cache_key(objectives: ObjectiveSet, frame: pd.DataFrame) -> str:
    """Hash of the objectives and the exact values they will read."""
    digest = hashlib.sha256()
    digest.update(json.dumps(objectives.as_dicts(), sort_keys=True, default=str).encode("utf-8"))
    fields = objectives.fields()
    if fields:
        values = frame[list(fields)].to_numpy(dtype=float, na_value=np.nan)
        digest.update(np.ascontiguousarray(values).tobytes())
    digest.update(str(list(frame.index)).encode("utf-8"))
    return digest.hexdigest()


def rank_candidates(frame: pd.DataFrame, objectives: ObjectiveSet, key: str = "") -> ParetoResult:
    """Compute Pareto ranks, distances and warnings for one candidate table."""
    matrix = objectives.matrix(frame)
    dominance = non_dominated_sort(matrix)

    table = pd.DataFrame(
        {
            "pareto_rank": dominance.ranks,
            "domination_count": dominance.domination_count,
            "dominated_count": dominance.dominated_count,
            "distance_to_ideal": distance_to_ideal(matrix),
            "crowding_distance": crowding_distance(matrix, dominance.ranks),
        },
        index=frame.index,
    )

    values = pd.DataFrame(
        matrix,
        index=frame.index,
        columns=[f"objective::{name}" for name in objectives.fields()],
    )

    return ParetoResult(
        table=table,
        objective_fields=objectives.fields(),
        warnings=tuple(_warnings(objectives, frame, table)),
        cache_key=key or cache_key(objectives, frame),
        objective_values=values,
    )


class ParetoRanker:
    """Ranks candidates, reusing the previous result when nothing changed."""

    def __init__(self, cache_size: int = 4) -> None:
        self._cache: dict[str, ParetoResult] = {}
        self._order: list[str] = []
        self._cache_size = cache_size

    def rank(self, frame: pd.DataFrame, objectives: ObjectiveSet) -> ParetoResult:
        key = cache_key(objectives, frame)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        result = rank_candidates(frame, objectives, key)
        self._cache[key] = result
        self._order.append(key)
        while len(self._order) > self._cache_size:
            self._cache.pop(self._order.pop(0), None)
        return result

    def clear(self) -> None:
        self._cache.clear()
        self._order.clear()


def _warnings(objectives: ObjectiveSet, frame: pd.DataFrame, table: pd.DataFrame) -> Sequence[str]:
    """Everything that would make this ranking misleading."""
    messages = list(objectives.diagnostics(frame))
    total = len(table)
    if total:
        share = int((table["pareto_rank"] == 1).sum()) / total
        if share > CROWDED_FRONT_FRACTION:
            messages.append(
                f"A primeira fronteira contém {share * 100:.0f}% das moléculas. "
                "Considere reduzir ou agrupar os objetivos para tornar a priorização "
                "mais informativa."
            )
    return messages


def order_within_front(result: ParetoResult, objectives: ObjectiveSet) -> pd.Index:
    """Candidates ordered for selection: front, then crowding, then distance.

    Weights enter only here, as the final tie-break, never in the dominance
    relation itself.
    """
    table = result.table.copy()
    weights = objectives.weights()
    if result.objective_values.empty or weights.size == 0:
        table["weighted"] = 0.0
    else:
        table["weighted"] = (result.objective_values.to_numpy() * weights).sum(axis=1)

    ordered = table.sort_values(
        by=["pareto_rank", "crowding_distance", "distance_to_ideal", "weighted"],
        ascending=[True, False, True, False],
        kind="stable",
    )
    return ordered.index
