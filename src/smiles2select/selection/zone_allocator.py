"""Deterministic allocation across overlapping chemical-space zones."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import pandas as pd

from smiles2select.chemical_space.zones import SelectionZone, membership_table


@dataclass(frozen=True)
class ZoneAllocationResult:
    """Final/reserve allocation plus every shortfall and assignment."""

    final_ids: tuple[object, ...]
    reserve_ids: tuple[object, ...]
    assignments: Mapping[object, str]
    allocation_table: pd.DataFrame
    memberships: pd.DataFrame
    unfulfilled_quotas: Mapping[str, int]

    @property
    def shortfall(self) -> int:
        return sum(self.unfulfilled_quotas.values())


def _rank_key(record_id: object, ranking: Mapping[object, float] | Callable[[object], float] | None):
    if ranking is None:
        return (0.0, str(record_id))
    score = ranking(record_id) if callable(ranking) else ranking.get(record_id, float("-inf"))
    try:
        numeric = float(score)
    except (TypeError, ValueError):
        numeric = float("-inf")
    return (-numeric, str(record_id))


def _ordered(ids: Iterable[object], ranking) -> list[object]:
    return sorted(ids, key=lambda record_id: _rank_key(record_id, ranking))


def _reserve_targets(zones: list[SelectionZone], reserve_count: int) -> dict[str, int]:
    quotas = {zone.zone_id: max(zone.quota or 0, 0) for zone in zones}
    total = sum(quotas.values())
    if reserve_count <= 0 or total == 0:
        return {zone.zone_id: 0 for zone in zones}
    raw = {zone_id: reserve_count * quota / total for zone_id, quota in quotas.items()}
    targets = {zone_id: int(value) for zone_id, value in raw.items()}
    remainder = reserve_count - sum(targets.values())
    order = sorted(quotas, key=lambda zone_id: (-(raw[zone_id] - targets[zone_id]), zone_id))
    for zone_id in order[:remainder]:
        targets[zone_id] += 1
    return targets


def allocate_zones(
    candidates: pd.DataFrame | Iterable[object],
    zones: Iterable[SelectionZone],
    *,
    final_count: int,
    reserve_count: int = 0,
    ranking: Mapping[object, float] | Callable[[object], float] | None = None,
) -> ZoneAllocationResult:
    """Allocate each candidate at most once and report quota shortfalls.

    Zones are processed by explicit priority, then by restrictive population,
    then by ID.  That ordering is deterministic and prevents an overlapping
    broad zone from consuming molecules needed by a rarer zone.
    """

    if final_count < 0 or reserve_count < 0:
        raise ValueError("final_count and reserve_count must not be negative")
    record_ids = list(candidates.index) if isinstance(candidates, pd.DataFrame) else list(candidates)
    zone_list = sorted(zones, key=lambda zone: (-zone.priority, zone.size, zone.zone_id))
    available = set(record_ids)
    final_ids: list[object] = []
    reserve_ids: list[object] = []
    assignments: dict[object, str] = {}
    rows: list[dict[str, object]] = []
    unfulfilled: dict[str, int] = {}

    for zone in zone_list:
        target = min(zone.quota if zone.quota is not None else 0, final_count)
        eligible = _ordered(available.intersection(zone.members), ranking)
        chosen = eligible[:target]
        final_ids.extend(chosen)
        available.difference_update(chosen)
        for record_id in chosen:
            assignments[record_id] = zone.zone_id
        shortfall = target - len(chosen)
        if shortfall:
            unfulfilled[zone.zone_id] = shortfall
        rows.append(
            {
                "zone_id": zone.zone_id,
                "zone_name": zone.name,
                "available": zone.size,
                "requested_final": target,
                "allocated_final": len(chosen),
                "requested_reserve": 0,
                "allocated_reserve": 0,
                "shortfall": shortfall,
            }
        )

    remaining_final = max(0, final_count - len(final_ids))
    fillers = _ordered(available, ranking)[:remaining_final]
    final_ids.extend(fillers)
    available.difference_update(fillers)

    reserve_targets = _reserve_targets(zone_list, reserve_count)
    row_by_zone = {row["zone_id"]: row for row in rows}
    for zone in zone_list:
        target = reserve_targets.get(zone.zone_id, 0)
        eligible = _ordered(available.intersection(zone.members), ranking)
        chosen = eligible[:target]
        reserve_ids.extend(chosen)
        available.difference_update(chosen)
        for record_id in chosen:
            assignments[record_id] = f"{zone.zone_id}:reserve"
        row = row_by_zone[zone.zone_id]
        row["requested_reserve"] = target
        row["allocated_reserve"] = len(chosen)

    remaining_reserve = max(0, reserve_count - len(reserve_ids))
    reserve_fillers = _ordered(available, ranking)[:remaining_reserve]
    reserve_ids.extend(reserve_fillers)
    available.difference_update(reserve_fillers)

    memberships = membership_table(zone_list, record_ids)
    allocation = pd.DataFrame(
        rows,
        columns=[
            "zone_id",
            "zone_name",
            "available",
            "requested_final",
            "allocated_final",
            "requested_reserve",
            "allocated_reserve",
            "shortfall",
        ],
    )
    return ZoneAllocationResult(
        final_ids=tuple(final_ids[:final_count]),
        reserve_ids=tuple(reserve_ids[:reserve_count]),
        assignments=assignments,
        allocation_table=allocation,
        memberships=memberships,
        unfulfilled_quotas=unfulfilled,
    )
