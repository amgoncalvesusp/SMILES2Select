"""Selection-zone definitions and auditable memberships."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from smiles2select.decision.expression_parser import evaluate as evaluate_expression


@dataclass(frozen=True)
class SelectionZone:
    """A named set of record IDs with optional final quota and priority."""

    zone_id: str
    name: str
    members: frozenset[object]
    kind: str = "manual"
    definition: str = ""
    priority: int = 0
    quota: int | None = None

    def __post_init__(self) -> None:
        if not self.zone_id.strip():
            raise ValueError("zone_id must not be empty")
        if not self.name.strip():
            raise ValueError("zone name must not be empty")
        if self.quota is not None and self.quota < 0:
            raise ValueError("zone quota must not be negative")

    @property
    def size(self) -> int:
        return len(self.members)

    def contains(self, record_id: object) -> bool:
        return record_id in self.members

    def as_dict(self) -> dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "members": list(self.members),
            "kind": self.kind,
            "definition": self.definition,
            "priority": self.priority,
            "quota": self.quota,
        }


def zone_from_mask(
    frame: pd.DataFrame,
    mask: pd.Series,
    *,
    zone_id: str,
    name: str,
    kind: str = "rule",
    definition: str = "",
    priority: int = 0,
    quota: int | None = None,
) -> SelectionZone:
    """Create a zone from a boolean mask without changing the source frame."""

    if not mask.index.equals(frame.index):
        mask = mask.reindex(frame.index, fill_value=False)
    members = frozenset(frame.index[mask.fillna(False).astype(bool)].tolist())
    return SelectionZone(zone_id, name, members, kind, definition, priority, quota)


def zone_from_expression(
    frame: pd.DataFrame,
    expression: str,
    *,
    zone_id: str,
    name: str,
    priority: int = 0,
    quota: int | None = None,
) -> SelectionZone:
    """Evaluate the same safe AND/OR/NOT expression language as the CLI."""

    mask = evaluate_expression(expression, frame)
    return zone_from_mask(
        frame,
        mask,
        zone_id=zone_id,
        name=name,
        kind="expression",
        definition=expression,
        priority=priority,
        quota=quota,
    )


def membership_table(zones: Iterable[SelectionZone], record_ids: Iterable[object]) -> pd.DataFrame:
    """Long-form table retaining every overlapping membership."""

    requested = list(record_ids)
    rows: list[dict[str, object]] = []
    for zone in zones:
        for record_id in requested:
            if record_id in zone.members:
                rows.append(
                    {
                        "record_id": record_id,
                        "zone_id": zone.zone_id,
                        "zone_name": zone.name,
                        "zone_kind": zone.kind,
                        "zone_priority": zone.priority,
                        "zone_quota": zone.quota,
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "record_id",
            "zone_id",
            "zone_name",
            "zone_kind",
            "zone_priority",
            "zone_quota",
        ],
    )


def zone_statistics(zones: Iterable[SelectionZone], record_ids: Iterable[object]) -> pd.DataFrame:
    """Counts and overlap status for the zone inspector."""

    zone_list = list(zones)
    ids = list(record_ids)
    membership = membership_table(zone_list, ids)
    counts = membership.groupby("record_id").size() if not membership.empty else pd.Series(dtype=int)
    rows = []
    for zone in zone_list:
        rows.append(
            {
                "zone_id": zone.zone_id,
                "name": zone.name,
                "kind": zone.kind,
                "available": zone.size,
                "quota": zone.quota,
                "priority": zone.priority,
                "overlapping_members": int(
                    sum(counts.get(record_id, 0) > 1 for record_id in zone.members)
                ),
            }
        )
    return pd.DataFrame(rows)
