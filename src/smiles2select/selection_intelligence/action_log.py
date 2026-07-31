"""Action history.

Every decision is an event carrying what changed and what it replaced. That is
what makes undo possible, and what lets a finished selection be explained months
later: which action, from which view, and why.

Timestamps are ISO 8601 in UTC (``2026-07-31T09:42:18+00:00``). A run may be
shared between people in different time zones, so a local clock reading would
make the order of events ambiguous.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ActionType(str, Enum):
    """Every action the basket can record."""

    ADD_TO_SHORTLIST = "ADD_TO_SHORTLIST"
    ADD_TO_FINAL = "ADD_TO_FINAL"
    REMOVE_FROM_FINAL = "REMOVE_FROM_FINAL"
    EXCLUDE = "EXCLUDE"
    PIN = "PIN"
    UNPIN = "UNPIN"
    ANNOTATE = "ANNOTATE"
    RESET = "RESET"
    APPLY_RECIPE = "APPLY_RECIPE"
    AUTOMATIC_SELECTION = "AUTOMATIC_SELECTION"


def utc_timestamp() -> str:
    """Current instant, ISO 8601 with an explicit UTC offset."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class SelectionAction:
    """One recorded decision, with enough state to undo it."""

    action_type: ActionType
    record_ids: tuple[int, ...]
    source: str = ""
    reason: str = ""
    timestamp: str = field(default_factory=utc_timestamp)
    previous_state: dict[int, dict[str, Any]] = field(default_factory=dict)
    new_state: dict[int, dict[str, Any]] = field(default_factory=dict)

    def as_row(self) -> dict[str, object]:
        """Row for the ``selection_actions`` table."""
        return {
            "timestamp": self.timestamp,
            "action_type": self.action_type.value,
            "payload_json": json.dumps(
                {
                    "record_ids": list(self.record_ids),
                    "source": self.source,
                    "reason": self.reason,
                },
                ensure_ascii=False,
            ),
            "previous_state_json": json.dumps(self.previous_state, ensure_ascii=False),
            "new_state_json": json.dumps(self.new_state, ensure_ascii=False),
        }

    def describe(self) -> str:
        parts = [self.timestamp, self.action_type.value, f"{len(self.record_ids)} molécula(s)"]
        if self.source:
            parts.append(f"origem: {self.source}")
        if self.reason:
            parts.append(f"motivo: {self.reason}")
        return " | ".join(parts)


class ActionLog:
    """Append-only history with undo/redo cursors.

    Undo does not delete anything: it moves a cursor, so a redo restores the
    same event rather than a reconstruction of it. A new action after an undo
    discards the redo tail, which is what every editor does and what users
    expect.
    """

    def __init__(self, actions: Iterable[SelectionAction] = ()) -> None:
        self._actions: list[SelectionAction] = list(actions)
        self._cursor = len(self._actions)

    def __len__(self) -> int:
        return len(self._actions)

    @property
    def applied(self) -> tuple[SelectionAction, ...]:
        """Actions currently in effect, oldest first."""
        return tuple(self._actions[: self._cursor])

    @property
    def can_undo(self) -> bool:
        return self._cursor > 0

    @property
    def can_redo(self) -> bool:
        return self._cursor < len(self._actions)

    def record(self, action: SelectionAction) -> SelectionAction:
        del self._actions[self._cursor :]  # a new action drops the redo tail
        self._actions.append(action)
        self._cursor = len(self._actions)
        return action

    def undo(self) -> SelectionAction | None:
        """Step back one action and return it, or None when there is nothing to undo."""
        if not self.can_undo:
            return None
        self._cursor -= 1
        return self._actions[self._cursor]

    def redo(self) -> SelectionAction | None:
        if not self.can_redo:
            return None
        action = self._actions[self._cursor]
        self._cursor += 1
        return action

    def history(self) -> list[dict[str, object]]:
        """Full history, marking which events are currently applied."""
        return [
            {**action.as_row(), "applied": index < self._cursor}
            for index, action in enumerate(self._actions)
        ]

    def rows(self) -> list[dict[str, object]]:
        return [action.as_row() for action in self._actions]


def from_rows(rows: Sequence[dict[str, Any]]) -> ActionLog:
    """Rebuild a log from stored rows (session restore)."""
    actions = []
    for row in rows:
        payload = json.loads(row["payload_json"])
        actions.append(
            SelectionAction(
                action_type=ActionType(row["action_type"]),
                record_ids=tuple(payload.get("record_ids", ())),
                source=payload.get("source", ""),
                reason=payload.get("reason", ""),
                timestamp=row["timestamp"],
                previous_state=_ints(json.loads(row.get("previous_state_json") or "{}")),
                new_state=_ints(json.loads(row.get("new_state_json") or "{}")),
            )
        )
    return ActionLog(actions)


def _ints(payload: dict[str, Any]) -> dict[int, Any]:
    """JSON object keys are strings; record ids are integers."""
    return {int(key): value for key, value in payload.items()}
