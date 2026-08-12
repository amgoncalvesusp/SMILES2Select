"""The selection basket.

Holds the current decision for every candidate and records how it got there.
All mutations go through :meth:`SelectionBasket.apply`, which is what keeps the
history complete enough for undo, redo and session restore.

Two invariants the class enforces rather than trusting callers to respect:

* a pinned molecule is never dropped by an automatic selection;
* keeping a molecule the rules rejected requires a written justification.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from smiles2select.selection_intelligence.action_log import ActionLog, ActionType, SelectionAction
from smiles2select.selection_intelligence.states import (
    ChemicalStatus,
    MoleculeState,
    SelectionOrigin,
    SelectionStatus,
)


class JustificationRequired(ValueError):
    """Raised when a human decision contradicts the chemistry without a reason."""


@dataclass(frozen=True)
class BasketCounters:
    """The numbers shown permanently above the basket."""

    candidates: int
    shortlisted: int
    final_selected: int
    reserve: int
    pinned: int
    manually_excluded: int
    target: int | None = None

    @property
    def remaining(self) -> int | None:
        if self.target is None:
            return None
        return max(0, self.target - self.final_selected)

    def as_rows(self) -> list[tuple[str, object]]:
        return [
            ("Candidates", self.candidates),
            ("Shortlist", self.shortlisted),
            (
                "Final selected",
                f"{self.final_selected} / {self.target}" if self.target else self.final_selected,
            ),
            ("Pinned", self.pinned),
            ("Reserve", self.reserve),
            ("Manually excluded", self.manually_excluded),
        ]


class SelectionBasket:
    """Current selection state plus its history."""

    def __init__(
        self,
        states: Iterable[MoleculeState] = (),
        log: ActionLog | None = None,
        target_count: int | None = None,
    ) -> None:
        self._states: dict[int, MoleculeState] = {state.record_id: state for state in states}
        self._log = log or ActionLog()
        self.target_count = target_count

    # -- reading ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._states)

    def __contains__(self, record_id: object) -> bool:
        return record_id in self._states

    @property
    def log(self) -> ActionLog:
        return self._log

    def state(self, record_id: int) -> MoleculeState:
        return self._states[record_id]

    def states(self) -> tuple[MoleculeState, ...]:
        return tuple(self._states[key] for key in sorted(self._states))

    def ids_with_status(self, status: SelectionStatus) -> tuple[int, ...]:
        return tuple(
            sorted(key for key, state in self._states.items() if state.selection_status is status)
        )

    def final_ids(self) -> tuple[int, ...]:
        return self.ids_with_status(SelectionStatus.FINAL_SELECTED)

    def pinned_ids(self) -> tuple[int, ...]:
        return tuple(sorted(key for key, state in self._states.items() if state.pinned))

    def excluded_ids(self) -> tuple[int, ...]:
        return self.ids_with_status(SelectionStatus.MANUALLY_EXCLUDED)

    def overrides(self) -> tuple[MoleculeState, ...]:
        """Selections that went against the chemical verdict."""
        return tuple(state for state in self.states() if state.contradicts_chemistry)

    def counters(self) -> BasketCounters:
        return BasketCounters(
            candidates=len(self._states),
            shortlisted=len(self.ids_with_status(SelectionStatus.SHORTLISTED)),
            final_selected=len(self.final_ids()),
            reserve=len(self.ids_with_status(SelectionStatus.RESERVE)),
            pinned=len(self.pinned_ids()),
            manually_excluded=len(self.excluded_ids()),
            target=self.target_count,
        )

    def preview(self, record_ids: Sequence[int]) -> dict[str, int]:
        """What a bulk action would do, shown before it is confirmed."""
        known = [key for key in record_ids if key in self._states]
        return {
            "requested": len(record_ids),
            "unknown": len(record_ids) - len(known),
            "already selected": sum(1 for key in known if self._states[key].is_selected),
            "manually excluded": sum(
                1
                for key in known
                if self._states[key].selection_status is SelectionStatus.MANUALLY_EXCLUDED
            ),
            "pinned": sum(1 for key in known if self._states[key].pinned),
        }

    # -- writing ------------------------------------------------------------

    def apply(
        self,
        action_type: ActionType,
        record_ids: Sequence[int],
        *,
        status: SelectionStatus | None = None,
        origin: SelectionOrigin = SelectionOrigin.MANUAL,
        pinned: bool | None = None,
        note: str | None = None,
        source: str = "",
        reason: str = "",
    ) -> SelectionAction:
        """Apply a decision to several molecules and record it.

        Unknown ids are ignored rather than created: the basket only decides
        about molecules the run actually produced.
        """
        targets = [key for key in dict.fromkeys(record_ids) if key in self._states]
        previous: dict[int, dict[str, Any]] = {}
        updated: dict[int, dict[str, Any]] = {}

        for record_id in targets:
            before = self._states[record_id]
            after = before
            if status is not None:
                self._require_justification(before, status, origin, reason)
                after = after.with_selection(status, origin, note or "")
            elif note is not None:
                after = replace(after, note=note)
            if pinned is not None:
                after = after.with_pin(pinned)
            if after == before:
                continue
            previous[record_id] = before.as_row()
            updated[record_id] = after.as_row()
            self._states[record_id] = after

        return self._log.record(
            SelectionAction(
                action_type=action_type,
                record_ids=tuple(targets),
                source=source,
                reason=reason,
                previous_state=previous,
                new_state=updated,
            )
        )

    def add_to_shortlist(self, record_ids: Sequence[int], **kwargs: Any) -> SelectionAction:
        return self.apply(
            ActionType.ADD_TO_SHORTLIST, record_ids, status=SelectionStatus.SHORTLISTED, **kwargs
        )

    def add_to_final(self, record_ids: Sequence[int], **kwargs: Any) -> SelectionAction:
        return self.apply(
            ActionType.ADD_TO_FINAL, record_ids, status=SelectionStatus.FINAL_SELECTED, **kwargs
        )

    def remove_from_final(self, record_ids: Sequence[int], **kwargs: Any) -> SelectionAction:
        return self.apply(
            ActionType.REMOVE_FROM_FINAL, record_ids, status=SelectionStatus.UNDECIDED, **kwargs
        )

    def exclude(self, record_ids: Sequence[int], **kwargs: Any) -> SelectionAction:
        return self.apply(
            ActionType.EXCLUDE, record_ids, status=SelectionStatus.MANUALLY_EXCLUDED, **kwargs
        )

    def pin(self, record_ids: Sequence[int], **kwargs: Any) -> SelectionAction:
        return self.apply(ActionType.PIN, record_ids, pinned=True, **kwargs)

    def unpin(self, record_ids: Sequence[int], **kwargs: Any) -> SelectionAction:
        return self.apply(ActionType.UNPIN, record_ids, pinned=False, **kwargs)

    def annotate(self, record_ids: Sequence[int], note: str, **kwargs: Any) -> SelectionAction:
        return self.apply(ActionType.ANNOTATE, record_ids, note=note, **kwargs)

    # -- undo / redo --------------------------------------------------------

    def undo(self) -> SelectionAction | None:
        """Restore the state each molecule had before the last applied action."""
        action = self._log.undo()
        if action is None:
            return None
        for record_id, row in action.previous_state.items():
            self._states[record_id] = _state_from_row(row)
        return action

    def redo(self) -> SelectionAction | None:
        action = self._log.redo()
        if action is None:
            return None
        for record_id, row in action.new_state.items():
            self._states[record_id] = _state_from_row(row)
        return action

    # -- internals ----------------------------------------------------------

    def _require_justification(
        self,
        state: MoleculeState,
        status: SelectionStatus,
        origin: SelectionOrigin,
        reason: str,
    ) -> None:
        """Selecting a chemically rejected molecule must be justified in writing."""
        if status is not SelectionStatus.FINAL_SELECTED:
            return
        if state.chemical_status.passed or reason.strip():
            return
        raise JustificationRequired(
            f"molecule {state.record_id} is {state.chemical_status.value}; "
            f"selecting it ({origin.value}) requires a justification"
        )


def _state_from_row(row: Mapping[str, Any]) -> MoleculeState:
    return MoleculeState(
        record_id=int(row["record_id"]),
        chemical_status=ChemicalStatus(row["chemical_status"]),
        selection_status=SelectionStatus(row["selection_status"]),
        origin=SelectionOrigin(row["selection_origin"]) if row.get("selection_origin") else None,
        pinned=bool(row.get("pinned", 0)),
        note=row.get("manual_note") or "",
    )


def basket_from_rows(
    rows: Sequence[Mapping[str, Any]], log: ActionLog | None = None
) -> SelectionBasket:
    """Rebuild a basket from stored ``selection_state`` rows."""
    return SelectionBasket((_state_from_row(row) for row in rows), log=log)
