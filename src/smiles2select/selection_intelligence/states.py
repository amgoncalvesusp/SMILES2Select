"""Molecule states.

Two independent axes, deliberately never merged:

* **chemical status** - what the rules concluded, owned by the pipeline;
* **selection status** - what the human decided, owned by this layer.

A molecule can be ``AUTO_FAIL`` and ``FINAL_SELECTED`` at the same time. That
combination is not a contradiction to be resolved; it is a rescue that has to
stay visible, with its origin and its justification attached.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class ChemicalStatus(str, Enum):
    """Verdict produced by the rules; never changed by a human decision."""

    AUTO_PASS = "AUTO_PASS"
    AUTO_FAIL = "AUTO_FAIL"
    INVALID = "INVALID"
    WARNING = "WARNING"
    BORDERLINE = "BORDERLINE"

    @property
    def passed(self) -> bool:
        """WARNING and BORDERLINE passed the rules; they only carry a flag."""
        return self in {
            ChemicalStatus.AUTO_PASS,
            ChemicalStatus.WARNING,
            ChemicalStatus.BORDERLINE,
        }


class SelectionStatus(str, Enum):
    """Where the molecule stands in the human decision."""

    UNDECIDED = "UNDECIDED"
    SHORTLISTED = "SHORTLISTED"
    FINAL_SELECTED = "FINAL_SELECTED"
    MANUALLY_EXCLUDED = "MANUALLY_EXCLUDED"


class SelectionOrigin(str, Enum):
    """What put the molecule in its current selection status."""

    AUTOMATIC = "AUTOMATIC"
    PARETO = "PARETO"
    DIVERSITY_PICKER = "DIVERSITY_PICKER"
    LASSO_SELECTION = "LASSO_SELECTION"
    MANUAL = "MANUAL"
    RESCUED = "RESCUED"
    IMPORTED_RECIPE = "IMPORTED_RECIPE"


#: Origins that represent a human acting directly, which require a justification
#: when they contradict the chemical verdict.
HUMAN_ORIGINS = frozenset(
    {SelectionOrigin.MANUAL, SelectionOrigin.LASSO_SELECTION, SelectionOrigin.RESCUED}
)


@dataclass(frozen=True)
class MoleculeState:
    """One molecule's chemical verdict and selection decision.

    ``pinned`` is a flag rather than a status: a molecule stays pinned while
    being shortlisted or finally selected, and pinning must survive every
    automatic re-selection.
    """

    record_id: int
    chemical_status: ChemicalStatus = ChemicalStatus.AUTO_PASS
    selection_status: SelectionStatus = SelectionStatus.UNDECIDED
    origin: SelectionOrigin | None = None
    pinned: bool = False
    note: str = ""

    @property
    def is_selected(self) -> bool:
        return self.selection_status is SelectionStatus.FINAL_SELECTED

    @property
    def contradicts_chemistry(self) -> bool:
        """True when a human kept a molecule the rules rejected."""
        if self.selection_status is SelectionStatus.FINAL_SELECTED:
            return not self.chemical_status.passed
        return False

    def with_selection(
        self,
        status: SelectionStatus,
        origin: SelectionOrigin,
        note: str = "",
    ) -> MoleculeState:
        """Return a copy with a new decision; the chemical verdict is untouched."""
        return replace(self, selection_status=status, origin=origin, note=note or self.note)

    def with_pin(self, pinned: bool) -> MoleculeState:
        return replace(self, pinned=pinned)

    def as_row(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "chemical_status": self.chemical_status.value,
            "selection_status": self.selection_status.value,
            "selection_origin": self.origin.value if self.origin else None,
            "pinned": int(self.pinned),
            "manual_note": self.note or None,
        }


def chemical_status_from_run(
    *, valid: bool, passed: bool, borderline: bool = False, alerted: bool = False
) -> ChemicalStatus:
    """Map a pipeline outcome onto a chemical status.

    Order matters: invalid input outranks everything, a failure outranks its
    flags, and BORDERLINE outranks WARNING because being near a threshold is
    the more actionable fact when reviewing a passing molecule.
    """
    if not valid:
        return ChemicalStatus.INVALID
    if not passed:
        return ChemicalStatus.AUTO_FAIL
    if borderline:
        return ChemicalStatus.BORDERLINE
    if alerted:
        return ChemicalStatus.WARNING
    return ChemicalStatus.AUTO_PASS
