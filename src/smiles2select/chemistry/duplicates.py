"""Duplicate detection over canonical structures.

Duplicates are decided on the canonical SMILES produced *after* standardization,
so a salt and its parent, or two different input spellings of the same graph,
collapse to a single structure.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class DuplicateReport:
    """Which records are first occurrences and which repeat an earlier one."""

    first_occurrence_index: dict[str, int]
    duplicate_of: dict[int, int]

    @property
    def duplicate_count(self) -> int:
        return len(self.duplicate_of)

    def is_duplicate(self, index: int) -> bool:
        return index in self.duplicate_of


def find_duplicates(canonical_smiles: Sequence[str | None]) -> DuplicateReport:
    """Map each record index to the index of its first identical structure.

    ``None`` and empty entries (invalid molecules) are never treated as
    duplicates of one another.
    """
    first: dict[str, int] = {}
    duplicate_of: dict[int, int] = {}
    for index, smiles in enumerate(canonical_smiles):
        if not smiles:
            continue
        if smiles in first:
            duplicate_of[index] = first[smiles]
        else:
            first[smiles] = index
    return DuplicateReport(first_occurrence_index=first, duplicate_of=duplicate_of)


def unique_indices(canonical_smiles: Sequence[str | None]) -> list[int]:
    """Indices of the first occurrence of each distinct structure."""
    report = find_duplicates(canonical_smiles)
    return [i for i in range(len(canonical_smiles)) if not report.is_duplicate(i)]


def group_by_structure(canonical_smiles: Iterable[str | None]) -> dict[str, list[int]]:
    """Structure -> every record index sharing it."""
    groups: dict[str, list[int]] = {}
    for index, smiles in enumerate(canonical_smiles):
        if not smiles:
            continue
        groups.setdefault(smiles, []).append(index)
    return groups
