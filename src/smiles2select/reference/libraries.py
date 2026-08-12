"""Explicit candidate, reference and background library roles.

The existing importer deliberately remains unchanged.  This module provides a
small, dependency-light boundary that turns any imported table into a
traceable molecular library before cross-library analysis begins.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum

import pandas as pd
from rdkit import Chem


class LibraryRole(str, Enum):
    """How a library participates in a selection session."""

    CANDIDATE = "candidate"
    REFERENCE = "reference"
    BACKGROUND = "background"


@dataclass(frozen=True)
class LibrarySpec:
    """Stable metadata for one library in a session."""

    library_id: str
    role: LibraryRole = LibraryRole.REFERENCE
    source: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not self.library_id.strip():
            raise ValueError("library_id must not be empty")

    def as_dict(self) -> dict[str, str]:
        return {
            "library_id": self.library_id,
            "role": self.role.value,
            "source": self.source,
            "description": self.description,
        }


@dataclass(frozen=True)
class ReferenceLibrary:
    """A prepared reference/background table plus its provenance."""

    spec: LibrarySpec
    molecules: pd.DataFrame

    def __post_init__(self) -> None:
        required = {"reference_id", "canonical_smiles", "valid", "inchikey"}
        missing = required.difference(self.molecules.columns)
        if missing:
            raise ValueError(f"prepared library is missing columns: {sorted(missing)}")

    @property
    def library_id(self) -> str:
        return self.spec.library_id

    @property
    def valid(self) -> pd.DataFrame:
        return self.molecules[self.molecules["valid"].astype(bool)]


def _first_column(frame: pd.DataFrame, requested: str | None, options: Iterable[str]) -> str:
    if requested:
        if requested not in frame.columns:
            raise KeyError(f"column '{requested}' not found; available: {list(frame.columns)}")
        return requested
    lowered = {str(column).lower(): column for column in frame.columns}
    for option in options:
        if option.lower() in lowered:
            return str(lowered[option.lower()])
    raise KeyError(f"could not infer a column from {list(frame.columns)}")


def _inchi_key(mol: Chem.Mol) -> str | None:
    try:
        value = Chem.MolToInchiKey(mol)
    except Exception:  # pragma: no cover - depends on the RDKit build
        return None
    return value or None


def _canonical_record(text: object) -> tuple[str | None, str | None, str | None]:
    original = "" if text is None else str(text).strip()
    if not original:
        return None, None, "empty SMILES"
    mol = Chem.MolFromSmiles(original)
    if mol is None:
        return None, None, "invalid SMILES"
    try:
        return Chem.MolToSmiles(mol), _inchi_key(mol), None
    except Exception as exc:  # pragma: no cover - defensive RDKit boundary
        return None, None, f"canonicalization failed: {exc}"


def prepare_library_frame(
    frame: pd.DataFrame,
    library_id: str,
    *,
    smiles_column: str | None = None,
    id_column: str | None = None,
    role: LibraryRole = LibraryRole.REFERENCE,
    source: str = "",
    description: str = "",
) -> ReferenceLibrary:
    """Normalize a table into a traceable molecular library.

    Invalid rows remain in the table and are marked ``valid=False``.  Keeping
    them makes reference counts auditable and prevents an invalid reference
    from being mistaken for an absent reference.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    smiles_name = _first_column(
        frame, smiles_column, ("canonical_smiles", "smiles", "SMILES", "structure")
    )
    id_name = None
    if id_column:
        id_name = _first_column(frame, id_column, ())
    else:
        lowered = {str(column).lower(): column for column in frame.columns}
        for option in ("reference_id", "molecule_id", "id", "compound_id", "access_code"):
            if option in lowered and lowered[option] != smiles_name:
                id_name = str(lowered[option])
                break

    canonical: list[str | None] = []
    inchikeys: list[str | None] = []
    errors: list[str | None] = []
    for value in frame[smiles_name].tolist():
        result = _canonical_record(value)
        canonical.append(result[0])
        inchikeys.append(result[1])
        errors.append(result[2])

    identifiers = (
        frame[id_name].astype("object").where(frame[id_name].notna(), None).tolist()
        if id_name
        else [None] * len(frame)
    )
    prepared = pd.DataFrame(
        {
            "reference_id": [
                str(value).strip() if value not in (None, "") else f"{library_id}_{index + 1:07d}"
                for index, value in enumerate(identifiers)
            ],
            "source_smiles": frame[smiles_name].fillna("").astype(str).str.strip().tolist(),
            "canonical_smiles": canonical,
            "inchikey": inchikeys,
            "valid": [value is not None for value in canonical],
            "invalid_reason": errors,
            "library_id": library_id,
        },
        index=frame.index,
    )
    spec = LibrarySpec(
        library_id=library_id,
        role=role,
        source=source,
        description=description,
    )
    return ReferenceLibrary(spec=spec, molecules=prepared)


def library_mapping(
    libraries: Iterable[ReferenceLibrary],
) -> Mapping[str, ReferenceLibrary]:
    """Return a mapping and reject duplicate library identifiers."""

    result: dict[str, ReferenceLibrary] = {}
    for library in libraries:
        if library.library_id in result:
            raise ValueError(f"duplicate library_id: {library.library_id}")
        result[library.library_id] = library
    return result
