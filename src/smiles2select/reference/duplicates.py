"""Exact candidate/reference overlap detection."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import pandas as pd

from smiles2select.reference.libraries import ReferenceLibrary

_ANNOTATION_COLUMNS = [
    "is_reference_duplicate",
    "reference_duplicate_library",
    "reference_duplicate_id",
]


@dataclass(frozen=True)
class ExactDuplicateReport:
    """Candidate annotations and one row per exact overlap."""

    annotations: pd.DataFrame
    overlaps: pd.DataFrame

    @property
    def duplicate_count(self) -> int:
        return int(self.annotations["is_reference_duplicate"].astype(bool).sum())


def _column(frame: pd.DataFrame, preferred: str, fallbacks: tuple[str, ...]) -> str:
    if preferred in frame.columns:
        return preferred
    lowered = {str(name).lower(): name for name in frame.columns}
    for name in (preferred, *fallbacks):
        if name.lower() in lowered:
            return str(lowered[name.lower()])
    return preferred


def _key_values(frame: pd.DataFrame, column: str) -> dict[object, list[object]]:
    if column not in frame.columns:
        return {}
    values: dict[object, list[object]] = {}
    for index, value in frame[column].items():
        if pd.isna(value) or str(value).strip() == "":
            continue
        values.setdefault(str(value), []).append(index)
    return values


def _as_libraries(
    references: Mapping[str, pd.DataFrame] | Iterable[ReferenceLibrary],
) -> list[ReferenceLibrary]:
    if isinstance(references, Mapping):
        from smiles2select.reference.libraries import prepare_library_frame

        result: list[ReferenceLibrary] = []
        for library_id, frame in references.items():
            if isinstance(frame, ReferenceLibrary):
                result.append(frame)
                continue
            if not isinstance(frame, pd.DataFrame):
                raise TypeError("reference mapping values must be pandas DataFrames")
            if {"reference_id", "canonical_smiles", "valid", "inchikey"}.issubset(frame.columns):
                from smiles2select.reference.libraries import LibraryRole, LibrarySpec

                result.append(
                    ReferenceLibrary(
                        LibrarySpec(str(library_id), LibraryRole.REFERENCE), frame.copy()
                    )
                )
            else:
                result.append(prepare_library_frame(frame, str(library_id)))
        return result
    return list(references)


def compare_exact_duplicates(
    candidates: pd.DataFrame,
    references: Mapping[str, pd.DataFrame] | Iterable[ReferenceLibrary],
    *,
    candidate_smiles_column: str = "canonical_smiles",
    candidate_id_column: str = "molecule_id",
) -> ExactDuplicateReport:
    """Annotate exact candidate/reference overlap by canonical SMILES or InChIKey.

    Matching libraries and IDs are joined with ``;`` in stable input order, so
    no source identifier is silently discarded when a candidate overlaps more
    than one reference library.
    """

    if not isinstance(candidates, pd.DataFrame):
        raise TypeError("candidates must be a pandas DataFrame")
    if "canonical_smiles" not in candidates.columns or "inchikey" not in candidates.columns:
        from smiles2select.reference.libraries import prepare_library_frame

        candidates = prepare_library_frame(
            candidates,
            "candidates",
            smiles_column=candidate_smiles_column,
            id_column=candidate_id_column if candidate_id_column in candidates.columns else None,
        ).molecules

    libraries = _as_libraries(references)
    canonical_maps = [
        (library, _key_values(library.valid, "canonical_smiles"), "canonical_smiles")
        for library in libraries
    ]
    inchikey_maps = [
        (library, _key_values(library.valid, "inchikey"), "inchikey") for library in libraries
    ]

    rows: list[dict[str, object]] = []
    overlaps: list[dict[str, object]] = []
    for candidate_index, candidate in candidates.iterrows():
        matches: list[tuple[ReferenceLibrary, object, str]] = []
        for library, key_map, key_type in (*canonical_maps, *inchikey_maps):
            value = candidate.get(key_type)
            if pd.isna(value) or str(value).strip() == "":
                continue
            for reference_index in key_map.get(str(value), []):
                reference_row = library.molecules.loc[reference_index]
                matches.append((library, reference_index, key_type))
                overlaps.append(
                    {
                        "candidate_index": candidate_index,
                        "candidate_id": candidate.get(candidate_id_column, candidate_index),
                        "reference_library": library.library_id,
                        "reference_index": reference_index,
                        "reference_id": reference_row["reference_id"],
                        "match_type": "InChIKey" if key_type == "inchikey" else "canonical_smiles",
                    }
                )
        unique_matches: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for library, reference_index, _ in matches:
            reference_id = str(library.molecules.loc[reference_index, "reference_id"])
            key = (library.library_id, reference_id)
            if key not in seen:
                seen.add(key)
                unique_matches.append(key)
        rows.append(
            {
                "record_id": candidate_index,
                "is_reference_duplicate": bool(unique_matches),
                "reference_duplicate_library": ";".join(item[0] for item in unique_matches) or None,
                "reference_duplicate_id": ";".join(item[1] for item in unique_matches) or None,
            }
        )

    annotations = pd.DataFrame(rows).set_index("record_id") if rows else pd.DataFrame(index=candidates.index)
    for column in _ANNOTATION_COLUMNS:
        if column not in annotations:
            annotations[column] = False if column == "is_reference_duplicate" else None
    annotations = annotations[_ANNOTATION_COLUMNS].reindex(candidates.index)
    overlap_frame = pd.DataFrame(overlaps)
    if overlap_frame.empty:
        overlap_frame = pd.DataFrame(
            columns=[
                "candidate_index",
                "candidate_id",
                "reference_library",
                "reference_index",
                "reference_id",
                "match_type",
            ]
        )
    return ExactDuplicateReport(annotations=annotations, overlaps=overlap_frame)
