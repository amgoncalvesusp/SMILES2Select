"""Reading SMILES libraries.

Provenance travels with every record: source file, sheet and row number are
carried through the whole pipeline so a rejected molecule can be located in the
user's original spreadsheet.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RECORD_COLUMNS = (
    "input_order",
    "molecule_id",
    "original_smiles",
    "source_file",
    "source_sheet",
    "source_row",
)

SPREADSHEET_SUFFIXES = {".xlsx", ".xlsm", ".xls"}
TEXT_SUFFIXES = {".csv", ".tsv", ".txt", ".smi", ".smiles"}


class SourceReadError(ValueError):
    """Raised when a source file cannot be read or mapped."""


@dataclass(frozen=True)
class ColumnMapping:
    """Which column holds the SMILES, and which (optionally) the identifier."""

    smiles: str
    molecule_id: str | None = None


@dataclass(frozen=True)
class SourceFile:
    """One input file, optionally one sheet of it."""

    path: Path
    mapping: ColumnMapping
    sheet: str | None = None

    @property
    def label(self) -> str:
        return self.path.name


def sheet_names(path: str | Path) -> list[str]:
    """Sheet names of a spreadsheet; empty list for text formats."""
    file_path = Path(path)
    if file_path.suffix.lower() not in SPREADSHEET_SUFFIXES:
        return []
    try:
        return list(pd.ExcelFile(file_path).sheet_names)
    except Exception as exc:
        raise SourceReadError(f"{file_path.name}: cannot read sheets ({exc})") from exc


def read_table(
    path: str | Path, sheet: str | None = None, nrows: int | None = None
) -> pd.DataFrame:
    """Read a source file into a DataFrame, choosing the reader by suffix."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    try:
        if suffix in SPREADSHEET_SUFFIXES:
            return pd.read_excel(file_path, sheet_name=sheet or 0, nrows=nrows, dtype=str)
        if suffix in {".smi", ".smiles"}:
            return _read_smi(file_path, nrows)
        separator = "\t" if suffix == ".tsv" else _detect_separator(file_path)
        return pd.read_csv(file_path, sep=separator, engine="python", nrows=nrows, dtype=str)
    except SourceReadError:
        raise
    except Exception as exc:
        raise SourceReadError(f"{file_path.name}: cannot be read ({exc})") from exc


def _detect_separator(path: Path) -> str:
    """Pick the delimiter from a fixed candidate list.

    Pandas' own sniffing (``sep=None``) inspects every character and happily
    decides that a single-column ``SMILES`` header is delimited by ``S``.
    Restricting the candidates to real delimiters keeps that from happening.
    """
    candidates = (",", ";", "\t", "|")
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            header = handle.readline()
    except OSError:
        return ","
    counts = {candidate: header.count(candidate) for candidate in candidates}
    best = max(counts, key=lambda candidate: counts[candidate])
    return best if counts[best] else ","


def _read_smi(path: Path, nrows: int | None) -> pd.DataFrame:
    """SMILES files: whitespace-separated ``SMILES [id]``, no header."""
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        nrows=nrows,
        dtype=str,
        comment="#",
        engine="python",
    )
    names = ["SMILES", "ID"] + [f"col_{index}" for index in range(2, frame.shape[1])]
    frame.columns = names[: frame.shape[1]]
    return frame


def preview_columns(path: str | Path, sheet: str | None = None, rows: int = 20) -> list[str]:
    """Column names, for the mapping screen."""
    return list(read_table(path, sheet, nrows=rows).columns)


def guess_mapping(columns: Sequence[str]) -> ColumnMapping:
    """Best-effort guess so the mapping screen opens pre-filled."""
    lowered = {str(column).strip().lower(): column for column in columns}
    smiles_column = next(
        (
            lowered[key]
            for key in ("smiles", "smile", "canonical_smiles", "structure")
            if key in lowered
        ),
        columns[0] if columns else "",
    )
    id_column = next(
        (
            lowered[key]
            for key in ("id", "molecule_id", "name", "compound_id", "cid")
            if key in lowered
        ),
        None,
    )
    if id_column == smiles_column:
        id_column = None
    return ColumnMapping(
        smiles=str(smiles_column), molecule_id=str(id_column) if id_column else None
    )


def load_records(sources: Iterable[SourceFile]) -> pd.DataFrame:
    """Concatenate every source into one record table indexed by ``record_id``.

    Rows with an empty SMILES cell are kept, not dropped: they are reported as
    invalid records so the input and output counts always reconcile.
    """
    frames: list[pd.DataFrame] = []
    for source in sources:
        table = read_table(source.path, source.sheet)
        if source.mapping.smiles not in table.columns:
            raise SourceReadError(
                f"{source.label}: column '{source.mapping.smiles}' not found; "
                f"available: {list(table.columns)}"
            )
        identifiers = (
            table[source.mapping.molecule_id]
            if source.mapping.molecule_id and source.mapping.molecule_id in table.columns
            else pd.Series([None] * len(table), index=table.index)
        )
        frames.append(
            pd.DataFrame(
                {
                    "molecule_id": identifiers.astype("object"),
                    # Always a string: an empty cell reads back as NaN (a float),
                    # which breaks every downstream consumer that treats the
                    # SMILES as text - the cache key among them.
                    "original_smiles": table[source.mapping.smiles]
                    .fillna("")
                    .astype(str)
                    .str.strip(),
                    "source_file": source.label,
                    "source_sheet": source.sheet or "",
                    # +2: one for the header row, one for 1-based spreadsheet rows.
                    "source_row": [index + 2 for index in range(len(table))],
                }
            )
        )

    if not frames:
        return pd.DataFrame(columns=list(RECORD_COLUMNS)).rename_axis("record_id")

    records = pd.concat(frames, ignore_index=True)
    records.insert(0, "input_order", range(1, len(records) + 1))
    records.index = pd.RangeIndex(start=1, stop=len(records) + 1, name="record_id")
    missing_id = records["molecule_id"].isna() | (
        records["molecule_id"].astype(str).str.strip() == ""
    )
    records.loc[missing_id, "molecule_id"] = [
        f"REC{record_id:07d}" for record_id in records.index[missing_id]
    ]
    return records[list(RECORD_COLUMNS)]
