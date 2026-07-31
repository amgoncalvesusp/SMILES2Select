"""Parquet export.

Complements the Excel report for libraries too large for a spreadsheet: Excel
tops out at about a million rows per sheet, Parquet does not, and it keeps the
column types instead of flattening everything to cells.
"""

from __future__ import annotations

from pathlib import Path

from smiles2select.export.excel import export_frame
from smiles2select.pipeline.runner import RunResult


class ParquetUnavailableError(RuntimeError):
    """Raised when no Parquet engine is installed."""


def export(result: RunResult, path: str | Path, *, selected_only: bool = False) -> Path:
    """Write the wide per-record table to Parquet and return its path."""
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:  # keep the dependency optional
        raise ParquetUnavailableError(
            "Parquet export requires pyarrow: pip install 'smiles2select[parquet]'"
        ) from exc

    frame = export_frame(result)
    if selected_only:
        frame = frame[frame["Final_Status"] == "SELECTED"]

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.reset_index(names="record_id").to_parquet(output, index=False)
    return output
