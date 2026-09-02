"""Hand-off to SMILES2Docking.

That application reads a spreadsheet and takes the SMILES from one column and
the identifier from another; its ``config/settings.yaml`` names them ``smiles``
and ``access_code`` by default. This module writes exactly that shape, so a
selection can go straight into docking without a manual reformatting step.

Only selected molecules are exported by default: sending the excluded ones to
docking would silently undo the selection the user just made.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from smiles2select.chemistry.preparability import estimated_3d_structures
from smiles2select.export.excel import export_frame
from smiles2select.pipeline.runner import RunResult

#: Column names expected by SMILES2Docking (config/settings.yaml).
SMILES_COLUMN = "smiles"
ACCESS_CODE_COLUMN = "access_code"


@dataclass(frozen=True)
class DockingExportOptions:
    """What to hand over, and under which column names."""

    selected_only: bool = True
    smiles_column: str = SMILES_COLUMN
    access_code_column: str = ACCESS_CODE_COLUMN
    #: Standardized structure by default: docking should receive the molecule
    #: the descriptors were computed on, not the raw input string.
    use_canonical: bool = True
    include_context: bool = False


def build_frame(result: RunResult, options: DockingExportOptions | None = None) -> pd.DataFrame:
    """Two-column table (plus optional context) ready for SMILES2Docking."""
    settings = options or DockingExportOptions()
    frame = export_frame(result)
    if settings.selected_only:
        frame = frame[frame["Final_Status"] == "SELECTED"]

    source = "Canonical_SMILES" if settings.use_canonical else "Original_SMILES"
    payload = pd.DataFrame(
        {
            settings.access_code_column: frame["ID"].astype(str),
            settings.smiles_column: frame[source].astype(str),
        }
    )
    payload = payload[payload[settings.smiles_column].str.strip() != ""]

    if settings.include_context:
        context = frame.loc[payload.index]
        for column in ("MW", "WLOGP", "TPSA", "QED"):
            if column in context.columns:
                payload[column] = context[column].to_numpy()

    if result.config.compute_preparability:
        undef_series = (
            result.descriptors["undefined_stereocenters"]
            .reindex(payload.index)
            .fillna(0)
            .astype(int)
        )
        payload["undefined_stereocenters"] = undef_series.to_numpy()

        taut_series = (
            result.descriptors["tautomer_count"].reindex(payload.index).fillna(1).astype(int)
            if "tautomer_count" in result.descriptors
            else pd.Series(1, index=payload.index)
        )
        est_structures = [
            estimated_3d_structures(int(u), int(t)) for u, t in zip(undef_series, taut_series)
        ]
        payload["estimated_3d_structures"] = est_structures

        if result.preparability is not None and not result.preparability.empty:
            flag_counts = (
                result.preparability.groupby("record_id")
                .size()
                .reindex(payload.index, fill_value=0)
                .astype(int)
            )
            payload["preparability_flags_count"] = flag_counts.to_numpy()
        else:
            payload["preparability_flags_count"] = 0

    return payload.reset_index(drop=True)


def export(
    result: RunResult, path: str | Path, options: DockingExportOptions | None = None
) -> Path:
    """Write the hand-off file; the suffix chooses spreadsheet or CSV."""
    settings = options or DockingExportOptions()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    frame = build_frame(result, settings)
    if frame.empty:
        raise ValueError(
            "nothing to export for docking: the selection is empty "
            "(use selected_only=False to hand over every valid molecule)"
        )

    if output.suffix.lower() in {".xlsx", ".xlsm"}:
        frame.to_excel(output, index=False, sheet_name="molecules")
    else:
        frame.to_csv(output, index=False)
    return output
