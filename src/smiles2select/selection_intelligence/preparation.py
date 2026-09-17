"""On-demand chemistry for final selection; safe to call from a background worker."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from smiles2select.selection_intelligence.constrained_selection import (
    SelectionConstraints,
    Strategy,
)


def ensure_selection_scaffolds(
    candidates: pd.DataFrame,
    constraints: SelectionConstraints,
    strategy: Strategy,
    *,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Fill missing Murcko values only when required, preserving IDs and cached values.

    The input is never changed. Empty scaffold SMILES are known acyclic molecules;
    null values are unknown. Repeated canonical SMILES are calculated once per call.
    Invalid structures fail explicitly rather than masquerading as acyclic cores.
    """
    required = (
        strategy == Strategy.SCAFFOLD_COVERAGE
        or constraints.max_per_scaffold is not None
        or constraints.min_scaffolds is not None
    )
    if not required:
        return candidates
    existing = candidates.get("murcko_scaffold", pd.Series(pd.NA, index=candidates.index))
    missing = existing.isna()
    if not missing.any():
        return candidates
    if "canonical_smiles" not in candidates:
        raise ValueError(
            "Molecular-core selection requires complete cached scaffolds or canonical SMILES "
            "to calculate missing values."
        )
    structures = candidates.loc[missing, "canonical_smiles"]
    if progress:
        progress(f"Calculating molecular cores for {len(structures):,} candidates...")
    cache: dict[str, str] = {}
    for record_id, text in structures.items():
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Missing canonical SMILES for record {record_id}.")
        if text in cache:
            continue
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            raise ValueError(f"Cannot calculate molecular core for record {record_id}: invalid SMILES.")
        try:
            core = MurckoScaffold.GetScaffoldForMol(mol)
            cache[text] = Chem.MolToSmiles(core)
        except Exception as exc:
            raise ValueError(f"Cannot calculate molecular core for record {record_id}.") from exc
        if progress and len(cache) % 1000 == 0:
            progress(f"Calculated molecular cores for {len(cache):,} unique structures...")
    completed = existing.where(~missing, candidates.canonical_smiles.map(cache))
    if progress:
        progress(f"Molecular cores ready for all {len(candidates):,} candidates.")
    return candidates.assign(murcko_scaffold=completed)
