"""Murcko scaffolds.

Grouping by scaffold answers a question the property filters cannot: how many
*distinct chemotypes* survived the selection. Two hundred analogues of one
scaffold and two hundred distinct scaffolds pass the same rules and are very
different libraries.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def murcko_scaffold(mol: Chem.Mol, *, generic: bool = False) -> str:
    """Canonical SMILES of the Bemis-Murcko scaffold.

    ``generic`` replaces every atom with carbon and every bond with a single
    bond, collapsing chemotypes that differ only in heteroatom placement.
    Returns an empty string for acyclic molecules, which have no scaffold.
    """
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return ""
        if generic:
            scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return ""


def scaffolds_from_smiles(smiles: Iterable[str], *, generic: bool = False) -> list[str]:
    """Scaffold of each SMILES; unparseable entries give an empty string."""
    results: list[str] = []
    for text in smiles:
        mol = Chem.MolFromSmiles(text) if text else None
        results.append(murcko_scaffold(mol, generic=generic) if mol is not None else "")
    return results


def scaffold_table(scaffold_smiles: pd.Series) -> pd.DataFrame:
    """One row per distinct scaffold, with its population, most frequent first.

    Acyclic molecules (empty scaffold) get their own row rather than being
    dropped, so the counts still add up to the library size.
    """
    counts = scaffold_smiles.fillna("").value_counts()
    return pd.DataFrame(
        {
            "scaffold_smiles": [key if key else "(acyclic)" for key in counts.index],
            "molecules": counts.to_numpy(),
        }
    )


def scaffold_ids(scaffold_smiles: pd.Series) -> pd.Series:
    """Stable integer id per scaffold, assigned by descending population."""
    order = {
        scaffold: index
        for index, scaffold in enumerate(scaffold_smiles.fillna("").value_counts().index, start=1)
    }
    return scaffold_smiles.fillna("").map(order).astype("Int64")


def scaffold_diversity(scaffold_smiles: pd.Series) -> dict[str, float]:
    """Summary of chemotype spread.

    ``scaffold_fraction`` is the share of distinct scaffolds;
    ``largest_scaffold_share`` exposes a library dominated by one chemotype,
    which the fraction alone can hide.
    """
    cleaned = scaffold_smiles.fillna("")
    total = len(cleaned)
    if total == 0:
        return {"molecules": 0, "scaffold_count": 0, "scaffold_fraction": 0.0}
    counts = cleaned.value_counts()
    return {
        "molecules": total,
        "scaffold_count": int(counts.size),
        "scaffold_fraction": round(float(counts.size) / total, 4),
        "largest_scaffold_share": round(float(counts.iloc[0]) / total, 4),
        "singleton_scaffolds": int((counts == 1).sum()),
    }
