"""Evaluation of docking preparability flags.

Produces a sparse table recording only raised flags with human-readable details.
"""

from __future__ import annotations

import pandas as pd
from rdkit import Chem

from smiles2select.chemistry.preparability import (
    amide_bond_count,
    estimated_3d_structures,
    fragment_count,
    largest_ring_size,
    uncommon_elements,
    undefined_stereocenters,
)
from smiles2select.preparability.engines import EngineProfile


def evaluate(descriptors: pd.DataFrame, engine: EngineProfile) -> pd.DataFrame:
    """One row per molecule and flag actually raised.

    Sparse on purpose: a library where nothing is flagged produces an empty
    table, not a million rows of "fine".
    """
    rows: list[dict[str, object]] = []
    flag_map = {f.id: f for f in engine.flags}

    for idx, row in descriptors.iterrows():
        if "valid" in row and not bool(row["valid"]):
            continue

        record_id = row["record_id"] if "record_id" in row else idx

        smiles = None
        for key in ("canonical_smiles", "standardized_smiles", "smiles", "original_smiles"):
            val = row.get(key)
            if pd.notna(val) and str(val).strip():
                smiles = str(val).strip()
                break

        mol = None
        if smiles:
            mol = Chem.MolFromSmiles(smiles)

        # 1. UNCOMMON_ELEMENT
        if "UNCOMMON_ELEMENT" in flag_map and mol is not None:
            flag = flag_map["UNCOMMON_ELEMENT"]
            uncommon = uncommon_elements(mol, engine.common_elements)
            if uncommon:
                elem_word = "elemento" if len(uncommon) == 1 else "elementos"
                rows.append(
                    {
                        "record_id": record_id,
                        "flag_id": flag.id,
                        "severity": flag.severity,
                        "detail": f"{elem_word} {', '.join(uncommon)} fora do conjunto comum",
                    }
                )

        # 2. MULTIPLE_FRAGMENTS
        if "MULTIPLE_FRAGMENTS" in flag_map:
            flag = flag_map["MULTIPLE_FRAGMENTS"]
            frags = row.get("fragment_count")
            if (frags is None or pd.isna(frags)) and mol is not None:
                frags = fragment_count(mol)
            thresh = flag.threshold if flag.threshold is not None else 1
            if frags is not None and pd.notna(frags) and frags > thresh:
                rows.append(
                    {
                        "record_id": record_id,
                        "flag_id": flag.id,
                        "severity": flag.severity,
                        "detail": f"{int(frags)} fragmentos após padronização",
                    }
                )

        # 3. MACROCYCLE
        if "MACROCYCLE" in flag_map:
            flag = flag_map["MACROCYCLE"]
            ring_size = row.get("largest_ring_size")
            if (ring_size is None or pd.isna(ring_size)) and mol is not None:
                ring_size = largest_ring_size(mol)
            thresh = flag.threshold if flag.threshold is not None else 12
            if ring_size is not None and pd.notna(ring_size) and ring_size >= thresh:
                rows.append(
                    {
                        "record_id": record_id,
                        "flag_id": flag.id,
                        "severity": flag.severity,
                        "detail": f"anel de {int(ring_size)} átomos",
                    }
                )

        # 4. PEPTIDE_LIKE
        if "PEPTIDE_LIKE" in flag_map:
            flag = flag_map["PEPTIDE_LIKE"]
            amides = row.get("amide_bond_count")
            if (amides is None or pd.isna(amides)) and mol is not None:
                amides = amide_bond_count(mol)
            thresh = flag.threshold if flag.threshold is not None else 4
            if amides is not None and pd.notna(amides) and amides >= thresh:
                rows.append(
                    {
                        "record_id": record_id,
                        "flag_id": flag.id,
                        "severity": flag.severity,
                        "detail": f"{int(amides)} ligações amídicas",
                    }
                )

        # 5. UNDEFINED_STEREO
        if "UNDEFINED_STEREO" in flag_map:
            flag = flag_map["UNDEFINED_STEREO"]
            undef = row.get("undefined_stereocenters")
            if (undef is None or pd.isna(undef)) and mol is not None:
                undef = undefined_stereocenters(mol)
            thresh = flag.threshold if flag.threshold is not None else 1
            if undef is not None and pd.notna(undef) and undef >= thresh:
                raw_taut = row.get("tautomer_count")
                tauts = int(raw_taut) if (raw_taut is not None and pd.notna(raw_taut)) else 1
                structures = estimated_3d_structures(int(undef), max(1, tauts))
                word = (
                    "estereocentros indefinidos" if int(undef) != 1 else "estereocentro indefinido"
                )
                rows.append(
                    {
                        "record_id": record_id,
                        "flag_id": flag.id,
                        "severity": flag.severity,
                        "detail": f"{int(undef)} {word} → {structures} estruturas",
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["record_id", "flag_id", "severity", "detail"])
    return pd.DataFrame(rows)
