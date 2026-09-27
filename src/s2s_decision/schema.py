"""Versioned feature order shared by chemistry, training, and inference."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

SCHEMA_VERSION = "s2s-tiny-features/1"
FP_BITS = 2048
DESCRIPTOR_NAMES = (
    "mol_wt",
    "rdkit_wlogp",
    "mol_mr",
    "tpsa",
    "hbd_lipinski",
    "hba_lipinski",
    "rotatable_bonds",
    "heavy_atom_count",
    "heteroatom_count",
    "ring_count",
    "aromatic_ring_count",
    "formal_charge",
    "fraction_csp3",
    "qed",
    "sa_score",
    "np_score",
)
PROPERTY_NAMES = DESCRIPTOR_NAMES + (
    "lipinski_violations",
    "veber_violations",
    "pains_count",
    "brenk_count",
)
CONTEXT_NAMES = (
    "similarity_active_max",
    "similarity_active_mean_top5",
    "similarity_inactive_max",
    "similarity_inactive_mean_top5",
    "novelty",
    "reference_scaffold_fraction",
    "reference_density_top5",
)
LOG_NAMES = (
    "hbd_lipinski",
    "hba_lipinski",
    "rotatable_bonds",
    "heavy_atom_count",
    "heteroatom_count",
    "ring_count",
    "aromatic_ring_count",
    "lipinski_violations",
    "veber_violations",
    "pains_count",
    "brenk_count",
)


@dataclass(frozen=True)
class FeatureSet:
    records: pd.DataFrame
    manifest: dict


def fingerprint_matrix(frame: pd.DataFrame, bits: int = FP_BITS) -> np.ndarray:
    """Decode RDKit packed binary text; never accept a truncated or absent fingerprint."""
    if bits not in (1024, 2048):
        raise ValueError("fingerprint bits must be 1024 or 2048")
    if "fingerprint_hex" not in frame:
        raise ValueError("fingerprint_hex is required")
    rows = []
    for value in frame["fingerprint_hex"]:
        if not isinstance(value, str) or len(value) != bits // 4:
            raise ValueError("fingerprint length does not match feature schema")
        try:
            packed = bytes.fromhex(value)
        except ValueError as exc:
            raise ValueError("fingerprint_hex must contain hexadecimal bytes") from exc
        if len(packed) != bits // 8:
            raise ValueError("fingerprint length does not match feature schema")
        rows.append(np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little"))
    return np.asarray(rows, dtype=np.float32).reshape(len(frame), bits)
