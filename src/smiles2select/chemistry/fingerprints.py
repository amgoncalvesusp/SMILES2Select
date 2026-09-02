"""Molecular fingerprints for similarity and diversity work.

Only Morgan (ECFP-like) fingerprints are exposed, with the radius and length
stated explicitly at every call: a similarity computed at radius 2 is not
comparable with one computed at radius 3, and silently changing either would
change every diversity result without changing a single threshold.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


@dataclass(frozen=True)
class FingerprintConfig:
    """Fingerprint parameters, recorded alongside any result derived from them."""

    radius: int = 2
    size: int = 2048
    use_chirality: bool = False

    def label(self) -> str:
        chirality = ", chiral" if self.use_chirality else ""
        return f"Morgan r={self.radius}, {self.size} bits{chirality}"


@lru_cache(maxsize=8)
def _generator(radius: int, size: int, use_chirality: bool):
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=size, includeChirality=use_chirality
    )


def fingerprint(mol: Chem.Mol, config: FingerprintConfig = FingerprintConfig()):
    """Morgan fingerprint of one molecule as an RDKit bit vector."""
    generator = _generator(config.radius, config.size, config.use_chirality)
    return generator.GetFingerprint(mol)


def fingerprints_from_smiles(
    smiles: Iterable[str], config: FingerprintConfig = FingerprintConfig()
) -> tuple[list, list[int]]:
    """Fingerprints plus the positions they came from.

    Unparseable entries are skipped, and the returned index list says which
    input positions the fingerprints correspond to - so a caller never has to
    assume the two lists line up.
    """
    vectors, positions = [], []
    for position, text in enumerate(smiles):
        mol = Chem.MolFromSmiles(text) if text else None
        if mol is None:
            continue
        vectors.append(fingerprint(mol, config))
        positions.append(position)
    return vectors, positions


def tanimoto(first, second) -> float:
    return float(DataStructs.TanimotoSimilarity(first, second))


def mean_pairwise_similarity(vectors: Sequence, sample_limit: int = 2000) -> float:
    """Average Tanimoto over all pairs; 0.0 for fewer than two molecules.

    Note: O(n^2) over the first ``sample_limit`` fingerprints. Exact for
    normal report sizes; for a million-molecule library, sample upstream rather
    than waiting for the full matrix.
    """
    chosen = list(vectors[:sample_limit])
    if len(chosen) < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for index, vector in enumerate(chosen[1:], start=1):
        similarities = DataStructs.BulkTanimotoSimilarity(vector, chosen[:index])
        total += float(sum(similarities))
        pairs += index
    return round(total / pairs, 4) if pairs else 0.0
