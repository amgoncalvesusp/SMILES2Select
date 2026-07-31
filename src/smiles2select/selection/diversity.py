"""Diverse subset selection.

Everything here judges a molecule *relative to the set*, which is why it lives
outside the rule engine: whether a compound is picked depends on what else was
picked, so it can never be expressed as a per-molecule pass/fail rule.

Diversity picking runs after selection, never instead of it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
from rdkit import DataStructs
from rdkit.SimDivFilters import rdSimDivPickers

from smiles2select.chemistry.fingerprints import (
    FingerprintConfig,
    fingerprints_from_smiles,
    mean_pairwise_similarity,
)
from smiles2select.chemistry.scaffolds import scaffold_diversity, scaffolds_from_smiles


@dataclass(frozen=True)
class DiversityReport:
    """Chosen subset plus the numbers that justify it."""

    picked_index: list[int]
    requested: int
    available: int
    fingerprint_label: str
    mean_similarity_before: float
    mean_similarity_after: float
    scaffolds_before: dict[str, float]
    scaffolds_after: dict[str, float]

    def summary_rows(self) -> list[tuple[str, object]]:
        return [
            ("moléculas disponíveis", self.available),
            ("moléculas selecionadas", len(self.picked_index)),
            ("fingerprint", self.fingerprint_label),
            ("similaridade média antes", self.mean_similarity_before),
            ("similaridade média depois", self.mean_similarity_after),
            ("scaffolds antes", self.scaffolds_before.get("scaffold_count", 0)),
            ("scaffolds depois", self.scaffolds_after.get("scaffold_count", 0)),
        ]


def max_min_pick(
    smiles: Sequence[str],
    count: int,
    config: FingerprintConfig = FingerprintConfig(),
    seed: int = 0xF00D,
) -> list[int]:
    """MaxMin pick of ``count`` molecules, returning positions in ``smiles``.

    MaxMin repeatedly takes the molecule furthest from everything already
    chosen, which spreads the subset over the space instead of clustering it.
    The seed is fixed so a run is reproducible.
    """
    if count <= 0:
        raise ValueError("count must be positive")

    vectors, positions = fingerprints_from_smiles(smiles, config)
    if len(vectors) <= count:
        return positions

    def distance(first: int, second: int) -> float:
        return 1.0 - DataStructs.TanimotoSimilarity(vectors[first], vectors[second])

    picker = rdSimDivPickers.MaxMinPicker()
    picked = picker.LazyPick(distance, len(vectors), count, seed=seed)
    return [positions[index] for index in picked]


def pick_diverse(
    smiles: Sequence[str],
    count: int,
    config: FingerprintConfig = FingerprintConfig(),
    seed: int = 0xF00D,
) -> DiversityReport:
    """MaxMin pick with before/after similarity and scaffold statistics."""
    picked = max_min_pick(smiles, count, config, seed)
    chosen = [smiles[index] for index in picked]

    all_vectors, _ = fingerprints_from_smiles(smiles, config)
    picked_vectors, _ = fingerprints_from_smiles(chosen, config)

    return DiversityReport(
        picked_index=picked,
        requested=count,
        available=len(smiles),
        fingerprint_label=config.label(),
        mean_similarity_before=mean_pairwise_similarity(all_vectors),
        mean_similarity_after=mean_pairwise_similarity(picked_vectors),
        scaffolds_before=scaffold_diversity(pd.Series(scaffolds_from_smiles(smiles))),
        scaffolds_after=scaffold_diversity(pd.Series(scaffolds_from_smiles(chosen))),
    )


def pick_by_scaffold(smiles: Sequence[str], per_scaffold: int = 1) -> list[int]:
    """Keep at most ``per_scaffold`` molecules of each Murcko scaffold.

    A cheaper alternative to MaxMin when the goal is only to stop one chemotype
    from dominating: no fingerprints, no distance matrix.
    """
    if per_scaffold < 1:
        raise ValueError("per_scaffold must be at least 1")
    seen: dict[str, int] = {}
    kept: list[int] = []
    for position, scaffold in enumerate(scaffolds_from_smiles(smiles)):
        taken = seen.get(scaffold, 0)
        if taken < per_scaffold:
            seen[scaffold] = taken + 1
            kept.append(position)
    return kept
