"""Selection strategies that use reference chemistry explicitly."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd
from rdkit import Chem, DataStructs

from smiles2select.chemistry.fingerprints import FingerprintConfig, fingerprint


@dataclass(frozen=True)
class ReferenceDiversityResult:
    """Audit information for a reference-seeded MaxMin selection."""

    picked_indices: tuple[int, ...]
    initial_reference_novelty: Mapping[int, float]
    selection_scores: Mapping[int, float]
    fingerprint: FingerprintConfig
    seed: int


def reference_seeded_maxmin(
    candidate_smiles: Sequence[str],
    reference_smiles: Sequence[str],
    count: int,
    *,
    fingerprint_config: FingerprintConfig = FingerprintConfig(),
    seed: int = 0xF00D,
) -> ReferenceDiversityResult:
    """Pick compounds far from references and from already selected compounds.

    The algorithm stores one current minimum distance per candidate.  It never
    appends reference fingerprints to the selected set and never builds a full
    candidate x candidate matrix, which keeps the memory behavior useful for
    large libraries.
    """

    if count <= 0:
        raise ValueError("count must be positive")
    candidate_vectors: list[object] = []
    candidate_positions: list[int] = []
    for position, smiles in enumerate(candidate_smiles):
        mol = Chem.MolFromSmiles(str(smiles)) if smiles else None
        if mol is not None:
            candidate_vectors.append(fingerprint(mol, fingerprint_config))
            candidate_positions.append(position)
    if not candidate_vectors:
        return ReferenceDiversityResult((), {}, {}, fingerprint_config, seed)

    reference_vectors: list[object] = []
    for smiles in reference_smiles:
        mol = Chem.MolFromSmiles(str(smiles)) if smiles else None
        if mol is not None:
            reference_vectors.append(fingerprint(mol, fingerprint_config))

    tie_order = list(range(len(candidate_vectors)))
    random.Random(seed).shuffle(tie_order)
    tie_rank = {position: rank for rank, position in enumerate(tie_order)}

    novelty: dict[int, float] = {}
    current_distance: list[float] = []
    for position, vector in enumerate(candidate_vectors):
        if reference_vectors:
            maximum_similarity = max(
                DataStructs.BulkTanimotoSimilarity(vector, reference_vectors)
            )
            distance = 1.0 - float(maximum_similarity)
        else:
            distance = 1.0
        original_index = candidate_positions[position]
        novelty[original_index] = distance
        current_distance.append(distance)

    target = min(count, len(candidate_vectors))
    remaining = set(range(len(candidate_vectors)))
    picked: list[int] = []
    selection_scores: dict[int, float] = {}
    while remaining and len(picked) < target:
        selected_position = max(
            remaining,
            key=lambda position: (current_distance[position], -tie_rank[position]),
        )
        remaining.remove(selected_position)
        original_index = candidate_positions[selected_position]
        picked.append(original_index)
        selection_scores[original_index] = current_distance[selected_position]
        selected_vector = candidate_vectors[selected_position]
        for other in remaining:
            similarity = DataStructs.TanimotoSimilarity(selected_vector, candidate_vectors[other])
            current_distance[other] = min(current_distance[other], 1.0 - float(similarity))

    return ReferenceDiversityResult(
        picked_indices=tuple(picked),
        initial_reference_novelty=novelty,
        selection_scores=selection_scores,
        fingerprint=fingerprint_config,
        seed=seed,
    )


def reference_novelty_pick(
    candidates: pd.DataFrame,
    count: int,
    *,
    novelty_column: str = "reference_novelty",
    duplicate_column: str = "is_reference_duplicate",
) -> list[object]:
    """Select the most reference-novel rows with stable tie-breaking."""

    if count <= 0:
        raise ValueError("count must be positive")
    if novelty_column not in candidates.columns:
        raise KeyError(f"missing novelty column '{novelty_column}'")
    pool = candidates.copy()
    if duplicate_column in pool.columns:
        pool = pool[~pool[duplicate_column].fillna(False).astype(bool)]
    pool = pool[pool[novelty_column].notna()]
    ranked = pool.sort_values(novelty_column, ascending=False, kind="mergesort")
    return ranked.index[:count].tolist()


def reference_neighborhood(
    candidates: pd.DataFrame,
    *,
    minimum_similarity: float = 0.0,
    maximum_similarity: float = 1.0,
    similarity_column: str = "max_reference_similarity",
) -> pd.Index:
    """Return candidates in a user-defined reference-similarity window."""

    if not 0.0 <= minimum_similarity <= maximum_similarity <= 1.0:
        raise ValueError("similarity window must satisfy 0 <= minimum <= maximum <= 1")
    if similarity_column not in candidates.columns:
        raise KeyError(f"missing similarity column '{similarity_column}'")
    mask = candidates[similarity_column].between(minimum_similarity, maximum_similarity)
    return candidates.index[mask.fillna(False)]


def bridge_compounds(
    candidates: pd.DataFrame,
    *,
    minimum_similarity: float = 0.4,
    maximum_similarity: float = 0.8,
    similarity_column: str = "max_reference_similarity",
) -> pd.Index:
    """Alias with bridge-oriented semantics and an explicit configurable window."""

    return reference_neighborhood(
        candidates,
        minimum_similarity=minimum_similarity,
        maximum_similarity=maximum_similarity,
        similarity_column=similarity_column,
    )
