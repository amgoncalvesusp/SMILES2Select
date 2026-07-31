"""Similarity search over the approved set.

Two stages, deliberately: an approximate pass to gather candidates, then an
exact Tanimoto pass to order them. The neighbours a chemist looks at are always
ranked by the real chemical metric - approximation is allowed to decide *what
to look at*, never *what is closest*.

``hnswlib`` accelerates the first stage when installed. Without it the exact
pass runs alone, which is slower on very large sets but never less correct.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import DataStructs

from smiles2select.chemistry.fingerprints import FingerprintConfig, fingerprints_from_smiles

#: Candidates pulled from the approximate stage before exact re-ranking.
DEFAULT_CANDIDATES = 200

#: Below this, brute force is already instant and a graph is pure overhead.
_APPROXIMATE_THRESHOLD = 100


@dataclass(frozen=True)
class Neighbour:
    """One approved molecule near the query."""

    record_id: int
    tanimoto: float


class SimilarityIndex:
    """Fingerprint index over a fixed set of molecules."""

    def __init__(
        self,
        smiles: Sequence[str],
        index: pd.Index,
        config: FingerprintConfig = FingerprintConfig(),
    ) -> None:
        self.config = config
        vectors, positions = fingerprints_from_smiles(smiles, config)
        self._vectors = vectors
        self._ids = [int(index[position]) for position in positions]
        self._approximate = self._build_approximate()

    def __len__(self) -> int:
        return len(self._vectors)

    @property
    def uses_approximate_stage(self) -> bool:
        return self._approximate is not None

    def _build_approximate(self):
        """Optional HNSW graph over the bit vectors."""
        if len(self._vectors) < _APPROXIMATE_THRESHOLD:
            return None
        try:
            import hnswlib
        except ImportError:
            return None

        matrix = np.array([list(vector) for vector in self._vectors], dtype=np.float32)
        graph = hnswlib.Index(space="cosine", dim=matrix.shape[1])
        graph.init_index(max_elements=matrix.shape[0], ef_construction=200, M=16)
        graph.add_items(matrix, np.arange(matrix.shape[0]))
        graph.set_ef(64)
        return graph

    def _candidate_positions(self, vector, candidates: int) -> Sequence[int]:
        if self._approximate is None:
            return range(len(self._vectors))
        query = np.array([list(vector)], dtype=np.float32)
        labels, _ = self._approximate.knn_query(query, k=min(candidates, len(self._vectors)))
        return [int(position) for position in labels[0]]

    def query(
        self,
        smiles: str,
        top: int = 10,
        minimum_similarity: float = 0.70,
        candidates: int = DEFAULT_CANDIDATES,
        exclude_ids: Sequence[int] = (),
    ) -> list[Neighbour]:
        """Nearest approved molecules, ordered by exact Tanimoto.

        The query molecule is excluded by id, not by similarity: a duplicate
        structure elsewhere in the library is a legitimate neighbour, while the
        molecule itself is not.
        """
        vectors, _ = fingerprints_from_smiles([smiles], self.config)
        if not vectors or not self._vectors:
            return []

        blocked = set(exclude_ids)
        positions = [
            position
            for position in self._candidate_positions(vectors[0], candidates)
            if self._ids[position] not in blocked
        ]
        if not positions:
            return []

        similarities = DataStructs.BulkTanimotoSimilarity(
            vectors[0], [self._vectors[position] for position in positions]
        )
        found = [
            Neighbour(self._ids[position], round(float(score), 4))
            for position, score in zip(positions, similarities, strict=True)
            if score >= minimum_similarity
        ]
        return sorted(found, key=lambda item: item.tanimoto, reverse=True)[:top]

    def settings(self) -> list[tuple[str, object]]:
        """What the search used, for the report."""
        return [
            ("fingerprint", self.config.label()),
            ("métrica final", "Tanimoto exato"),
            ("busca aproximada", "HNSW" if self.uses_approximate_stage else "exaustiva"),
            ("moléculas indexadas", len(self._vectors)),
        ]
