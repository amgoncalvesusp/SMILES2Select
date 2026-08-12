"""Reference-aware nearest-neighbour similarity.

Exact mode is chunk-friendly in memory: it computes one candidate against the
packed reference vectors at a time and never materializes a candidate x
reference similarity matrix.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs

from smiles2select.chemistry.fingerprints import FingerprintConfig, fingerprint
from smiles2select.reference.libraries import ReferenceLibrary


class SimilaritySearchUnavailable(RuntimeError):
    """Raised when an explicitly requested optional search backend is absent."""


@dataclass(frozen=True)
class ReferenceSimilarityResult:
    """Nearest reference annotation plus the exact method metadata."""

    annotations: pd.DataFrame
    pairs: pd.DataFrame
    fingerprint: FingerprintConfig
    search_mode: str

    @property
    def method_description(self) -> str:
        return f"{self.search_mode}; {self.fingerprint.label()}; Tanimoto"


def _prepare_libraries(
    references: Mapping[str, pd.DataFrame] | Iterable[ReferenceLibrary],
) -> list[ReferenceLibrary]:
    if isinstance(references, Mapping):
        from smiles2select.reference.libraries import prepare_library_frame

        return [
            value
            if isinstance(value, ReferenceLibrary)
            else prepare_library_frame(value, str(library_id))
            for library_id, value in references.items()
        ]
    return list(references)


def _candidate_columns(candidates: pd.DataFrame, smiles_column: str, id_column: str) -> tuple[str, str]:
    if smiles_column in candidates.columns:
        smiles_name = smiles_column
    elif "smiles" in candidates.columns:
        smiles_name = "smiles"
    else:
        raise KeyError(f"candidate SMILES column '{smiles_column}' not found")
    identifier = id_column if id_column in candidates.columns else "record_id"
    return smiles_name, identifier


def _reference_vectors(libraries: list[ReferenceLibrary], config: FingerprintConfig):
    prepared: list[tuple[ReferenceLibrary, list[object], list[object]]] = []
    for library in libraries:
        vectors: list[object] = []
        row_indices: list[object] = []
        for row_index, row in library.valid.iterrows():
            mol = Chem.MolFromSmiles(str(row["canonical_smiles"]))
            if mol is None:
                continue
            vectors.append(fingerprint(mol, config))
            row_indices.append(row_index)
        if vectors:
            prepared.append((library, vectors, row_indices))
    return prepared


class _HnswReferenceIndex:
    """Approximate discovery index with exact Tanimoto re-ranking.

    HNSW does not expose a Tanimoto space.  It is therefore used only to find
    a small pool of bit-vector neighbours using cosine distance; the returned
    pool is always re-ranked with RDKit's exact Tanimoto implementation before
    it is reported.  This keeps the approximation explicit in provenance and
    prevents an approximate score from leaking into the scientific result.
    """

    def __init__(self, prepared, config: FingerprintConfig) -> None:
        import hnswlib

        self._prepared = prepared
        self._vectors = [vector for _library, vectors, _indices in prepared for vector in vectors]
        self._locations = [
            (library, row_index)
            for library, _vectors, indices in prepared
            for row_index in indices
        ]
        matrix = np.zeros((len(self._vectors), config.size), dtype=np.float32)
        for position, vector in enumerate(self._vectors):
            DataStructs.ConvertToNumpyArray(vector, matrix[position])
        self._graph = hnswlib.Index(space="cosine", dim=config.size)
        self._graph.init_index(
            max_elements=len(self._vectors),
            ef_construction=200,
            M=16,
        )
        self._graph.add_items(matrix, np.arange(len(self._vectors)))
        self._graph.set_ef(min(len(self._vectors), 128))

    def positions(self, query, limit: int) -> list[int]:
        query_array = np.zeros((1, query.GetNumBits()), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(query, query_array[0])
        labels, _ = self._graph.knn_query(
            query_array, k=min(max(1, limit), len(self._vectors))
        )
        return [int(position) for position in labels[0]]


def compute_reference_similarity(
    candidates: pd.DataFrame,
    references: Mapping[str, pd.DataFrame] | Iterable[ReferenceLibrary],
    *,
    candidate_smiles_column: str = "canonical_smiles",
    candidate_id_column: str = "molecule_id",
    fingerprint_config: FingerprintConfig = FingerprintConfig(),
    search: str = "exact",
    chunk_size: int = 1024,
) -> ReferenceSimilarityResult:
    """Compute nearest-reference similarity and ``reference_novelty``.

    ``chunk_size`` controls progress granularity for callers.  In ``fast`` mode
    HNSW discovers a bounded neighbour pool and RDKit re-ranks that pool using
    exact Tanimoto similarity.  Fast mode is rejected when the optional backend
    is not installed rather than silently falling back to a different method.
    """

    if search not in {"exact", "fast"}:
        raise ValueError("search must be 'exact' or 'fast'")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    hnsw_index = None
    if search == "fast":
        try:
            import hnswlib  # noqa: F401
        except ImportError as exc:
            raise SimilaritySearchUnavailable(
                "fast reference search requires the optional 'hnswlib' dependency"
            ) from exc
    if not isinstance(candidates, pd.DataFrame):
        raise TypeError("candidates must be a pandas DataFrame")

    smiles_name, id_name = _candidate_columns(
        candidates, candidate_smiles_column, candidate_id_column
    )
    reference_vectors = _reference_vectors(_prepare_libraries(references), fingerprint_config)
    if search == "fast" and reference_vectors:
        hnsw_index = _HnswReferenceIndex(reference_vectors, fingerprint_config)
    # A bounded pool keeps the fast path predictable while retaining enough
    # neighbours for exact re-ranking.  The number is recorded indirectly by
    # the method label; it is not a scientific threshold.
    discovery_limit = min(200, max(32, int(chunk_size)))
    rows: list[dict[str, object]] = []
    pairs: list[dict[str, object]] = []
    for row_index, candidate in candidates.iterrows():
        mol = Chem.MolFromSmiles(str(candidate[smiles_name])) if pd.notna(candidate[smiles_name]) else None
        best_similarity = -1.0
        best_library: str | None = None
        best_reference_id: object = None
        best_reference_index: object = None
        if mol is not None:
            candidate_vector = fingerprint(mol, fingerprint_config)
            if hnsw_index is None:
                locations = [
                    (library, vector, reference_index)
                    for library, vectors, reference_indices in reference_vectors
                    for vector, reference_index in zip(vectors, reference_indices, strict=True)
                ]
            else:
                locations = [
                    (
                        hnsw_index._locations[position][0],
                        hnsw_index._vectors[position],
                        hnsw_index._locations[position][1],
                    )
                    for position in hnsw_index.positions(candidate_vector, discovery_limit)
                ]
            if locations:
                similarities = DataStructs.BulkTanimotoSimilarity(
                    candidate_vector, [vector for _library, vector, _index in locations]
                )
                best_position = max(range(len(similarities)), key=similarities.__getitem__)
                best_library, _vector, reference_index = locations[best_position]
                best_similarity = float(similarities[best_position])
                best_library = best_library.library_id
                best_reference_index = reference_index
                best_reference_id = next(
                    library.molecules.loc[reference_index, "reference_id"]
                    for library, _vector, row_index in locations
                    if library.library_id == best_library and row_index == reference_index
                )
        valid_neighbour = best_library is not None
        rows.append(
            {
                "record_id": row_index,
                "candidate_id": candidate.get(id_name, row_index),
                "nearest_reference_id": best_reference_id,
                "nearest_reference_library": best_library,
                "nearest_reference_index": best_reference_index,
                "max_reference_similarity": best_similarity if valid_neighbour else float("nan"),
                "reference_novelty": (1.0 - best_similarity) if valid_neighbour else float("nan"),
                "reference_search": (
                    "HNSW discovery + exact Tanimoto verification"
                    if hnsw_index is not None
                    else "Exact search"
                ),
                "fingerprint": fingerprint_config.label(),
            }
        )
        if valid_neighbour:
            pairs.append(
                {
                    "record_id": row_index,
                    "candidate_id": candidate.get(id_name, row_index),
                    "reference_library": best_library,
                    "reference_index": best_reference_index,
                    "reference_id": best_reference_id,
                    "similarity": best_similarity,
                }
            )

    annotation_frame = pd.DataFrame(rows).set_index("record_id")
    pair_frame = pd.DataFrame(pairs)
    if pair_frame.empty:
        pair_frame = pd.DataFrame(
            columns=[
                "record_id",
                "candidate_id",
                "reference_library",
                "reference_index",
                "reference_id",
                "similarity",
            ]
        )
    return ReferenceSimilarityResult(
        annotations=annotation_frame,
        pairs=pair_frame,
        fingerprint=fingerprint_config,
        search_mode=(
            "HNSW discovery + exact Tanimoto verification"
            if hnsw_index is not None
            else "Exact search"
        ),
    )
