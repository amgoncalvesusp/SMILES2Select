"""UMAP projection (optional).

Offered, not default. UMAP arranges neighbourhoods well but its output depends
on parameters and on a random seed, so two runs can tell different visual
stories about the same library. Every result therefore carries its parameters
and its seed, and the interface shows the proximity disclaimer next to it.

The dependency is optional: without ``umap-learn`` installed the caller is told
to use PCA rather than being handed a silently different map.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from smiles2select.chemical_space.pca_projection import Projection, standardise
from smiles2select.chemistry.fingerprints import FingerprintConfig, fingerprints_from_smiles


class UmapUnavailableError(RuntimeError):
    """Raised when umap-learn is not installed."""


@dataclass(frozen=True)
class UmapParameters:
    """Everything that changes the shape of the result."""

    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "jaccard"
    seed: int = 0xF00D

    def as_dict(self) -> dict[str, object]:
        return {
            "n_neighbors": self.n_neighbors,
            "min_dist": self.min_dist,
            "metric": self.metric,
            "seed": self.seed,
        }


def _umap_module():
    try:
        import umap
    except ImportError as exc:
        raise UmapUnavailableError(
            "UMAP requires umap-learn: pip install umap-learn. "
            "The PCA map is available and deterministic."
        ) from exc
    return umap


def project_fingerprints(
    smiles: Sequence[str],
    index: pd.Index,
    parameters: UmapParameters = UmapParameters(),
    fingerprint: FingerprintConfig = FingerprintConfig(),
) -> Projection:
    """Structural UMAP over Morgan fingerprints."""
    umap = _umap_module()
    vectors, positions = fingerprints_from_smiles(smiles, fingerprint)
    if len(vectors) < 3:
        raise ValueError("UMAP needs at least three parseable molecules")

    matrix = np.array([list(vector) for vector in vectors], dtype=np.float32)
    reducer = umap.UMAP(
        n_neighbors=min(parameters.n_neighbors, len(vectors) - 1),
        min_dist=parameters.min_dist,
        metric=parameters.metric,
        random_state=parameters.seed,
    )
    embedding = reducer.fit_transform(matrix)

    coordinates = pd.DataFrame(
        {"x": embedding[:, 0].astype(np.float32), "y": embedding[:, 1].astype(np.float32)},
        index=index[positions],
    )
    return Projection(
        coordinates=coordinates,
        method="umap",
        features=(fingerprint.label(),),
        parameters=parameters.as_dict(),
    )


def project_descriptors(
    descriptors: pd.DataFrame,
    features: Sequence[str],
    parameters: UmapParameters = UmapParameters(),
) -> Projection:
    """Property-space UMAP over standardised descriptors."""
    requested = tuple(features)
    chosen = tuple(feature for feature in requested if feature in descriptors.columns)
    if not chosen:
        raise ValueError(f"none of the requested descriptors are available: {list(requested)}")

    umap = _umap_module()
    matrix, index, imputed = standardise(descriptors[list(chosen)])
    if matrix.shape[0] < 3:
        raise ValueError("UMAP needs at least three molecules with descriptors")

    reducer = umap.UMAP(
        n_neighbors=min(parameters.n_neighbors, matrix.shape[0] - 1),
        min_dist=parameters.min_dist,
        metric="euclidean",
        random_state=parameters.seed,
    )
    embedding = reducer.fit_transform(matrix)
    coordinates = pd.DataFrame(
        {"x": embedding[:, 0].astype(np.float32), "y": embedding[:, 1].astype(np.float32)},
        index=index,
    )
    return Projection(
        coordinates=coordinates,
        method="umap",
        features=chosen,
        parameters={**parameters.as_dict(), "metric": "euclidean"},
        imputed=imputed,
    )


def is_available() -> bool:
    try:
        _umap_module()
    except UmapUnavailableError:
        return False
    return True
