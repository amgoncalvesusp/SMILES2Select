"""Reproducible projection orchestration for the Chemical Space Hub."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from smiles2select.app_metadata import rdkit_version
from smiles2select.chemical_space import pca_projection, umap_projection
from smiles2select.chemical_space.pca_projection import Projection
from smiles2select.chemistry.fingerprints import FingerprintConfig


@dataclass(frozen=True)
class ProjectionConfig:
    """All settings that can change a chemical-space projection."""

    method: str = "property_pca"
    features: tuple[str, ...] = pca_projection.DEFAULT_FEATURES
    fingerprint: FingerprintConfig = field(default_factory=FingerprintConfig)
    parameters: dict[str, Any] = field(default_factory=dict)
    seed: int = 0xF00D

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "features": list(self.features),
            "fingerprint": {
                "radius": self.fingerprint.radius,
                "bits": self.fingerprint.size,
                "use_chirality": self.fingerprint.use_chirality,
            },
            "parameters": dict(self.parameters),
            "seed": self.seed,
        }


@dataclass(frozen=True)
class ProjectionResult:
    """Coordinates plus the provenance required to reload the same map."""

    projection: Projection
    config: ProjectionConfig
    input_hash: str
    software_versions: dict[str, str]

    def recipe_block(self) -> dict[str, Any]:
        return {
            **self.config.as_dict(),
            "input_hash": self.input_hash,
            "software_versions": dict(self.software_versions),
        }


def input_hash(frame: pd.DataFrame, features: tuple[str, ...]) -> str:
    """Hash projected values and row IDs, not just the feature names."""

    digest = hashlib.sha256()
    digest.update(json.dumps(list(features), sort_keys=True).encode("utf-8"))
    values = frame.reindex(columns=list(features)).copy()
    values = values.apply(pd.to_numeric, errors="coerce")
    digest.update(values.to_numpy(dtype="float64", na_value=float("nan")).tobytes())
    digest.update(json.dumps([str(value) for value in frame.index]).encode("utf-8"))
    return digest.hexdigest()


def project(frame: pd.DataFrame, config: ProjectionConfig = ProjectionConfig()) -> ProjectionResult:
    """Dispatch to the existing projection implementations with provenance."""

    if config.method == "property_pca":
        projection = pca_projection.project(frame, config.features)
    elif config.method == "property_umap":
        parameters = umap_projection.UmapParameters(
            n_neighbors=int(config.parameters.get("n_neighbors", 15)),
            min_dist=float(config.parameters.get("min_dist", 0.1)),
            metric="euclidean",
            seed=config.seed,
        )
        projection = umap_projection.project_descriptors(frame, config.features, parameters)
    elif config.method == "structural_umap":
        if "canonical_smiles" not in frame.columns:
            raise KeyError("structural_umap requires a canonical_smiles column")
        parameters = umap_projection.UmapParameters(
            n_neighbors=int(config.parameters.get("n_neighbors", 15)),
            min_dist=float(config.parameters.get("min_dist", 0.1)),
            metric=str(config.parameters.get("metric", "jaccard")),
            seed=config.seed,
        )
        projection = umap_projection.project_fingerprints(
            frame["canonical_smiles"].fillna("").tolist(), frame.index, parameters, config.fingerprint
        )
    else:
        raise ValueError(
            f"unknown projection method '{config.method}'; expected property_pca, "
            "property_umap or structural_umap"
        )
    return ProjectionResult(
        projection=projection,
        config=config,
        input_hash=input_hash(frame, config.features),
        software_versions={"rdkit": rdkit_version()},
    )
