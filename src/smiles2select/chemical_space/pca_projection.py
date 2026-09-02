"""PCA projection.

The default map because it is *deterministic*: the same molecules and the same
descriptors always give the same picture, and the axes carry a stated share of
the variance. Neighbourhood methods produce prettier islands but different ones
on every parameter change, which is a poor foundation for a decision someone
has to defend later.

Implemented with a plain SVD so the projection adds no dependency and cannot
drift between library versions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Descriptors used when the caller does not choose.
DEFAULT_FEATURES = (
    "mol_wt",
    "rdkit_wlogp",
    "tpsa",
    "hbd_lipinski",
    "hba_lipinski",
    "rotatable_bonds",
    "fraction_csp3",
    "qed",
)


@dataclass(frozen=True)
class Projection:
    """2D coordinates plus everything needed to reproduce them."""

    coordinates: pd.DataFrame
    method: str
    features: tuple[str, ...]
    explained_variance: tuple[float, ...] = ()
    parameters: dict[str, object] | None = None
    imputed: pd.Series | None = None

    @property
    def projection_id(self) -> str:
        """Stable id for the ``chemical_space_coordinates`` table."""
        return f"{self.method}::{'-'.join(self.features)}"

    def rows_for_storage(self, clusters: pd.Series | None = None) -> list[dict[str, object]]:
        assigned = clusters.reindex(self.coordinates.index) if clusters is not None else None
        return [
            {
                "record_id": int(record_id),
                "x": float(row.x),
                "y": float(row.y),
                "cluster_id": (
                    None
                    if assigned is None or pd.isna(assigned.loc[record_id])
                    else int(assigned.loc[record_id])
                ),
            }
            for record_id, row in self.coordinates.iterrows()
        ]

    def describe(self) -> list[tuple[str, object]]:
        rows: list[tuple[str, object]] = [
            ("method", self.method),
            ("descriptors", ", ".join(self.features)),
            ("molecules", len(self.coordinates)),
        ]
        if self.explained_variance:
            rows.append(("explained variance", f"{sum(self.explained_variance) * 100:.1f}%"))
        if self.imputed is not None and int(self.imputed.sum()) > 0:
            rows.append(("moléculas com descritor imputado", int(self.imputed.sum())))
        return rows


def standardise(frame: pd.DataFrame) -> tuple[np.ndarray, pd.Index, pd.Series]:
    """Z-score the descriptors, dropping rows that have none.

    Without standardising, molecular weight (hundreds) would dominate every
    axis and QED (0-1) would never influence the map at all.
    """
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    usable = numeric.dropna(how="all")
    if usable.empty:
        return (
            np.zeros((0, numeric.shape[1])),
            usable.index,
            pd.Series(dtype=bool, index=usable.index),
        )

    imputed = usable.isna().any(axis=1)
    filled = usable.fillna(usable.mean())
    values = filled.to_numpy(dtype=float)
    centred = values - values.mean(axis=0)
    spread = centred.std(axis=0)
    spread[spread == 0] = 1.0
    return centred / spread, usable.index, imputed


def project(
    descriptors: pd.DataFrame, features: Sequence[str] | None = None, components: int = 2
) -> Projection:
    """Project descriptors onto their first principal components."""
    requested = tuple(features or DEFAULT_FEATURES)
    chosen = tuple(feature for feature in requested if feature in descriptors.columns)
    if not chosen:
        raise ValueError(f"none of the requested descriptors are available: {list(requested)}")

    matrix, index, imputed = standardise(descriptors[list(chosen)])
    if matrix.shape[0] == 0:
        return Projection(
            pd.DataFrame(columns=["x", "y"], index=index),
            "pca",
            chosen,
            (),
            imputed=imputed,
        )

    # Sign convention: the largest-magnitude loading of each component is made
    # positive, so the same data never produces a mirrored map.
    _, singular, right = np.linalg.svd(matrix, full_matrices=False)
    wanted = min(components, right.shape[0])
    directions = right[:wanted].copy()
    for row in range(wanted):
        if directions[row][np.argmax(np.abs(directions[row]))] < 0:
            directions[row] = -directions[row]

    scores = matrix @ directions.T
    variance = (singular**2) / max(matrix.shape[0] - 1, 1)
    total_variance = variance.sum()
    explained = (
        tuple(float(value / total_variance) for value in variance[:wanted])
        if total_variance > 0
        else ()
    )

    coordinates = pd.DataFrame(
        {
            "x": scores[:, 0].astype(np.float32),
            "y": (
                scores[:, 1].astype(np.float32)
                if wanted > 1
                else np.zeros(len(index), dtype=np.float32)
            ),
        },
        index=index,
    )
    return Projection(
        coordinates, "pca", chosen, explained, {"components": wanted}, imputed=imputed
    )
