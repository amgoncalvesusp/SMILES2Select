"""Coordinate cache and method selection.

A projection is expensive and must not be recomputed every time the workspace
opens. Coordinates are keyed by (method, representation, parameters, data), so
a cached map is only reused when it would have been identical anyway.

Also picks the default method by library size, following the scale bands in the
specification, and always says which method it chose and why.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from smiles2select.chemical_space import pca_projection, tmap_projection, umap_projection
from smiles2select.chemical_space.pca_projection import Projection

#: Scale bands from the specification.
SMALL_LIBRARY = 50_000
LARGE_LIBRARY = 250_000


@dataclass(frozen=True)
class MethodChoice:
    """Which projection to use, and the reason to show the user."""

    method: str
    reason: str
    alternatives: tuple[str, ...] = ()


def available_methods() -> tuple[str, ...]:
    """Projection methods this installation can actually run."""
    methods = ["pca"]
    if umap_projection.is_available():
        methods.append("umap")
    if tmap_projection.is_available():
        methods.append("tmap")
    return tuple(methods)


def choose_method(molecule_count: int) -> MethodChoice:
    """Default method for a library of this size.

    TMAP is preferred above 50k because it was designed for that scale; when it
    is not installed the choice degrades to PCA *explicitly*, rather than
    silently attempting a layout that would not finish.
    """
    methods = available_methods()
    if molecule_count <= SMALL_LIBRARY:
        return MethodChoice(
            "pca",
            f"{molecule_count} molecules: PCA is deterministic and immediate at this scale",
            tuple(method for method in methods if method != "pca"),
        )

    if "tmap" in methods:
        band = "50 mil a 250 mil" if molecule_count <= LARGE_LIBRARY else "acima de 250 mil"
        return MethodChoice(
            "tmap", f"{band} molecules: TMAP was designed for this scale", methods
        )

    return MethodChoice(
        "pca",
        (
            f"{molecule_count} molecules: TMAP would be the default at this scale, but it is "
            "not installed; using deterministic PCA"
        ),
        methods,
    )


def cache_key(
    method: str, features: Sequence[str], parameters: dict[str, object] | None, frame: pd.DataFrame
) -> str:
    """Hash of the method, its settings and the exact values projected."""
    digest = hashlib.sha256()
    digest.update(method.encode("utf-8"))
    digest.update(json.dumps(sorted(features), default=str).encode("utf-8"))
    digest.update(json.dumps(parameters or {}, sort_keys=True, default=str).encode("utf-8"))
    usable = [feature for feature in features if feature in frame.columns]
    if usable:
        values = frame[usable].to_numpy(dtype=float, na_value=np.nan)
        digest.update(np.ascontiguousarray(values).tobytes())
    digest.update(str(list(frame.index)).encode("utf-8"))
    return digest.hexdigest()


class ProjectionCache:
    """In-memory cache of computed projections."""

    def __init__(self, size: int = 3) -> None:
        self._entries: dict[str, Projection] = {}
        self._order: list[str] = []
        self._size = size

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str) -> Projection | None:
        return self._entries.get(key)

    def put(self, key: str, projection: Projection) -> Projection:
        self._entries[key] = projection
        self._order.append(key)
        while len(self._order) > self._size:
            self._entries.pop(self._order.pop(0), None)
        return projection

    def project_descriptors(
        self, descriptors: pd.DataFrame, features: Sequence[str] | None = None
    ) -> Projection:
        """PCA over descriptors, reusing the cached map when nothing changed."""
        chosen = tuple(features or pca_projection.DEFAULT_FEATURES)
        key = cache_key("pca", chosen, None, descriptors)
        cached = self.get(key)
        if cached is not None:
            return cached
        return self.put(key, pca_projection.project(descriptors, chosen))

    def clear(self) -> None:
        self._entries.clear()
        self._order.clear()


def load_coordinates(rows: Sequence[dict[str, object]]) -> pd.DataFrame:
    """Rebuild a coordinate frame from stored rows.

    Coordinates come back as float32: the map only needs display precision, and
    halving the array keeps a large library comfortably in memory.
    """
    if not rows:
        return pd.DataFrame(columns=["x", "y", "cluster_id"])
    frame = pd.DataFrame(rows).set_index("record_id")
    frame["x"] = frame["x"].astype(np.float32)
    frame["y"] = frame["y"].astype(np.float32)
    if "cluster_id" in frame.columns:
        frame["cluster_id"] = frame["cluster_id"].astype("Int64")
    return frame
