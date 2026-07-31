"""TMAP projection (optional).

TMAP was built for exactly the case this tool hits at scale: very many
high-dimensional points laid out as a tree rather than a cloud, so dense
regions stay legible instead of collapsing into one blob.

Optional dependency. When it is missing the caller is told so and pointed at
PCA, which always works.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from smiles2select.chemical_space.pca_projection import Projection
from smiles2select.chemistry.fingerprints import FingerprintConfig, fingerprints_from_smiles


class TmapUnavailableError(RuntimeError):
    """Raised when tmap is not installed."""


@dataclass(frozen=True)
class TmapParameters:
    """Layout parameters, recorded with the coordinates."""

    lsh_bits: int = 1024
    permutations: int = 512
    k: int = 20
    node_size: float = 1 / 37

    def as_dict(self) -> dict[str, object]:
        return {
            "lsh_bits": self.lsh_bits,
            "permutations": self.permutations,
            "k": self.k,
            "node_size": self.node_size,
        }


def _tmap_module():
    try:
        import tmap
    except ImportError as exc:
        raise TmapUnavailableError(
            "TMAP requires the tmap package: conda install -c tmap tmap. "
            "The PCA map is available and deterministic."
        ) from exc
    return tmap


def project_fingerprints(
    smiles: Sequence[str],
    index: pd.Index,
    parameters: TmapParameters = TmapParameters(),
    fingerprint: FingerprintConfig = FingerprintConfig(),
) -> Projection:
    """TMAP layout over Morgan fingerprints."""
    tmap = _tmap_module()
    vectors, positions = fingerprints_from_smiles(smiles, fingerprint)
    if len(vectors) < 3:
        raise ValueError("TMAP needs at least three parseable molecules")

    encoder = tmap.Minhash(fingerprint.size, 42, parameters.permutations)
    storage = tmap.LSHForest(parameters.lsh_bits, parameters.permutations)
    storage.batch_add(encoder.batch_from_binary_array([list(vector) for vector in vectors]))
    storage.index()

    layout = tmap.LayoutConfiguration()
    layout.k = min(parameters.k, len(vectors) - 1)
    layout.node_size = parameters.node_size
    x, y, _, _, _ = tmap.layout_from_lsh_forest(storage, layout)

    coordinates = pd.DataFrame(
        {"x": np.asarray(x, dtype=np.float32), "y": np.asarray(y, dtype=np.float32)},
        index=index[positions],
    )
    return Projection(
        coordinates=coordinates,
        method="tmap",
        features=(fingerprint.label(),),
        parameters=parameters.as_dict(),
    )


def is_available() -> bool:
    try:
        _tmap_module()
    except TmapUnavailableError:
        return False
    return True
