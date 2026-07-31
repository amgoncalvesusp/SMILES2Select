"""Structural clustering.

Butina clustering over Morgan fingerprints: deterministic and threshold-based,
needing no target number of clusters. That matters here because nobody knows in
advance how many chemotypes a library contains, and forcing a count would
invent structure that is not there.

Cluster membership is what the final-selection quotas and the coverage report
are built on.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina

from smiles2select.chemistry.fingerprints import FingerprintConfig, fingerprints_from_smiles

#: Tanimoto distance below which two molecules join the same cluster.
DEFAULT_CUTOFF = 0.35


@dataclass(frozen=True)
class ClusterResult:
    """Cluster of each molecule, plus the settings that produced it."""

    labels: pd.Series
    cutoff: float
    fingerprint_label: str

    @property
    def cluster_count(self) -> int:
        return int(self.labels.nunique())

    @property
    def singleton_count(self) -> int:
        counts = self.labels.value_counts()
        return int((counts == 1).sum())

    def sizes(self) -> pd.DataFrame:
        counts = self.labels.value_counts().sort_values(ascending=False)
        return pd.DataFrame({"cluster_id": counts.index, "molecules": counts.to_numpy()})

    def summary_rows(self) -> list[tuple[str, object]]:
        return [
            ("moléculas agrupadas", int(self.labels.notna().sum())),
            ("clusters", self.cluster_count),
            ("clusters unitários", self.singleton_count),
            ("corte de distância", self.cutoff),
            ("fingerprint", self.fingerprint_label),
        ]


def cluster(
    smiles: Sequence[str],
    index: pd.Index,
    cutoff: float = DEFAULT_CUTOFF,
    fingerprint: FingerprintConfig = FingerprintConfig(),
) -> ClusterResult:
    """Butina clustering; unparseable molecules get no cluster.

    ponytail: builds the full lower-triangular distance list, which is O(n^2)
    in memory. Fine for a selection-sized set; cluster a sample upstream for a
    whole screening library.
    """
    if not 0 < cutoff < 1:
        raise ValueError("cutoff must lie in (0, 1)")

    vectors, positions = fingerprints_from_smiles(smiles, fingerprint)
    labels = pd.Series(pd.NA, index=index, dtype="Int64")
    if not vectors:
        return ClusterResult(labels, cutoff, fingerprint.label())

    distances: list[float] = []
    for position in range(1, len(vectors)):
        similarities = DataStructs.BulkTanimotoSimilarity(vectors[position], vectors[:position])
        distances.extend(1.0 - value for value in similarities)

    clusters = Butina.ClusterData(distances, len(vectors), cutoff, isDistData=True)
    for cluster_id, members in enumerate(clusters, start=1):
        for member in members:
            labels.iloc[positions[member]] = cluster_id
    return ClusterResult(labels, cutoff, fingerprint.label())


def coverage(labels: pd.Series, selected_ids: Sequence[int]) -> dict[str, float]:
    """How much of the clustered space the selection actually represents."""
    known = labels.dropna()
    total = int(known.nunique())
    if total == 0:
        return {"clusters": 0, "clusters_representados": 0, "cobertura": 0.0}
    represented = int(known.reindex(pd.Index(selected_ids)).dropna().nunique())
    return {
        "clusters": total,
        "clusters_representados": represented,
        "cobertura": round(represented / total, 4),
    }


def unrepresented(labels: pd.Series, selected_ids: Sequence[int]) -> list[int]:
    """Clusters with no molecule in the selection - the coverage gaps."""
    known = labels.dropna()
    covered = set(known.reindex(pd.Index(selected_ids)).dropna().tolist())
    return sorted(int(value) for value in set(known.tolist()) - covered)
