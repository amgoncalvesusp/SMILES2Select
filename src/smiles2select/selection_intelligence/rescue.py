"""Rescue explorer.

For a molecule the rules rejected, finds structurally similar molecules that
*were* approved. It never generates new structures: everything it shows already
exists in the library.

The point is not to overturn the filter. It is to show a chemist that a
rejected compound has close approved relatives - which usually means the
rejection was about a threshold, not about the chemotype.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFMCS

from smiles2select.selection_intelligence.similarity_index import Neighbour, SimilarityIndex

#: Properties compared side by side with an analogue.
DELTA_PROPERTIES = (
    "mol_wt",
    "rdkit_wlogp",
    "tpsa",
    "hbd_lipinski",
    "hba_lipinski",
    "qed",
    "sa_score",
)

#: MCS is exponential in the worst case; the pair view must stay responsive.
MCS_TIMEOUT_SECONDS = 5

RESCUE_COLUMNS = (
    "Excluded_ID",
    "Approved_Analog_ID",
    "Tanimoto",
    "Failed_Rules",
    "Property_Deltas",
    "Review_Status",
)


@dataclass(frozen=True)
class AnalogueComparison:
    """One excluded molecule against one approved analogue."""

    excluded_id: int
    analog_id: int
    tanimoto: float
    failed_rules: tuple[str, ...] = ()
    deltas: dict[str, float] = field(default_factory=dict)
    common_substructure: str | None = None
    review_status: str = "pendente"

    def describe_deltas(self) -> str:
        return "; ".join(f"Δ{name}={value:+.2f}" for name, value in self.deltas.items())

    def as_row(self) -> dict[str, object]:
        return {
            "Excluded_ID": self.excluded_id,
            "Approved_Analog_ID": self.analog_id,
            "Tanimoto": self.tanimoto,
            "Failed_Rules": ", ".join(self.failed_rules),
            "Property_Deltas": self.describe_deltas(),
            "Review_Status": self.review_status,
        }


def property_deltas(
    descriptors: pd.DataFrame, excluded_id: int, analog_id: int
) -> dict[str, float]:
    """Analogue minus excluded, for the properties both molecules have."""
    deltas: dict[str, float] = {}
    for name in DELTA_PROPERTIES:
        if name not in descriptors.columns:
            continue
        if excluded_id not in descriptors.index or analog_id not in descriptors.index:
            continue
        first = descriptors.at[excluded_id, name]
        second = descriptors.at[analog_id, name]
        if pd.isna(first) or pd.isna(second):
            continue
        deltas[name] = round(float(second) - float(first), 3)
    return deltas


def common_substructure(
    first_smiles: str, second_smiles: str, timeout: int = MCS_TIMEOUT_SECONDS
) -> str | None:
    """Maximum common substructure of one pair, computed on demand.

    Only ever for the pair the user is looking at, never precomputed for every
    pair, and always under a timeout: MCS can run for an unbounded time on
    large flexible molecules.
    """
    first = Chem.MolFromSmiles(first_smiles or "")
    second = Chem.MolFromSmiles(second_smiles or "")
    if first is None or second is None:
        return None
    try:
        result = rdFMCS.FindMCS([first, second], timeout=timeout)
    except Exception:
        return None
    if result.canceled or not result.smartsString:
        return None
    return result.smartsString


def _failed_rules(failures: pd.DataFrame | None, record_id: int) -> tuple[str, ...]:
    if failures is None or failures.empty:
        return ()
    rows = failures[failures["record_id"] == record_id]
    return tuple(sorted(set(rows["failure_code"])))


def find_analogues(
    excluded_id: int,
    smiles: str,
    index: SimilarityIndex,
    descriptors: pd.DataFrame,
    failures: pd.DataFrame | None = None,
    top: int = 10,
    minimum_similarity: float = 0.70,
) -> list[AnalogueComparison]:
    """Approved analogues of one rejected molecule, best first."""
    neighbours: list[Neighbour] = index.query(
        smiles, top=top, minimum_similarity=minimum_similarity, exclude_ids=[excluded_id]
    )
    broken = _failed_rules(failures, excluded_id)
    return [
        AnalogueComparison(
            excluded_id=excluded_id,
            analog_id=neighbour.record_id,
            tanimoto=neighbour.tanimoto,
            failed_rules=broken,
            deltas=property_deltas(descriptors, excluded_id, neighbour.record_id),
        )
        for neighbour in neighbours
    ]


def neighbourhood(comparisons: Sequence[AnalogueComparison]) -> pd.DataFrame:
    """Edges of the small neighbour graph: excluded molecule at the centre."""
    if not comparisons:
        return pd.DataFrame(columns=["source", "target", "tanimoto", "distance"])
    return pd.DataFrame(
        [
            {
                "source": item.excluded_id,
                "target": item.analog_id,
                "tanimoto": item.tanimoto,
                # Visual distance: similar molecules are drawn closer.
                "distance": round(1.0 - item.tanimoto, 4),
            }
            for item in comparisons
        ]
    )


def rescue_table(comparisons: Sequence[AnalogueComparison]) -> pd.DataFrame:
    """Rows for the RESCUE_ANALYSIS sheet."""
    if not comparisons:
        return pd.DataFrame(columns=list(RESCUE_COLUMNS))
    return pd.DataFrame([item.as_row() for item in comparisons])


def build_index(
    descriptors: pd.DataFrame,
    approved_ids: Sequence[int],
    smiles_column: str = "canonical_smiles",
) -> SimilarityIndex:
    """Index over the approved molecules only.

    Searching only the approved set is the point: an analogue that was itself
    rejected explains nothing.
    """
    approved = descriptors.loc[descriptors.index.intersection(pd.Index(approved_ids))]
    return SimilarityIndex(approved[smiles_column].fillna("").tolist(), approved.index)
