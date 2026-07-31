"""Structural alert scanning.

Alerts are evaluated independently of the physicochemical rules and produce
flags, counts and - only where the user asked for it - exclusions. The
distinction is enforced here: :func:`exclusion_mask` is the single place that
can turn an alert into a removal.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import pandas as pd
from rdkit import Chem

from smiles2select.alerts import rdkit_catalogs
from smiles2select.alerts.custom_smarts import CustomSmartsCatalog
from smiles2select.alerts.policies import AlertPolicy

ALERT_COLUMNS = (
    "record_id",
    "catalog_id",
    "alert_name",
    "alert_description",
    "occurrence_count",
    "action",
)


@dataclass(frozen=True)
class AlertHit:
    """One catalogue entry matching one molecule."""

    catalog_id: str
    alert_name: str
    alert_description: str
    occurrence_count: int


class AlertEngine:
    """Runs the selected catalogues over molecules."""

    def __init__(
        self,
        catalog_ids: Sequence[str] = (),
        custom_catalog: CustomSmartsCatalog | None = None,
        policy: AlertPolicy | None = None,
    ) -> None:
        available = rdkit_catalogs.available_catalogs()
        unknown = [catalog_id for catalog_id in catalog_ids if catalog_id not in available]
        if unknown:
            raise rdkit_catalogs.UnknownCatalogError(
                f"unavailable catalog(s): {unknown}; available: {list(available)}"
            )
        self.catalog_ids = tuple(catalog_ids)
        self.custom_catalog = custom_catalog
        self.policy = policy or AlertPolicy()

    @property
    def is_active(self) -> bool:
        return bool(self.catalog_ids) or bool(self.custom_catalog and len(self.custom_catalog))

    def scan(self, mol: Chem.Mol) -> list[AlertHit]:
        """All hits for one molecule across every selected catalogue."""
        hits: list[AlertHit] = []
        for catalog_id in self.catalog_ids:
            for name, description in rdkit_catalogs.match_catalog(mol, catalog_id):
                hits.append(AlertHit(catalog_id, name, description, 1))
        if self.custom_catalog is not None:
            for name, description, count in self.custom_catalog.match(mol):
                hits.append(AlertHit(self.custom_catalog.catalog_id, name, description, count))
        return hits

    def scan_to_rows(self, record_id: int, mol: Chem.Mol) -> list[dict[str, object]]:
        return [
            {
                "record_id": record_id,
                "catalog_id": hit.catalog_id,
                "alert_name": hit.alert_name,
                "alert_description": hit.alert_description,
                "occurrence_count": hit.occurrence_count,
                "action": self.policy.action_for(hit.catalog_id),
            }
            for hit in self.scan(mol)
        ]


def alerts_frame(rows: Iterable[dict[str, object]]) -> pd.DataFrame:
    """Long table of alerts, one row per molecule x catalogue entry."""
    materialised = list(rows)
    if not materialised:
        return pd.DataFrame(columns=list(ALERT_COLUMNS))
    return pd.DataFrame(materialised, columns=list(ALERT_COLUMNS))


def alert_counts(alerts: pd.DataFrame, index: pd.Index) -> pd.Series:
    """Total alerts per record, zero-filled for molecules with none."""
    if alerts.empty:
        return pd.Series(0, index=index, dtype="int64")
    counted = alerts.groupby("record_id")["occurrence_count"].sum()
    return counted.reindex(index).fillna(0).astype("int64")


def counts_by_catalog(alerts: pd.DataFrame, index: pd.Index, catalog_id: str) -> pd.Series:
    """Alert count for one catalogue (e.g. the PAINS_Count export column)."""
    if alerts.empty:
        return pd.Series(0, index=index, dtype="int64")
    subset = alerts[alerts["catalog_id"] == catalog_id]
    if subset.empty:
        return pd.Series(0, index=index, dtype="int64")
    counted = subset.groupby("record_id")["occurrence_count"].sum()
    return counted.reindex(index).fillna(0).astype("int64")


def exclusion_mask(alerts: pd.DataFrame, index: pd.Index, policy: AlertPolicy) -> pd.Series:
    """True where an alert whose action is ``exclude`` was hit.

    This is the only path from an alert to a removal. Catalogues set to
    inform, warn or penalize never appear here.
    """
    excluded = pd.Series(False, index=index)
    if alerts.empty:
        return excluded
    excluding = [
        catalog_id
        for catalog_id in alerts["catalog_id"].unique()
        if policy.excludes(str(catalog_id))
    ]
    if not excluding:
        return excluded
    hit_records = alerts.loc[alerts["catalog_id"].isin(excluding), "record_id"].unique()
    excluded.loc[excluded.index.isin(hit_records)] = True
    return excluded
