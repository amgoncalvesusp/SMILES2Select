"""User-supplied SMARTS alerts.

Patterns are compiled once and validated up front: an unparseable SMARTS is
reported before the run starts, not discovered as a silent zero-hit pattern
after the library has been scanned.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from rdkit import Chem


@dataclass(frozen=True)
class SmartsAlert:
    """One named substructure pattern."""

    id: str
    name: str
    smarts: str
    description: str = ""

    def compile_pattern(self) -> Chem.Mol:
        pattern = Chem.MolFromSmarts(self.smarts)
        if pattern is None:
            raise ValueError(f"alert '{self.id}': invalid SMARTS '{self.smarts}'")
        return pattern


class CustomSmartsCatalog:
    """A catalogue of user patterns, compiled once."""

    def __init__(self, alerts: Iterable[SmartsAlert], catalog_id: str = "custom_smarts") -> None:
        self.catalog_id = catalog_id
        self._alerts = tuple(alerts)
        self._patterns = {alert.id: alert.compile_pattern() for alert in self._alerts}

    def __len__(self) -> int:
        return len(self._alerts)

    @property
    def alerts(self) -> tuple[SmartsAlert, ...]:
        return self._alerts

    def match(self, mol: Chem.Mol) -> list[tuple[str, str, int]]:
        """``(alert_name, description, occurrence_count)`` for every matching pattern."""
        hits: list[tuple[str, str, int]] = []
        for alert in self._alerts:
            matches = mol.GetSubstructMatches(self._patterns[alert.id], uniquify=True)
            if matches:
                hits.append((alert.name, alert.description, len(matches)))
        return hits


def alerts_from_dicts(payload: Iterable[dict[str, str]]) -> list[SmartsAlert]:
    """Build alerts from JSON-shaped dicts, failing loudly on bad SMARTS."""
    alerts: list[SmartsAlert] = []
    for index, item in enumerate(payload):
        alert = SmartsAlert(
            id=item.get("id") or f"custom_{index + 1}",
            name=item.get("name") or item.get("id") or f"custom_{index + 1}",
            smarts=item["smarts"],
            description=item.get("description", ""),
        )
        alert.compile_pattern()  # validate now, not mid-run
        alerts.append(alert)
    return alerts
