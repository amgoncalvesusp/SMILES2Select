"""Chemistry workers.

These functions run in joblib worker processes, so they must be importable at
module level and take only picklable arguments. Expensive per-process objects
(descriptor registry, filter catalogues) are built once per worker and cached,
not rebuilt per molecule.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from rdkit import Chem

from smiles2select.alerts.custom_smarts import CustomSmartsCatalog, SmartsAlert
from smiles2select.alerts.engine import AlertEngine
from smiles2select.alerts.policies import AlertPolicy
from smiles2select.chemistry.descriptor_registry import DescriptorRegistry, default_registry
from smiles2select.chemistry.parsing import canonical_smiles, parse_smiles
from smiles2select.chemistry.standardization import StandardizationConfig, standardize


@dataclass(frozen=True)
class RecordResult:
    """Per-molecule output of one worker call."""

    record_id: int
    valid: bool
    invalid_reason: str | None
    standardized_smiles: str
    canonical_smiles: str
    descriptors: dict[str, Any]
    substructure_flags: dict[str, bool]
    alert_rows: list[dict[str, Any]]


@lru_cache(maxsize=1)
def _registry() -> DescriptorRegistry:
    return default_registry()


@lru_cache(maxsize=8)
def _alert_engine(
    catalog_ids: tuple[str, ...],
    custom_alerts: tuple[tuple[str, str, str, str], ...],
) -> AlertEngine | None:
    """Build (once per worker) the alert engine for these catalogues."""
    if not catalog_ids and not custom_alerts:
        return None
    custom_catalog = (
        CustomSmartsCatalog(
            SmartsAlert(id=item[0], name=item[1], smarts=item[2], description=item[3])
            for item in custom_alerts
        )
        if custom_alerts
        else None
    )
    return AlertEngine(catalog_ids, custom_catalog, AlertPolicy())


@lru_cache(maxsize=64)
def _compiled_smarts(smarts: str) -> Chem.Mol | None:
    return Chem.MolFromSmarts(smarts)


def process_record(
    record_id: int,
    smiles: str,
    descriptor_ids: tuple[str, ...],
    standardization: StandardizationConfig,
    catalog_ids: tuple[str, ...] = (),
    custom_alerts: tuple[tuple[str, str, str, str], ...] = (),
    substructure_rules: tuple[tuple[str, str], ...] = (),
) -> RecordResult:
    """Parse, standardize, describe and scan one molecule.

    Invalid input produces a result with ``valid=False`` and a reason, never an
    exception: one unparseable row must not abort the batch.
    """
    parsed = parse_smiles(smiles)
    if not parsed.valid or parsed.mol is None:
        return RecordResult(record_id, False, parsed.error or "invalid SMILES", "", "", {}, {}, [])

    standardized = standardize(parsed.mol, standardization)
    if not standardized.valid or standardized.mol is None:
        return RecordResult(
            record_id, False, standardized.error or "standardization failed", "", "", {}, {}, []
        )

    mol = standardized.mol
    descriptors = _registry().compute(mol, descriptor_ids)

    flags: dict[str, bool] = {}
    for rule_id, smarts in substructure_rules:
        pattern = _compiled_smarts(smarts)
        flags[f"smarts__{rule_id}"] = bool(pattern is not None and mol.HasSubstructMatch(pattern))

    engine = _alert_engine(catalog_ids, custom_alerts)
    alert_rows = engine.scan_to_rows(record_id, mol) if engine is not None else []

    return RecordResult(
        record_id=record_id,
        valid=True,
        invalid_reason=None,
        standardized_smiles=standardized.standardized_smiles,
        canonical_smiles=canonical_smiles(mol),
        descriptors=descriptors,
        substructure_flags=flags,
        alert_rows=alert_rows,
    )


def process_chunk(
    chunk: Sequence[tuple[int, str]],
    descriptor_ids: tuple[str, ...],
    standardization: StandardizationConfig,
    catalog_ids: tuple[str, ...] = (),
    custom_alerts: tuple[tuple[str, str, str, str], ...] = (),
    substructure_rules: tuple[tuple[str, str], ...] = (),
) -> list[RecordResult]:
    """Process a batch of records in one worker call.

    Batching amortises the inter-process handoff: sending 2000 molecules at
    once costs one pickle round trip instead of 2000.
    """
    return [
        process_record(
            record_id,
            smiles,
            descriptor_ids,
            standardization,
            catalog_ids,
            custom_alerts,
            substructure_rules,
        )
        for record_id, smiles in chunk
    ]


def alerts_to_tuples(alerts: Sequence[SmartsAlert]) -> tuple[tuple[str, str, str, str], ...]:
    """Make custom alerts hashable so workers can cache the catalogue."""
    return tuple((alert.id, alert.name, alert.smarts, alert.description) for alert in alerts)
