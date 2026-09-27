"""Isolated, record-ID based input from a live SMILES2Select workspace."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace

import pandas as pd

from smiles2select.pipeline.runner import RunResult
from smiles2select.selection_intelligence.basket import SelectionBasket
from smiles2select.selection_intelligence.constrained_selection import (
    SelectionConstraints,
    Strategy,
)
from smiles2select.selection_intelligence.preparation import ensure_selection_scaffolds
from smiles2select.selection_intelligence.scenarios import _reference_mask, data_fingerprint

from .features import featurize
from .schema import FeatureSet


@dataclass(frozen=True)
class SessionSnapshot:
    session_id: str
    revision: str
    original_ids: tuple[int, ...]
    pinned_ids: tuple[int, ...]
    excluded_ids: tuple[int, ...]
    constraints: SelectionConstraints
    _records: pd.DataFrame = field(repr=False, compare=False)
    _records_hash: str = field(repr=False, compare=False)

    def prepared(self, bits: int = 2048) -> FeatureSet:
        """Return fresh model features; never expose mutable snapshot rows."""
        if _frame_hash(self._records) != self._records_hash:
            raise ValueError("Session snapshot rows changed after capture")
        records = self._records.copy(deep=True)
        records.attrs["provenance"] = {
            "input_scope": "workspace_snapshot",
            "session_id": self.session_id,
            "session_revision": self.revision,
            "original_ids": list(self.original_ids),
        }
        return featurize(records, bits)


def _frame_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256(json.dumps(list(frame.columns)).encode())
    digest.update(pd.util.hash_pandas_object(frame.index).values.tobytes())
    for name in frame:
        digest.update(pd.util.hash_pandas_object(frame[name], index=False).values.tobytes())
    return digest.hexdigest()


def snapshot_from_workspace(
    result: RunResult,
    candidates: pd.DataFrame,
    basket: SelectionBasket,
    constraints: SelectionConstraints,
    *,
    session_id: str | None = None,
) -> SessionSnapshot:
    """Freeze the full eligible pool, current basket and native quota labels."""
    if result.zone_allocation is not None:
        raise ValueError("Model proposals are unavailable for runs with zone allocation")
    if not result.descriptors.index.is_unique or not candidates.index.is_unique:
        raise ValueError("Workspace record IDs must be unique")
    if not result.descriptors.index.isin([state.record_id for state in basket.states()]).all():
        raise ValueError("Basket is missing workspace record IDs")
    states = {state.record_id: state for state in basket.states()}
    eligible_ids = [
        record_id
        for record_id, state in states.items()
        if state.chemical_status.passed or (state.pinned and state.is_selected)
    ]
    allowed = _reference_mask(result, pd.Index(eligible_ids))
    eligible = set(allowed.index[allowed]) - set(basket.excluded_ids())
    if not eligible:
        raise ValueError("No eligible molecules remain for a model proposal")
    if not eligible.issubset(candidates.index):
        raise ValueError(f"Eligible records are missing from native candidates: {sorted(eligible - set(candidates.index))}")
    # Selection always needs native cores, even when no quota was requested.
    core_constraints = replace(constraints, max_per_scaffold=constraints.max_per_scaffold or 1)
    prepared = ensure_selection_scaffolds(
        candidates.loc[candidates.index.isin(eligible)].copy(deep=True),
        core_constraints,
        Strategy.BALANCED,
    )
    native = prepared.reindex(result.descriptors.index)
    source = result.descriptors.copy(deep=True)
    source["record_id"] = source.index
    source["eligible"] = source.index.isin(eligible)
    source["is_final"] = source.index.isin(basket.final_ids())
    source["pinned"] = source.index.isin(basket.pinned_ids())
    source["chemical_status"] = [states[int(key)].chemical_status.value for key in source.index]
    source["selection_status"] = [states[int(key)].selection_status.value for key in source.index]
    source["selection_note"] = [states[int(key)].note for key in source.index]
    source["session_scaffold"] = native.get("murcko_scaffold", pd.Series(pd.NA, index=source.index))
    source["cluster_id"] = native.get("cluster_id", pd.Series(pd.NA, index=source.index))
    original_ids = basket.final_ids()
    pinned_ids = basket.pinned_ids()
    excluded_ids = basket.excluded_ids()
    source_hash = data_fingerprint(result, candidates)
    frozen_records = source.reset_index(drop=True)
    records_hash = _frame_hash(frozen_records)
    revision_input = {
        "source_hash": source_hash,
        "records_hash": records_hash,
        "basket": [state.as_row() for state in basket.states()],
        "constraints": asdict(constraints),
        "eligible_ids": sorted(map(int, eligible)),
        "session_scaffold_hash": hashlib.sha256(
            pd.util.hash_pandas_object(prepared["murcko_scaffold"], index=True).values.tobytes()
        ).hexdigest(),
    }
    revision = hashlib.sha256(json.dumps(revision_input, sort_keys=True).encode()).hexdigest()
    return SessionSnapshot(
        session_id=session_id or source_hash,
        revision=revision,
        original_ids=original_ids,
        pinned_ids=pinned_ids,
        excluded_ids=excluded_ids,
        constraints=constraints,
        _records=frozen_records,
        _records_hash=records_hash,
    )


def reject_model_identity_collisions(features: FeatureSet) -> None:
    """Model standardization must not merge independently eligible records."""
    pool = features.records.loc[features.records.eligible & features.records.valid]
    duplicate = pool.loc[pool.identity.duplicated(keep=False), ["record_id", "identity"]]
    if not duplicate.empty:
        groups = duplicate.groupby("identity").record_id.apply(lambda ids: sorted(map(int, ids)))
        raise ValueError(f"Model identity collisions among record IDs: {groups.tolist()}")
