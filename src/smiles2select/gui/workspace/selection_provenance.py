"""Immutable applied criteria, detached from editable workspace controls.

Create a snapshot when submitting a selection job and associate it with the
resulting basket action. Undo/redo then identifies the applied snapshot from
the action log, without reconstructing scientific provenance from live widgets.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from smiles2select.app_metadata import APP_VERSION, rdkit_version
from smiles2select.pipeline.runner import RunResult
from smiles2select.selection_intelligence.constrained_selection import (
    SelectionConstraints,
    Strategy,
)
from smiles2select.selection_intelligence.objectives import ObjectiveSet
from smiles2select.selection_intelligence.pareto_ranking import ParetoResult


def input_data_hash(*frames: pd.DataFrame) -> str:
    """SHA-256 over ordered tables, their schemas, row identities and values.

    The versioned encoding uses pandas' value hashes, one column at a time to
    bound scratch memory. It is a data fingerprint, not a source-file checksum;
    pandas' version is recorded alongside it for future audit/reproduction.
    """
    digest = hashlib.sha256(b"smiles2select-dataframe-v1\0")
    for frame in frames:
        schema = [list(frame.columns), [str(dtype) for dtype in frame.dtypes],
                  list(frame.index.names), str(frame.index.dtype), frame.shape]
        digest.update(json.dumps(schema, default=str, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\0")
        digest.update(pd.util.hash_pandas_object(frame.index).to_numpy().tobytes())
        for column in frame:
            digest.update(pd.util.hash_pandas_object(frame[column], index=False).to_numpy().tobytes())
        digest.update(b"\0end-table\0")
    return digest.hexdigest()


def run_provenance(result: RunResult, candidates: pd.DataFrame) -> dict[str, Any]:
    """Capture data identity, complete screening policy and software versions."""
    frames = (candidates, result.alerts, result.scores, result.evaluation.status)
    references = tuple(result.reference_libraries) + tuple(result.background_libraries)
    reference_frames = tuple(library.molecules for library in references)
    annotation_frames = tuple(
        report.annotations for report in (result.reference_similarity, result.reference_duplicates)
        if report is not None
    )
    return {
        "input_hash": input_data_hash(result.records, result.descriptors),
        "selection_data_hash": input_data_hash(*frames, *reference_frames, *annotation_frames),
        "hash_encoding": "smiles2select-dataframe-v1 (SHA-256 of pandas value hashes)",
        "run_config": json.loads(json.dumps(asdict(result.config), default=str)),
        "profiles": [profile.as_dict() for profile in result.profiles],
        "libraries": [library.spec.as_dict() for library in references],
        "software_versions": {
            "smiles2select": APP_VERSION, "rdkit": rdkit_version(),
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        },
        "projection_affects_selection": False,
    }


@dataclass(frozen=True)
class _FrozenFrame:
    """Numeric Pareto table stored without exposing a mutable DataFrame."""

    columns: tuple
    index: tuple
    index_name: Any
    column_name: Any
    dtypes: tuple[str, ...]
    rows: tuple[tuple, ...]

    @classmethod
    def capture(cls, frame: pd.DataFrame) -> _FrozenFrame:
        return cls(tuple(frame.columns), tuple(frame.index), frame.index.name, frame.columns.name,
                   tuple(str(dtype) for dtype in frame.dtypes),
                   tuple(frame.itertuples(index=False, name=None)))

    def restore(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.rows, columns=self.columns,
                             index=pd.Index(self.index, name=self.index_name))
        frame = frame.astype(dict(zip(self.columns, self.dtypes)))
        frame.columns.name = self.column_name
        return frame


@dataclass(frozen=True)
class AppliedSelection:
    """The settings and ranking actually used by one automatic basket action."""

    constraints: SelectionConstraints
    strategy: str
    input_hash: str
    _objectives_json: str
    _provenance_json: str
    _table: _FrozenFrame | None
    _objective_values: _FrozenFrame | None
    _objective_fields: tuple[str, ...]
    _warnings: tuple[str, ...]
    _cache_key: str

    @classmethod
    def capture(
        cls, *, constraints: SelectionConstraints, objectives: ObjectiveSet,
        pareto: ParetoResult | None, strategy: Strategy | str, input_hash: str,
        provenance: dict[str, Any] | None = None,
    ) -> AppliedSelection:
        return cls(
            constraints, strategy.value if isinstance(strategy, Strategy) else strategy, input_hash,
            json.dumps(objectives.as_dicts(), sort_keys=True, allow_nan=False),
            json.dumps(provenance or {}, sort_keys=True, allow_nan=False),
            _FrozenFrame.capture(pareto.table) if pareto is not None else None,
            _FrozenFrame.capture(pareto.objective_values) if pareto is not None else None,
            tuple(pareto.objective_fields) if pareto is not None else (),
            tuple(pareto.warnings) if pareto is not None else (),
            pareto.cache_key if pareto is not None else "",
        )

    @property
    def objectives(self) -> tuple[dict[str, Any], ...]:
        return tuple(json.loads(self._objectives_json))

    @property
    def provenance(self) -> dict[str, Any]:
        return json.loads(self._provenance_json)

    @property
    def pareto(self) -> ParetoResult | None:
        if self._table is None or self._objective_values is None:
            return None
        return ParetoResult(self._table.restore(), self._objective_fields, self._warnings,
                            self._cache_key, self._objective_values.restore())
