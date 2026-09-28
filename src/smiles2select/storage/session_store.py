"""Portable, versioned workspace snapshots without source files or pickle."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Mapping
from contextlib import closing
from dataclasses import fields, is_dataclass, replace
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from smiles2select.alerts.custom_smarts import SmartsAlert
from smiles2select.alerts.policies import AlertPolicy
from smiles2select.chemistry.descriptor_planner import DescriptorPlan
from smiles2select.chemistry.fingerprints import FingerprintConfig
from smiles2select.chemistry.standardization import StandardizationConfig
from smiles2select.decision.engine import DecisionResult
from smiles2select.decision.policies import DecisionPolicy
from smiles2select.gui.workspace.criteria_state import CriteriaState
from smiles2select.gui.workspace.selection_provenance import AppliedSelection, _FrozenFrame
from smiles2select.io.importer import ColumnMapping, SourceFile
from smiles2select.pipeline.config import RunConfig
from smiles2select.pipeline.diagnostics import ParallelDiagnosticsConfig
from smiles2select.pipeline.runner import RunResult, _persist
from smiles2select.profiles.registry import Profile
from smiles2select.reference.duplicates import ExactDuplicateReport
from smiles2select.reference.libraries import LibraryRole, LibrarySpec, ReferenceLibrary
from smiles2select.reference.similarity import ReferenceSimilarityResult
from smiles2select.rules.engine import Rule
from smiles2select.rules.evaluator import ProfileEvaluation
from smiles2select.scores.qed import QedSelection
from smiles2select.selection.diversity import DiversityReport
from smiles2select.selection.zone_allocator import ZoneAllocationResult
from smiles2select.selection_intelligence.basket import SelectionBasket
from smiles2select.selection_intelligence.constrained_selection import (
    SelectionConstraints,
    SelectionOutcome,
    Strategy,
)
from smiles2select.selection_intelligence.objectives import Direction, Objective
from smiles2select.selection_intelligence.scenarios import ScenarioSnapshot, ScenarioSpec
from smiles2select.storage.selection_store import SelectionStore

SCHEMA_VERSION = 2
_DATACLASSES = {
    cls.__name__: cls
    for cls in (
        RunResult,
        ProfileEvaluation,
        DecisionResult,
        DescriptorPlan,
        Profile,
        Rule,
        RunConfig,
        SourceFile,
        ColumnMapping,
        StandardizationConfig,
        FingerprintConfig,
        SmartsAlert,
        DecisionPolicy,
        QedSelection,
        AlertPolicy,
        ParallelDiagnosticsConfig,
        DiversityReport,
        LibrarySpec,
        ReferenceLibrary,
        ExactDuplicateReport,
        ReferenceSimilarityResult,
        ZoneAllocationResult,
        CriteriaState,
        AppliedSelection,
        _FrozenFrame,
        SelectionConstraints,
        SelectionOutcome,
        Objective,
        ScenarioSnapshot,
        ScenarioSpec,
    )
}
_ENUMS = {cls.__name__: cls for cls in (LibraryRole, Strategy, Direction)}
_WORKSPACE_KEYS = frozenset(
    {
        "applied_selections",
        "criteria_snapshots",
        "selection_outcomes",
        "model_scores_by_action",
        "initial_criteria",
        "draft_criteria",
        "run_provenance",
        "scenario_snapshots",
        "candidates",
    }
)
_ACTION_MAP_KEYS = _WORKSPACE_KEYS - {
    "initial_criteria", "draft_criteria", "run_provenance", "candidates",
}


def _encode(value: Any, frame_sink: sqlite3.Connection | None = None) -> Any:
    if isinstance(value, Enum):
        if value.__class__.__name__ not in _ENUMS:
            raise TypeError(f"unsupported session enum {value.__class__.__name__}")
        return {"$": "enum", "name": value.__class__.__name__, "value": value.value}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if value is pd.NA:
        return {"$": "pd.NA"}
    if value is pd.NaT:
        return {"$": "pd.NaT"}
    if isinstance(value, np.generic):
        return _encode(value.item(), frame_sink)
    if isinstance(value, float):
        if np.isnan(value):
            return {"$": "nan"}
        if np.isposinf(value):
            return {"$": "+inf"}
        if np.isneginf(value):
            return {"$": "-inf"}
        return value
    if isinstance(value, pd.DataFrame):
        if frame_sink is None:
            raise ValueError("session frame storage is required")
        return _write_frame(value, frame_sink)
    if isinstance(value, pd.Series):
        return {
            "$": "series",
            "name": _encode(value.name, frame_sink),
            "frame": _encode(value.to_frame(name="__series__"), frame_sink),
        }
    if isinstance(value, pd.Index):
        if isinstance(value, (pd.MultiIndex, pd.CategoricalIndex)):
            raise ValueError(f"{type(value).__name__} is unsupported in saved sessions")
        return {
            "$": "index",
            "name": _encode(value.name),
            "dtype": str(value.dtype),
            "values": [_encode(item, frame_sink) for item in value],
        }
    if isinstance(value, pd.Timestamp):
        return {"$": "timestamp", "value": value.isoformat()}
    if isinstance(value, datetime):
        return {"$": "datetime", "value": value.isoformat()}
    if isinstance(value, date):
        return {"$": "date", "value": value.isoformat()}
    if isinstance(value, Path):
        return {"$": "path", "value": str(value)}
    if is_dataclass(value) and not isinstance(value, type):
        name = value.__class__.__name__
        if _DATACLASSES.get(name) is not value.__class__:
            raise TypeError(f"unsupported session dataclass {name}")
        return {
            "$": "object",
            "name": name,
            "fields": {
                field.name: _encode(getattr(value, field.name), frame_sink)
                for field in fields(value)
            },
        }
    if isinstance(value, tuple):
        return {"$": "tuple", "items": [_encode(item, frame_sink) for item in value]}
    if isinstance(value, list):
        return [_encode(item, frame_sink) for item in value]
    if isinstance(value, Mapping):
        return {
            "$": "mapping",
            "items": [[_encode(k, frame_sink), _encode(v, frame_sink)] for k, v in value.items()],
        }
    raise TypeError(f"unsupported session value {type(value).__name__}")


def _write_frame(frame: pd.DataFrame, connection: sqlite3.Connection) -> dict[str, str]:
    for axis in (frame.index, frame.columns):
        if isinstance(axis, (pd.MultiIndex, pd.CategoricalIndex)):
            raise ValueError(f"{type(axis).__name__} is unsupported in saved sessions")
    if not frame.columns.is_unique:
        raise ValueError("duplicate DataFrame columns are unsupported in saved sessions")
    frame_id = uuid4().hex
    metadata = {
        "columns": [_encode(column) for column in frame.columns],
        "columns_dtype": str(frame.columns.dtype),
        "column_name": _encode(frame.columns.name),
        "index_name": _encode(frame.index.name),
        "index_dtype": str(frame.index.dtype),
        "range_index": (
            [frame.index.start, frame.index.stop, frame.index.step]
            if isinstance(frame.index, pd.RangeIndex)
            else None
        ),
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "categoricals": {
            str(position): {
                "categories": [_encode(item) for item in dtype.categories],
                "ordered": dtype.ordered,
            }
            for position, dtype in enumerate(frame.dtypes)
            if isinstance(dtype, pd.CategoricalDtype)
        },
        "row_count": len(frame),
    }
    metadata_json = json.dumps(metadata, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    digest = hashlib.sha256(metadata_json.encode("utf-8"))
    connection.execute(
        "INSERT INTO session_frames (frame_id, metadata_json) VALUES (?, ?)",
        (frame_id, metadata_json),
    )
    batch: list[tuple[str, int, str]] = []
    for position, (index, cells) in enumerate(
        zip(frame.index, frame.itertuples(index=False, name=None), strict=True)
    ):
        row_json = json.dumps(
            [_encode(index), [_encode(cell) for cell in cells]],
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        digest.update(row_json.encode("utf-8"))
        batch.append((frame_id, position, row_json))
        if len(batch) == 1000:
            connection.executemany(
                "INSERT INTO session_frame_rows (frame_id, row_number, row_json) VALUES (?, ?, ?)",
                batch,
            )
            batch.clear()
    if batch:
        connection.executemany(
            "INSERT INTO session_frame_rows (frame_id, row_number, row_json) VALUES (?, ?, ?)",
            batch,
        )
    return {"$": "frame_ref", "id": frame_id, "sha256": digest.hexdigest()}


def _read_frame(reference: dict[str, str], connection: sqlite3.Connection) -> pd.DataFrame:
    row = connection.execute(
        "SELECT metadata_json FROM session_frames WHERE frame_id = ?", (reference["id"],)
    ).fetchone()
    if row is None:
        raise ValueError("session frame is missing")
    metadata_json = row[0]
    digest = hashlib.sha256(metadata_json.encode("utf-8"))
    metadata = json.loads(metadata_json)
    rows: list[list[Any]] = []
    indexes: list[Any] = []
    for number, row_json in connection.execute(
        "SELECT row_number, row_json FROM session_frame_rows WHERE frame_id = ? ORDER BY row_number",
        (reference["id"],),
    ):
        if number != len(rows):
            raise ValueError("session frame row order is invalid")
        digest.update(row_json.encode("utf-8"))
        index, cells = json.loads(row_json)
        indexes.append(_decode(index, connection))
        rows.append([_decode(cell, connection) for cell in cells])
    if digest.hexdigest() != reference["sha256"] or len(rows) != metadata["row_count"]:
        raise ValueError("session frame integrity check failed")
    columns = [_decode(column, connection) for column in metadata["columns"]]
    column_index = pd.Index(columns, dtype=metadata.get("columns_dtype"),
                            name=_decode(metadata.get("column_name"), connection))
    frame = pd.DataFrame(rows, columns=column_index, dtype=object)
    for position, (column, dtype) in enumerate(zip(columns, metadata["dtypes"], strict=True)):
        if dtype == "category":
            details = metadata.get("categoricals", {}).get(str(position))
            if details is None:
                raise ValueError("session category metadata is missing")
            frame[column] = pd.Categorical(
                frame[column],
                categories=[_decode(item, connection) for item in details["categories"]],
                ordered=details["ordered"],
            )
        else:
            frame[column] = frame[column].astype(dtype)
    name = _decode(metadata["index_name"], connection)
    if metadata["range_index"] is not None:
        frame.index = pd.RangeIndex(*metadata["range_index"], name=name)
    else:
        frame.index = pd.Index(indexes, dtype=metadata["index_dtype"], name=name)
    if len(frame.index) != len(frame):
        raise ValueError("session frame index length differs from data")
    return frame


def _decode(value: Any, frame_source: sqlite3.Connection | None = None) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, list):
        return [_decode(item, frame_source) for item in value]
    if not isinstance(value, dict):
        raise ValueError("invalid session value")
    kind = value.get("$")
    if kind == "pd.NA":
        return pd.NA
    if kind == "pd.NaT":
        return pd.NaT
    if kind == "nan":
        return float("nan")
    if kind == "+inf":
        return float("inf")
    if kind == "-inf":
        return float("-inf")
    if kind == "path":
        return Path(value["value"])
    if kind == "timestamp":
        return pd.Timestamp(value["value"])
    if kind == "datetime":
        return datetime.fromisoformat(value["value"])
    if kind == "date":
        return date.fromisoformat(value["value"])
    if kind == "tuple":
        return tuple(_decode(item, frame_source) for item in value["items"])
    if kind == "series":
        series = _decode(value["frame"], frame_source)["__series__"]
        series.name = _decode(value["name"], frame_source)
        return series
    if kind == "index":
        return pd.Index(
            [_decode(item, frame_source) for item in value["values"]],
            dtype=value["dtype"],
            name=_decode(value["name"], frame_source),
        )
    if kind == "mapping":
        return {_decode(k, frame_source): _decode(v, frame_source) for k, v in value["items"]}
    if kind == "enum":
        enum_type = _ENUMS.get(value["name"])
        if enum_type is None:
            raise ValueError(f"unknown session enum {value['name']}")
        return enum_type(value["value"])
    if kind == "object":
        cls = _DATACLASSES.get(value["name"])
        if cls is None:
            raise ValueError(f"unknown session object {value['name']}")
        expected = {field.name for field in fields(cls)}
        saved_fields = value["fields"]
        # Version 3.4.0 saved criteria before the minimum-core control existed.
        if cls is CriteriaState and set(saved_fields) == expected - {"minimum_scaffolds"}:
            saved_fields = {**saved_fields, "minimum_scaffolds": 0}
        if set(saved_fields) != expected:
            raise ValueError(f"session object fields changed for {value['name']}")
        return cls(**{name: _decode(item, frame_source) for name, item in saved_fields.items()})
    if kind == "frame_ref":
        if frame_source is None:
            raise ValueError("session frame source is unavailable")
        return _read_frame(value, frame_source)
    raise ValueError(f"unknown session value kind {kind!r}")


def _materialize_run_database(path: Path, result: RunResult) -> None:
    # The in-memory RunResult is authoritative; an old run.sqlite may have changed.
    config = replace(result.config, database_path=path)
    _persist(
        config,
        result.records,
        result.descriptors,
        result.evaluation,
        result.alerts,
        result.scores,
        result.decision,
        reference_libraries=result.reference_libraries,
        background_libraries=result.background_libraries,
        reference_duplicates=result.reference_duplicates,
        reference_similarity=result.reference_similarity,
        reserve_ids=result.reserve_ids,
        zone_allocation=result.zone_allocation,
        preparability=result.preparability,
    )


def _basket_hash(basket: SelectionBasket) -> str:
    digest = hashlib.sha256(b"smiles2select-basket-v2\0")
    for section, entries in (
        (b"states", (state.as_row() for state in basket.states())),
        (b"actions", basket.log.rows()),
    ):
        digest.update(section + b"\0")
        for entry in entries:
            row = json.dumps(entry, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            digest.update(row.encode("utf-8") + b"\0")
    digest.update(f"cursor={basket.log.cursor};target={basket.target_count}".encode("ascii"))
    return digest.hexdigest()


def _workspace_state(
    state: Mapping[str, Any] | None,
    basket: SelectionBasket,
    *,
    prune_orphans: bool = False,
) -> dict[str, Any]:
    if state is None:
        return {}
    if not isinstance(state, Mapping):
        raise TypeError("workspace_state must be a mapping")
    unknown = set(state) - _WORKSPACE_KEYS
    if unknown:
        raise ValueError(f"unknown workspace state fields: {sorted(unknown)}")
    normalized = dict(state)
    for name in _ACTION_MAP_KEYS.intersection(state):
        actions = state[name]
        if not isinstance(actions, Mapping) or any(
            type(index) is not int or index < 0 for index in actions
        ):
            raise ValueError(f"{name} must map stored action indices to values")
        orphans = {index for index in actions if index >= len(basket.log)}
        if orphans and not prune_orphans:
            raise ValueError(f"{name} contains actions absent from history")
        normalized[name] = {
            index: value for index, value in actions.items() if index not in orphans
        }
    return normalized


def save_session(
    path: str | Path,
    result: RunResult,
    basket: SelectionBasket,
    *,
    workspace_state: Mapping[str, Any] | None = None,
) -> Path:
    """Publish a complete snapshot atomically; prior session survives failure."""
    if not isinstance(result, RunResult):
        raise TypeError("result must be RunResult")
    if not isinstance(basket, SelectionBasket):
        raise TypeError("basket must be SelectionBasket")
    normalized_workspace = _workspace_state(workspace_state, basket, prune_orphans=True)
    basket_digest = _basket_hash(basket)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        _materialize_run_database(temporary, result)
        with SelectionStore(temporary) as store:
            store.save_basket(basket)
            store._connection.execute(
                "CREATE TABLE session_frames (frame_id TEXT PRIMARY KEY, metadata_json TEXT NOT NULL)"
            )
            store._connection.execute(
                "CREATE TABLE session_frame_rows (frame_id TEXT NOT NULL, "
                "row_number INTEGER NOT NULL, row_json TEXT NOT NULL, "
                "PRIMARY KEY (frame_id, row_number))"
            )
            document = {
                "run": _encode(result, store._connection),
                "basket_sha256": basket_digest,
                "workspace": _encode(normalized_workspace, store._connection),
            }
            payload = json.dumps(
                document, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode("utf-8")
            digest = hashlib.sha256(payload).hexdigest()
            store._connection.execute(
                "CREATE TABLE IF NOT EXISTS session_payload ("
                "session_key INTEGER PRIMARY KEY CHECK (session_key = 1), "
                "schema_version INTEGER NOT NULL, payload BLOB NOT NULL, "
                "sha256 TEXT NOT NULL)"
            )
            store._connection.execute(
                "INSERT INTO session_payload VALUES (1, ?, ?, ?) "
                "ON CONFLICT(session_key) DO UPDATE SET schema_version=excluded.schema_version, "
                "payload=excluded.payload, sha256=excluded.sha256",
                (SCHEMA_VERSION, payload, digest),
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_session(path: str | Path) -> tuple[RunResult, SelectionBasket, dict[str, Any]]:
    """Read a saved snapshot and verify its payload before decoding classes."""
    source = Path(path).resolve()
    try:
        with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as connection:
            row = connection.execute(
                "SELECT schema_version, payload, sha256 FROM session_payload WHERE session_key = 1"
            ).fetchone()
            if row is None:
                raise ValueError("session has no payload")
            version, payload, digest = row
            if version != SCHEMA_VERSION:
                raise ValueError(f"unsupported session schema {version}")
            if not isinstance(payload, bytes) or hashlib.sha256(payload).hexdigest() != digest:
                raise ValueError("session payload integrity check failed")
            document = json.loads(payload)
            result = _decode(document["run"], connection)
            workspace = _decode(document["workspace"], connection)
    except sqlite3.DatabaseError as exc:
        raise ValueError("invalid SMILES2Select session database") from exc
    if not isinstance(result, RunResult):
        raise ValueError("session payload is not a RunResult")
    with SelectionStore(source, readonly=True) as store:
        basket = store.load_basket()
    if _basket_hash(basket) != document["basket_sha256"]:
        raise ValueError("session basket integrity check failed")
    return replace(result, database_path=source), basket, _workspace_state(workspace, basket)
