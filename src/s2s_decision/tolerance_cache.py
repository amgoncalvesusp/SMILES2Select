"""Optional durable chemical annotations; never stores biological/source metadata."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from collections.abc import Callable, Iterable, Mapping
from os import PathLike
from pathlib import Path
from types import TracebackType
from typing import Any

from .tolerance_schema import _finite_tree

LOGGER = logging.getLogger(__name__)
CACHE_SCHEMA = "s2s-tolerance-cache-entry/1"
BATCH_SIZE = 400
MAX_ENTRY_BYTES = 2 * 1024 * 1024
Annotation = dict[str, Any]
Validator = Callable[[Annotation], None]


def json_text(value: Any) -> str:
    """Canonical JSON rejects arbitrary objects, nonfinite values and deep trees."""
    _finite_tree(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(json_text(value).encode("utf-8")).hexdigest()


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate annotation cache JSON key")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError(f"Nonfinite annotation cache JSON constant: {value}")


class ToleranceAnnotationCache:
    """Bounded parameterized SQLite access with explicit corruption recovery."""

    def __init__(self, path: str | PathLike[str], recipe: Mapping[str, Any]) -> None:
        self.recipe_sha256 = content_hash(recipe)
        self.connection: sqlite3.Connection | None = None
        try:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(target, timeout=30)
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS tolerance_annotation_cache "
                "(cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL, payload_sha256 TEXT NOT NULL)"
            )
            self.connection.commit()
        except (OSError, TypeError, ValueError, sqlite3.Error) as exc:
            self.close()
            raise ValueError("Cannot open tolerance annotation cache") from exc

    def __enter__(self) -> ToleranceAnnotationCache:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 traceback: TracebackType | None) -> None:
        self.close()

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def _open_connection(self) -> sqlite3.Connection:
        if self.connection is None:
            raise ValueError("The tolerance annotation cache is closed")
        return self.connection

    def _decode(self, key: str, payload: object, digest: object, validate: Validator) -> Annotation:
        if not isinstance(payload, str) or len(payload.encode("utf-8")) > MAX_ENTRY_BYTES:
            raise ValueError("Annotation cache entry exceeds size/type limits")
        if hashlib.sha256(payload.encode("utf-8")).hexdigest() != digest:
            raise ValueError("Annotation cache checksum mismatch")
        body = json.loads(payload, object_pairs_hook=_object, parse_constant=_constant)
        _finite_tree(body)
        if (not isinstance(body, dict) or set(body) != {"schema", "cache_key", "recipe_sha256", "annotation"}
                or body["schema"] != CACHE_SCHEMA or body["cache_key"] != key
                or body["recipe_sha256"] != self.recipe_sha256 or not isinstance(body["annotation"], dict)):
            raise ValueError("Annotation cache identity/recipe mismatch")
        validate(body["annotation"])
        return body["annotation"]

    def fetch(self, keys: Iterable[str], validate: Validator) -> dict[str, Annotation]:
        connection = self._open_connection()
        found = {}
        unique = list(dict.fromkeys(keys))
        try:
            for start in range(0, len(unique), BATCH_SIZE):
                group = unique[start:start + BATCH_SIZE]
                slots = ",".join("?" for _ in group)
                rows = connection.execute(
                    "SELECT cache_key, CASE WHEN length(CAST(payload AS BLOB)) <= ? THEN payload ELSE NULL END, "
                    "substr(payload_sha256,1,65) FROM tolerance_annotation_cache "
                    f"WHERE cache_key IN ({slots})", [MAX_ENTRY_BYTES, *group],
                )
                for key, payload, digest in rows:
                    try:
                        found[key] = self._decode(key, payload, digest, validate)
                    except (ValueError, TypeError, RecursionError) as exc:
                        LOGGER.warning("Corrupt tolerance annotation cache entry %s; recomputing (%s)", key, exc)
        except sqlite3.Error as exc:
            raise ValueError("Cannot read tolerance annotation cache") from exc
        return found

    def store(self, entries: Mapping[str, Annotation], validate: Validator) -> None:
        connection = self._open_connection()
        rows = []
        for key, annotation in entries.items():
            validate(annotation)
            body = {"schema": CACHE_SCHEMA, "cache_key": key,
                    "recipe_sha256": self.recipe_sha256, "annotation": annotation}
            payload = json_text(body)
            if len(payload.encode("utf-8")) > MAX_ENTRY_BYTES:
                raise ValueError("Tolerance annotation cache entry exceeds size limit")
            rows.append((key, payload, hashlib.sha256(payload.encode("utf-8")).hexdigest()))
        try:
            for start in range(0, len(rows), BATCH_SIZE):
                with connection:
                    connection.executemany(
                        "INSERT OR REPLACE INTO tolerance_annotation_cache (cache_key,payload,payload_sha256) VALUES (?,?,?)",
                        rows[start:start + BATCH_SIZE],
                    )
        except sqlite3.Error as exc:
            raise ValueError("Cannot write tolerance annotation cache") from exc
