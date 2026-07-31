"""Transactional per-chunk checkpoint, so a crash never loses finished work.

A chunk is only ever considered done once its results and its COMPLETED
status are committed in the same SQLite transaction. Anything registered but
not completed when the process last exited (PENDING, RUNNING, FAILED,
ISOLATING) is pruned on open: its record range gets re-sliced and
re-registered under a fresh chunk id by ``ChunkPlanner`` rather than
reconstructed, which would require assuming the previous run's chunk
boundaries are still valid. A COMPLETED chunk's boundary is the only thing
ever treated as permanent.

Resume is keyed on ``input_hash`` (the source SMILES, order-sensitive) and
``config_hash`` (``RunConfig.fingerprint()``, the same hash the descriptor
cache uses): if either has changed since the checkpoint was written, the
recorded progress refers to different data and the checkpoint starts fresh
instead of silently reusing it.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from smiles2select.pipeline.chunking import MoleculeChunk
from smiles2select.pipeline.workers import RecordResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoint_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS processing_chunks (
    chunk_id INTEGER PRIMARY KEY,
    first_input_order INTEGER NOT NULL,
    last_input_order INTEGER NOT NULL,
    record_count INTEGER NOT NULL,
    status TEXT NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    worker_pid INTEGER,
    started_at TEXT,
    completed_at TEXT,
    error_code TEXT,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS chunk_results (
    chunk_id INTEGER NOT NULL,
    record_id INTEGER PRIMARY KEY,
    payload TEXT NOT NULL
);
"""


def _now() -> str:
    # Imported lazily: datetime.now() is fine here (this is wall-clock
    # logging, not something a workflow replay needs to be deterministic).
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _require_checkpoint_database(path: Path) -> None:
    """Refuse to delete a file that is not one of our checkpoints.

    Mirrors ``storage.sqlite_store._require_own_database``: the path is
    user-supplied, and a typo must not destroy an unrelated file.
    """
    connection = None
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    except sqlite3.DatabaseError as exc:
        raise ValueError(
            f"{path} exists and is not a SQLite database; refusing to overwrite it"
        ) from exc
    finally:
        if connection is not None:
            connection.close()
    if "processing_chunks" not in tables:
        raise ValueError(
            f"{path} is a SQLite database but not a SMILES2Select checkpoint; "
            "refusing to overwrite it"
        )


def _matches_signature(path: Path, input_hash: str, config_hash: str) -> bool:
    connection = None
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        rows = dict(connection.execute("SELECT key, value FROM checkpoint_meta").fetchall())
    except sqlite3.DatabaseError:
        return False
    finally:
        if connection is not None:
            connection.close()
    return rows.get("input_hash") == input_hash and rows.get("config_hash") == config_hash


def _serialize(result: RecordResult) -> str:
    payload = {
        "record_id": result.record_id,
        "valid": result.valid,
        "invalid_reason": result.invalid_reason,
        "standardized_smiles": result.standardized_smiles,
        "canonical_smiles": result.canonical_smiles,
        "descriptors": result.descriptors,
        "substructure_flags": result.substructure_flags,
        "alert_rows": result.alert_rows,
        "error_code": result.error_code,
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


def _deserialize(payload: str) -> RecordResult:
    return RecordResult(**json.loads(payload))


class ChunkCheckpointStore:
    """One checkpoint file per run. Safe to reopen after a crash."""

    def __init__(self, path: str | Path, *, input_hash: str, config_hash: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if self.path.exists() and not _matches_signature(self.path, input_hash, config_hash):
            _require_checkpoint_database(self.path)
            self.path.unlink()

        self._connection = sqlite3.connect(self.path)
        self._connection.executescript(SCHEMA)
        self._connection.commit()
        self._store_meta(input_hash, config_hash)
        self._prune_incomplete()

    def __enter__(self) -> ChunkCheckpointStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.commit()
        self._connection.close()

    def discard(self) -> None:
        """Delete the checkpoint file entirely; called after a fully
        successful run when nothing is left to resume."""
        self.close()
        if self.path.exists():
            self.path.unlink()

    def _store_meta(self, input_hash: str, config_hash: str) -> None:
        self._connection.executemany(
            "INSERT OR REPLACE INTO checkpoint_meta (key, value) VALUES (?, ?)",
            [("input_hash", input_hash), ("config_hash", config_hash)],
        )
        self._connection.commit()

    def _prune_incomplete(self) -> None:
        """Drop any chunk that never finished. Its record range was never
        treated as permanent, so the planner will re-slice and re-register it
        under a fresh chunk id."""
        self._connection.execute("DELETE FROM processing_chunks WHERE status != 'COMPLETED'")
        self._connection.execute(
            "DELETE FROM chunk_results WHERE chunk_id NOT IN "
            "(SELECT chunk_id FROM processing_chunks)"
        )
        self._connection.commit()

    def planned_record_count(self) -> int:
        """Records already covered by a COMPLETED chunk - where the planner resumes."""
        cursor = self._connection.execute(
            "SELECT COALESCE(SUM(record_count), 0) FROM processing_chunks"
        )
        return int(cursor.fetchone()[0])

    def next_chunk_id(self) -> int:
        cursor = self._connection.execute(
            "SELECT COALESCE(MAX(chunk_id), -1) FROM processing_chunks"
        )
        return int(cursor.fetchone()[0]) + 1

    def last_chunk_size(self) -> int | None:
        cursor = self._connection.execute(
            "SELECT record_count FROM processing_chunks ORDER BY chunk_id DESC LIMIT 1"
        )
        row = cursor.fetchone()
        return int(row[0]) if row else None

    def completed_chunk_count(self) -> int:
        cursor = self._connection.execute(
            "SELECT COUNT(*) FROM processing_chunks WHERE status = 'COMPLETED'"
        )
        return int(cursor.fetchone()[0])

    def completed_results(self) -> list[RecordResult]:
        cursor = self._connection.execute("SELECT payload FROM chunk_results")
        return [_deserialize(row[0]) for row in cursor.fetchall()]

    def register_chunks(self, chunks: Sequence[MoleculeChunk]) -> None:
        rows = [
            (chunk.chunk_id, chunk.first_input_order, chunk.last_input_order, len(chunk), "PENDING")
            for chunk in chunks
        ]
        self._connection.executemany(
            "INSERT OR REPLACE INTO processing_chunks "
            "(chunk_id, first_input_order, last_input_order, record_count, status) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        self._connection.commit()

    def mark_running(self, chunk_id: int, pid: int | None) -> None:
        self._connection.execute(
            "UPDATE processing_chunks SET status='RUNNING', worker_pid=?, started_at=?, "
            "attempt_count = attempt_count + 1 WHERE chunk_id=?",
            (pid, _now(), chunk_id),
        )
        self._connection.commit()

    def mark_isolating(self, chunk_id: int) -> None:
        self._connection.execute(
            "UPDATE processing_chunks SET status='ISOLATING' WHERE chunk_id=?", (chunk_id,)
        )
        self._connection.commit()

    def mark_failed(self, chunk_id: int, error_code: str, error_message: str) -> None:
        self._connection.execute(
            "UPDATE processing_chunks SET status='FAILED', error_code=?, error_message=? "
            "WHERE chunk_id=?",
            (error_code, error_message, chunk_id),
        )
        self._connection.commit()

    def commit_chunk(self, chunk_id: int, results: Sequence[RecordResult]) -> None:
        """Insert results and mark the chunk COMPLETED, atomically.

        Only after this returns is the chunk considered done; a crash before
        the commit leaves it not-COMPLETED, and the next open prunes it back
        to unclaimed.
        """
        connection = self._connection
        try:
            connection.execute("BEGIN")
            connection.executemany(
                "INSERT OR REPLACE INTO chunk_results (chunk_id, record_id, payload) VALUES (?, ?, ?)",
                [(chunk_id, result.record_id, _serialize(result)) for result in results],
            )
            connection.execute(
                "UPDATE processing_chunks SET status='COMPLETED', completed_at=? WHERE chunk_id=?",
                (_now(), chunk_id),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
