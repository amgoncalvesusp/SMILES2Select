"""Temporary SQLite database for one run.

Results go to disk rather than staying in memory so a library larger than RAM
can be processed, and so the Excel export reads from a stable snapshot instead
of recomputing anything.

Two tables are deliberately sparse: ``rule_failures`` stores only broken rules
and ``structural_alerts`` only actual hits. Storing one row per satisfied rule
would multiply the database size by the rule count for no information gain.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS molecule_descriptors (
    record_id INTEGER PRIMARY KEY,
    input_order INTEGER NOT NULL,
    molecule_id TEXT,
    original_smiles TEXT,
    standardized_smiles TEXT,
    canonical_smiles TEXT,
    mol_wt REAL,
    rdkit_wlogp REAL,
    mol_mr REAL,
    hbd INTEGER,
    hba INTEGER,
    tpsa REAL,
    rotatable_bonds INTEGER,
    heavy_atom_count INTEGER,
    total_atom_count INTEGER,
    carbon_count INTEGER,
    heteroatom_count INTEGER,
    ring_count INTEGER,
    aromatic_ring_count INTEGER,
    formal_charge INTEGER,
    fraction_csp3 REAL,
    qed REAL,
    valid INTEGER NOT NULL,
    invalid_reason TEXT,
    duplicate_of INTEGER,
    source_file TEXT,
    source_sheet TEXT,
    source_row INTEGER
);

CREATE TABLE IF NOT EXISTS profile_results (
    record_id INTEGER NOT NULL,
    profile_id TEXT NOT NULL,
    passed INTEGER NOT NULL,
    violation_count INTEGER NOT NULL,
    failure_mask INTEGER,
    PRIMARY KEY (record_id, profile_id)
);

CREATE TABLE IF NOT EXISTS rule_failures (
    record_id INTEGER NOT NULL,
    profile_id TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    failure_code TEXT NOT NULL,
    observed_value REAL,
    lower_limit REAL,
    upper_limit REAL
);

CREATE TABLE IF NOT EXISTS structural_alerts (
    record_id INTEGER NOT NULL,
    catalog_id TEXT NOT NULL,
    alert_name TEXT NOT NULL,
    alert_description TEXT,
    occurrence_count INTEGER,
    action TEXT
);

CREATE TABLE IF NOT EXISTS final_decisions (
    record_id INTEGER PRIMARY KEY,
    selected INTEGER NOT NULL,
    decision_policy_id TEXT NOT NULL,
    hard_failure_count INTEGER NOT NULL,
    alert_count INTEGER NOT NULL,
    consensus_score REAL,
    exclusion_reasons TEXT,
    selection_status TEXT
);

CREATE TABLE IF NOT EXISTS reference_libraries (
    library_id TEXT NOT NULL,
    role TEXT NOT NULL,
    source TEXT,
    description TEXT
);

CREATE TABLE IF NOT EXISTS reference_overlap (
    candidate_index INTEGER,
    candidate_id TEXT,
    reference_library TEXT,
    reference_index INTEGER,
    reference_id TEXT,
    match_type TEXT
);

CREATE TABLE IF NOT EXISTS reference_similarity (
    record_id INTEGER,
    candidate_id TEXT,
    reference_library TEXT,
    reference_index INTEGER,
    reference_id TEXT,
    similarity REAL
);

CREATE TABLE IF NOT EXISTS reserve_selection (
    record_id INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS zone_memberships (
    record_id TEXT,
    zone_id TEXT,
    zone_name TEXT,
    zone_kind TEXT,
    zone_priority INTEGER,
    zone_quota INTEGER
);

CREATE TABLE IF NOT EXISTS zone_allocation (
    zone_id TEXT,
    zone_name TEXT,
    available INTEGER,
    requested_final INTEGER,
    allocated_final INTEGER,
    requested_reserve INTEGER,
    allocated_reserve INTEGER,
    shortfall INTEGER
);

CREATE TABLE IF NOT EXISTS run_config (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_rule_failures_record ON rule_failures (record_id);
CREATE INDEX IF NOT EXISTS idx_rule_failures_code ON rule_failures (failure_code);
CREATE INDEX IF NOT EXISTS idx_alerts_record ON structural_alerts (record_id);
CREATE INDEX IF NOT EXISTS idx_alerts_catalog ON structural_alerts (catalog_id);
"""


def _require_own_database(path: Path) -> None:
    """Refuse to delete a file that is not a SMILES2Select run database.

    A run database is disposable, but the path comes from the user and a typo
    must not destroy an unrelated file.
    """
    # `with sqlite3.connect(...)` only manages the transaction; the handle must
    # be closed explicitly or Windows refuses to unlink the file afterwards.
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
    if "molecule_descriptors" not in tables:
        raise ValueError(
            f"{path} is a SQLite database but not a SMILES2Select run database; "
            "refusing to overwrite it"
        )


class SqliteStore:
    """Thin wrapper over the run database.

    Usable as a context manager; the connection is closed on exit even if the
    run fails halfway through, so the partial database stays readable.
    """

    def __init__(self, path: str | Path, *, overwrite: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite and self.path.exists():
            _require_own_database(self.path)
            self.path.unlink()
        self._connection = sqlite3.connect(self.path)
        self._connection.executescript(SCHEMA)
        self._connection.commit()

    def __enter__(self) -> SqliteStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    def close(self) -> None:
        self._connection.commit()
        self._connection.close()

    def table_columns(self, table: str) -> list[str]:
        cursor = self._connection.execute(f"PRAGMA table_info({table})")
        return [row[1] for row in cursor.fetchall()]

    def write_frame(
        self, frame: pd.DataFrame, table: str, *, index_label: str | None = None
    ) -> int:
        """Append a DataFrame to a table, keeping only the table's own columns."""
        if frame.empty:
            return 0
        payload = frame.copy()
        if index_label is not None:
            payload.index.name = index_label
            payload = payload.reset_index()
        known = self.table_columns(table)
        payload = payload[[column for column in payload.columns if column in known]]
        payload.to_sql(table, self._connection, if_exists="append", index=False)
        self._connection.commit()
        return len(payload)

    def read_frame(self, query: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
        return pd.read_sql_query(query, self._connection, params=params)

    def read_table(self, table: str) -> pd.DataFrame:
        return self.read_frame(f"SELECT * FROM {table}")

    def write_config(self, config: Mapping[str, Any]) -> None:
        """Store the run configuration as key/value JSON for the CONFIG sheet."""
        rows = [
            (key, json.dumps(value, ensure_ascii=False, default=str))
            for key, value in config.items()
        ]
        self._connection.executemany(
            "INSERT OR REPLACE INTO run_config (key, value) VALUES (?, ?)", rows
        )
        self._connection.commit()

    def read_config(self) -> dict[str, Any]:
        cursor = self._connection.execute("SELECT key, value FROM run_config")
        return {key: json.loads(value) for key, value in cursor.fetchall()}

    def count(self, table: str, where: str = "") -> int:
        clause = f" WHERE {where}" if where else ""
        cursor = self._connection.execute(f"SELECT COUNT(*) FROM {table}{clause}")
        return int(cursor.fetchone()[0])

    def failure_codes(self) -> list[str]:
        """Distinct failure codes present, for the per-rule export sheets."""
        cursor = self._connection.execute(
            "SELECT DISTINCT failure_code FROM rule_failures ORDER BY failure_code"
        )
        return [row[0] for row in cursor.fetchall()]
