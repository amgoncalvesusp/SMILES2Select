"""Persistence for the Selection Intelligence layer.

Lives in its own tables, attached to the same run database. Nothing here
overwrites a pipeline table: the chemical result and the human decision are
stored side by side, which is what allows a selection to be audited later.

Timestamps are ISO 8601 in UTC.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from smiles2select.selection_intelligence.action_log import ActionLog, from_rows, utc_timestamp
from smiles2select.selection_intelligence.basket import SelectionBasket, basket_from_rows
from smiles2select.selection_intelligence.recipes import SelectionRecipe, from_dict

SCHEMA = """
CREATE TABLE IF NOT EXISTS pareto_results (
    record_id INTEGER NOT NULL,
    recipe_id TEXT NOT NULL,
    pareto_rank INTEGER NOT NULL,
    domination_count INTEGER,
    dominated_count INTEGER,
    distance_to_ideal REAL,
    PRIMARY KEY (record_id, recipe_id)
);

CREATE TABLE IF NOT EXISTS rule_margins (
    record_id INTEGER NOT NULL,
    rule_id TEXT NOT NULL,
    observed_value REAL,
    threshold_low REAL,
    threshold_high REAL,
    normalized_margin REAL,
    margin_status TEXT,
    PRIMARY KEY (record_id, rule_id)
);

CREATE TABLE IF NOT EXISTS chemical_space_coordinates (
    record_id INTEGER NOT NULL,
    projection_id TEXT NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    cluster_id INTEGER,
    PRIMARY KEY (record_id, projection_id)
);

CREATE TABLE IF NOT EXISTS selection_state (
    record_id INTEGER PRIMARY KEY,
    chemical_status TEXT NOT NULL,
    selection_status TEXT NOT NULL,
    selection_origin TEXT,
    pinned INTEGER NOT NULL DEFAULT 0,
    manual_note TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS selection_actions (
    action_id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    action_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    previous_state_json TEXT,
    new_state_json TEXT
);

CREATE TABLE IF NOT EXISTS selection_recipes (
    recipe_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    config_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS hub_libraries (
    library_id TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    source TEXT,
    description TEXT
);

CREATE TABLE IF NOT EXISTS hub_reference_overlap (
    candidate_index TEXT,
    candidate_id TEXT,
    reference_library TEXT,
    reference_index TEXT,
    reference_id TEXT,
    match_type TEXT
);

CREATE TABLE IF NOT EXISTS hub_reference_similarity (
    record_id TEXT,
    candidate_id TEXT,
    reference_library TEXT,
    reference_index TEXT,
    reference_id TEXT,
    similarity REAL
);

CREATE TABLE IF NOT EXISTS hub_zones (
    zone_id TEXT PRIMARY KEY,
    label TEXT,
    expression TEXT,
    quota INTEGER,
    priority INTEGER,
    color TEXT
);

CREATE TABLE IF NOT EXISTS hub_projections (
    projection_id TEXT PRIMARY KEY,
    method TEXT NOT NULL,
    input_hash TEXT,
    parameters_json TEXT NOT NULL,
    recipe_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS hub_projection_coordinates (
    record_id TEXT NOT NULL,
    projection_id TEXT NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    cluster_id TEXT,
    PRIMARY KEY (record_id, projection_id)
);

CREATE INDEX IF NOT EXISTS idx_pareto_rank ON pareto_results (recipe_id, pareto_rank);
CREATE INDEX IF NOT EXISTS idx_margin_status ON rule_margins (margin_status);
CREATE INDEX IF NOT EXISTS idx_selection_status ON selection_state (selection_status);
"""


class SelectionStore:
    """Read/write access to the selection tables of one run database."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(SCHEMA)
        self._connection.commit()

    def __enter__(self) -> SelectionStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.commit()
        self._connection.close()

    # -- selection state ----------------------------------------------------

    def save_basket(self, basket: SelectionBasket) -> int:
        """Write the current decision of every molecule, replacing what was there."""
        stamp = utc_timestamp()
        rows = [
            (
                state.record_id,
                state.chemical_status.value,
                state.selection_status.value,
                state.origin.value if state.origin else None,
                int(state.pinned),
                state.note or None,
                stamp,
            )
            for state in basket.states()
        ]
        self._connection.executemany(
            "INSERT OR REPLACE INTO selection_state (record_id, chemical_status, "
            "selection_status, selection_origin, pinned, manual_note, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._save_actions(basket.log)
        self._connection.commit()
        return len(rows)

    def load_basket(self, target_count: int | None = None) -> SelectionBasket:
        """Restore a saved session: decisions plus the history behind them."""
        states = [dict(row) for row in self._connection.execute("SELECT * FROM selection_state")]
        basket = basket_from_rows(states, log=self.load_log())
        basket.target_count = target_count
        return basket

    def _save_actions(self, log: ActionLog) -> None:
        """Rewrite the history; the log is the authority on its own order."""
        self._connection.execute("DELETE FROM selection_actions")
        self._connection.executemany(
            "INSERT INTO selection_actions (timestamp, action_type, payload_json, "
            "previous_state_json, new_state_json) VALUES (?, ?, ?, ?, ?)",
            [
                (
                    row["timestamp"],
                    row["action_type"],
                    row["payload_json"],
                    row["previous_state_json"],
                    row["new_state_json"],
                )
                for row in log.rows()
            ],
        )

    def load_log(self) -> ActionLog:
        rows = [
            dict(row)
            for row in self._connection.execute(
                "SELECT * FROM selection_actions ORDER BY action_id"
            )
        ]
        return from_rows(rows)

    # -- recipes ------------------------------------------------------------

    def save_recipe(self, recipe_id: str, recipe: SelectionRecipe) -> str:
        payload = recipe.as_dict()
        self._connection.execute(
            "INSERT OR REPLACE INTO selection_recipes (recipe_id, name, schema_version, "
            "config_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                recipe_id,
                recipe.name,
                payload["schema_version"],
                json.dumps(payload, ensure_ascii=False),
                recipe.created_at,
                utc_timestamp(),
            ),
        )
        self._connection.commit()
        return recipe_id

    def load_recipe(self, recipe_id: str) -> SelectionRecipe:
        row = self._connection.execute(
            "SELECT config_json FROM selection_recipes WHERE recipe_id = ?", (recipe_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown recipe '{recipe_id}'")
        return from_dict(json.loads(row["config_json"]), source=recipe_id)

    def recipe_ids(self) -> list[str]:
        return [
            row["recipe_id"]
            for row in self._connection.execute(
                "SELECT recipe_id FROM selection_recipes ORDER BY updated_at DESC"
            )
        ]

    # -- Chemical Space Hub -----------------------------------------------

    def save_hub_libraries(self, libraries: Iterable[dict[str, Any]]) -> int:
        rows = [
            (
                str(row["library_id"]),
                str(row.get("role", "reference")),
                row.get("source", ""),
                row.get("description", ""),
            )
            for row in libraries
        ]
        self._connection.executemany(
            "INSERT OR REPLACE INTO hub_libraries "
            "(library_id, role, source, description) VALUES (?, ?, ?, ?)",
            rows,
        )
        self._connection.commit()
        return len(rows)

    def save_hub_analysis(
        self,
        overlaps: Iterable[dict[str, Any]] = (),
        similarities: Iterable[dict[str, Any]] = (),
    ) -> tuple[int, int]:
        overlap_rows = [
            tuple(row.get(key) for key in (
                "candidate_index", "candidate_id", "reference_library",
                "reference_index", "reference_id", "match_type",
            ))
            for row in overlaps
        ]
        similarity_rows = [
            tuple(row.get(key) for key in (
                "record_id", "candidate_id", "reference_library",
                "reference_index", "reference_id", "similarity",
            ))
            for row in similarities
        ]
        self._connection.executemany(
            "INSERT INTO hub_reference_overlap VALUES (?, ?, ?, ?, ?, ?)", overlap_rows
        )
        self._connection.executemany(
            "INSERT INTO hub_reference_similarity VALUES (?, ?, ?, ?, ?, ?)", similarity_rows
        )
        self._connection.commit()
        return len(overlap_rows), len(similarity_rows)

    def save_hub_zones(self, zones: Iterable[dict[str, Any]]) -> int:
        rows = [
            (
                str(row["zone_id"]), row.get("label", ""), row.get("expression", ""),
                row.get("quota"), row.get("priority", 0), row.get("color", ""),
            )
            for row in zones
        ]
        self._connection.executemany(
            "INSERT OR REPLACE INTO hub_zones "
            "(zone_id, label, expression, quota, priority, color) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._connection.commit()
        return len(rows)

    def save_hub_projection(self, projection: Any) -> str:
        block = projection.recipe_block()
        projection_id = str(projection.projection.projection_id)
        self._connection.execute(
            "INSERT OR REPLACE INTO hub_projections "
            "(projection_id, method, input_hash, parameters_json, recipe_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                projection_id,
                str(block.get("method", projection.projection.method)),
                block.get("input_hash", ""),
                json.dumps(block.get("parameters", {}), ensure_ascii=False),
                json.dumps(block, ensure_ascii=False),
            ),
        )
        coordinates = projection.projection.coordinates
        rows = [
            (str(record_id), projection_id, float(row["x"]), float(row["y"]), None)
            for record_id, row in coordinates.iterrows()
        ]
        self._connection.executemany(
            "INSERT OR REPLACE INTO hub_projection_coordinates "
            "(record_id, projection_id, x, y, cluster_id) VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        self._connection.commit()
        return projection_id

    # -- analysis tables ----------------------------------------------------

    def save_pareto(self, recipe_id: str, rows: Iterable[dict[str, Any]]) -> int:
        payload = [
            (
                int(row["record_id"]),
                recipe_id,
                int(row["pareto_rank"]),
                row.get("domination_count"),
                row.get("dominated_count"),
                row.get("distance_to_ideal"),
            )
            for row in rows
        ]
        self._connection.executemany(
            "INSERT OR REPLACE INTO pareto_results (record_id, recipe_id, pareto_rank, "
            "domination_count, dominated_count, distance_to_ideal) VALUES (?, ?, ?, ?, ?, ?)",
            payload,
        )
        self._connection.commit()
        return len(payload)

    def save_margins(self, rows: Iterable[dict[str, Any]]) -> int:
        payload = [
            (
                int(row["record_id"]),
                row["rule_id"],
                row.get("observed_value"),
                row.get("threshold_low"),
                row.get("threshold_high"),
                row.get("normalized_margin"),
                row.get("margin_status"),
            )
            for row in rows
        ]
        self._connection.executemany(
            "INSERT OR REPLACE INTO rule_margins (record_id, rule_id, observed_value, "
            "threshold_low, threshold_high, normalized_margin, margin_status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            payload,
        )
        self._connection.commit()
        return len(payload)

    def save_coordinates(self, projection_id: str, rows: Iterable[dict[str, Any]]) -> int:
        payload = [
            (
                int(row["record_id"]),
                projection_id,
                float(row["x"]),
                float(row["y"]),
                row.get("cluster_id"),
            )
            for row in rows
        ]
        self._connection.executemany(
            "INSERT OR REPLACE INTO chemical_space_coordinates "
            "(record_id, projection_id, x, y, cluster_id) VALUES (?, ?, ?, ?, ?)",
            payload,
        )
        self._connection.commit()
        return len(payload)

    def read(self, query: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        return [dict(row) for row in self._connection.execute(query, tuple(params))]

    def count(self, table: str) -> int:
        return int(self._connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
