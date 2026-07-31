"""Persistent descriptor cache.

Parsing, standardizing and describing a molecule is the expensive part of a
run. This cache lets a second run over an overlapping library skip that work.

A cached value is only reused when everything that could change it is
identical: the RDKit version, the standardization profile (salts, charges,
tautomers, stereo), and the alert catalogues in force. Those are folded into
the cache key, so a value computed under one configuration can never surface
under another.

The descriptor set is *not* part of the key. It is checked separately: an entry
is reused only if it already holds every descriptor the current run asked for,
and a molecule whose entry is missing one is recomputed and stored again with
the union. That way adding a profile re-uses the work done for the others.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path

from smiles2select.app_metadata import rdkit_version
from smiles2select.chemistry.standardization import StandardizationConfig
from smiles2select.pipeline.workers import RecordResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS descriptor_cache (
    cache_key TEXT PRIMARY KEY,
    original_smiles TEXT NOT NULL,
    standardized_smiles TEXT,
    canonical_smiles TEXT,
    valid INTEGER NOT NULL,
    invalid_reason TEXT,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cache_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""

#: SQLite caps a statement at 999 variables by default.
_QUERY_CHUNK = 400


class DescriptorCache:
    """Key/value store of per-molecule chemistry results."""

    def __init__(
        self,
        path: str | Path,
        standardization: StandardizationConfig,
        catalog_ids: Sequence[str] = (),
        custom_smarts: Sequence[str] = (),
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.executescript(SCHEMA)
        self._connection.commit()

        self._signature = _signature(standardization, catalog_ids, custom_smarts)
        self._hits = 0
        self._misses = 0
        self._store_meta()

    def __enter__(self) -> DescriptorCache:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.commit()
        self._connection.close()

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def key_for(self, smiles: str) -> str:
        """Cache key: the molecule plus everything that could change its values."""
        digest = hashlib.sha256()
        digest.update(smiles.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(self._signature.encode("utf-8"))
        return digest.hexdigest()

    def fetch(
        self, smiles_list: Iterable[str], descriptor_ids: Sequence[str]
    ) -> dict[str, RecordResult]:
        """Entries usable for this run, keyed by the original SMILES.

        An entry that does not hold every requested descriptor is skipped, so
        the caller recomputes that molecule rather than evaluating a rule
        against a missing value.
        """
        wanted = set(descriptor_ids)
        unique = list(dict.fromkeys(smiles_list))
        by_key = {self.key_for(smiles): smiles for smiles in unique}
        found: dict[str, RecordResult] = {}

        keys = list(by_key)
        for start in range(0, len(keys), _QUERY_CHUNK):
            chunk = keys[start : start + _QUERY_CHUNK]
            placeholders = ",".join("?" * len(chunk))
            rows = self._connection.execute(
                "SELECT cache_key, standardized_smiles, canonical_smiles, valid, "
                f"invalid_reason, payload FROM descriptor_cache WHERE cache_key IN ({placeholders})",
                chunk,
            ).fetchall()
            for cache_key, standardized, canonical, valid, reason, payload in rows:
                data = json.loads(payload)
                descriptors = data.get("descriptors", {})
                if valid and not wanted.issubset(descriptors):
                    continue  # incomplete for this run; recompute instead
                found[by_key[cache_key]] = RecordResult(
                    record_id=-1,
                    valid=bool(valid),
                    invalid_reason=reason,
                    standardized_smiles=standardized or "",
                    canonical_smiles=canonical or "",
                    descriptors={key: descriptors.get(key) for key in descriptor_ids},
                    substructure_flags=data.get("substructure_flags", {}),
                    alert_rows=data.get("alert_rows", []),
                )

        self._hits += len(found)
        self._misses += len(unique) - len(found)
        return found

    def store(self, smiles_by_record: dict[int, str], results: Iterable[RecordResult]) -> int:
        """Persist freshly computed results.

        Alert rows are stored without their ``record_id``: that number belongs
        to one run, not to the molecule.
        """
        rows = []
        seen: set[str] = set()
        for result in results:
            smiles = smiles_by_record.get(result.record_id)
            if smiles is None or smiles in seen:
                continue
            seen.add(smiles)
            payload = {
                "descriptors": result.descriptors,
                "substructure_flags": result.substructure_flags,
                "alert_rows": [
                    {key: value for key, value in row.items() if key != "record_id"}
                    for row in result.alert_rows
                ],
            }
            rows.append(
                (
                    self.key_for(smiles),
                    smiles,
                    result.standardized_smiles,
                    result.canonical_smiles,
                    int(result.valid),
                    result.invalid_reason,
                    json.dumps(payload, ensure_ascii=False, default=str),
                )
            )

        if not rows:
            return 0
        self._connection.executemany(
            "INSERT OR REPLACE INTO descriptor_cache "
            "(cache_key, original_smiles, standardized_smiles, canonical_smiles, valid, "
            "invalid_reason, payload) VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._connection.commit()
        return len(rows)

    def size(self) -> int:
        return int(self._connection.execute("SELECT COUNT(*) FROM descriptor_cache").fetchone()[0])

    def _store_meta(self) -> None:
        self._connection.executemany(
            "INSERT OR REPLACE INTO cache_meta (key, value) VALUES (?, ?)",
            [("rdkit_version", rdkit_version()), ("signature", self._signature)],
        )
        self._connection.commit()


def _signature(
    standardization: StandardizationConfig,
    catalog_ids: Sequence[str],
    custom_smarts: Sequence[str],
) -> str:
    """Everything that invalidates a cached value, as one hash."""
    payload = json.dumps(
        {
            "standardization": standardization.fingerprint(),
            "rdkit": rdkit_version(),
            "catalogs": sorted(catalog_ids),
            "custom_smarts": sorted(custom_smarts),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def rebind(result: RecordResult, record_id: int) -> RecordResult:
    """Attach a cached result to the record that requested it."""
    alert_rows = [{**row, "record_id": record_id} for row in result.alert_rows]
    return replace(result, record_id=record_id, alert_rows=alert_rows)
