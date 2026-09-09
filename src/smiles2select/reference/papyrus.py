"""Optional, local Papyrus evidence: bounded import and indexed identity lookup.

No downloading, similarity calculation or activity prediction occurs here.
Mixed endpoints and censored records are excluded, never silently averaged.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import tempfile
from collections.abc import Callable, Iterable
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

_ENDPOINTS = ("IC50", "EC50", "KD", "Ki")
_FLAGS = tuple(f"type_{value}" for value in (*_ENDPOINTS, "other"))
_KEY = re.compile(r"[A-Z]{14}-[A-Z]{10}-[A-Z]")
_CONNECTIVITY = re.compile(r"[A-Z]{14}")


@dataclass(frozen=True)
class ImportReport:
    index_path: Path
    rows_read: int
    rows_indexed: int
    mixed_endpoint_rows: int
    censored_rows: int


def _options(target_id, endpoint, dataset_version, qualities, identity_level, chunk_size):
    if any(
        not isinstance(value, str) or not value.strip() for value in (target_id, dataset_version)
    ):
        raise ValueError("target_id and dataset_version must be explicit and nonempty")
    if endpoint not in _ENDPOINTS:
        raise ValueError(f"endpoint must be one of {_ENDPOINTS}")
    if not qualities or not set(qualities).issubset({"high", "medium", "low"}):
        raise ValueError("qualities must explicitly contain high, medium or low")
    if identity_level not in {"connectivity", "inchikey"}:
        raise ValueError("identity_level must be connectivity or inchikey")
    if not isinstance(chunk_size, int) or not 1 <= chunk_size <= 100_000:
        raise ValueError("chunk_size must be between 1 and 100000")


def _schema(source: Path, identity_level: str):
    uncompressed = source.with_suffix("") if source.suffix in {".gz", ".xz"} else source
    sep = "," if uncompressed.suffix.lower() == ".csv" else "\t"
    columns = list(pd.read_csv(source, sep=sep, nrows=0).columns)
    identity_column = "InChIKey" if identity_level == "inchikey" else "connectivity"
    if identity_column not in columns and identity_level == "connectivity":
        identity_column = "InChIKey"
    required = {
        identity_column,
        "target_id",
        "Quality",
        "relation",
        "source",
        "Activity_ID",
        "pchembl_value_Mean",
        *_FLAGS,
    }
    missing = required.difference(columns)
    if missing:
        raise ValueError(f"Papyrus is missing required columns: {sorted(missing)}")
    return sep, sorted(required), identity_column


def _key(value: object, level: str) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if level == "connectivity":
        if _KEY.fullmatch(text):
            return text[:14]
        return text if _CONNECTIVITY.fullmatch(text) else None
    return text if _KEY.fullmatch(text) else None


def _record(row: dict, identity_column: str, level: str, endpoint: str):
    flags = {name: set(row[name].split(";")) for name in _FLAGS}
    if any(not values.issubset({"0", "1", "0.0", "1.0"}) for values in flags.values()):
        raise ValueError("Malformed Papyrus endpoint flag")
    active = {name for name, values in flags.items() if values.intersection({"1", "1.0"})}
    if f"type_{endpoint}" not in active:
        return "other", None
    # ponytail: exclude mixed aggregates; endpoint-aware unnesting needs original assays.
    if len(active) != 1:
        return "mixed", None
    relations = set(row["relation"].split(";"))
    if not relations.issubset({"=", "<", ">", "<=", ">=", "~"}):
        raise ValueError("Malformed Papyrus measurement relation")
    if relations != {"="}:
        return "censored", None
    identity = _key(row[identity_column], level)
    if identity is None:
        raise ValueError("Malformed or missing Papyrus molecular identity")
    try:
        value = float(row["pchembl_value_Mean"])
    except (ValueError, TypeError) as exc:
        raise ValueError("Papyrus pChEMBL mean must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError("Papyrus pChEMBL mean must be finite")
    if not row["source"].strip() or not row["Activity_ID"].strip():
        raise ValueError("Papyrus source and Activity_ID must be nonempty")
    return "kept", (identity, value, row["Quality"].lower(), row["source"], row["Activity_ID"])


def import_index(
    source: str | Path,
    index_path: str | Path,
    *,
    target_id: str,
    endpoint: str,
    dataset_version: str,
    qualities: tuple[str, ...] = ("high",),
    identity_level: str = "connectivity",
    chunk_size: int = 10_000,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> ImportReport:
    """Stream a local TSV/CSV (.gz/.xz supported) into a NEW SQLite index.

    Target ID must match exactly (including Papyrus mutation suffix). Qualities
    are an explicit allowlist, not an ordinal threshold. Only pure endpoint,
    exact relation records are retained. Progress receives rows read/indexed.
    Failure/cancellation removes the temporary index; existing files are refused.
    """
    _options(target_id, endpoint, dataset_version, qualities, identity_level, chunk_size)
    source, destination = Path(source).resolve(), Path(index_path).resolve()
    if destination.exists():
        raise FileExistsError(f"Index already exists: {destination}")
    sep, columns, identity_column = _schema(source, identity_level)
    stat = source.stat()
    metadata = {
        "schema_version": 1,
        "target_id": target_id,
        "endpoint": endpoint,
        "dataset_version": dataset_version,
        "qualities": list(qualities),
        "identity_level": identity_level,
        "source_file": source.name,
        "source_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
    }
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".partial-", dir=destination.parent)
    os.close(fd)
    try:
        report = _populate(
            Path(temporary),
            source,
            sep,
            columns,
            identity_column,
            metadata,
            chunk_size,
            progress,
            cancelled,
        )
        # Windows rename refuses an existing destination, including a concurrent writer.
        if destination.exists():
            raise FileExistsError(f"Index already exists: {destination}")
        os.rename(temporary, destination)
        return ImportReport(destination, *report)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _populate(
    temporary, source, sep, columns, identity_column, metadata, chunk_size, progress, cancelled
):
    seen = kept = mixed = censored = 0
    with closing(sqlite3.connect(temporary)) as connection:
        connection.execute("PRAGMA cache_size=-8192")
        connection.execute(
            "CREATE TABLE evidence (identity TEXT, pchembl REAL, quality TEXT, "
            "source TEXT, activity_id TEXT)"
        )
        with pd.read_csv(
            source, sep=sep, usecols=columns, dtype=str, keep_default_na=False, chunksize=chunk_size
        ) as chunks:
            for chunk in chunks:
                if cancelled and cancelled():
                    raise InterruptedError("Papyrus import cancelled")
                seen += len(chunk)
                target = chunk.loc[chunk["target_id"].eq(metadata["target_id"])]
                target = target.loc[target["Quality"].str.lower().isin(metadata["qualities"])]
                records = []
                for row in target.to_dict("records"):
                    status, record = _record(
                        row, identity_column, metadata["identity_level"], metadata["endpoint"]
                    )
                    mixed += status == "mixed"
                    censored += status == "censored"
                    if record is not None:
                        records.append(record)
                connection.executemany("INSERT INTO evidence VALUES (?, ?, ?, ?, ?)", records)
                kept += len(records)
                connection.commit()
                if progress:
                    progress(seen, kept)
        if cancelled and cancelled():
            raise InterruptedError("Papyrus import cancelled")
        connection.execute("CREATE INDEX evidence_identity ON evidence(identity)")
        connection.execute("CREATE TABLE metadata (payload TEXT NOT NULL)")
        payload = dict(
            metadata,
            rows_read=seen,
            rows_indexed=kept,
            mixed_endpoint_rows=mixed,
            censored_rows=censored,
        )
        connection.execute("INSERT INTO metadata VALUES (?)", (json.dumps(payload),))
        connection.commit()
    return seen, kept, mixed, censored


def _connect(index_path):
    path = Path(index_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Papyrus index not found: {path}")
    return sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)


def read_index_metadata(index_path: str | Path) -> dict:
    """Read provenance and scope without reading the evidence table."""
    try:
        with closing(_connect(index_path)) as connection:
            row = connection.execute("SELECT payload FROM metadata").fetchone()
            payload = json.loads(row[0]) if row else {}
            if not isinstance(payload, dict) or payload.get("schema_version") != 1:
                raise ValueError("Unsupported Papyrus index schema")
            _options(
                payload["target_id"],
                payload["endpoint"],
                payload["dataset_version"],
                payload["qualities"],
                payload["identity_level"],
                10000,
            )
            connection.execute(
                "SELECT identity, pchembl, quality, source, activity_id FROM evidence LIMIT 0"
            )
            return payload
    except (sqlite3.DatabaseError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("Invalid Papyrus index") from exc


def lookup_evidence(index_path: str | Path, inchikeys: Iterable[str | None]) -> pd.DataFrame:
    """Return one summary per input identity, in input order; absent is unknown.

    Min/max describe stored record means, not raw assay ranges or predictions.
    Lookup queries batches of 500; call per candidate chunk for bounded output.
    Full records remain in SQLite for audit. No chemical calculation occurs.
    """
    metadata = read_index_metadata(index_path)
    keys = [_key(value, metadata["identity_level"]) for value in inchikeys]
    rows = []
    with closing(_connect(index_path)) as connection:
        for start in range(0, len(keys), 500):
            batch = keys[start : start + 500]
            valid = list(dict.fromkeys(key for key in batch if key is not None))
            found = {}
            if valid:
                placeholders = ",".join("?" for _ in valid)
                query = (
                    "SELECT identity, COUNT(*), MIN(pchembl), MAX(pchembl), "
                    "GROUP_CONCAT(DISTINCT quality), GROUP_CONCAT(DISTINCT source) "
                    f"FROM evidence WHERE identity IN ({placeholders}) GROUP BY identity"
                )
                found = {record[0]: record[1:] for record in connection.execute(query, valid)}
            rows.extend(found.get(key, (0, None, None, None, None)) for key in batch)
    frame = pd.DataFrame(
        rows,
        columns=[
            "papyrus_records",
            "papyrus_pchembl_min",
            "papyrus_pchembl_max",
            "papyrus_quality",
            "papyrus_source",
        ],
    )
    return frame.assign(
        papyrus_identity_level=metadata["identity_level"],
        papyrus_target_id=metadata["target_id"],
        papyrus_endpoint=metadata["endpoint"],
        papyrus_dataset_version=metadata["dataset_version"],
    )


def annotate_candidates(
    candidates: pd.DataFrame, index_path: str | Path, *, inchikey_column: str = "inchikey"
) -> pd.DataFrame:
    """Return annotations only, preserving candidate index and input unchanged."""
    if inchikey_column not in candidates:
        raise ValueError(f"Candidate identity column is required: {inchikey_column}")
    result = lookup_evidence(index_path, candidates[inchikey_column])
    return result.set_axis(candidates.index)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Index local Papyrus target evidence; no download."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("index", type=Path)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--endpoint", choices=_ENDPOINTS, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument(
        "--identity-level", choices=("connectivity", "inchikey"), default="connectivity"
    )
    args = parser.parse_args(argv)
    try:
        report = import_index(
            args.source,
            args.index,
            target_id=args.target_id,
            endpoint=args.endpoint,
            dataset_version=args.dataset_version,
            identity_level=args.identity_level,
            progress=lambda seen, kept: print(f"Read {seen:,}; indexed {kept:,}", flush=True),
        )
    except (ValueError, OSError, sqlite3.DatabaseError) as exc:
        parser.exit(1, f"Papyrus import failed: {exc}\n")
    print(json.dumps(asdict(report), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
