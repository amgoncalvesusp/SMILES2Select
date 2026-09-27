"""Small, hash-verified local data bundles. No pickle is accepted."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from .schema import SCHEMA_VERSION, FeatureSet


def file_hash(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: str | Path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    handle, temporary = tempfile.mkstemp(prefix=".s2s-", dir=destination.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(serialized)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_bundle(features: FeatureSet, path: str | Path) -> Path:
    destination = Path(path)
    if destination.exists():
        raise ValueError(f"output already exists: {destination}; choose a new directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".s2s-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        records = staging / "records.jsonl"
        features.records.to_json(records, orient="records", lines=True, double_precision=15)
        manifest = {
            **features.manifest,
            "bundle_version": 1,
            "feature_schema": SCHEMA_VERSION,
            "records_sha256": file_hash(records),
            "row_count": len(features.records),
        }
        write_json(staging / "manifest.json", manifest)
        staging.rename(destination)
    return destination


def read_bundle(path: str | Path) -> FeatureSet:
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("bundle_version") != 1 or manifest.get("feature_schema") != SCHEMA_VERSION:
        raise ValueError("unsupported bundle or feature schema")
    records = root / "records.jsonl"
    if file_hash(records) != manifest.get("records_sha256"):
        raise ValueError("records checksum does not match manifest")
    # dtype=False preserves numeric-looking molecule IDs and all original strings.
    frame = pd.read_json(records, orient="records", lines=True, dtype=False, convert_dates=False)
    if len(frame) != manifest.get("row_count"):
        raise ValueError("bundle row count does not match manifest")
    frame.attrs["provenance"] = manifest
    return FeatureSet(frame, manifest)


def spreadsheet_safe(frame: pd.DataFrame) -> pd.DataFrame:
    """Escape spreadsheet formula text only; the JSONL bundle remains lossless."""

    def escape(value):
        if isinstance(value, str) and value.lstrip(" \t\r\n\ufeff").startswith(
            ("=", "+", "-", "@")
        ):
            return "'" + value
        return value

    return frame.apply(lambda column: column.map(escape)).rename(columns=escape)
