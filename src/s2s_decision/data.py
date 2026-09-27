"""Local, streamed bioactivity ingestion. Unknown measurements stay unknown."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

_ENDPOINTS = {"IC50": "IC50", "EC50": "EC50", "KD": "Kd", "KI": "Ki"}
_RELATIONS = {"=", "<", "<=", ">", ">=", "~"}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _chunks(path, chunk_size):
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    path = Path(path)
    suffixes = [s.lower() for s in path.suffixes]
    if suffixes and suffixes[-1] in {".gz", ".xz"}:
        suffixes = suffixes[:-1]
    if not suffixes or suffixes[-1] not in {".csv", ".tsv", ".txt"}:
        raise ValueError("Source must be CSV/TSV/TXT, optionally gz/xz compressed")
    return pd.read_csv(
        path,
        sep="," if suffixes[-1] == ".csv" else "\t",
        dtype=str,
        keep_default_na=False,
        chunksize=chunk_size,
    )


def _endpoint(value):
    return _ENDPOINTS.get(str(value).strip().upper(), "")


def _number(value):
    if str(value).strip().lower() in {"", "nan", "na", "n/a", "none", "null", "<na>"}:
        return np.nan
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("pactivity must be numeric or missing") from exc
    if not math.isfinite(result):
        raise ValueError("pactivity must be finite")
    return result


def _label(value, relation, threshold):
    if math.isnan(value) or relation == "~":
        return np.nan
    if relation == "=":
        return float(value >= threshold)
    if relation in {">", ">="} and value >= threshold:
        return 1.0
    if relation == "<" and value <= threshold:
        return 0.0
    if relation == "<=" and value < threshold:
        return 0.0
    return np.nan


def _normalize(row, ordinal, threshold):
    papyrus = "pchembl_value_Mean" in row
    if papyrus:
        active = []
        for name in ("IC50", "EC50", "KD", "Ki", "other"):
            values = set(str(row.get("type_" + name, "")).split(";"))
            if name == "other" and values == {""}:
                values = {"0"}
            if not values.issubset({"0", "1", "0.0", "1.0"}):
                raise ValueError("Malformed Papyrus endpoint flag")
            if values & {"1", "1.0"}:
                active.append(name)
        endpoint = _endpoint(active[0]) if len(active) == 1 else ""
        mixed = len(active) > 1
    else:
        endpoint = _endpoint(row.get("endpoint", ""))
        mixed = ";" in row.get("endpoint", "")
    value = _number(row.get("pchembl_value_Mean" if papyrus else "pactivity", ""))
    relations = set(row.get("relation", "=").strip().split(";"))
    if not relations.issubset(_RELATIONS):
        raise ValueError("Malformed measurement relation")
    relation = next(iter(relations)) if len(relations) == 1 else "~"
    scale = row.get("relation_scale", "concentration" if papyrus else "pactivity")
    if scale not in {"concentration", "pactivity"}:
        raise ValueError("relation_scale must be concentration or pactivity")
    model_relation = (
        {"<": ">", "<=": ">=", ">": "<", ">=": "<="}.get(relation, relation)
        if scale == "concentration"
        else relation
    )
    # Papyrus Year is earliest publication; its aggregate may include later evidence.
    year_text = ";".join(
        filter(
            None, (row.get("year", row.get("Year", "")).strip(), row.get("all_years", "").strip())
        )
    )
    raw_years = [_number(v) for v in year_text.split(";")] if year_text else []
    if any(math.isfinite(y) and not y.is_integer() for y in raw_years):
        raise ValueError("Year must be an integer")
    years = [int(y) for y in raw_years if math.isfinite(y)]
    if any(y < 1800 or y > 3000 for y in years):
        raise ValueError("Invalid year")
    date = row.get("date", "").strip()
    parsed_date = pd.to_datetime(date, errors="raise") if date else None
    year = max(years) if years else (parsed_date.year if parsed_date is not None else np.nan)
    original_smiles = row.get("original_smiles", row.get("SMILES", row.get("smiles", "")))
    result = dict(row)
    result.update(
        record_id=ordinal,
        source_row=ordinal,
        molecule_id=row.get("molecule_id", row.get("connectivity", row.get("InChIKey", ""))),
        measurement_id=row.get("measurement_id", row.get("Activity_ID", f"row:{ordinal}")),
        target_id=row.get("target_id", "").strip(),
        endpoint=endpoint,
        measured_pactivity=value,
        pactivity=value if relation == "=" else np.nan,
        relation=relation,
        relation_scale=scale,
        y_active=_label(value, model_relation, threshold) if endpoint and not mixed else np.nan,
        year=year,
        date=date,
        source=row.get("source", ""),
        quality=row.get("quality", row.get("Quality", "")).lower(),
        original_smiles=original_smiles,
        mixed_endpoint=mixed,
        raw_record_json=json.dumps(row, sort_keys=True, ensure_ascii=False),
    )
    return result


def _normalized(path, threshold, chunk_size):
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    ordinal = 0
    with _chunks(path, chunk_size) as chunks:
        for chunk in chunks:
            if "target_id" not in chunk or not (
                {"endpoint", "pchembl_value_Mean"} & set(chunk.columns)
            ):
                raise ValueError("Source requires target_id and endpoint/pchembl_value_Mean")
            records = []
            for row in chunk.to_dict("records"):
                ordinal += 1
                records.append(_normalize(row, ordinal, threshold))
            yield records


def audit_source(path, threshold=6.0, chunk_size=10000):
    """Audit all records; no structures or classes are invented for missing fields."""
    counts = Counter()
    targets, endpoints, quality = Counter(), Counter(), Counter()
    task_counts = {}
    scaffold_values = set()
    has_scaffolds = False
    for records in _normalized(path, threshold, chunk_size):
        for row in records:
            counts["rows"] += 1
            counts["dated_records"] += int(pd.notna(row["year"]))
            counts["missing_structures"] += int(not row["original_smiles"].strip())
            counts["mixed_endpoint_rows"] += int(row["mixed_endpoint"])
            counts["censored_rows"] += int(row["relation"] != "=")
            label = row["y_active"]
            counts[
                "unknown_labels"
                if pd.isna(label)
                else ("active_records" if label else "inactive_records")
            ] += 1
            targets[row["target_id"] or "(missing)"] += 1
            endpoints[row["endpoint"] or "(mixed/unknown)"] += 1
            quality[row["quality"] or "(unspecified)"] += 1
            task = task_counts.setdefault((row["target_id"], row["endpoint"]), Counter())
            task["rows"] += 1
            task["dated_records"] += int(pd.notna(row["year"]))
            task["exact_records"] += int(pd.notna(row["pactivity"]))
            task["high_quality_records"] += int(row["quality"] == "high")
            task[
                "unknown_labels"
                if pd.isna(label)
                else ("active_records" if label else "inactive_records")
            ] += 1
            if "murcko_scaffold" in row:
                has_scaffolds = True
                scaffold_values.add(row["murcko_scaffold"])
    keys = (
        "rows",
        "dated_records",
        "missing_structures",
        "mixed_endpoint_rows",
        "censored_rows",
        "unknown_labels",
        "active_records",
        "inactive_records",
    )
    return {
        **{key: counts[key] for key in keys},
        "targets": dict(targets),
        "endpoints": dict(endpoints),
        "qualities": dict(quality),
        "target_endpoint_counts": [
            dict(
                target_id=target,
                endpoint=endpoint,
                **{
                    key: value[key]
                    for key in (
                        "rows",
                        "dated_records",
                        "exact_records",
                        "active_records",
                        "inactive_records",
                        "unknown_labels",
                        "high_quality_records",
                    )
                },
            )
            for (target, endpoint), value in sorted(task_counts.items())
        ],
        "scaffold_count": len(scaffold_values) if has_scaffolds else None,
        "source_sha256": _sha256(path),
        "threshold": threshold,
        "scaffold_status": "provided_in_source" if has_scaffolds else "requires_feature_generation",
    }


def load_measurements(path, target_id, endpoint, threshold=6.0, max_rows=250000):
    """Load one endpoint; Papyrus defaults to high quality. Preserve censored boundaries."""
    endpoint = _endpoint(endpoint)
    if not target_id or not endpoint or not isinstance(max_rows, int) or max_rows < 1:
        raise ValueError("Explicit target_id, supported endpoint and positive max_rows required")
    selected = []
    for records in _normalized(path, threshold, min(max_rows + 1, 10000)):
        for row in records:
            if (
                row["target_id"] != target_id
                or row["endpoint"] != endpoint
                or row["mixed_endpoint"]
            ):
                continue
            if "pchembl_value_Mean" in row and row["quality"] != "high":
                continue
            selected.append(row)
            if len(selected) > max_rows:
                raise ValueError(
                    "Selected measurements exceed max_rows; narrow target/endpoint or explicitly raise bound"
                )
    if not selected:
        raise ValueError("No matching measurements after endpoint and quality filtering")
    frame = pd.DataFrame.from_records(selected)
    frame.attrs.update(
        source_sha256=_sha256(path),
        threshold=threshold,
        target_id=target_id,
        endpoint=endpoint,
        quality_policy="Papyrus high only; custom quality preserved",
    )
    return frame
