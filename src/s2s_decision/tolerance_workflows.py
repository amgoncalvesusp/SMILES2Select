"""Local custom-data preparation and profile fitting with explicit activity provenance."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import file_hash, spreadsheet_safe, write_json
from .data import _normalize
from .features import featurize
from .tolerance_features import annotate_tolerance

_UNITS = {"m": 1.0, "mm": 1e-3, "um": 1e-6, "nm": 1e-9, "pm": 1e-12}
_ROLES = ("fit", "tune", "policy_cal", "test")


def tolerance_context(target_id=None, species="unknown", endpoint=None):
    """Unspecified defaults do not override known input or package context."""
    return {key: value for key, value in
            (("target_id", target_id), ("species", species), ("endpoint", endpoint))
            if value and value != "unknown"}


def _check_calibration_overlap(original, calibration):
    protected = original.loc[original.split.ne("policy_cal") & original.valid]
    identities = set(protected.identity.dropna()) & set(calibration.identity.dropna())
    scaffolds = set(protected.murcko_scaffold.dropna()) & set(calibration.murcko_scaffold.dropna())
    if identities or scaffolds - {""}:
        raise ValueError("External calibration identity/scaffold overlap with fit/tune/test groups")


def _context_mask(records, context):
    result = pd.Series(True, index=records.index)
    for key, value in context.items():
        if key not in records:
            return pd.Series(False, index=records.index)
        result = result & records[key].fillna("").astype(str).eq(str(value))
    return result


def _text(value):
    return "" if pd.isna(value) else str(value).strip()


def curate_custom(frame, *, target_id=None, species="unknown", endpoint=None):
    """Convert units/censoring without inventing activity or experimental metadata."""
    aliases = {"SMILES": "smiles", "pActivity": "pactivity", "target": "target_id",
               "units": "unit", "activity_value": "value"}
    if any(old in frame and new in frame for old, new in aliases.items()):
        raise ValueError("Ambiguous custom column aliases; retain one name per field")
    source = frame.rename(columns={old: new for old, new in aliases.items() if old in frame})
    if not {"original_smiles", "smiles", "model_smiles"} & set(source):
        raise ValueError("Custom data requires a SMILES column")
    rows = []
    for ordinal, raw in enumerate(source.to_dict("records"), 1):
        row = {key: _text(value) for key, value in raw.items()}
        row["original_smiles"] = row.get("original_smiles") or row.get("smiles") or row.get("model_smiles", "")
        row["target_id"] = row.get("target_id") or target_id or "unspecified"
        row["endpoint"] = row.get("endpoint") or endpoint or ""
        row["species"] = row.get("species") or species or "unknown"
        row["relation"] = row.get("relation") or "="
        concentration = row.get("value", "")
        if concentration:
            unit = row.get("unit", "").replace("μ", "u").replace("µ", "u").lower()
            if unit not in _UNITS:
                raise ValueError(f"Row {ordinal}: concentration unit must be M/mM/uM/nM/pM")
            value = float(concentration) * _UNITS[unit]
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Row {ordinal}: concentration must be finite and positive")
            potency = -math.log10(value)
            if row.get("pactivity") and not math.isclose(float(row["pactivity"]), potency, abs_tol=1e-6):
                raise ValueError(f"Row {ordinal}: concentration and pactivity conflict")
            row["pactivity"] = str(potency)
            row["relation_scale"] = "concentration"
        else:
            row["relation_scale"] = row.get("relation_scale") or "pactivity"
        normalized = _normalize(row, ordinal, 6.5)
        normalized.update(species=row["species"],
            task_id="|".join((row["target_id"], row["species"], normalized["endpoint"])),
            pActivity=normalized["pactivity"],
            activity_status=("exact" if row["relation"] == "=" and pd.notna(normalized["pactivity"])
                             else "censored" if pd.notna(normalized["measured_pactivity"]) else "unknown"),
            document_id=row.get("document_id", ""), assay_id=row.get("assay_id", ""))
        rows.append(normalized)
    if not rows:
        raise ValueError("Custom data contains no rows")
    records = pd.DataFrame(rows)
    audit = {"input_rows": len(records), "threshold": 6.5, "comparison": ">=",
        "exact_positive_rows": int((records.y_active.eq(1) & records.activity_status.eq("exact")).sum()),
        "censored_positive_rows": int((records.y_active.eq(1) & records.activity_status.eq("censored")).sum()),
        "unknown_activity_rows": int(records.y_active.isna().sum()),
        "defaults": {"target_id": target_id or "unspecified", "species": species or "unknown",
                     "endpoint": endpoint, "relation": "= when not supplied"},
        "status": "custom_experimental; no external validation"}
    return records, audit


def _load_table(path):
    root = Path(path)
    if root.is_dir():
        metadata = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if metadata.get("schema") != "s2s-tolerance-custom-data/1":
            raise ValueError("Not a prepared tolerance dataset")
        records = root / "records.jsonl"
        if file_hash(records) != metadata.get("records_sha256"):
            raise ValueError("Prepared dataset checksum mismatch")
        return pd.read_json(records, orient="records", lines=True, dtype=False, convert_dates=False), metadata
    if root.suffix.lower() == ".parquet":
        return pd.read_parquet(root), {}
    delimiter = "\t" if root.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(root, sep=delimiter, dtype=str, keep_default_na=False), {}


def _custom_records(path, target_id=None, species="unknown", endpoint=None):
    raw, metadata = _load_table(path)
    if metadata:
        return raw, metadata
    curated, audit = curate_custom(raw, target_id=target_id, species=species, endpoint=endpoint)
    annotated, schema = annotate_tolerance(featurize(curated).records)
    conflicts = annotated.groupby(["identity", "task_id"], dropna=False).y_active.transform(
        lambda values: values.dropna().nunique() > 1)
    annotated = annotated.assign(
        pactivity=annotated.pactivity.mask(conflicts), pActivity=annotated.pActivity.mask(conflicts),
        y_active=annotated.y_active.mask(conflicts),
        curation_status=np.where(conflicts, "conflicting_activity", "retained"))
    groups = annotated.murcko_scaffold.fillna("").where(
        annotated.murcko_scaffold.fillna("").ne(""), "acyclic:" + annotated.identity.fillna("invalid"))
    buckets = groups.map(lambda value: int(hashlib.sha256(("20261009:" + value).encode()).hexdigest()[:8], 16) % 100)
    roles = pd.Series(np.select([buckets < 60, buckets < 75, buckets < 85], _ROLES[:3], default="test"),
                      index=annotated.index)
    # Identity takes precedence when one identity has incompatible supplied scaffold labels.
    roles = roles.groupby(annotated.identity).transform("first").fillna("fit")
    annotated = annotated.assign(split=roles)
    audit = {**audit, "conflicting_rows": int(conflicts.sum()),
             "invalid_rows": int((~annotated.valid).sum()),
             "split_counts": annotated.split.value_counts().to_dict(),
             "unique_identities": int(annotated.identity.nunique())}
    return annotated, {"schema": "s2s-tolerance-custom-data/1", "source_sha256": file_hash(path),
        "source_path": str(Path(path).resolve()), "annotation_schema": schema,
        "audit": audit, "split_method": "global_scaffold_hash_60_15_10_15",
        "seed": 20261009, "temporal_status": "not evaluated"}


def prepare_tolerance(path, output, *, target_id=None, species="unknown", endpoint=None):
    destination = Path(output)
    if destination.exists():
        raise ValueError("output already exists")
    records, metadata = _custom_records(path, target_id, species, endpoint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".s2s-tolerance-data-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "prepared"
        staging.mkdir()
        record_file = staging / "records.jsonl"
        records.to_json(record_file, orient="records", lines=True, double_precision=15)
        write_json(staging / "manifest.json", {**metadata, "records_sha256": file_hash(record_file),
                                               "row_count": len(records)})
        write_json(staging / "validation_report.json", metadata["audit"])
        staging.rename(destination)
    return {"output": str(destination), "rows": len(records), "validation_report": metadata["audit"]}


def train_tolerance(path, output, *, mode="general", recipe="A", target_id=None,
                    species="unknown", endpoint=None, positive_cal=None, background_cal=None,
                    target_retention=.95, max_expansion=.1, base_profile=None,
                    reconsidered_criteria=None):
    from .tolerance_profile import calibrate_tolerance, fit_tolerance, write_tolerance_package
    from .tolerance_schema import seal_profile

    if (positive_cal is None) != (background_cal is None):
        raise ValueError("Provide both positive and background calibration files")
    records, metadata = _custom_records(path, target_id, species, endpoint)
    context = tolerance_context(target_id, species, endpoint)
    positives = records.y_active.eq(1) & records.activity_status.eq("exact") & records.valid
    matched = _context_mask(records, context)
    fit = records.loc[positives & matched & records.split.eq("fit")].copy()
    if fit.empty:
        raise ValueError("No exact positive training records; inspect validation and split report")
    if base_profile is None:
        profile = fit_tolerance(fit, mode=mode, recipe=recipe, context=context, provenance=metadata)
    else:
        from .tolerance_profile import adapt_tolerance, read_tolerance_package

        parent = read_tolerance_package(base_profile)
        reserved = records.loc[records.split.ne("fit") & records.valid]
        identities = {row["identity"] for row in parent["references"]}
        scaffolds = {row["scaffold"] for row in parent["references"]} - {""}
        if identities & set(reserved.identity) or scaffolds & set(reserved.murcko_scaffold):
            raise ValueError("Shared profile training overlap with custom tune/calibration/test groups")
        profile = adapt_tolerance(parent, fit, mode=mode, context=context, provenance=metadata)
        if profile["recipe"] != recipe:
            raise ValueError("Adaptation recipe must match shared profile frozen representation")
    calibration = records.loc[records.split.eq("policy_cal") & records.valid]
    if positive_cal is not None:
        positive_rows, positive_metadata = _custom_records(positive_cal, target_id, species, endpoint)
        background_rows, background_metadata = _custom_records(background_cal, target_id, species, endpoint)
        _check_calibration_overlap(records, positive_rows)
        _check_calibration_overlap(records, background_rows)
        if base_profile is not None:
            for external in (positive_rows, background_rows):
                if identities & set(external.identity) or scaffolds & set(external.murcko_scaffold):
                    raise ValueError("Shared profile training overlap with external calibration groups")
        positive_rows = positive_rows.loc[positive_rows.y_active.eq(1)
                                          & positive_rows.activity_status.eq("exact") & positive_rows.valid
                                          & _context_mask(positive_rows, context)]
        background_rows = background_rows.loc[background_rows.valid & ~background_rows.y_active.eq(1)]
        positive_rows = positive_rows.assign(split="policy_cal")
        background_rows = background_rows.assign(split="policy_cal")
        calibration_sources = {"positive_source_sha256": positive_metadata.get("source_sha256"),
                               "background_source_sha256": background_metadata.get("source_sha256")}
    else:
        positive_rows = calibration.loc[calibration.y_active.eq(1)
                                        & calibration.activity_status.eq("exact")
                                        & _context_mask(calibration, context)].copy()
        background_rows = calibration.loc[~calibration.y_active.eq(1)].copy()
        calibration_sources = {"source": "reserved custom scaffold policy_cal partition"}
    known_active = set(records.loc[records.y_active.eq(1), "identity"].dropna())
    known_active |= set(positive_rows.identity.dropna())
    known_active |= {row["identity"] for row in profile["references"]}
    removed_background = int(background_rows.identity.isin(known_active).sum())
    background_rows = background_rows.loc[~background_rows.identity.isin(known_active)].copy()
    calibration_sources = {**calibration_sources,
                           "known_active_background_rows_removed": removed_background}
    if len(positive_rows):
        profile = calibrate_tolerance(profile, positive_rows, background_rows if len(background_rows) else None,
                                      target_retention=target_retention, max_expansion=max_expansion,
                                      reconsidered_criteria=reconsidered_criteria)
    else:
        profile = seal_profile({**profile, "custom_validation": {
            "status": "insufficient_calibration", "positive_cal_count": len(positive_rows),
            "background_cal_count": len(background_rows),
            "reason": "Independent positive retention and expansion could not both be assessed"}})
    profile = seal_profile({**profile, "provenance": {**profile["provenance"],
        "custom_calibration_sources": calibration_sources,
        "custom_requested_context": context, "context_matched_rows": int(matched.sum()),
        "context_excluded_rows": int((~matched).sum())}})
    write_tolerance_package(profile, output)
    return {"output": str(output), "mode": mode, "recipe": recipe,
        "status": "experimental_custom", "training_count": len(fit),
        "training_mode": "adapt_shared" if base_profile is not None else "from_scratch",
        "calibration_status": profile["policy"],
        "validation_report": metadata["audit"]}


def predict_tolerance_file(path, output, model, *, context=None):
    from .tolerance_profile import predict_tolerance, read_tolerance_package

    records, _metadata = _load_table(path)
    if not _metadata:
        names = {name: "original_smiles" for name in ("SMILES", "smiles", "model_smiles")
                 if name in records and "original_smiles" not in records}
        if len(names) > 1:
            raise ValueError("Prediction contains ambiguous SMILES columns")
        records = featurize(records.rename(columns=names)).records
    profile = read_tolerance_package(model)
    predicted = predict_tolerance(profile, records, context=context)
    if not predicted.record_id.is_unique or set(predicted.record_id) != set(records.record_id):
        raise ValueError("Tolerance inference changed input record identities")
    aligned = predicted.set_index("record_id").reindex(records.record_id)
    predicted = records.assign(**{name: aligned[name].to_numpy() for name in aligned})
    destination = Path(output)
    if destination.exists():
        raise ValueError("output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".s2s-tolerance-predict-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "prediction"
        staging.mkdir()
        predicted.to_json(staging / "records.jsonl", orient="records", lines=True, double_precision=15)
        spreadsheet_safe(predicted.drop(columns=["fingerprint_hex"], errors="ignore")).to_csv(
            staging / "decisions.csv", index=False)
        report = {"rows": len(predicted), "state_counts": predicted.tolerance_state.value_counts().to_dict(),
                  "meaning": "Chemical compatibility, not activity or safety probability"}
        write_json(staging / "report.json", report)
        staging.rename(destination)
    return {"output": str(destination), **report}
