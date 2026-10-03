"""Prepare an audited target panel from complete direct ChEMBL activity snapshots.

Assay decisions are supplied by a scientific audit. This module never infers
mechanism from endpoint names and never assigns missing measurements to class 0.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd

from s2s_decision.artifacts import file_hash, write_json
from s2s_decision.features import chemistry_manifest, featurize
from s2s_decision.preprocessing import Preprocessor
from s2s_decision.schema import CONTEXT_NAMES, fingerprint_matrix
from s2s_decision.splits import assign_splits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_multitask import collapse_observations, make_labels, select_tasks


def load_snapshot(manifest_path: Path) -> tuple[list[dict], dict]:
    """Verify hashes and a complete, contiguous API pagination chain before use."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pages = manifest["pages"]
    collection = manifest["collection"]
    total = manifest["total_count"]
    if (
        not pages
        or collection != "activities"
        or not isinstance(total, int)
        or total < 0
    ):
        raise ValueError("Invalid activity snapshot manifest")
    records: list[dict] = []
    for index, entry in enumerate(pages):
        path = (manifest_path.parent / entry["path"]).resolve()
        if not path.is_relative_to(manifest_path.parent):
            raise ValueError("Snapshot path escapes manifest directory")
        if file_hash(path) != entry["sha256"]:
            raise ValueError(f"Snapshot checksum mismatch: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        meta, batch = payload["page_meta"], payload[collection]
        if meta["offset"] != len(records) or meta["total_count"] != total:
            raise ValueError("Snapshot pagination offset/total mismatch")
        if not isinstance(batch, list) or len(batch) > meta["limit"]:
            raise ValueError("Invalid snapshot page size")
        expected = pages[index + 1]["url"] if index + 1 < len(pages) else None
        actual = urljoin(entry["url"], meta["next"]) if meta["next"] else None
        if actual != expected or (not batch and expected is not None):
            raise ValueError("Snapshot pagination chain incomplete")
        records.extend(batch)
    if len(records) != total:
        raise ValueError("Snapshot record count differs from total")
    return records, manifest


def validate_allowlist(allowed_assays: dict, targets: dict) -> None:
    """Fail closed if an audit decision lacks the required unannotated human context."""
    required = {
        "organism": "Homo sapiens",
        "target_type": "SINGLE PROTEIN",
        "confidence_score": 9,
        "relationship_type": "D",
        "mode": "biochemical_enzyme_inhibition",
        "decision": "include",
        "variant": "no_annotated_or_explicit_variant",
    }
    for assay, audit in allowed_assays.items():
        task = audit.get("task_id")
        if (
            not assay
            or task not in targets
            or audit.get("target_chembl_id") != targets[task]["target_chembl_id"]
            or any(audit.get(key) != value for key, value in required.items())
            or not str(audit.get("document_chembl_id") or "").strip()
            or not str(audit.get("rationale") or "").strip()
        ):
            raise ValueError(f"Invalid scientific assay allowlist decision: {assay}")


def _finite(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def _record_reason(record: dict, audit: dict | None, targets: dict) -> str | None:
    if audit is None:
        return "assay_not_allowed"
    target = targets[audit["task_id"]]
    if record.get("target_chembl_id") != target["target_chembl_id"]:
        return "target_mismatch"
    if record.get("document_chembl_id") != audit["document_chembl_id"]:
        return "document_mismatch"
    if record.get("target_organism") != audit["organism"]:
        return "organism_mismatch"
    checks = [
        ("standard_type", target["endpoint"], "endpoint_mismatch"),
        ("standard_relation", "=", "nonexact_relation"),
        ("standard_units", "nM", "nonstandard_units"),
        ("standard_flag", 1, "nonstandard_record"),
        ("potential_duplicate", 0, "potential_duplicate"),
    ]
    for key, expected, reason in checks:
        if record.get(key) != expected:
            return reason
    if record.get("data_validity_comment") not in (None, ""):
        return "validity_comment"
    if record.get("assay_variant_mutation") or record.get("assay_variant_accession"):
        return "variant_annotation"
    activity = _finite(record.get("pchembl_value"))
    if activity is None:
        return "invalid_pactivity"
    value = _finite(record.get("standard_value"))
    if value is None or value <= 0:
        return "invalid_standard_value"
    if abs(activity - (9 - math.log10(value))) > 0.011:
        return "inconsistent_pactivity"
    year = _finite(record.get("document_year"))
    if year is None or year != int(year) or not 1800 <= year <= 2100:
        return "unknown_year"
    for key in (
        "activity_id",
        "molecule_chembl_id",
        "canonical_smiles",
        "document_chembl_id",
    ):
        if record.get(key) is None or not str(record[key]).strip():
            return "missing_provenance"
    return None


def curate_records(
    records: list[dict], allowed_assays: dict, targets: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retain direct activity provenance; reject ambiguous/invalid observations."""
    validate_allowlist(allowed_assays, targets)
    accepted: list[dict] = []
    excluded: list[dict] = []
    seen: dict[str, str] = {}
    reason: str | None
    for source_row, record in enumerate(records):
        raw = json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False)
        activity_id = str(record.get("activity_id"))
        if activity_id in seen:
            if seen[activity_id] != raw:
                raise ValueError(f"Conflicting activity_id: {activity_id}")
            reason = "repeated_activity_id"
        else:
            seen[activity_id] = raw
            reason = _record_reason(
                record, allowed_assays.get(record.get("assay_chembl_id")), targets
            )
        evidence = dict(record, source_row=source_row, raw_record_json=raw)
        if reason:
            excluded.append(dict(evidence, reason=reason))
            continue
        task = allowed_assays[record["assay_chembl_id"]]["task_id"]
        computed = 9 - math.log10(float(record["standard_value"]))
        accepted.append(
            dict(
                evidence,
                task_id=task,
                target_id=targets[task]["accession"] + "_HUMAN",
                endpoint=targets[task]["endpoint"],
                measurement_id=activity_id,
                original_smiles=record["canonical_smiles"],
                year=int(record["document_year"]),
                pactivity=computed,
                y_active=int(float(record["standard_value"]) <= 1000),
            )
        )
    return pd.DataFrame(accepted), pd.DataFrame(excluded)


def reserve_observations(
    joined: pd.DataFrame, historical_identities=(), external_start: int = 2024
):
    """Reserve later identity-novel records without using them to curate training."""
    development = joined.loc[joined.year.lt(external_start)].copy()
    later = joined.loc[joined.year.ge(external_start)].copy()
    exposed = set(development.identity) | set(historical_identities)
    overlap = later.loc[later.identity.isin(exposed)].assign(
        reason="previous_identity_exposure"
    )
    novel = later.loc[~later.identity.isin(exposed)].copy()
    dev, dev_conflicts = collapse_observations(development)
    external, external_conflicts = collapse_observations(novel)
    conflicts = pd.concat(
        [
            dev_conflicts.assign(cohort="development"),
            external_conflicts.assign(cohort="external"),
        ],
        ignore_index=True,
    )
    return dev, external, overlap, conflicts


def prepare_dataset(
    accepted: pd.DataFrame,
    output: Path,
    *,
    expected_task_ids: list[str],
    historical_identities=(),
    historical_scaffolds=(),
    train_min: int = 100,
    holdout_min: int = 20,
) -> dict:
    """Create shared chemistry arrays; retain all tasks or fail the support gate.

    Output must be a new directory. No model is fit and no external score is
    computed. Preprocessing statistics use development training molecules only.
    """
    output = Path(output)
    if output.exists():
        raise ValueError("Preparation output must not already exist")
    if accepted.empty:
        raise ValueError("No accepted direct activity records")
    required_tasks = sorted(expected_task_ids)
    if not required_tasks or len(required_tasks) != len(set(required_tasks)):
        raise ValueError("Expected tasks must be nonempty and unique")
    if set(accepted.task_id) != set(required_tasks):
        raise ValueError("Accepted tasks differ from prespecified expected tasks")
    smiles = sorted(accepted.original_smiles.unique())
    features = featurize(
        pd.DataFrame(
            {"original_smiles": smiles, "record_id": range(1, len(smiles) + 1)}
        ),
        2048,
    ).records
    valid = features.loc[features.eligible].copy()
    joined = accepted.merge(
        valid[["original_smiles", "identity"]],
        on="original_smiles",
        validate="many_to_one",
    )
    dev, external, overlaps, conflicts = reserve_observations(
        joined, historical_identities
    )
    dev_molecules = (
        valid.loc[valid.identity.isin(dev.identity)]
        .drop_duplicates("identity")
        .sort_values("identity")
        .reset_index(drop=True)
    )
    output.mkdir(parents=True)
    initial_frames = {
        "accepted-direct-records": accepted,
        "features": features,
        "reserved-external-observations": external,
        "external-exposure-exclusions": overlaps,
        "conflicting-pairs": conflicts,
        "source-identity-map": joined[
            ["source_row", "original_smiles", "identity", "task_id"]
        ],
    }
    for name, frame in initial_frames.items():
        clean = frame.copy()
        clean.attrs = {}
        clean.to_parquet(output / f"{name}.parquet", index=False)
    development_ids = set(joined.loc[joined.year.lt(2024), "identity"])
    development_scaffolds = set(
        valid.loc[valid.identity.isin(development_ids), "murcko_scaffold"]
    )
    write_json(
        output / "development-exposure.json",
        {
            "identities": sorted(development_ids),
            "scaffolds": sorted(development_scaffolds),
            "definition": "All valid accepted pre-2024 identities, including conflicting identity-task pairs",
        },
    )
    try:
        molecules = assign_splits(dev_molecules, seed=42)
    except ValueError as error:
        write_json(
            output / "failure.json",
            {"stage": "split", "error": str(error), "expected_tasks": required_tasks},
        )
        raise
    dev = dev.merge(molecules[["identity", "split"]], validate="many_to_one")
    tasks, support = select_tasks(dev, train_min=train_min, validation_min=holdout_min)
    failing = [
        task
        for task in required_tasks
        if task not in tasks
        or any(
            min(support[task][split].values()) < holdout_min
            for split in ("calibration", "test")
        )
    ]
    if failing:
        write_json(
            output / "failure.json",
            {
                "stage": "support",
                "failing_tasks": failing,
                "support": support,
                "expected_tasks": required_tasks,
            },
        )
        dev.to_parquet(output / "observations.parquet", index=False)
        molecules.to_parquet(output / "molecules.parquet", index=False)
        raise ValueError(
            f"Tasks fail frozen support gates: {failing}; support={support}"
        )
    molecules = molecules.assign(**dict.fromkeys(CONTEXT_NAMES, np.nan))
    preprocessor = Preprocessor.fit(molecules.loc[molecules.split.eq("train")])
    properties, _ = preprocessor.transform(molecules)
    x = np.concatenate([fingerprint_matrix(molecules, 2048), properties], axis=1)
    labels = make_labels(molecules.identity.tolist(), tasks, dev)
    external_molecules = (
        valid.loc[valid.identity.isin(external.identity)]
        .drop_duplicates("identity")
        .sort_values("identity")
        .reset_index(drop=True)
    )
    development_ids = set(joined.loc[joined.year.lt(2024), "identity"])
    exposed_scaffolds = set(
        valid.loc[valid.identity.isin(development_ids), "murcko_scaffold"]
    ) | set(historical_scaffolds)
    external_source_rows = {row for rows in external.source_rows for row in rows}
    external_years = (
        joined.loc[joined.source_row.isin(external_source_rows)]
        .groupby("identity")
        .year.min()
    )
    external_molecules = external_molecules.assign(
        year=external_molecules.identity.map(external_years)
    )
    external_molecules = external_molecules.assign(
        **dict.fromkeys(CONTEXT_NAMES, np.nan),
        scaffold_novel=~external_molecules.murcko_scaffold.isin(exposed_scaffolds),
    )
    external_properties, _ = preprocessor.transform(external_molecules)
    external_x = np.concatenate(
        [fingerprint_matrix(external_molecules, 2048), external_properties], axis=1
    )
    external_labels = make_labels(external_molecules.identity.tolist(), tasks, external)
    np.savez_compressed(
        output / "external-arrays.npz",
        x=external_x,
        labels=external_labels,
        identities=external_molecules.identity.to_numpy(dtype=str),
        scaffold_novel=external_molecules.scaffold_novel.to_numpy(dtype=bool),
    )
    np.savez_compressed(
        output / "arrays.npz",
        x=x,
        labels=labels,
        splits=molecules.split.to_numpy(dtype=str),
        identities=molecules.identity.to_numpy(dtype=str),
    )
    frames = {
        "molecules": molecules,
        "observations": dev,
        "external-molecules": external_molecules,
    }
    for name, frame in frames.items():
        clean = frame.copy()
        clean.attrs = {}
        clean.to_parquet(output / f"{name}.parquet", index=False)
    write_json(output / "preprocess.json", preprocessor.to_dict())
    write_json(output / "chemistry.json", chemistry_manifest(2048))
    task_source = accepted.drop_duplicates("task_id").set_index("task_id")
    write_json(
        output / "tasks.json",
        [
            {
                "task_id": task,
                "target_id": task_source.loc[task, "target_id"],
                "endpoint": task_source.loc[task, "endpoint"],
                "support": support[task],
            }
            for task in tasks
        ],
    )
    audit = {
        "tasks": tasks,
        "support": support,
        "molecules": len(molecules),
        "observed_labels": int(np.isfinite(labels).sum()),
        "reserved_external_labels": len(external),
        "external_exposure_exclusions": len(overlaps),
        "conflicting_pairs": len(conflicts),
        "invalid_structures": int((~features.eligible).sum()),
        "class_definition": "Exact standard_value <= 1000 nM; rounded pChEMBL is consistency check only",
        "external_start_year": 2024,
        "split_seed": 42,
        "variant_scope": "No annotated or explicit variant; not proof of wild type",
        "external_scaffold_novel_molecules": int(
            external_molecules.scaffold_novel.sum()
        ),
        "historically_exposed_identities": len(set(historical_identities)),
    }
    write_json(output / "audit.json", audit)
    write_json(
        output / "completion.json",
        {
            "files": {p.name: file_hash(p) for p in output.iterdir() if p.is_file()},
            "audit": audit,
        },
    )
    return audit
