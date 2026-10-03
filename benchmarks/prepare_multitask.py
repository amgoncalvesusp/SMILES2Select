"""Prepare a traceable Papyrus multitask research dataset using shared chemistry."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from s2s_decision.artifacts import file_hash
from s2s_decision.data import _normalized
from s2s_decision.features import chemistry_manifest, featurize
from s2s_decision.preprocessing import Preprocessor
from s2s_decision.schema import CONTEXT_NAMES, fingerprint_matrix
from s2s_decision.splits import assign_splits

SOURCE_HASH = "8004e0d1027a760f205b45264386f792e7d49658da39f77f52e660a6f19760dd"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def seal(path: Path, payload: Any) -> None:
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != payload:
        raise ValueError(f"Frozen preparation settings changed: {path}")
    if not path.exists():
        write_json(path, payload)


def collapse_observations(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Discard conflicting identity-task pairs, retaining unrelated observed labels."""
    grouped = frame.groupby(["identity", "task_id"], sort=True)
    summary = grouped.agg(
        y_active=("y_active", "first"),
        distinct_labels=("y_active", "nunique"),
        source_rows=("source_row", lambda s: sorted({int(v) for v in s})),
        measurement_ids=("measurement_id", lambda s: sorted(set(s.astype(str)))),
    ).reset_index()
    conflicts = summary.loc[summary.distinct_labels.gt(1)].copy()
    clean = summary.loc[summary.distinct_labels.eq(1)].drop(columns="distinct_labels")
    return clean.reset_index(drop=True), conflicts.reset_index(drop=True)


def make_labels(identities: list[str], task_ids: list[str], observations: pd.DataFrame) -> np.ndarray:
    matrix = np.full((len(identities), len(task_ids)), np.nan, dtype=np.float32)
    row_map, col_map = dict(zip(identities, range(len(identities)))), dict(zip(task_ids, range(len(task_ids))))
    selected = observations.loc[observations.task_id.isin(col_map)]
    if selected.duplicated(["identity", "task_id"]).any():
        raise ValueError("Duplicate identity-task observations")
    if not selected.y_active.isin([0, 1]).all():
        raise ValueError("Observed labels must be binary")
    rows = selected.identity.map(row_map)
    if rows.isna().any():
        raise ValueError("Observed identity is absent from molecular features")
    matrix[rows.to_numpy(dtype=int), selected.task_id.map(col_map).to_numpy(dtype=int)] = selected.y_active
    return matrix


def select_tasks(observations: pd.DataFrame, train_min: int = 100,
                 validation_min: int = 10) -> tuple[list[str], dict]:
    counts = observations.groupby(["task_id", "split", "y_active"]).size()
    support = {
        task: {
            split: {"negative": int(counts.get((task, split, 0), 0)),
                    "positive": int(counts.get((task, split, 1), 0))}
            for split in ("train", "validation", "calibration", "test")
        }
        for task in sorted(observations.task_id.unique())
    }
    selected = [task for task, values in support.items()
                if min(values["train"].values()) >= train_min
                and min(values["validation"].values()) >= validation_min]
    return selected, support


def extract_source(source: Path, candidates: set[tuple[str, str]], output: Path) -> pd.DataFrame:
    path = output / "raw-selected-measurements.parquet"
    if path.exists():
        return pd.read_parquet(path)
    selected = []
    total = 0
    started = time.monotonic()
    for records in _normalized(source, 6.0, 10000):
        total += len(records)
        selected.extend(row for row in records
                        if (row["target_id"], row["endpoint"]) in candidates
                        and row["quality"] == "high" and row["relation"] == "="
                        and not row["mixed_endpoint"] and pd.notna(row["y_active"]))
        if total % 100000 == 0:
            print(f"source_rows={total} selected={len(selected)} elapsed_s={time.monotonic()-started:.1f}", flush=True)
    frame = pd.DataFrame(selected)
    if frame.empty:
        raise ValueError("No eligible measurements")
    frame["task_id"] = frame.target_id + "_" + frame.endpoint
    frame.to_parquet(path, index=False)
    print(f"extracted={len(frame)} unique_smiles={frame.original_smiles.nunique()}", flush=True)
    return frame


def feature_chunk(frame: pd.DataFrame, path: Path) -> tuple[str, int, float]:
    started = time.monotonic()
    result = featurize(frame, 2048).records
    result.attrs = {}
    result.to_parquet(path, index=False)
    return str(path), len(result), round(time.monotonic() - started, 2)


def generate_features(raw: pd.DataFrame, output: Path, workers: int) -> pd.DataFrame:
    final = output / "features-before-identity-collapse.parquet"
    if final.exists():
        return pd.read_parquet(final)
    molecules = pd.DataFrame({"original_smiles": sorted(raw.original_smiles.unique())})
    molecules["record_id"] = np.arange(1, len(molecules) + 1)
    chunk_dir = output / "feature-chunks"
    chunk_dir.mkdir(exist_ok=True)
    paths = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = {}
        for start in range(0, len(molecules), 1000):
            path = chunk_dir / f"{start:07d}.parquet"
            paths.append(path)
            if not path.exists():
                future = executor.submit(feature_chunk, molecules.iloc[start:start+1000].copy(), path)
                pending[future] = path
        for future in as_completed(pending):
            path, size, elapsed = future.result()
            print(f"features_chunk={Path(path).name} rows={size} elapsed_s={elapsed}", flush=True)
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    frame.attrs = {}
    frame.to_parquet(final, index=False)
    return frame


def exposure_audit(study: Path, molecules: pd.DataFrame) -> dict:
    """Describe historical identity exposure; do not change the frozen partition."""
    previous = study / "artifacts/data-capacity-v1/excluded-previous-identities.json"
    identities = set()
    sources = {}
    if previous.exists():
        identities.update(json.loads(previous.read_text(encoding="utf-8"))["identities"])
        sources[str(previous)] = file_hash(previous)
    # Training/holdout feature bundles contain exposure beyond prediction CSVs.
    for path in sorted((study / "artifacts/data-capacity-v1/datasets").rglob("records.jsonl")):
        with path.open(encoding="utf-8") as stream:
            identities.update(json.loads(line)["identity"] for line in stream if line.strip())
        sources[str(path)] = file_hash(path)
    for path in sorted((study / "artifacts/data-capacity-v1").rglob("*.csv")):
        columns = pd.read_csv(path, nrows=0).columns
        if "identity" in columns:
            identities.update(pd.read_csv(path, usecols=["identity"]).identity.dropna().astype(str))
            sources[str(path)] = file_hash(path)
    return {
        "interpretation": "internal exploratory scaffold evaluation; prior study exposure is disclosed, not removed",
        "historically_exposed_identities": len(identities), "source_files": sources,
        "overlap_by_split": molecules.assign(exposed=molecules.identity.isin(identities))
        .groupby("split").exposed.sum().astype(int).to_dict(),
    }


def finish(raw: pd.DataFrame, features: pd.DataFrame, study: Path, output: Path) -> None:
    valid = features.loc[features.eligible].copy()
    joined = raw.merge(valid[["original_smiles", "identity"]], on="original_smiles", validate="many_to_one")
    observations, conflicts = collapse_observations(joined)
    molecules = valid.drop_duplicates("identity").sort_values("identity").reset_index(drop=True)
    molecules = assign_splits(molecules, seed=42)
    observations = observations.merge(molecules[["identity", "split"]], validate="many_to_one")
    selected, support = select_tasks(observations)
    if not selected:
        raise ValueError("No tasks passed training/validation support gates")
    # Partition precedes support gating; discarded tasks cannot move retained molecules.
    observations = observations.loc[observations.task_id.isin(selected)].reset_index(drop=True)
    molecules = molecules.loc[molecules.identity.isin(observations.identity)].reset_index(drop=True)
    molecules = molecules.assign(**dict.fromkeys(CONTEXT_NAMES, np.nan))
    preprocessor = Preprocessor.fit(molecules.loc[molecules.split.eq("train")])
    properties, _ = preprocessor.transform(molecules)
    x = np.concatenate((fingerprint_matrix(molecules, 2048), properties), axis=1)
    labels = make_labels(molecules.identity.tolist(), selected, observations)
    np.savez_compressed(output / "arrays.npz", x=x, labels=labels,
                        splits=molecules.split.to_numpy(dtype=str),
                        identities=molecules.identity.to_numpy(dtype=str))
    molecules.attrs = {}
    molecules.to_parquet(output / "molecules.parquet", index=False)
    observations.to_parquet(output / "observations.parquet", index=False)
    conflicts.to_parquet(output / "conflicting-pairs.parquet", index=False)
    joined[["source_row", "original_smiles", "identity", "task_id"]].to_parquet(
        output / "source-identity-map.parquet", index=False)
    task_source = raw.drop_duplicates("task_id").set_index("task_id")
    tasks = [{"task_id": task, "target_id": task_source.loc[task, "target_id"],
              "endpoint": task_source.loc[task, "endpoint"], "support": support[task]}
             for task in selected]
    write_json(output / "tasks.json", tasks)
    write_json(output / "task-support-all.json", support)
    write_json(output / "preprocess.json", preprocessor.to_dict())
    write_json(output / "chemistry.json", chemistry_manifest(2048))
    audit = {
        "model_name": "S2 Decision", "source_rows_selected": len(raw),
        "unique_source_smiles": len(features), "invalid_structures": int((~features.eligible).sum()),
        "identity_task_conflicts": len(conflicts), "molecules": len(molecules),
        "observed_labels": int(np.isfinite(labels).sum()), "tasks": len(tasks),
        "label_density": float(np.isfinite(labels).mean()), "input_dimensions": int(x.shape[1]),
        "split_molecule_counts": molecules.split.value_counts().astype(int).to_dict(),
        "scaffold_count": int(molecules.murcko_scaffold.nunique()),
        "class_definition": "Papyrus exact aggregate mean pActivity >= 6; lower values below threshold, not universal inactivity",
        "aggregation": "Existing Papyrus source aggregates retained; conflicting post-standardization identity-task labels removed",
        "evaluation": "Internal exploratory scaffold holdout, not independent-source or prospective validation",
        "historical_exposure": exposure_audit(study, molecules),
    }
    write_json(output / "audit.json", audit)
    files = {path.name: file_hash(path) for path in output.iterdir() if path.is_file() and path.name != "completion.json"}
    write_json(output / "completion.json", {"files": files, "audit": audit})
    print(json.dumps(audit, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        parser.error("workers must be between 1 and 6")
    study = args.study_root.resolve()
    source = study / "data/papyrus/05.7++_combined_set_without_stereochemistry.tsv.xz"
    audit_path = study / "artifacts/papyrus-audit/audit.json"
    if file_hash(source) != SOURCE_HASH:
        raise ValueError("Papyrus source checksum mismatch")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["source_sha256"] != SOURCE_HASH or audit["threshold"] != 6.0:
        raise ValueError("Source audit disagrees with source or threshold")
    candidates = {(row["target_id"], row["endpoint"]) for row in audit["target_endpoint_counts"]
                  if row["target_id"].endswith("_WT") and row["endpoint"] in {"IC50", "Ki", "Kd", "EC50"}
                  and row["rows"] >= 500 and row["active_records"] >= 100 and row["inactive_records"] >= 100}
    output = study / "artifacts/s2-decision-multitask-v1/data"
    output.mkdir(parents=True, exist_ok=True)
    settings = {"source_sha256": SOURCE_HASH, "audit_sha256": file_hash(audit_path),
                "candidate_tasks": sorted([list(pair) for pair in candidates]), "threshold": 6.0,
                "quality": "high", "relation": "=", "target_variant": "WT", "split_seed": 42,
                "split_method": "global Murcko scaffold", "train_min_per_class": 100,
                "validation_min_per_class": 10, "chemistry_hash": chemistry_manifest(2048)["chemistry_hash"],
                "missing_label": "NaN", "script_sha256": file_hash(Path(__file__))}
    settings_path = output / "preparation-settings.json"
    if (output / "completion.json").exists() and settings_path.exists():
        original_settings = json.loads(settings_path.read_text(encoding="utf-8"))
        if original_settings["script_sha256"] != settings["script_sha256"]:
            amendment = json.loads((output / "preparation-amendment.json").read_text(encoding="utf-8"))
            if (amendment["original_script_sha256"] != original_settings["script_sha256"]
                    or amendment["current_script_sha256"] != settings["script_sha256"]
                    or amendment["unchanged_arrays_sha256"] != file_hash(output / "arrays.npz")):
                raise ValueError("Preparation amendment disagrees with script or arrays")
            settings = {**settings, "script_sha256": original_settings["script_sha256"]}
    seal(settings_path, settings)
    if (output / "completion.json").exists():
        completion = json.loads((output / "completion.json").read_text(encoding="utf-8"))
        if any(file_hash(output / name) != digest for name, digest in completion["files"].items()):
            raise ValueError("Completed preparation artifact changed")
        print("Preparation already complete; hashes verified", flush=True)
        return
    print(f"candidate_tasks={len(candidates)}", flush=True)
    raw = extract_source(source, candidates, output)
    features = generate_features(raw, output, args.workers)
    finish(raw, features, study, output)


if __name__ == "__main__":
    main()
