"""Mask contradictory TRAIN labels while preserving all frozen B09 heldouts.

This ablation does not resolve assay semantics: source constituent values must
all fall on the same side of pActivity 6. Unknown labels stay unknown.
"""

import argparse
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from s2s_decision.artifacts import file_hash, write_json

_spec = importlib.util.spec_from_file_location(
    "consensus_base_runner", Path(__file__).with_name("train_multitask.py")
)
if _spec is None or _spec.loader is None:
    raise ImportError("Cannot load existing train_multitask dataset verifier")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
load_dataset = _base.load_dataset
MEAN_ATOL = 1e-8  # Source decimal serialization; observed B09 error <= 3.56e-15.


def _constituents(row: dict) -> tuple[float, float, int, float]:
    """Reject malformed provenance instead of converting it into missing labels."""
    value = row["pchembl_value"]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Missing constituent pActivity values")
    try:
        values = np.array([float(item) for item in value.split(";")])
        count = float(row["pchembl_value_N"])
        mean = float(row["pchembl_value_Mean"])
        pactivity = float(row["pactivity"])
    except (ValueError, TypeError) as exc:
        raise ValueError("Malformed constituent provenance") from exc
    if not np.isfinite(values).all() or not np.isfinite([count, mean, pactivity]).all():
        raise ValueError("Constituent values, mean and N must be finite")
    if count <= 0 or not count.is_integer() or count != len(values):
        raise ValueError("Constituent N must match a positive integer count")
    if not np.isclose(values.mean(), mean, rtol=0, atol=MEAN_ATOL):
        raise ValueError("Constituent mean differs from source aggregate")
    if not np.isclose(values.mean(), pactivity, rtol=0, atol=MEAN_ATOL):
        raise ValueError("Constituent mean differs from training pActivity")
    if row["y_active"] not in (0, 1) or int(mean >= 6) != row["y_active"]:
        raise ValueError("Source aggregate label disagrees with threshold")
    return float(values.min()), float(values.max()), len(values), mean


def _source_index(frame: pd.DataFrame, columns: list[str]) -> dict:
    if not set(columns) <= set(frame):
        raise ValueError("Missing source provenance columns")
    if frame.source_row.duplicated().any():
        raise ValueError("Duplicate source row")
    return {row["source_row"]: row for row in frame[columns].to_dict("records")}


def _audit_observation(observation: Any, raw: dict, mapping: dict) -> dict:
    rows = list(observation.source_rows)
    if (
        not rows
        or len(set(rows)) != len(rows)
        or any(
            isinstance(row, (bool, np.bool_)) or not isinstance(row, (int, np.integer))
            for row in rows
        )
    ):
        raise ValueError("Source rows must be nonempty unique integer IDs")
    minimum, maximum, counts, means, measurements = [], [], [], [], []
    for source_id in rows:
        if source_id not in raw or source_id not in mapping:
            raise ValueError("Observation references missing source row")
        source, mapped = raw[source_id], mapping[source_id]
        if (
            source["task_id"] != observation.task_id
            or mapped["task_id"] != observation.task_id
            or mapped["identity"] != observation.identity
            or source["original_smiles"] != mapped["original_smiles"]
        ):
            raise ValueError("Source identity, task or SMILES provenance mismatch")
        low, high, count, mean = _constituents(source)
        if source["y_active"] != observation.y_active:
            raise ValueError("Observation disagrees with source label")
        minimum.append(low)
        maximum.append(high)
        counts.append(count)
        means.append(mean)
        measurements.append(source["measurement_id"])
    if sorted(measurements) != sorted(observation.measurement_ids):
        raise ValueError("Observation measurement IDs mismatch")
    low, high = min(minimum), max(maximum)
    crossing = low < 6 <= high
    return {
        "identity": observation.identity,
        "task_id": observation.task_id,
        "split": observation.split,
        "source_rows": rows,
        "measurement_ids": measurements,
        "source_aggregate_means": means,
        "original_label": float(observation.y_active),
        "constituent_min": low,
        "constituent_max": high,
        "constituent_range": high - low,
        "constituent_count": sum(counts),
        "crossing": crossing,
        "reason": "constituents_cross_threshold" if crossing else "consensus",
        "status": (
            "masked_train"
            if crossing and observation.split == "train"
            else (
                "heldout_unchanged"
                if observation.split != "train"
                else "retained_train"
            )
        ),
    }


def consensus_labels(
    arrays: dict,
    tasks: list[dict],
    observations: pd.DataFrame,
    raw: pd.DataFrame,
    mapping: pd.DataFrame,
    min_train_per_class: int = 100,
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Return new labels, complete observation audit, and post-mask task support."""
    if not isinstance(min_train_per_class, int) or min_train_per_class < 1:
        raise ValueError("Minimum training support must be positive integer")
    source = _source_index(
        raw,
        [
            "source_row",
            "task_id",
            "original_smiles",
            "measurement_id",
            "pchembl_value",
            "pchembl_value_N",
            "pchembl_value_Mean",
            "pactivity",
            "y_active",
        ],
    )
    mapped = _source_index(
        mapping, ["source_row", "task_id", "original_smiles", "identity"]
    )
    row_index = {identity: i for i, identity in enumerate(arrays["identities"])}
    task_index = {task["task_id"]: i for i, task in enumerate(tasks)}
    labels = arrays["labels"].copy()
    visited = np.zeros(labels.shape, dtype=bool)
    records = []
    for observation in observations.itertuples(index=False):
        if (
            observation.identity not in row_index
            or observation.task_id not in task_index
        ):
            raise ValueError("Unknown observation identity or task")
        i, j = row_index[observation.identity], task_index[observation.task_id]
        if (
            visited[i, j]
            or arrays["splits"][i] != observation.split
            or labels[i, j] != observation.y_active
        ):
            raise ValueError("Observation duplicate, split or label mismatch")
        record = _audit_observation(observation, source, mapped)
        visited[i, j] = True
        if record["status"] == "masked_train":
            labels[i, j] = np.nan
        records.append(record)
    if not np.array_equal(visited, np.isfinite(arrays["labels"])):
        raise ValueError(
            "Observations do not cover exactly the original observed labels"
        )
    train = arrays["splits"] == "train"
    support = [
        {
            "task_id": task["task_id"],
            "negative": int((labels[train, j] == 0).sum()),
            "positive": int((labels[train, j] == 1).sum()),
        }
        for j, task in enumerate(tasks)
    ]
    if any(
        min(item["negative"], item["positive"]) < min_train_per_class
        for item in support
    ):
        raise ValueError("Consensus masking violates frozen minimum training support")
    audit = pd.DataFrame(records)
    summary = {
        "schema": "s2-decision-consensus-data/1",
        "tasks": len(tasks),
        "observed_labels_before": int(visited.sum()),
        "observed_labels_after": int(np.isfinite(labels).sum()),
        "train_labels_before": int(visited[train].sum()),
        "train_labels_after": int(np.isfinite(labels[train]).sum()),
        "crossing_observations": int(audit.crossing.sum()),
        "masked_train_labels": int((audit.status == "masked_train").sum()),
        "train_rows_without_labels": int(
            (~np.isfinite(labels[train]).any(axis=1)).sum()
        ),
        "mean_absolute_tolerance": MEAN_ATOL,
        "mean_relative_tolerance": 0,
        "threshold": 6.0,
        "train_support": support,
        "tasks_json_support": "Original B09 support; updated TRAIN support in audit.json",
        "evaluation": "All heldout labels unchanged; constituent consensus is not assay equivalence",
    }
    return labels, audit, summary


def prepare(source: Path, output: Path, min_train_per_class: int = 100) -> dict:
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    arrays, tasks = load_dataset(source)
    names = [
        "observations.parquet",
        "raw-selected-measurements.parquet",
        "source-identity-map.parquet",
    ]
    completion = json.loads((source / "completion.json").read_text(encoding="utf-8"))
    completion_hash = file_hash(source / "completion.json")
    copied = [
        "tasks.json",
        "preprocess.json",
        "chemistry.json",
        "molecules.parquet",
        "observations.parquet",
    ]
    if not set(names + copied) <= set(completion["files"]):
        raise ValueError(
            "Source completion must include observation and provenance hashes"
        )
    frames = [pd.read_parquet(source / name) for name in names]
    labels, audit, summary = consensus_labels(
        arrays, tasks, frames[0], frames[1], frames[2], min_train_per_class
    )
    mask = np.zeros(labels.shape, dtype=bool)
    rows = {identity: i for i, identity in enumerate(arrays["identities"])}
    columns = {task["task_id"]: i for i, task in enumerate(tasks)}
    for item in audit.loc[audit.crossing].itertuples(index=False):
        mask[rows[item.identity], columns[item.task_id]] = True
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "arrays.npz", **{**arrays, "labels": labels})
    np.savez_compressed(output / "consensus-mask.npz", mask=mask)
    audit.to_parquet(output / "mask-audit.parquet", index=False)
    for name in copied:
        shutil.copyfile(source / name, output / name)
        if file_hash(output / name) != completion["files"][name]:
            raise ValueError(f"Copied source changed: {name}")
    _base.verify_preparation(source)
    if file_hash(source / "completion.json") != completion_hash:
        raise ValueError("Source completion changed during preparation")
    write_json(output / "audit.json", summary)
    write_json(
        output / "source-lineage.json",
        {
            "source": str(source),
            "completion_sha256": completion_hash,
            "source_hashes": completion["files"],
            "preparation_code_sha256": file_hash(Path(__file__)),
            "heldout_labels": "bit-identical",
            "features_splits_identity_order": "unchanged",
            "observations_parquet": "Original B09 observations; mask-audit.parquet describes every change",
        },
    )
    write_json(
        output / "completion.json",
        {
            "files": {
                path.name: file_hash(path)
                for path in sorted(output.iterdir())
                if path.is_file()
            },
            "audit": summary,
        },
    )
    load_dataset(output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
