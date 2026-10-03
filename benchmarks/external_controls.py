"""Frozen B09 single-task controls; no external labels enter any fitting step."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from s2s_decision.artifacts import file_hash, read_bundle, write_bundle, write_json
from s2s_decision.schema import CONTEXT_NAMES, FeatureSet, fingerprint_matrix

TASKS = ("P03372_WT_IC50", "P37231_WT_EC50")
SEEDS = (42, 43, 44)


def observed_subset(arrays, molecules, task_index):
    """Keep observed head rows without changing global features or partitions."""
    if not np.array_equal(arrays["identities"], molecules.identity.to_numpy()) or not np.array_equal(
        arrays["splits"], molecules.split.to_numpy()
    ):
        raise ValueError("Molecules and arrays must align exactly")
    if molecules.identity.duplicated().any():
        raise ValueError("Duplicate molecular identity")
    if molecules.groupby("murcko_scaffold").split.nunique().gt(1).any():
        raise ValueError("Global scaffold leakage")
    labels = arrays["labels"][:, task_index:task_index + 1]
    if labels.shape[1] != 1 or not np.all(np.isnan(labels) | (labels == 0) | (labels == 1)):
        raise ValueError("Labels must be binary or unknown")
    observed = np.isfinite(labels[:, 0])
    selected = {key: value[observed] for key, value in arrays.items() if key != "labels"}
    selected["labels"] = labels[observed]
    if set(selected["splits"]) != {"train", "validation", "calibration", "test"}:
        raise ValueError("All four original partitions must retain observed rows")
    return selected, molecules.loc[observed].reset_index(drop=True)


def join_activity(frame, labels, observations, mapping, raw, task_id):
    """Join exact source aggregates; take median of same-label identity replicates."""
    obs = observations.loc[observations.task_id.eq(task_id)].copy()
    if obs.identity.duplicated().any() or set(obs.identity) != set(frame.identity):
        raise ValueError("Observation identities disagree with the observed array rows")
    declared = obs[["identity", "task_id", "source_rows"]].explode("source_rows").rename(
        columns={"source_rows": "source_row"})
    source_map = mapping.loc[mapping.task_id.eq(task_id), ["source_row", "task_id", "identity"]]
    joined = declared.merge(source_map, on=["source_row", "task_id"], how="left",
                            validate="one_to_one", suffixes=("", "_source"))
    if not joined.identity.eq(joined.identity_source).all():
        raise ValueError("Source identity does not match declared observation")
    values = raw.loc[raw.task_id.eq(task_id), ["source_row", "task_id", "pactivity", "relation"]]
    joined = joined.merge(values, on=["source_row", "task_id"], how="left", validate="one_to_one")
    if not joined.relation.eq("=").all() or not np.isfinite(joined.pactivity.to_numpy(float)).all():
        raise ValueError("Each declared source row must have exact finite pActivity")
    y = obs.set_index("identity").y_active
    if not joined.pactivity.ge(6).astype(float).eq(joined.identity.map(y)).all():
        raise ValueError("Exact source pActivity disagrees with retained binary label")
    activity = joined.groupby("identity").pactivity.median()
    aligned = obs.set_index("identity").loc[frame.identity]
    if not np.array_equal(aligned.y_active.to_numpy(), labels[:, 0]) or not np.array_equal(
        aligned.split.to_numpy(), frame.split.to_numpy()
    ):
        raise ValueError("Observed label or split differs from frozen B09 arrays")
    return frame.assign(y_active=labels[:, 0], pactivity=frame.identity.map(activity), relation="=")


def score_control(features, model_dir, kind, b09_preprocessor=None, context_features=None):
    """Return raw logits, raw sigmoid and calibration aligned with feature rows.

    Tiny references are the persisted task-training bundle, with own identity
    exclusion in build_context. Matched MLP inputs use the frozen B09 preprocessor.
    This research interface never consumes external labels.
    """
    from s2s_decision.context import build_context
    from s2s_decision.preprocessing import Preprocessor

    folder = Path(model_dir)
    if kind == "matched":
        import torch

        from s2s_decision.multitask import S2Decision
        from s2s_decision.training import _require_safe_checkpoint_runtime

        _require_safe_checkpoint_runtime()
        checkpoint = torch.load(folder / "best.pt", weights_only=True, map_location="cpu")
        prep = Preprocessor.from_dict(b09_preprocessor)
        clean = features.assign(**dict.fromkeys(CONTEXT_NAMES, np.nan))
        properties, _ = prep.transform(clean)
        x = np.concatenate((fingerprint_matrix(clean, 2048), properties), axis=1)
        model = S2Decision(checkpoint["inputs"], 1, checkpoint["hidden"]).eval()
        model.load_state_dict(checkpoint["state_dict"])
        with torch.no_grad():
            logits = np.concatenate([model(torch.from_numpy(x[start:start + 1024])).numpy()
                                     for start in range(0, len(x), 1024)])[:, 0]
        calibration = json.loads((folder / "calibration.json").read_text())[0]
        return {"raw_logits": logits, "raw_probabilities": expit(logits),
                "probabilities": expit(logits * calibration["slope"] + calibration["intercept"])}
    if kind != "tiny":
        raise ValueError("Unknown control kind")
    import onnxruntime as ort

    from s2s_decision.inference import validate_model_manifest

    manifest = json.loads((folder / "manifest.json").read_text())
    prep = validate_model_manifest(manifest)
    if file_hash(folder / "model.onnx") != manifest["onnx_sha256"]:
        raise ValueError("Tiny ONNX checksum changed")
    references = read_bundle(folder / "references")
    if references.manifest["records_sha256"] != manifest["reference_records_sha256"]:
        raise ValueError("Tiny reference checksum changed")
    context = build_context(features, references.records) if context_features is None else context_features
    if not context.index.equals(features.index) or tuple(context.columns) != CONTEXT_NAMES:
        raise ValueError("Context must align with feature rows and ordered context columns")
    complete = pd.concat([features.drop(columns=list(CONTEXT_NAMES), errors="ignore"), context], axis=1)
    properties, contexts = prep.transform(complete)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(folder / "model.onnx"), sess_options=options,
                                  providers=["CPUExecutionProvider"])
    logits = session.run(None, {"properties": properties, "context": contexts,
                                "fingerprint": fingerprint_matrix(complete, 2048)})[0][:, 0]
    calibration = manifest["calibrator"]
    return {"raw_logits": logits, "raw_probabilities": expit(logits),
            "probabilities": expit(logits * calibration["slope"] + calibration["intercept"])}


def freeze_protocol(data, output):
    """Seal the controls, original data and executable source before fitting."""
    source = Path(__file__).resolve().parents[1]
    paths = [path for path in data.iterdir() if path.is_file()]
    code = [Path(__file__).resolve(), *sorted((source / "src/s2s_decision").glob("*.py"))]
    payload = {
        "schema": "s2-decision-external-controls/1", "created_utc": datetime.now(UTC).isoformat(),
        "tasks": list(TASKS), "seeds": list(SEEDS), "source_data": str(data.resolve()),
        "source_hashes": {str(path.resolve()): file_hash(path) for path in paths},
        "code_hashes": {str(path): file_hash(path) for path in code},
        "shared_settings": {"epochs": 40, "patience": 6, "batch_size": 512,
                            "learning_rate": .001, "weight_decay": .0001},
        "matched": {"hidden": [512, 256], "inputs": "B09 global train-only preprocessed inputs",
                    "unknowns": "drop unobserved rows for the head", "validation": "average_precision"},
        "tiny": {"width_multiplier": 1, "inputs": "native properties, fingerprint and reference context",
                 "context": "scaffold OOF seed42 five folds; held-out uses task train only",
                 "preprocessing": "native task-train-only", "regression_weight": .2,
                 "regression_values": "median of declared exact source-aggregate means within concordant identity-task",
                 "scheduler": "native ReduceLROnPlateau factor0.5 patience5",
                 "validation": "native PR-AUC checkpoint criterion"},
        "split": "Original global B09 scaffold assignments; no resplitting",
        "test": "No B09 or external test labels enter training, checkpoint selection or calibration",
        "comparison_limit": "Equal epoch budget does not equal optimizer update counts; single-task has fewer observed rows",
        "external_primary": "raw logit ranking; no external probability calibration claims",
        "postfit_lineage_predictions": "Full task-train reference context with own identity excluded; training rows are not OOF predictions",
    }
    parent_plan = output.parent / "plan.json"
    if parent_plan.exists():
        payload["parent_plan_sha256"] = file_hash(parent_plan)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", payload)
    return payload


def run(data, output):
    import torch

    # Sibling runner verifies every original preparation hash before loading arrays.
    from train_multitask import load_dataset

    from s2s_decision.context import training_context
    from s2s_decision.multitask import train_multitask
    from s2s_decision.training import TrainingConfig
    from s2s_decision.workflows import train_dataset

    arrays, tasks = load_dataset(data)
    molecules = pd.read_parquet(data / "molecules.parquet")
    observations = pd.read_parquet(data / "observations.parquet")
    mapping = pd.read_parquet(data / "source-identity-map.parquet")
    raw = pd.read_parquet(data / "raw-selected-measurements.parquet")
    chemistry = json.loads((data / "chemistry.json").read_text())
    protocol = freeze_protocol(data, output)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    summaries = []
    for task_id in TASKS:
        task_index = next(i for i, task in enumerate(tasks) if task["task_id"] == task_id)
        task = tasks[task_index]
        subset, frame = observed_subset(arrays, molecules, task_index)
        measured = join_activity(frame, subset["labels"], observations, mapping, raw, task_id)
        measured = measured.assign(target_id=task["target_id"], endpoint=task["endpoint"])
        context = training_context(measured, seed=42)
        complete = pd.concat([measured.drop(columns=list(CONTEXT_NAMES)), context], axis=1)
        destination = output / task_id
        destination.mkdir()
        measured[["identity", "record_id", "split", "y_active", "pactivity"]].to_csv(
            destination / "lineage.csv", index=False)
        provenance = {**chemistry, "target_id": task["target_id"], "endpoint": task["endpoint"],
                      "threshold": 6., "context_method": "exact_tanimoto_scaffold_oof_v1",
                      "split_method": "B09_original_global_scaffold", "split_seed": 42,
                      "source_protocol_sha256": file_hash(output / "protocol.json"),
                      "activity_aggregation": protocol["tiny"]["regression_values"]}
        dataset = destination / "tiny-dataset"
        write_bundle(FeatureSet(complete, provenance), dataset)
        for seed in SEEDS:
            print(f"CONTROL task={task_id} kind=matched seed={seed}", flush=True)
            matched = train_multitask(subset["x"], subset["labels"], subset["splits"],
                                     destination / "matched" / f"seed-{seed}", seed=seed)
            print(f"CONTROL task={task_id} kind=tiny seed={seed}", flush=True)
            tiny_dir = destination / "tiny" / f"seed-{seed}"
            tiny = train_dataset(dataset, tiny_dir, TrainingConfig(epochs=40, patience=6,
                                batch_size=512, seed=seed))
            scores = score_control(frame, tiny_dir, "tiny")
            np.savez_compressed(tiny_dir / "lineage-predictions.npz", identities=subset["identities"],
                                **scores)
            n_train = int((subset["splits"] == "train").sum())
            summaries.append({"task_id": task_id, "seed": seed, "observed_train": n_train,
                              "updates_per_epoch": int(np.ceil(n_train / 512)),
                              "matched_epochs": len(matched["history"]), "tiny": tiny})
            write_json(output / "progress.json", {"completed_pairs": summaries})
    for path, digest in {**protocol["source_hashes"], **protocol["code_hashes"]}.items():
        if file_hash(path) != digest:
            raise ValueError(f"Frozen input changed during controls training: {path}")
    hashes = {str(path.relative_to(output)): file_hash(path) for path in output.rglob("*") if path.is_file()}
    write_json(output / "completion.json", {"pairs": summaries, "files": hashes})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.data.resolve(), args.output.resolve())
