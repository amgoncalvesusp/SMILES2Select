"""CLI workflows binding chemistry, data, references, and model provenance."""

import json
import tempfile
from pathlib import Path

import pandas as pd

from .artifacts import file_hash, read_bundle, write_bundle, write_json
from .schema import CONTEXT_NAMES, FeatureSet


def prepare_candidates(source, destination, bits=2048):
    from .features import featurize
    from .smiles2select_io import read_smiles2select

    incoming = read_smiles2select(source)
    prepared = featurize(incoming.records, bits)
    write_bundle(prepared, destination)
    return {
        "output": str(destination),
        "rows": len(prepared.records),
        "valid": prepared.manifest["valid_count"],
        "eligible": prepared.manifest["eligible_count"],
        "input_scope": incoming.manifest["input_scope"],
    }


def prepare_dataset(
    source,
    destination,
    target,
    endpoint,
    threshold=6.0,
    method="scaffold",
    seed=42,
    bits=2048,
    max_rows=250000,
):
    from .context import training_context
    from .data import load_measurements
    from .features import featurize
    from .splits import assign_splits

    measured = load_measurements(source, target, endpoint, threshold, max_rows)
    metadata = dict(measured.attrs)
    if measured.original_smiles.fillna("").eq("").any():
        raise ValueError(
            "measurements lack structures; join the source structure table before dataset preparation"
        )
    features = featurize(measured, bits)
    valid = features.records.loc[features.records.valid].copy()
    labeled = valid.y_active.notna() | valid.pactivity.notna()
    accepted = valid.loc[labeled].copy()
    if accepted.empty:
        raise ValueError("no valid measured molecules")
    if accepted.identity.duplicated().any():
        raise ValueError(
            "duplicate molecular identities: curate compatible assay replicates explicitly before training"
        )
    split = assign_splits(accepted, method=method, seed=seed)
    split.attrs["threshold"] = threshold
    context = training_context(split, seed=seed, temporal=method == "temporal")
    complete = pd.concat(
        [split.drop(columns=list(CONTEXT_NAMES), errors="ignore"), context], axis=1
    )
    manifest = {
        **features.manifest,
        **metadata,
        "input_path": str(Path(source).resolve()),
        "split_method": method,
        "split_seed": seed,
        "context_method": "exact_tanimoto_scaffold_oof_v1"
        if method != "temporal"
        else "exact_tanimoto_strict_past_v1",
        "source_measurements": len(measured),
        "invalid_measurements": int((~features.records.valid).sum()),
        "unknown_measurements": int((~labeled).sum()),
        "split_counts": {str(k): int(v) for k, v in complete.split.value_counts().items()},
        "split_classes": {
            name: {
                "active": int(group.y_active.eq(1).sum()),
                "inactive": int(group.y_active.eq(0).sum()),
                "scaffolds": int(group.murcko_scaffold.nunique()),
            }
            for name, group in complete.groupby("split")
        },
        "temporal_interpretation": "retrospective grouped latest evidence; not prospective validation"
        if method == "temporal"
        else None,
    }
    complete.attrs["provenance"] = manifest
    write_bundle(FeatureSet(complete, manifest), destination)
    return manifest


def train_dataset(source, destination, config, threads=4):
    import torch

    from .training import train_model

    dataset = read_bundle(source)
    provenance = dataset.manifest
    required = ("target_id", "endpoint", "threshold", "chemistry_hash", "context_method")
    if any(key not in provenance for key in required):
        raise ValueError(
            "training requires a prepared measured dataset with task/chemistry/context provenance"
        )
    if threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    refs = dataset.records.loc[dataset.records.split.eq("train")].copy()
    reference_path = Path(destination) / "references"
    if reference_path.exists():
        saved = read_bundle(reference_path)
        if saved.manifest.get("dataset_records_sha256") != provenance["records_sha256"]:
            raise ValueError("existing model references belong to another dataset")
    report = train_model(
        dataset.records,
        destination,
        config,
        fingerprint_bits=provenance["fingerprint_bits"],
        provenance=provenance,
    )
    if not reference_path.exists():
        write_bundle(
            FeatureSet(
                refs,
                {
                    **provenance,
                    "dataset_records_sha256": provenance["records_sha256"],
                    "reference_scope": "training_only",
                },
            ),
            reference_path,
        )
    references = read_bundle(reference_path)
    report = {
        **report,
        "reference_records_sha256": references.manifest["records_sha256"],
        "runtime_ready": True,
        "release_status": "experimental_not_biologically_validated",
    }
    write_json(Path(destination) / "manifest.json", report)
    (Path(destination) / "MODEL_CARD.md").write_text(
        "# S2S-Decision Tiny experimental model\n\n"
        f"Target: {provenance['target_id']}; endpoint: {provenance['endpoint']}; "
        f"threshold: pActivity >= {provenance['threshold']}.\n\n"
        "This model ranks candidates using experimental activity supervision. It does not estimate "
        "P(advance), prove binding, or predict clinical efficacy. Test-set evaluation is a separate "
        "command. Training reference identities, dataset checksums, chemistry settings, calibration "
        "status and software versions are recorded in manifest.json.\n",
        encoding="utf-8",
    )
    return {
        "output": str(destination),
        "parameter_count": report["parameter_count"],
        "epochs": len(report["history"]),
        "calibration": report["calibrator"]["status"],
        "onnx_parity_max_abs": report["onnx_parity_max_abs"],
        "heldout_test_evaluated": False,
    }


def score_candidates(source, model_dir, destination, batch_size=1024):
    from .context import _references, build_context
    from .inference import predict

    features = read_bundle(source)
    model = json.loads((Path(model_dir) / "manifest.json").read_text(encoding="utf-8"))
    if not model.get("runtime_ready"):
        raise ValueError("model has no complete reference package")
    task = model["provenance"]
    references = read_bundle(Path(model_dir) / "references")
    if references.manifest["records_sha256"] != model.get("reference_records_sha256"):
        raise ValueError("model reference checksum mismatch")
    from .features import validate_chemistry_compatibility
    from .session_adapter import reject_model_identity_collisions

    validate_chemistry_compatibility(task, features.manifest, references.records)
    reject_model_identity_collisions(features)
    valid = features.records.loc[features.records.valid & features.records.eligible].copy()
    for field in ("target_id", "endpoint"):
        if field in valid:
            provided = valid[field].replace(r"^\s*$", None, regex=True).dropna()
            if not provided.eq(task[field]).all():
                raise ValueError(f"candidate {field} conflicts with model")
        valid[field] = task[field]
    contexts = build_context(valid, references.records, threshold=task["threshold"])
    inputs = pd.concat([valid.drop(columns=list(CONTEXT_NAMES), errors="ignore"), contexts], axis=1)
    predictions = predict(inputs, model_dir, batch_size=batch_size)
    evidence = _references(references.records, task["threshold"])
    predictions = predictions.assign(
        reference_count=(
            len(evidence) - valid.identity.isin(evidence.identity).astype(int)
        ).to_numpy(),
        reference_similarity_max=(1 - contexts.novelty).to_numpy(),
        reference_distance_status=contexts.novelty.notna()
        .map({True: "measured_similarity_not_confidence", False: "no_external_reference"})
        .to_numpy(),
    )
    scored = features.records.drop(
        columns=[*CONTEXT_NAMES, *predictions.columns.drop("record_id")], errors="ignore"
    )
    scored = scored.merge(predictions, on="record_id", how="left", validate="one_to_one")
    context_rows = contexts.assign(record_id=valid.record_id.to_numpy())
    scored = scored.merge(context_rows, on="record_id", how="left", validate="one_to_one")
    scored["priority_percentile"] = scored.priority_score.rank(method="average", pct=True) * 100
    scored["score_status"] = scored.priority_score.notna().map(
        {True: "experimental", False: "not_scored"}
    )
    manifest = {
        **features.manifest,
        "model_sha256": model["onnx_sha256"],
        "model_manifest_sha256": file_hash(Path(model_dir) / "manifest.json"),
        "reference_records_sha256": references.manifest["records_sha256"],
        "task": {key: task[key] for key in ("target_id", "endpoint", "threshold")},
        "meaning": "priority_percentile is relative to the eligible pool, never P(advance)",
        "calibration_status": model["calibrator"]["status"],
        "estimator": model.get("estimator", "tiny"),
        "reference_interpretation": "labeled training references excluding own identity; similarity is not probability confidence",
        "scored_count": len(predictions),
    }
    write_bundle(FeatureSet(scored, manifest), destination)
    return {"output": str(destination), "scored": len(predictions), "total": len(scored)}


def train_baseline_dataset(
    source: str | Path,
    destination: str | Path,
    estimator: str = "logistic",
    seed: int = 42,
    threads: int = 4,
    input_layout: str | None = None,
) -> dict:
    """Publish a complete classifier/reference package atomically; never overwrite."""
    from .baseline_training import train_baseline

    folder = Path(destination)
    if folder.exists():
        raise ValueError("Output directory already exists; choose a new directory")
    dataset = read_bundle(source)
    provenance = dataset.manifest
    required = (
        "target_id",
        "endpoint",
        "threshold",
        "chemistry_hash",
        "context_method",
        "fingerprint_bits",
    )
    if any(key not in provenance for key in required):
        raise ValueError("training requires prepared task/chemistry/context provenance")
    folder.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".s2s-model-", dir=folder.parent) as temporary:
        staging = Path(temporary) / "package"
        staging.mkdir()
        report = train_baseline(
            dataset.records, staging, provenance, estimator, seed, threads, input_layout
        )
        refs = dataset.records.loc[dataset.records.split.eq("train")].copy()
        write_bundle(
            FeatureSet(
                refs,
                {
                    **provenance,
                    "dataset_records_sha256": provenance["records_sha256"],
                    "reference_scope": "training_only",
                },
            ),
            staging / "references",
        )
        references = read_bundle(staging / "references")
        report = {
            **report,
            "reference_records_sha256": references.manifest["records_sha256"],
            "runtime_ready": True,
            "release_status": "experimental_not_biologically_validated",
        }
        write_json(staging / "manifest.json", report)
        (staging / "MODEL_CARD.md").write_text(
            f"# S2S-Decision experimental {estimator} model\n\n"
            f"Target: {provenance['target_id']}; endpoint: {provenance['endpoint']}; "
            f"threshold: pActivity >= {provenance['threshold']}.\n\n"
            f"Input layout: {report['input_layout']}. Fixed estimator trained on train rows only. "
            "Validation metrics are descriptive; no automatic model selection. Separate calibration "
            "partition fits probability correction. Held-out test was not evaluated.\n\n"
            "Predicts experimental activity, never P(advance), binding proof or clinical efficacy. "
            "No regression prediction is supported. ONNX export parity, training-only references, "
            "dataset hashes, chemistry settings and calibration status are recorded in manifest.json. "
            "No prior benchmark outcome constitutes validation of this newly trained package.\n",
            encoding="utf-8",
        )
        staging.rename(folder)
    return {"output": str(destination), **report}


def evaluate_dataset(source, model_dir, n=10):
    from .inference import predict
    from .metrics import evaluate_predictions

    dataset = read_bundle(source)
    model = json.loads((Path(model_dir) / "manifest.json").read_text(encoding="utf-8"))
    if dataset.manifest["records_sha256"] != model["provenance"].get("records_sha256"):
        raise ValueError("evaluation dataset differs from the frozen training protocol")
    held = dataset.records.loc[dataset.records.split.eq("test")]
    predictions = predict(held, model_dir)
    regression = predictions.predicted_pactivity if model["regression_supported"] else None
    report = evaluate_predictions(
        held.y_active,
        predictions.activity_probability,
        regression,
        held.pactivity if regression is not None else None,
        n=n,
    )
    import numpy as np

    common = (
        np.isfinite(pd.to_numeric(held.qed, errors="raise"))
        & np.isfinite(pd.to_numeric(held.similarity_active_max, errors="raise"))
    ).to_numpy()
    paired = evaluate_predictions(
        held.y_active.to_numpy()[common], predictions.activity_probability.to_numpy()[common], n=n
    )
    estimator = model.get("estimator", "tiny")
    return {
        "estimator": estimator,
        "model": report,
        "model_comparison_pool": paired,
        **({"tiny": report, "tiny_comparison_pool": paired} if estimator == "tiny" else {}),
        "comparison_pool": "finite QED and similarity_active_max, same as baselines",
        "dataset_records_sha256": dataset.manifest["records_sha256"],
        "onnx_sha256": model["onnx_sha256"],
        "split": "test",
        "calibration": model["calibrator"],
        "limitation": "single-run retrospective metrics; not a validated release or confidence interval",
    }
