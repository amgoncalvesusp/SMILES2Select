"""CPU training, held-out calibration, and verified ONNX export."""

import hashlib
import json
import platform
import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import expit
from torch import nn

from .metrics import evaluate_predictions
from .model import Tiny
from .preprocessing import Preprocessor
from .schema import CONTEXT_NAMES, PROPERTY_NAMES, SCHEMA_VERSION, fingerprint_matrix


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 200
    batch_size: int = 1024
    patience: int = 20
    seed: int = 42
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    resume: bool = False
    width_multiplier: int = 1

    def __post_init__(self):
        if type(self.width_multiplier) is not int or self.width_multiplier not in (1, 2):
            raise ValueError("width_multiplier must be integer 1 or 2")
        if min(self.epochs, self.batch_size, self.patience) < 1:
            raise ValueError("epochs, batch_size and patience must be positive")
        if (
            not np.isfinite(self.learning_rate)
            or self.learning_rate <= 0
            or not np.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise ValueError("Invalid learning rate or weight decay")


def _json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _require_safe_checkpoint_runtime():
    # weights_only was vulnerable before 2.10.0; reject prereleases with unknown fixes.
    # https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p
    version = str(torch.__version__)
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:\+[A-Za-z0-9._-]+)?", version)
    if match is None or tuple(map(int, match.groups())) < (2, 10, 0):
        raise ValueError(
            f"Resume requires PyTorch >= 2.10.0 stable for safe checkpoint loading; found {version}"
        )


def _validate(frame):
    required = {
        "record_id",
        "identity",
        "split",
        "y_active",
        "pactivity",
        "fingerprint_hex",
        *PROPERTY_NAMES,
        *CONTEXT_NAMES,
    }
    if required - set(frame):
        raise ValueError(f"Missing training columns: {sorted(required - set(frame))}")
    if (
        frame.empty
        or frame.record_id.isna().any()
        or frame.record_id.duplicated().any()
        or frame.identity.isna().any()
    ):
        raise ValueError("Training requires unique record_id and nonmissing identity")
    if not frame.split.isin(("train", "validation", "calibration", "test")).all():
        raise ValueError("Invalid split")
    if set(frame.split) != {"train", "validation", "calibration", "test"}:
        raise ValueError("All four splits must contain records")
    if frame.groupby("identity").split.nunique().gt(1).any():
        raise ValueError("Identity leakage between splits")
    if frame.identity.duplicated().any():
        raise ValueError(
            "Duplicate molecular identity: curate assay replicates explicitly before training"
        )
    labels = pd.to_numeric(frame.y_active, errors="raise").to_numpy(float)
    if np.isinf(labels).any() or not np.isin(labels[np.isfinite(labels)], (0, 1)).all():
        raise ValueError("y_active must be binary or missing")
    if np.isinf(pd.to_numeric(frame.pactivity, errors="raise").to_numpy(float)).any():
        raise ValueError("Infinite pactivity is invalid")
    for split in ("train", "validation"):
        observed = frame.loc[frame.split.eq(split), "y_active"].dropna()
        if set(observed) != {0, 1}:
            raise ValueError(f"{split} requires both activity classes")


def _loss(output, labels, activity):
    classified, measured = torch.isfinite(labels), torch.isfinite(activity)
    loss = output.sum() * 0
    if classified.any():
        loss = loss + nn.functional.binary_cross_entropy_with_logits(
            output[classified, 0], labels[classified]
        )
    if measured.any():
        loss = loss + 0.2 * nn.functional.huber_loss(output[measured, 1], activity[measured])
    return loss


def _raw(model, arrays, batch_size):
    model.eval()
    with torch.no_grad():
        batches = [
            model(*(value[start : start + batch_size] for value in arrays)).numpy()
            for start in range(0, len(arrays[0]), batch_size)
        ]
    return np.concatenate(batches) if batches else np.empty((0, 2), np.float32)


def _calibrate(logits, labels):
    valid = np.isfinite(labels)
    x, y = logits[valid], labels[valid]
    if len(y) < 20 or min(np.sum(y == 0), np.sum(y == 1)) < 5:
        return {
            "slope": 1.0,
            "intercept": 0.0,
            "status": "uncalibrated_insufficient_data",
            "count": len(y),
        }

    def objective(parameters):
        values = parameters[0] * x + parameters[1]
        return np.mean(np.logaddexp(0, values) - y * values)

    result = minimize(
        objective, [1.0, 0.0], bounds=((1e-6, 100.0), (-100.0, 100.0)), method="L-BFGS-B"
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Calibration failed: {result.message}")
    return {
        "slope": float(result.x[0]),
        "intercept": float(result.x[1]),
        "status": "fitted_held_out_not_prospectively_validated",
        "count": len(y),
    }


def _export(model, arrays, folder):
    import onnxruntime as ort

    temporary = folder / "model.pending.onnx"
    sample = tuple(value[: min(3, len(value))] for value in arrays)
    torch.onnx.export(
        model,
        sample,
        temporary,
        input_names=["properties", "fingerprint", "context"],
        output_names=["outputs"],
        opset_version=17,
        dynamo=False,
        dynamic_axes={
            name: {0: "batch"} for name in ("properties", "fingerprint", "context", "outputs")
        },
    )
    session = ort.InferenceSession(str(temporary), providers=["CPUExecutionProvider"])
    expected = _raw(model, sample, 3)
    observed = session.run(
        None,
        dict(zip(("properties", "fingerprint", "context"), (value.numpy() for value in sample))),
    )[0]
    np.testing.assert_allclose(observed, expected, atol=1e-5, rtol=1e-5)
    temporary.replace(folder / "model.onnx")
    return float(np.max(np.abs(observed - expected)))


def train_model(frame, output_dir, config=TrainingConfig(), fingerprint_bits=2048, provenance=None):
    """Use development rows only; test rows enter schema checks and provenance hash only."""
    _validate(frame)
    folder = Path(output_dir)
    checkpoint_path = folder / "checkpoint.pt"
    if not config.resume and folder.exists() and any(folder.iterdir()):
        raise ValueError("Output directory already exists and is not empty")
    if config.resume and not checkpoint_path.is_file():
        raise ValueError("Resume checkpoint does not exist")
    columns = [
        "record_id",
        "identity",
        "split",
        "y_active",
        "pactivity",
        "fingerprint_hex",
        *PROPERTY_NAMES,
        *CONTEXT_NAMES,
    ]
    data_hash = hashlib.sha256(
        pd.util.hash_pandas_object(frame[columns], index=False).values.tobytes()
    ).hexdigest()
    configuration = {k: v for k, v in asdict(config).items() if k not in ("resume", "epochs")}
    contract = {
        "data_hash": data_hash,
        "config": configuration,
        "fingerprint_bits": fingerprint_bits,
        "schema": SCHEMA_VERSION,
        "provenance": provenance or {},
    }
    # Canonical JSON also rejects nonserializable provenance before training.
    contract = json.loads(json.dumps(contract, sort_keys=True, allow_nan=False))
    if config.resume:
        _require_safe_checkpoint_runtime()
    checkpoint = (
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if config.resume
        else None
    )
    if checkpoint:
        saved_contract = checkpoint["contract"]
        # Historical checkpoints used the original width without an explicit size field.
        saved_contract = {
            **saved_contract,
            "config": {"width_multiplier": 1, **saved_contract["config"]},
        }
        if saved_contract != contract:
            raise ValueError(
                "Resume incompatible with dataset, schema, configuration or provenance"
            )
    # Seal test features as well as labels before any transform, forward pass, or export.
    # Retain development order and the full-input resume contract above.
    frame = frame.loc[~frame.split.eq("test")].copy()
    torch.manual_seed(config.seed)
    prep = Preprocessor.fit(frame.loc[frame.split.eq("train")])
    properties, context = prep.transform(frame)
    arrays = tuple(
        torch.from_numpy(value)
        for value in (properties, fingerprint_matrix(frame, fingerprint_bits), context)
    )
    labels = torch.tensor(frame.y_active.to_numpy(float), dtype=torch.float32)
    regression = frame.pactivity.to_numpy(float)
    measured_train = regression[frame.split.eq("train").to_numpy() & np.isfinite(regression)]
    mean = float(measured_train.mean()) if len(measured_train) else 0.0
    scale = float(measured_train.std()) if len(measured_train) and measured_train.std() > 0 else 1.0
    activity = torch.tensor((regression - mean) / scale, dtype=torch.float32)
    train_indices = torch.tensor(np.flatnonzero(frame.split.eq("train")), dtype=torch.long)
    validation_indices = np.flatnonzero(frame.split.eq("validation"))
    model = Tiny(fingerprint_bits, width_multiplier=config.width_multiplier)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    start, best, stale, best_state, history = 0, -1.0, 0, deepcopy(model.state_dict()), []
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        torch.set_rng_state(checkpoint["rng"])
        start, best, stale = checkpoint["epoch"] + 1, checkpoint["best"], checkpoint["stale"]
        best_state, history = checkpoint["best_model"], checkpoint["history"]
    folder.mkdir(parents=True, exist_ok=True)
    for epoch in range(start, config.epochs):
        if stale >= config.patience:
            break
        model.train()
        order = train_indices[torch.randperm(len(train_indices))]
        losses = []
        for batch in order.split(config.batch_size):
            optimizer.zero_grad()
            loss = _loss(model(*(value[batch] for value in arrays)), labels[batch], activity[batch])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        raw = _raw(model, tuple(value[validation_indices] for value in arrays), config.batch_size)
        score = evaluate_predictions(labels[validation_indices].numpy(), expit(raw[:, 0]))["pr_auc"]
        scheduler.step(score)
        if score > best:
            best, stale, best_state = score, 0, deepcopy(model.state_dict())
        else:
            stale += 1
        history.append(
            {"epoch": epoch + 1, "loss": float(np.mean(losses)), "validation_pr_auc": score}
        )
        checkpoint = {
            "contract": contract,
            "epoch": epoch,
            "model": model.state_dict(),
            "best_model": best_state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": torch.get_rng_state(),
            "best": best,
            "stale": stale,
            "history": history,
        }
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(checkpoint_path)
    model.load_state_dict(best_state)
    raw = _raw(model, arrays, config.batch_size)
    calibration_indices = frame.split.eq("calibration").to_numpy()
    calibrator = _calibrate(raw[calibration_indices, 0], labels.numpy()[calibration_indices])
    parity = _export(model, arrays, folder)
    onnx_hash = hashlib.sha256((folder / "model.onnx").read_bytes()).hexdigest()
    manifest = {
        **contract,
        "format_version": 1,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "preprocessing": prep.to_dict(),
        "regression_mean": mean,
        "regression_scale": scale,
        "regression_supported": bool(len(measured_train)),
        "calibrator": calibrator,
        "onnx_sha256": onnx_hash,
        "onnx_parity_max_abs": parity,
        "history": history,
        "heldout_test_evaluated": False,
        "versions": {
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
        "warnings": ["Missing context throughout training: " + ", ".join(prep.context_all_missing)]
        if prep.context_all_missing
        else [],
        "meaning": "Activity for the target, endpoint and threshold in provenance; never P(advance).",
    }
    _json(folder / "manifest.json", manifest)
    return manifest


def baseline_predictions(frame, fingerprint_bits=2048, seed=42):
    """Fit fixed baselines on train only and return ID-aligned scores for all test rows."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    _validate(frame)
    train = frame.split.eq("train").to_numpy() & frame.y_active.notna().to_numpy()
    test = frame.split.eq("test").to_numpy()
    prep = Preprocessor.fit(frame.loc[frame.split.eq("train")])
    properties, context = prep.transform(frame)
    scalar = np.concatenate((properties, context), axis=1)
    fingerprint = fingerprint_matrix(frame, fingerprint_bits)
    predictions = frame.loc[test, ["record_id"]].copy()
    for name, estimator, inputs in (
        (
            "logistic",
            LogisticRegression(max_iter=1000, random_state=seed),
            np.concatenate((scalar, fingerprint), axis=1),
        ),
        (
            "gradient_boosting",
            HistGradientBoostingClassifier(max_iter=100, random_state=seed),
            scalar,
        ),
    ):
        estimator.fit(inputs[train], frame.loc[train, "y_active"])
        predictions[name] = (
            estimator.predict_proba(inputs[test])[:, 1] if test.any() else np.empty(0)
        )
    for name, column in (("similarity", "similarity_active_max"), ("qed", "qed")):
        predictions[name] = pd.to_numeric(frame.loc[test, column], errors="raise")
    return predictions


def benchmark_baselines(frame, n=10, fingerprint_bits=2048):
    """Same prepared train/test rows for simple fixed baselines; no model selection on test."""
    predictions = baseline_predictions(frame, fingerprint_bits)
    methods = ["logistic", "gradient_boosting", "similarity", "qed"]
    common = np.isfinite(predictions[methods].to_numpy(float)).all(axis=1)
    scores = predictions.loc[common]
    labels = frame.set_index("record_id").loc[scores.record_id, "y_active"]
    comparison_counts = {
        "full_test_count": len(predictions),
        "common_count": int(common.sum()),
        "excluded_missing_score": int((~common).sum()),
    }
    report = {}
    for name in methods:
        report[name] = evaluate_predictions(labels, scores[name], n=n)
        report[name].update(comparison_counts, score_kind="uncalibrated_model_probability")
        if name in ("similarity", "qed"):
            report[name].update(score_kind="heuristic_ranking_score", brier=None, ece=None)
    return report
