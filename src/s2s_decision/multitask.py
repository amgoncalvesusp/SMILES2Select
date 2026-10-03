"""S2 Decision: sparse-label shared molecular MLP for research training.

Inputs are finite, pretransformed molecular features. Fit and persist preprocessing
using training molecules only. Task order and molecule/scaffold separation belong
to the dataset manifest; this module enforces row split and observed-label contracts.
"""

import json
import warnings
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch import nn


class S2Decision(nn.Module):
    """Shared molecular representation with one independent logit per task."""

    def __init__(self, inputs: int, tasks: int, hidden: tuple[int, ...] = (512, 256)) -> None:
        super().__init__()
        if any(type(v) is not int or v < 1 for v in (inputs, tasks, *hidden)) or not hidden:
            raise ValueError("Network dimensions must be positive integers")
        widths = (inputs, *hidden)
        layers = []
        for before, after in zip(widths[:-1], widths[1:]):
            layers.extend((nn.Linear(before, after), nn.ReLU(), nn.Dropout(.1)))
        self.network = nn.Sequential(*layers, nn.Linear(hidden[-1], tasks))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def masked_task_loss(logits: torch.Tensor, labels: torch.Tensor,
                     train_counts: torch.Tensor, train_size: int) -> torch.Tensor:
    """Unbiased minibatch estimate of mean observed BCE, equally weighted by task.

    Global training counts avoid overweighting a sparse task merely because one
    batch contains its only observed measurement. Unknown cells have zero gradient.
    """
    observed = torch.isfinite(labels)
    targets = torch.where(observed, labels, torch.zeros_like(labels))
    losses = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    per_task = (losses * observed).sum(dim=0) / train_counts
    return per_task.mean() * train_size / len(labels)


def _validate_labels(labels):
    labels = np.asarray(labels, dtype=np.float32)
    if labels.ndim != 2 or 0 in labels.shape:
        raise ValueError("Labels must be a nonempty molecule-by-task matrix")
    if not np.all(np.isnan(labels) | (labels == 0) | (labels == 1)):
        raise ValueError("Labels must be 0, 1, or NaN for unknown")
    return labels


def _validate_splits(splits, rows):
    splits = np.asarray(splits)
    names = {"train", "validation", "calibration", "test"}
    if splits.shape != (rows,) or set(splits.tolist()) != names:
        raise ValueError("All four row splits train/validation/calibration/test are required")
    return splits


def calibrate_multitask(logits: np.ndarray, labels: np.ndarray, splits: np.ndarray,
                        min_per_class: int = 20) -> tuple[np.ndarray, list[dict]]:
    """Fit per-task sigmoid calibration exclusively on calibration measurements."""
    labels = _validate_labels(labels)
    logits = np.asarray(logits, dtype=np.float64)
    splits = _validate_splits(splits, len(labels))
    if logits.shape != labels.shape or not np.isfinite(logits).all():
        raise ValueError("Finite logits must have the label matrix shape")
    if type(min_per_class) is not int or min_per_class < 1:
        raise ValueError("min_per_class must be a positive integer")
    coefficients = []
    calibrated = np.empty_like(logits, dtype=np.float32)
    for task in range(labels.shape[1]):
        mask = (splits == "calibration") & np.isfinite(labels[:, task])
        y = labels[mask, task]
        negative, positive = int((y == 0).sum()), int((y == 1).sum())
        slope, intercept = 1., 0.
        method = "identity_insufficient_support"
        if min(negative, positive) >= min_per_class:
            fit = LogisticRegression(C=1e3, max_iter=1000, random_state=0)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                fit.fit(logits[mask, task:task + 1], y)
            slope, intercept = float(fit.coef_[0, 0]), float(fit.intercept_[0])
            method = "platt_calibration_only"
        coefficients.append({"task_index": task, "method": method, "slope": slope,
                             "intercept": intercept, "negative": negative, "positive": positive})
        calibrated[:, task] = expit(logits[:, task] * slope + intercept)
    return calibrated, coefficients


def evaluate_multitask(labels: np.ndarray, probabilities: np.ndarray,
                       ks: tuple[int, ...] = (50, 100)) -> dict:
    """Evaluate only measured task cells; K applies to each observed task pool."""
    labels = _validate_labels(labels)
    probabilities = np.asarray(probabilities)
    if probabilities.shape != labels.shape or not np.isfinite(probabilities).all():
        raise ValueError("Finite probabilities must have the label matrix shape")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be between zero and one")
    if any(type(k) is not int or k < 1 for k in ks):
        raise ValueError("K must be a positive integer")
    tasks = []
    for task in range(labels.shape[1]):
        mask = np.isfinite(labels[:, task])
        y, p = labels[mask, task], probabilities[mask, task]
        n, positive = len(y), int(y.sum())
        both = 0 < positive < n
        entry = {"task_index": task, "observed": n, "positive": positive,
                 "prevalence": positive / n if n else None,
                 "average_precision": float(average_precision_score(y, p)) if both else None,
                 "roc_auc": float(roc_auc_score(y, p)) if both else None,
                 "brier": float(brier_score_loss(y, p)) if n else None, "selection": {}}
        order = np.argsort(-p, kind="stable")
        for k in ks:
            effective = min(k, n)
            hits = int(y[order[:effective]].sum())
            precision = hits / effective if effective else None
            entry["selection"][str(k)] = {
                "effective_k": effective, "hits": hits, "precision": precision,
                "recall": hits / positive if positive else None,
                "enrichment_factor": precision / (positive / n) if positive else None,
            }
        tasks.append(entry)
    result = {"tasks": tasks}
    for key in ("average_precision", "roc_auc", "brier"):
        values = [task[key] for task in tasks if task[key] is not None]
        result[f"macro_{key}"] = float(np.mean(values)) if values else None
    return result


def _predict(model, x, batch_size):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(x[start:start + batch_size]).numpy()
                               for start in range(0, len(x), batch_size)])


class _Calibrated(nn.Module):
    def __init__(self, model, coefficients):
        super().__init__()
        self.model = model
        self.register_buffer("slope", torch.tensor([c["slope"] for c in coefficients]))
        self.register_buffer("intercept", torch.tensor([c["intercept"] for c in coefficients]))

    def forward(self, features):
        return torch.sigmoid(self.model(features) * self.slope + self.intercept)


def _export(model, coefficients, x, folder):
    import onnxruntime as ort

    calibrated = _Calibrated(model, coefficients).eval()
    sample = x[:min(7, len(x))]
    path = folder / "model.onnx"
    torch.onnx.export(calibrated, (sample,), path, input_names=["features"],
                      output_names=["probabilities"], opset_version=17, dynamo=False,
                      dynamic_axes={"features": {0: "batch"}, "probabilities": {0: "batch"}})
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(path), sess_options=options,
                                  providers=["CPUExecutionProvider"])
    error = 0.
    for size in (1, len(sample)):
        with torch.no_grad():
            expected = calibrated(sample[:size]).numpy()
        actual = session.run(None, {"features": sample[:size].numpy()})[0]
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
        error = max(error, float(np.max(np.abs(actual - expected))))
    if error > 1e-5:
        raise RuntimeError(f"ONNX absolute error exceeds 1e-5: {error}")
    return error


def train_multitask(x: np.ndarray, labels: np.ndarray, splits: np.ndarray,
                    output_dir: str | Path, seed: int = 42, max_epochs: int = 40,
                    patience: int = 6, batch_size: int = 512,
                    hidden: tuple[int, ...] = (512, 256), learning_rate: float = .001) -> dict:
    """Train shared MLP; checkpoint selection uses validation macro AP only.

    Test labels never enter fitting, early stopping, or calibration. No test metrics
    are produced here: evaluate after the experimental comparison is frozen.
    Existing output directories containing files are rejected to preserve evidence.
    """
    x = np.asarray(x, dtype=np.float32)
    labels = _validate_labels(labels)
    splits = _validate_splits(splits, len(labels))
    if x.ndim != 2 or len(x) != len(labels) or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError("Features must be a finite molecule-by-feature matrix")
    if any(type(v) is not int or v < 1 for v in (max_epochs, patience, batch_size)):
        raise ValueError("Epochs, patience and batch size must be positive integers")
    if type(seed) is not int or seed < 0 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Invalid seed or learning rate")
    train, validation = splits == "train", splits == "validation"
    for task in range(labels.shape[1]):
        if set(labels[train, task][np.isfinite(labels[train, task])].tolist()) != {0., 1.}:
            raise ValueError(f"Task {task} requires both observed training classes")
    eligible = [task for task in range(labels.shape[1])
                if set(labels[validation, task][np.isfinite(labels[validation, task])]) == {0., 1.}]
    if not eligible:
        raise ValueError("Validation requires at least one task with both classes")
    model = S2Decision(x.shape[1], labels.shape[1], hidden)
    folder = Path(output_dir)
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("Output directory must be empty; existing evidence is preserved")
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    model = S2Decision(x.shape[1], labels.shape[1], hidden)
    features, targets = torch.from_numpy(x), torch.from_numpy(labels)
    train_x, train_y = features[train], targets[train]
    counts = torch.isfinite(train_y).sum(dim=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best_score, best_state, best_epoch, stale, history = -np.inf, None, 0, 0, []
    for epoch in range(1, max_epochs + 1):
        started = perf_counter()
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        total_loss = 0.
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = masked_task_loss(model(train_x[indices]), train_y[indices], counts, len(train_y))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite multitask training loss")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices) / len(train_y)
        scores = expit(_predict(model, features[validation], batch_size))
        metric = evaluate_multitask(labels[validation], scores, ks=())["macro_average_precision"]
        history.append({"epoch": epoch, "train_loss": total_loss, "validation_macro_ap": metric})
        duration = perf_counter() - started
        progress = {**history[-1], "seconds": duration,
                    "train_molecules_per_second": len(train_y) / duration}
        with (folder / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(progress, allow_nan=False) + "\n")
        print(json.dumps({"seed": seed, **progress}), flush=True)
        if metric > best_score:
            best_score, best_state, best_epoch, stale = metric, deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    torch.save({"state_dict": best_state, "inputs": x.shape[1], "tasks": labels.shape[1],
                "hidden": tuple(hidden), "seed": seed, "best_epoch": best_epoch}, folder / "best.pt")
    logits = _predict(model, features, batch_size)
    probabilities, coefficients = calibrate_multitask(logits, labels, splits)
    for name, values in (("logits", logits), ("raw_probabilities", expit(logits)),
                         ("probabilities", probabilities)):
        np.save(folder / f"{name}.npy", values)
    (folder / "calibration.json").write_text(json.dumps(coefficients, indent=2), encoding="utf-8")
    error = _export(model, coefficients, features, folder)
    summary = {"name": "S2 Decision", "schema": "s2-decision-multitask/1", "seed": seed,
               "inputs": x.shape[1], "tasks": labels.shape[1], "hidden": list(hidden),
               "parameters": sum(p.numel() for p in model.parameters()),
               "dropout": .1, "optimizer": "AdamW", "weight_decay": 1e-4,
               "max_epochs": max_epochs, "patience": patience, "batch_size": batch_size,
               "learning_rate": learning_rate, "best_epoch": best_epoch,
               "best_validation_macro_ap": best_score, "validation_eligible_tasks": eligible,
               "selection": "validation_macro_average_precision_only", "history": history,
               "onnx_max_absolute_error": error, "input_contract": "finite_pretransformed_features",
               "calibration_minimum_per_class": 20, "torch_version": str(torch.__version__)}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary
