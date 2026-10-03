"""Train frozen S2 Decision seeds and a paired logistic control, then evaluate.

This research runner never selects a model or task using test performance.
Missing labels remain unknown; each metric uses that task's observed test rows.
"""

import argparse
import json
import platform
import warnings
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from s2s_decision.artifacts import file_hash, write_json


def verify_preparation(data: Path) -> None:
    root = Path(data).resolve()
    completion = json.loads((root / "completion.json").read_text(encoding="utf-8"))
    expected = completion["files"]
    if not {"arrays.npz", "tasks.json", "preprocess.json", "chemistry.json"} <= set(expected):
        raise ValueError("Prepared completion manifest lacks required data hashes")
    for name, digest in expected.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or file_hash(path) != digest:
            raise ValueError(f"Prepared data changed or unsafe path: {name}")


def load_dataset(data: Path) -> tuple[dict, list]:
    verify_preparation(data)
    with np.load(Path(data) / "arrays.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in ("x", "labels", "splits", "identities")}
    x, labels, splits, identities = (arrays[k] for k in ("x", "labels", "splits", "identities"))
    tasks = json.loads((Path(data) / "tasks.json").read_text(encoding="utf-8"))
    if x.ndim != 2 or labels.ndim != 2 or len(x) != len(labels) or not len(x):
        raise ValueError("Features and labels must be nonempty aligned matrices")
    if splits.shape != (len(x),) or identities.shape != (len(x),):
        raise ValueError("Splits and identities must align with feature rows")
    if len(set(identities)) != len(identities):
        raise ValueError("Molecular identities must be globally unique across splits")
    if set(splits) != {"train", "validation", "calibration", "test"}:
        raise ValueError("All four known split names must be present")
    if not np.isfinite(x).all() or np.isinf(labels).any():
        raise ValueError("Features must be finite; labels may be missing but not infinite")
    if not np.isin(labels[np.isfinite(labels)], [0, 1]).all():
        raise ValueError("Observed labels must be binary")
    if not isinstance(tasks, list) or len(tasks) != labels.shape[1] or not tasks:
        raise ValueError("Task definitions must match label columns")
    if len({item["task_id"] for item in tasks}) != len(tasks):
        raise ValueError("Task identifiers must be unique")
    return arrays, tasks


def freeze_protocol(data: Path, output: Path, seeds: list[int], max_epochs: int,
                    plan_path: Path | None = None) -> dict:
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Training seeds must be distinct nonnegative integers")
    if max_epochs < 1:
        raise ValueError("max_epochs must be positive")
    plan = None
    if plan_path is not None:
        plan_path = Path(plan_path).resolve()
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan["model_name"] != "S2 Decision" or plan["model"]["seeds"] != list(seeds):
            raise ValueError("Model name and seeds must match the frozen experiment plan")
        if plan["model"]["max_epochs"] != max_epochs:
            raise ValueError("Epoch budget must match the frozen experiment plan")
    source = Path(__file__).resolve().parents[1]
    data_files = sorted(path for path in Path(data).resolve().rglob("*") if path.is_file())
    code_files = sorted((source / "src").rglob("*.py"))
    code_files += [Path(__file__).resolve()]
    prepare = source / "benchmarks/prepare_multitask.py"
    if prepare.exists():
        code_files.append(prepare)
    if plan_path is not None:
        data_files.append(plan_path)
    protocol = {
        "schema": "s2-decision-training-run/1", "model_name": "S2 Decision",
        "created_utc": datetime.now(UTC).isoformat(), "seeds": list(seeds),
        "data": str(Path(data).resolve()),
        "training": {"max_epochs": max_epochs, "patience": 6, "batch_size": 512,
                     "hidden": [512, 256], "learning_rate": 0.001},
        "resolved_defaults": {"dropout": 0.1, "optimizer": "AdamW", "weight_decay": 0.0001,
                              "optimizer_betas": [0.9, 0.999], "optimizer_epsilon": 1e-8,
                              "calibrator": {"C": 1000, "max_iter": 1000,
                                             "solver": "lbfgs", "random_state": 0},
                              "torch_threads": 4, "deterministic_algorithms": True,
                              "onnx_opset": 17, "onnx_absolute_tolerance": 0.00001},
        "baseline": {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
        "calibration_min_per_class": 20, "selection_k": [50, 100],
        "test_usage": "evaluation_only_after_all_fits",
        "evaluation_scope": "internal_global_scaffold_split_observed_labels_only",
        "promotion": "none; external validation required",
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {name: version(name) for name in (
            "numpy", "scipy", "pandas", "rdkit", "torch", "scikit-learn",
            "onnx", "onnxruntime", "threadpoolctl",
        )},
        "data_hashes": {str(path): file_hash(path) for path in data_files},
        "code_hashes": {str(path): file_hash(path) for path in code_files},
    }
    if plan is not None:
        for key, value in protocol["training"].items():
            if plan["model"][key] != value:
                raise ValueError(f"Training setting {key} differs from frozen experiment plan")
        for key, value in protocol["baseline"].items():
            if plan["baseline"][key] != value:
                raise ValueError(f"Baseline setting {key} differs from frozen experiment plan")
        if plan["evaluation"]["K"] != protocol["selection_k"]:
            raise ValueError("Selection K differs from frozen experiment plan")
        protocol = {**protocol, "experiment_plan": str(plan_path),
                    "research_version": plan["research_version"]}
    Path(output).mkdir(parents=True, exist_ok=False)
    write_json(Path(output) / "protocol.json", protocol)
    return protocol


def verify_inputs(protocol: dict) -> None:
    for path, expected in {**protocol["data_hashes"], **protocol["code_hashes"]}.items():
        if file_hash(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")


def fit_baseline(x: np.ndarray, labels: np.ndarray, splits: np.ndarray,
                 output: Path, config: dict, tasks: list[dict]) -> dict:
    from s2s_decision.multitask import calibrate_multitask

    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    weights, intercepts, statuses = [], [], []
    for task in range(labels.shape[1]):
        observed = (splits == "train") & np.isfinite(labels[:, task])
        if set(labels[observed, task]) != {0, 1}:
            raise ValueError(f"Task {task} lacks both training classes")
        with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=4):
            warnings.simplefilter("always", ConvergenceWarning)
            estimator = LogisticRegression(**config).fit(x[observed], labels[observed, task])
        weights.append(estimator.coef_[0])
        intercepts.append(float(estimator.intercept_[0]))
        status = {
            "task_index": task, "task_id": tasks[task]["task_id"],
            "observed_train": int(observed.sum()),
            "iterations": int(estimator.n_iter_[0]),
            "converged": not any(issubclass(w.category, ConvergenceWarning) for w in captured),
            "warnings": [str(w.message) for w in captured],
        }
        statuses.append(status)
        print(f"LOGISTIC task={task + 1}/{labels.shape[1]} converged={status['converged']}", flush=True)
    coefficients = np.asarray(weights)
    intercept = np.asarray(intercepts)
    np.savez_compressed(output / "coefficients.npz", coefficients=coefficients, intercept=intercept)
    with threadpool_limits(limits=4):
        logits = x @ coefficients.T + intercept
    probability, calibration = calibrate_multitask(logits, labels, splits, min_per_class=20)
    np.save(output / "logits.npy", logits, allow_pickle=False)
    np.save(output / "raw_probabilities.npy", expit(logits).astype(np.float32), allow_pickle=False)
    np.save(output / "probabilities.npy", probability, allow_pickle=False)
    write_json(output / "calibration.json", {"tasks": calibration})
    summary = {"method": "logistic", "fit_seconds": perf_counter() - started, "tasks": statuses}
    write_json(output / "training.json", summary)
    return summary


def _metric(task, name):
    if name.startswith("ef_at_"):
        return task["selection"][name.removeprefix("ef_at_")]["enrichment_factor"]
    return task[name]


def macro_summary(report: dict) -> dict:
    output = {}
    for name in ("average_precision", "roc_auc", "brier", "ef_at_50", "ef_at_100"):
        values = [_metric(task, name) for task in report["tasks"]]
        defined = [value for value in values if value is not None and np.isfinite(value)]
        output[name] = {"mean": float(np.mean(defined)) if defined else None,
                        "tasks": len(defined), "total_tasks": len(values)}
    return output


def paired_summary(neural: dict, baseline: dict) -> dict:
    reference = {task["task_index"]: task for task in baseline["tasks"]}
    if set(reference) != {task["task_index"] for task in neural["tasks"]}:
        raise ValueError("Paired reports must contain identical tasks")
    output = {"total_tasks": len(reference)}
    for name in ("average_precision", "roc_auc", "brier", "ef_at_50", "ef_at_100"):
        differences = []
        for task in neural["tasks"]:
            value, control = _metric(task, name), _metric(reference[task["task_index"]], name)
            if value is not None and control is not None and np.isfinite([value, control]).all():
                differences.append(value - control)
        direction = -1 if name == "brier" else 1
        output[name] = {
            "paired_tasks": len(differences),
            "mean_delta": float(np.mean(differences)) if differences else None,
            "median_delta": float(np.median(differences)) if differences else None,
            "wins": sum(direction * delta > 0 for delta in differences),
            "ties": sum(delta == 0 for delta in differences),
            "losses": sum(direction * delta < 0 for delta in differences),
            "delta_definition": "S2 Decision minus logistic; lower Brier is better",
        }
    return output


def evaluate_runs(arrays: dict, tasks: list[dict], output: Path,
                  methods: list[str], ks: list[int]) -> dict:
    from s2s_decision.multitask import evaluate_multitask

    held = arrays["splits"] == "test"
    reports = {}
    for method in methods:
        for state, filename in (("raw", "raw_probabilities.npy"), ("calibrated", "probabilities.npy")):
            probabilities = np.load(output / method / filename, allow_pickle=False)
            report = evaluate_multitask(arrays["labels"][held], probabilities[held], ks=ks)
            report["tasks"] = [
                {**item, "task_id": tasks[item["task_index"]]["task_id"]}
                for item in report["tasks"]
            ]
            reports[f"{method}/{state}"] = {**report, "macro": macro_summary(report)}
    paired = {
        key: paired_summary(report, reports[f"logistic/{key.rsplit('/', 1)[1]}"])
        for key, report in reports.items() if not key.startswith("logistic/")
    }
    return {"scope": "internal test; task-specific observed-label denominators",
            "test_molecules": int(held.sum()), "methods": reports, "paired": paired}


def run(data: Path, output: Path, seeds: list[int], max_epochs: int,
        plan_path: Path | None = None) -> dict:
    from s2s_decision.multitask import train_multitask

    arrays, tasks = load_dataset(data)
    protocol = freeze_protocol(data, output, seeds, max_epochs, plan_path)
    output = Path(output)
    summaries = {}
    methods = [f"s2_decision_seed{seed}" for seed in seeds]
    for seed, method in zip(seeds, methods, strict=True):
        verify_inputs(protocol)
        print(f"TRAIN {method}", flush=True)
        summaries[method] = train_multitask(
            arrays["x"], arrays["labels"], arrays["splits"], output / method,
            seed=seed, **protocol["training"],
        )
    verify_inputs(protocol)
    summaries["logistic"] = fit_baseline(
        arrays["x"], arrays["labels"], arrays["splits"], output / "logistic", protocol["baseline"],
        tasks,
    )
    write_json(output / "fits-complete.json", {"methods": summaries})
    verify_inputs(protocol)
    print("EVALUATE frozen test after all fits", flush=True)
    metrics = evaluate_runs(arrays, tasks, output, [*methods, "logistic"], protocol["selection_k"])
    write_json(output / "metrics.json", metrics)
    products = sorted(path for path in output.rglob("*") if path.is_file())
    write_json(output / "completion.json", {
        "model_name": "S2 Decision", "completed_utc": datetime.now(UTC).isoformat(),
        "files": {str(path.relative_to(output)): file_hash(path) for path in products},
    })
    print(f"COMPLETE {output.resolve()}", flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--plan", type=Path, help="Frozen experiment plan; checked and hashed")
    args = parser.parse_args()
    run(args.data, args.output, args.seeds, args.max_epochs, args.plan)


if __name__ == "__main__":
    main()
