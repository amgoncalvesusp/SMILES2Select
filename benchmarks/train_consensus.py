"""B11 train-only measurement-consensus ablation; no external data are accessed."""

import argparse
import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from train_multitask import fit_baseline, load_dataset, verify_inputs

from s2s_decision.artifacts import file_hash, write_json

SEEDS = [42, 43, 44]
KS = [20, 50, 100, 500]
TRAINING = {"max_epochs": 40, "patience": 6, "batch_size": 512,
            "hidden": [512, 256], "learning_rate": .001}
LOGISTIC = {"C": 1., "max_iter": 1000, "solver": "lbfgs"}


def validate_pair(original: dict, clean: dict, excluded: np.ndarray) -> None:
    """Only observed TRAIN labels flagged by the source-consensus mask may change."""
    for key in ("x", "splits", "identities"):
        if (original[key].dtype != clean[key].dtype or original[key].shape != clean[key].shape
                or original[key].tobytes() != clean[key].tobytes()):
            raise ValueError(f"Original features, identities and splits must be unchanged: {key}")
    labels, curated = original["labels"], clean["labels"]
    if (labels.shape != curated.shape or labels.dtype != curated.dtype
            or excluded.shape != labels.shape or excluded.dtype != bool):
        raise ValueError("Consensus mask and labels must have the original shape; mask must be bool")
    if np.any(excluded & ~np.isfinite(labels)):
        raise ValueError("Consensus mask cannot exclude an already unknown observation")
    held = original["splits"] != "train"
    if labels[held].tobytes() != curated[held].tobytes():
        raise ValueError("Original held-out labels must be unchanged")
    expected = np.where(excluded & ~held[:, None], np.nan, labels)
    if not np.array_equal(expected, curated, equal_nan=True):
        raise ValueError("Curated train labels must exactly implement the declared consensus mask")
    for task in range(labels.shape[1]):
        if set(curated[~held, task][np.isfinite(curated[~held, task])]) != {0., 1.}:
            raise ValueError(f"Task {task} lacks both curated training classes")


def method_paths(baseline: Path, output: Path) -> dict[str, Path]:
    return {**{f"baseline_seed{s}": baseline / f"s2_decision_seed{s}" for s in SEEDS},
            **{f"consensus_seed{s}": output / f"s2_decision_seed{s}" for s in SEEDS},
            "baseline_logistic": baseline / "logistic", "consensus_logistic": output / "logistic"}


def freeze(data: Path, baseline_data: Path, baseline: Path, output: Path, plan_path: Path) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan["training"]["seeds"] != SEEDS or any(
        plan["training"][key] != value for key, value in TRAINING.items()
    ):
        raise ValueError("Training settings differ from the frozen plan")
    if plan["comparison"]["K"] != KS or plan["comparison"]["primary_K"] != 50:
        raise ValueError("Evaluation settings differ from the frozen plan")
    inputs = [plan_path]
    for root in (data, baseline_data):
        inputs.extend(path for path in root.rglob("*") if path.is_file())
    completion_path = baseline / "completion.json"
    completed = {name.replace("\\", "/"): digest for name, digest in
                 json.loads(completion_path.read_text(encoding="utf-8"))["files"].items()}
    inputs.append(completion_path)
    historical = baseline / "protocol.json"
    if file_hash(historical) != completed.get("protocol.json"):
        raise ValueError("Original training protocol fails its completion hash")
    historical_protocol = json.loads(historical.read_text(encoding="utf-8"))
    if historical_protocol["baseline"] != LOGISTIC:
        raise ValueError("Logistic settings differ from the original training protocol")
    if historical_protocol["training"] != TRAINING or historical_protocol["seeds"] != SEEDS:
        raise ValueError("Neural training settings differ from the original training protocol")
    historical_hashes = {name.replace("\\", "/"): digest
                         for name, digest in historical_protocol["data_hashes"].items()}
    for name in ("arrays.npz", "tasks.json", "preprocess.json", "chemistry.json"):
        historical_path = historical_protocol["data"].replace("\\", "/").rstrip("/") + "/" + name
        if file_hash(baseline_data / name) != historical_hashes.get(historical_path):
            raise ValueError(f"Baseline data do not match original prediction provenance: {name}")
    inputs.append(historical)
    for method, folder in method_paths(baseline, output).items():
        if not method.startswith("baseline_"):
            continue
        names = ["logits.npy", "raw_probabilities.npy", "probabilities.npy"]
        names += ["coefficients.npz"] if method.endswith("logistic") else ["best.pt", "model.onnx"]
        for name in names:
            path = folder / name
            relative = path.relative_to(baseline).as_posix()
            digest = completed.get(relative)
            if not digest or file_hash(path) != digest:
                raise ValueError(f"Original model or prediction fails its completion hash: {path}")
            inputs.append(path)
    source = Path(__file__).resolve().parents[1]
    code = list((source / "src").rglob("*.py"))
    code.extend(source / "benchmarks" / name for name in (
        "train_consensus.py", "train_multitask.py", "prepare_consensus.py")
                if (source / "benchmarks" / name).exists())
    protocol = {
        "schema": "s2-decision-consensus-run/1", "created_utc": datetime.now(UTC).isoformat(),
        "training": TRAINING, "seeds": SEEDS, "logistic": LOGISTIC, "selection_k": KS,
        "resolved_defaults": {"dropout": .1, "optimizer": "AdamW", "weight_decay": .0001,
                              "torch_threads": 4, "deterministic_algorithms": True,
                              "checkpoint": "validation macro average precision",
                              "calibration_min_per_class": 20},
        "candidate_rule": plan["comparison"]["candidate_rule"],
        "evaluation_order": "All fits complete before evaluation; select on full validation only",
        "external_usage": "None; B10 is not loaded or rescored",
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {name: version(name) for name in (
            "numpy", "scipy", "pandas", "torch", "scikit-learn", "onnx", "onnxruntime")},
        "data_hashes": {str(path.resolve()): file_hash(path) for path in inputs},
        "code_hashes": {str(path.resolve()): file_hash(path) for path in code},
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", protocol)
    return protocol


def validate_prepared(data: Path, baseline_data: Path, plan_path: Path,
                      arrays: dict, excluded: np.ndarray) -> None:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    completion = json.loads((data / "completion.json").read_text(encoding="utf-8"))
    if not {"consensus-mask.npz", "source-lineage.json", "mask-audit.parquet"} <= set(completion["files"]):
        raise ValueError("Prepared completion must seal the consensus mask, audit and source lineage")
    lineage = json.loads((data / "source-lineage.json").read_text(encoding="utf-8"))
    prepare = Path(__file__).with_name("prepare_consensus.py")
    if lineage["preparation_code_sha256"] != file_hash(prepare):
        raise ValueError("Preparation code changed since consensus data were prepared")
    if lineage["completion_sha256"] != file_hash(baseline_data / "completion.json"):
        raise ValueError("Consensus source completion differs from baseline data")
    for name in ("tasks.json", "preprocess.json", "chemistry.json"):
        if file_hash(data / name) != file_hash(baseline_data / name):
            raise ValueError(f"Unchanged source contract differs: {name}")
    train = arrays["splits"] == "train"
    labels = arrays["labels"][train]
    expected = plan["curation"]
    if (labels.shape[1] != expected["expected_tasks"]
            or int(excluded[train].sum()) != expected["expected_training_masked"]
            or int(np.isfinite(labels).sum()) != expected["expected_training_remaining"]):
        raise ValueError("Prepared training counts differ from the frozen plan")
    minimum = expected["minimum_training_support"]
    if any(np.any((labels == value).sum(axis=0) < minimum) for value in (0, 1)):
        raise ValueError("Curated tasks fail frozen minimum training support")


def task_metrics(labels: np.ndarray, logits: np.ndarray, raw: np.ndarray,
                 calibrated: np.ndarray, identities: np.ndarray, ks: list[int]) -> dict:
    observed = np.isfinite(labels)
    y, score, ids = labels[observed], logits[observed], identities[observed]
    raw, calibrated = raw[observed], calibrated[observed]
    if not np.isfinite(score).all() or not np.isfinite(raw).all() or not np.isfinite(calibrated).all():
        raise ValueError("Predictions must be finite")
    if np.any((raw < 0) | (raw > 1)) or np.any((calibrated < 0) | (calibrated > 1)):
        raise ValueError("Probability predictions must lie in [0, 1]")
    n, positive = len(y), int(y.sum())
    both = 0 < positive < n
    order = np.lexsort((ids, -score))
    selection = {}
    for k in ks:
        effective = min(k, n)
        hits = int(y[order[:effective]].sum())
        precision = hits / effective if effective else None
        selection[str(k)] = {"requested_k": k, "effective_k": effective, "hits": hits,
                             "precision": precision, "recall": hits / positive if positive else None,
                             "enrichment_factor": hits * n / (effective * positive) if positive else None}
    return {"observed": n, "positive": positive,
            "average_precision": float(average_precision_score(y, score)) if both else None,
            "roc_auc": float(roc_auc_score(y, score)) if both else None,
            "brier_raw": float(np.mean((raw - y) ** 2)) if n else None,
            "brier_calibrated": float(np.mean((calibrated - y) ** 2)) if n else None,
            "selection": selection}


def metric_value(task: dict, name: str) -> float | None:
    if name.startswith("ef_at_"):
        return task["selection"][name.removeprefix("ef_at_")]["enrichment_factor"]
    return task[name]


def macro(tasks: list[dict]) -> dict:
    result = {}
    for name in ["average_precision", "roc_auc", "brier_raw", "brier_calibrated",
                 *(f"ef_at_{k}" for k in KS)]:
        values = [metric_value(task, name) for task in tasks]
        defined = [value for value in values if value is not None]
        result[name] = {"mean": float(np.mean(defined)) if defined else None,
                        "defined_tasks": len(defined), "total_tasks": len(tasks)}
    return result


def candidate_gate(validation: dict) -> dict:
    """Accept full validation only; seed variability is not biological replication."""
    pairs, task_sets = [], []
    for seed in SEEDS:
        reference = {item["task_id"]: item for item in validation[f"baseline_seed{seed}"]["tasks"]}
        candidate = {item["task_id"]: item for item in validation[f"consensus_seed{seed}"]["tasks"]}
        if reference.keys() != candidate.keys():
            raise ValueError("Paired validation reports must have identical task definitions")
        task_sets.append(set(reference))
        summary: dict = {"seed": seed}
        for metric in ("average_precision", "ef_at_50"):
            common = [key for key in reference if metric_value(reference[key], metric) is not None
                      and metric_value(candidate[key], metric) is not None]
            if not common:
                raise ValueError(f"No paired validation tasks define {metric}")
            summary[metric] = {
                "paired_tasks": len(common), "total_tasks": len(reference),
                "baseline": float(np.mean([metric_value(reference[key], metric) for key in common])),
                "consensus": float(np.mean([metric_value(candidate[key], metric) for key in common])),
            }
        pairs.append(summary)
    if any(task_set != task_sets[0] for task_set in task_sets):
        raise ValueError("Task definitions must be identical across seeds")
    means = {metric: {method: float(np.mean([item[metric][method] for item in pairs]))
                      for method in ("baseline", "consensus")}
             for metric in ("average_precision", "ef_at_50")}
    accepted = (means["ef_at_50"]["consensus"] > means["ef_at_50"]["baseline"]
                and means["average_precision"]["consensus"] >= means["average_precision"]["baseline"])
    return {"selected": "consensus" if accepted else "baseline", "validation_seed_means": means,
            "per_seed": pairs, "test_used_for_selection": False, "production_promotion": False,
            "rule": "Validation mean EF50 strictly improves and AP does not decrease; ties retain baseline"}


def paired_report(reports: dict) -> dict:
    result = {}
    for suffix in [*(f"seed{s}" for s in SEEDS), "logistic"]:
        reference = reports[f"baseline_{suffix}"]["tasks"]
        candidate = reports[f"consensus_{suffix}"]["tasks"]
        comparison = {}
        for name in ["average_precision", "roc_auc", "brier_raw", "brier_calibrated",
                     *(f"ef_at_{k}" for k in KS)]:
            deltas = []
            for c, b in zip(candidate, reference, strict=True):
                current, previous = metric_value(c, name), metric_value(b, name)
                if current is not None and previous is not None:
                    deltas.append(current - previous)
            direction = -1 if name.startswith("brier") else 1
            comparison[name] = {
                "paired_tasks": len(deltas), "total_tasks": len(reference),
                "mean_delta": float(np.mean(deltas)) if deltas else None,
                "median_delta": float(np.median(deltas)) if deltas else None,
                "wins": sum(direction * value > 0 for value in deltas),
                "ties": sum(value == 0 for value in deltas),
                "losses": sum(direction * value < 0 for value in deltas),
            }
        result[suffix] = comparison
    return result


def evaluate(arrays: dict, tasks: list[dict], excluded: np.ndarray,
             baseline: Path, output: Path) -> dict:
    reports: dict = {split: {"full": {}, "consensus_sensitivity": {}}
               for split in ("validation", "calibration", "test")}
    for method, folder in method_paths(baseline, output).items():
        predictions = [np.load(folder / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                       for name in ("logits", "raw_probabilities", "probabilities")]
        if any(values.shape != arrays["labels"].shape for values in predictions):
            raise ValueError(f"Prediction shape differs from original labels: {method}")
        for split, cohorts in reports.items():
            selected = arrays["splits"] == split
            original = arrays["labels"][selected]
            for cohort, methods in cohorts.items():
                labels = (original if cohort == "full" else
                          np.where(excluded[selected], np.nan, original))
                values = [item[selected] for item in predictions]
                results = [{"task_index": index, "task_id": task["task_id"],
                            **task_metrics(labels[:, index], values[0][:, index], values[1][:, index],
                                           values[2][:, index],
                                           arrays["identities"][selected], KS)}
                           for index, task in enumerate(tasks)]
                methods[method] = {"tasks": results, "macro": macro(results)}
    return {"scope": "Internal reused splits; test is exploratory, no external confirmation",
            "ranking": "Raw logits descending, identity ascending for ties",
            "selection": candidate_gate(reports["validation"]["full"]), "splits": reports,
            "paired": {split: {cohort: paired_report(methods) for cohort, methods in cohorts.items()}
                       for split, cohorts in reports.items()},
            "uncertainty": "Seed variation describes optimization, not biological confidence intervals"}


def run(data: Path, baseline_data: Path, baseline: Path, output: Path, plan_path: Path) -> dict:
    from s2s_decision.multitask import train_multitask

    arrays, tasks = load_dataset(data)
    original, original_tasks = load_dataset(baseline_data)
    if tasks != original_tasks:
        raise ValueError("Task definitions must equal B09 exactly")
    with np.load(data / "consensus-mask.npz", allow_pickle=False) as saved:
        excluded = saved["mask"]
    validate_pair(original, arrays, excluded)
    validate_prepared(data, baseline_data, plan_path, arrays, excluded)
    protocol = freeze(data, baseline_data, baseline, output, plan_path)
    summaries = {}
    for seed in SEEDS:
        verify_inputs(protocol)
        method = f"s2_decision_seed{seed}"
        print(f"TRAIN {method}", flush=True)
        summaries[method] = train_multitask(arrays["x"], arrays["labels"], arrays["splits"],
                                           output / method, seed=seed, **TRAINING)
    verify_inputs(protocol)
    summaries["logistic"] = fit_baseline(arrays["x"], arrays["labels"], arrays["splits"],
                                         output / "logistic", LOGISTIC, tasks)
    fit_files = [path for folder in method_paths(baseline, output).values()
                 if folder.is_relative_to(output) for path in folder.rglob("*") if path.is_file()]
    fit_hashes = {str(path.resolve()): file_hash(path) for path in fit_files}
    write_json(output / "fits-complete.json", {"methods": summaries, "files": fit_hashes})
    verify_inputs(protocol)
    metrics = evaluate(original, tasks, excluded, baseline, output)
    verify_inputs(protocol)
    verify_inputs({"data_hashes": fit_hashes, "code_hashes": {}})
    write_json(output / "metrics.json", metrics)
    products = sorted(path for path in output.rglob("*") if path.is_file())
    write_json(output / "completion.json", {
        "completed_utc": datetime.now(UTC).isoformat(),
        "files": {str(path.relative_to(output)): file_hash(path) for path in products}})
    print(f"COMPLETE {output.resolve()}", flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("data", "baseline-data", "baseline-run", "output", "plan"):
        parser.add_argument(f"--{argument}", required=True, type=Path)
    args = parser.parse_args()
    run(args.data.resolve(), args.baseline_data.resolve(), args.baseline_run.resolve(),
        args.output.resolve(), args.plan.resolve())


if __name__ == "__main__":
    main()
