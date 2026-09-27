"""Frozen, resumable benchmark using the existing chemistry and training pipeline."""

import argparse
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from .artifacts import file_hash, read_bundle, write_bundle, write_json
from .schema import CONTEXT_NAMES, FeatureSet

TASKS = (
    ("P24941_WT", "IC50"),
    ("O60885_WT", "IC50"),
    ("P08684_WT", "IC50"),
    ("Q12809_WT", "IC50"),
    ("P37231_WT", "EC50"),
    ("O14842_WT", "EC50"),
    ("O43613_WT", "Ki"),
    ("P00742_WT", "Ki"),
    ("P30542_WT", "Ki"),
    ("O96013_WT", "Ki"),
)


def make_plan(source):
    return {
        "format_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "source": str(Path(source).resolve()),
        "source_sha256": file_hash(source),
        "tasks": [
            {"id": f"{target}_{endpoint}", "target_id": target, "endpoint": endpoint}
            for target, endpoint in TASKS
        ],
        "selection_basis": "Preselected by endpoint, support and prevalence from source audit; no model outcomes. Not a random sample of all Papyrus tasks.",
        "excluded_legacy_target": "P22303_WT: previously observed pilot test",
        "seeds": [11, 23, 42, 71, 101],
        "split_methods": ["scaffold", "temporal"],
        "fingerprint_bits": 2048,
        "threshold": 6.0,
        "n": 50,
        "max_per_scaffold": 3,
        "training": {
            "epochs": 80,
            "batch_size": 256,
            "patience": 15,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
        },
        "threads": 4,
        "curation": "Exclude invalid, unknown labels, and ALL records of repeated standardized identities; never average or keep first.",
        "minimum_support": {
            "train": [200, 20],
            "validation": [30, 5],
            "calibration": [20, 5],
            "test": [50, 10],
        },
        "temporal": "Fixed chronological partitions for all model seeds; strict earlier train references. Year groups, not molecule quantiles; retrospective only.",
        "missing_or_infeasible": "Record exclusions; no seed retries, replacement tasks or outcome-based changes.",
        "aggregation": "Separate split protocols; mean across five seeds within task then equal-weight task macro means. Complete five-seed tasks primary; paired task bootstrap descriptive only.",
    }


def curate_features(features):
    frame = features.records
    invalid = ~frame.valid.astype(bool)
    unlabeled = ~invalid & frame.y_active.isna()
    usable = frame.loc[~invalid & ~unlabeled]
    duplicate = usable.identity.duplicated(keep=False)
    excluded = {
        "invalid_records": frame.loc[invalid, "record_id"].tolist(),
        "unlabeled_records": frame.loc[unlabeled, "record_id"].tolist(),
        "duplicate_identity_records": usable.loc[duplicate, "record_id"].tolist(),
    }
    accepted = usable.loc[~duplicate].copy().reset_index(drop=True)
    return FeatureSet(accepted, {**features.manifest, "curated_count": len(accepted)}), excluded


def feasibility(frame):
    limits = {"train": (200, 20), "validation": (30, 5), "calibration": (20, 5), "test": (50, 10)}
    counts, failures = {}, []
    for name, (minimum_rows, minimum_class) in limits.items():
        group = frame.loc[frame.split.eq(name)]
        negative, positive = int(group.y_active.eq(0).sum()), int(group.y_active.eq(1).sum())
        counts[name] = {"rows": len(group), "active": positive, "inactive": negative}
        if len(group) < minimum_rows or min(negative, positive) < minimum_class:
            failures.append(f"{name}: {len(group)} rows, {positive} active, {negative} inactive")
    return {"eligible": not failures, "reason": "; ".join(failures), "counts": counts}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def freeze_runtime(root):
    """Refuse mixed code/dependency versions when continuing a benchmark."""
    package = Path(__file__).parent
    files = sorted(package.glob("*.py"))
    current = {
        "sources": {path.name: file_hash(path) for path in files},
        "versions": {
            name: version(name)
            for name in (
                "numpy",
                "pandas",
                "rdkit",
                "torch",
                "scikit-learn",
                "onnx",
                "onnxruntime",
                "smiles2select",
                "scipy",
                "threadpoolctl",
            )
        },
    }
    path = Path(root) / "execution-source.json"
    if path.exists():
        if _json(path) != current:
            raise ValueError(
                "code or dependencies changed after benchmark freeze; use a new experiment"
            )
        return
    snapshot = Path(root) / "source-snapshot"
    snapshot.mkdir(exist_ok=True)
    for source in files:
        (snapshot / source.name).write_bytes(source.read_bytes())
    write_json(path, current)


def _extract_sources(root, plan):
    marker = root / "source-extraction.json"
    if marker.exists():
        saved = _json(marker)
        for relative, digest in saved["files"].items():
            if file_hash(root / relative) != digest:
                raise ValueError("extracted source checksum mismatch")
        return
    folders = root / "sources"
    folders.mkdir(exist_ok=True)
    targets = {task["target_id"] for task in plan["tasks"]}
    outputs = {target: folders / f"{target}.tsv" for target in targets}
    if any(path.exists() for path in outputs.values()):
        raise ValueError("incomplete source extraction; preserve it and choose a new output root")
    ordinal = 0
    # One source pass, rather than normalizing all 707k rows once per task and seed.
    with pd.read_csv(
        plan["source"],
        sep="\t",
        compression="infer",
        dtype=str,
        keep_default_na=False,
        chunksize=10000,
    ) as chunks:
        for chunk in chunks:
            chunk = chunk.assign(papyrus_source_row=range(ordinal + 1, ordinal + len(chunk) + 1))
            ordinal += len(chunk)
            for target, selected in chunk.loc[chunk.target_id.isin(targets)].groupby("target_id"):
                path = outputs[target]
                selected.to_csv(path, sep="\t", index=False, mode="a", header=not path.exists())
    write_json(
        marker,
        {
            "source_sha256": plan["source_sha256"],
            "source_rows": ordinal,
            "files": {str(path.relative_to(root)): file_hash(path) for path in outputs.values()},
        },
    )


def _task_features(root, task, plan):
    from .data import load_measurements
    from .features import featurize

    destination = root / "features" / task["id"]
    source = root / "sources" / f"{task['target_id']}.tsv"
    if destination.exists():
        cached = read_bundle(destination)
        expected = {
            "target_id": task["target_id"],
            "endpoint": task["endpoint"],
            "threshold": plan["threshold"],
            "fingerprint_bits": plan["fingerprint_bits"],
            "full_source_sha256": plan["source_sha256"],
            "source_sha256": file_hash(source),
        }
        if any(cached.manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("cached task features differ from frozen source or task contract")
        return cached
    measured = load_measurements(source, task["target_id"], task["endpoint"], plan["threshold"])
    features = featurize(measured, plan["fingerprint_bits"])
    accepted, exclusions = curate_features(features)
    provenance = {
        **accepted.manifest,
        **measured.attrs,
        "full_source_sha256": plan["source_sha256"],
        "source_measurements": len(measured),
        "curation": plan["curation"],
        "excluded_records": exclusions,
    }
    write_bundle(FeatureSet(accepted.records, provenance), destination)
    return read_bundle(destination)


def _prepare_partition(root, task, features, method, seed, plan):
    from .context import training_context
    from .splits import assign_splits

    # Temporal partitions and references do not depend on model initialization seed.
    split_seed = seed if method == "scaffold" else plan["seeds"][0]
    destination = root / "datasets" / task["id"] / method / str(split_seed)
    status_path = destination.parent / f"{split_seed}-feasibility.json"
    if status_path.exists():
        saved = _json(status_path)
        if saved.get("plan_sha256") != file_hash(root / "plan.json"):
            raise ValueError("cached partition belongs to another experiment plan")
        if saved["eligible"]:
            cached = read_bundle(destination)
            if cached.manifest.get("experiment_plan_sha256") != saved["plan_sha256"]:
                raise ValueError("cached dataset belongs to another experiment plan")
        return saved
    try:
        frame = assign_splits(features.records, method=method, seed=split_seed)
    except ValueError as exc:
        status = {"eligible": False, "reason": str(exc), "counts": {}}
    else:
        status = feasibility(frame)
        if status["eligible"]:
            frame.attrs["threshold"] = plan["threshold"]
            context = training_context(frame, seed=split_seed, temporal=method == "temporal")
            complete = pd.concat(
                [frame.drop(columns=list(CONTEXT_NAMES), errors="ignore"), context], axis=1
            )
            manifest = {
                **features.manifest,
                "split_method": method,
                "split_seed": split_seed,
                "context_method": "exact_tanimoto_strict_past_v1"
                if method == "temporal"
                else "exact_tanimoto_scaffold_oof_v1",
                "split_counts": status["counts"],
                "experiment_plan_sha256": file_hash(root / "plan.json"),
            }
            write_bundle(FeatureSet(complete, manifest), destination)
    status = {
        **status,
        "dataset": str(destination.resolve()),
        "split_seed": split_seed,
        "plan_sha256": file_hash(root / "plan.json"),
    }
    write_json(status_path, status)
    return status


def prepare_experiment(source, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "plan.json"
    if plan_path.exists():
        plan = _json(plan_path)
        if (
            str(Path(source).resolve()) != plan["source"]
            or file_hash(source) != plan["source_sha256"]
        ):
            raise ValueError("source differs from frozen experiment plan")
    else:
        plan = make_plan(source)
        write_json(plan_path, plan)
    if (root / "prepared.json").exists():
        prepared = _json(root / "prepared.json")
        _validate_prepared(root, plan, prepared)
        return prepared
    _extract_sources(root, plan)
    runs = []
    for task in plan["tasks"]:
        print(f"PREPARE {task['id']}", flush=True)
        features = _task_features(root, task, plan)
        for method in plan["split_methods"]:
            for seed in plan["seeds"]:
                status = _prepare_partition(root, task, features, method, seed, plan)
                run = {"task_id": task["id"], "split_method": method, "seed": seed, **status}
                runs.append(run)
                if not status["eligible"]:
                    path = root / "runs" / task["id"] / method / str(seed) / "result.json"
                    write_json(path, {**run, "status": "excluded"})
                print(
                    f"  {method} seed={seed} eligible={status['eligible']} {status['reason']}",
                    flush=True,
                )
    prepared = {
        "plan_sha256": file_hash(plan_path),
        "runs": runs,
        "eligible_runs": sum(run["eligible"] for run in runs),
        "total_runs": len(runs),
    }
    write_json(root / "prepared.json", prepared)
    return prepared


def _validate_prepared(root, plan, prepared):
    digest = file_hash(root / "plan.json")
    if prepared["plan_sha256"] != digest:
        raise ValueError("experiment plan changed after preparation")
    tasks = {task["id"]: task for task in plan["tasks"]}
    expected_runs = {
        (task, method, seed)
        for task in tasks
        for method in plan["split_methods"]
        for seed in plan["seeds"]
    }
    actual_runs = [(run["task_id"], run["split_method"], run["seed"]) for run in prepared["runs"]]
    if len(actual_runs) != len(expected_runs) or set(actual_runs) != expected_runs:
        raise ValueError("prepared runs differ from predeclared tasks and seeds")
    for run in prepared["runs"]:
        if run["eligible"]:
            saved = read_bundle(run["dataset"])
            expected = {
                "experiment_plan_sha256": digest,
                "split_method": run["split_method"],
                "split_seed": run.get("split_seed", run["seed"]),
                "target_id": tasks[run["task_id"]]["target_id"],
                "endpoint": tasks[run["task_id"]]["endpoint"],
            }
            if any(saved.manifest.get(key) != value for key, value in expected.items()):
                raise ValueError("prepared dataset differs from frozen partition contract")


def run_experiment(root):
    from threadpoolctl import threadpool_limits

    from .benchmark_evaluation import evaluate_benchmark
    from .training import TrainingConfig
    from .workflows import train_dataset

    root = Path(root)
    plan, prepared = _json(root / "plan.json"), _json(root / "prepared.json")
    _validate_prepared(root, plan, prepared)
    freeze_runtime(root)
    for run in prepared["runs"]:
        if not run["eligible"]:
            continue
        folder = root / "runs" / run["task_id"] / run["split_method"] / str(run["seed"])
        result_path = folder / "result.json"
        if result_path.exists():
            result = _json(result_path)
            if result["status"] == "completed":
                if file_hash(folder / "evaluation.json") != result["evaluation_sha256"]:
                    raise ValueError("completed evaluation checksum mismatch")
                continue
            if result["status"] == "failed":
                continue  # Failures remain visible, never silently retried after observing outcomes.
        started = time.perf_counter()
        folder.mkdir(parents=True, exist_ok=True)
        print(f"RUN {run['task_id']} {run['split_method']} seed={run['seed']}", flush=True)
        try:
            source, model = Path(run["dataset"]), folder / "model"
            config = TrainingConfig(
                **plan["training"], seed=run["seed"], resume=(model / "checkpoint.pt").exists()
            )
            with threadpool_limits(limits=plan["threads"]):
                trained = train_dataset(source, model, config, threads=plan["threads"])
                dataset = read_bundle(source)
                evaluation = evaluate_benchmark(
                    dataset.records,
                    model,
                    fingerprint_bits=plan["fingerprint_bits"],
                    seed=run["seed"],
                    n=plan["n"],
                    max_per_scaffold=plan["max_per_scaffold"],
                )
            write_json(folder / "evaluation.json", evaluation)
            result = {
                **run,
                "status": "completed",
                "training": trained,
                "config": asdict(config),
                "dataset_sha256": dataset.manifest["records_sha256"],
                "evaluation_sha256": file_hash(folder / "evaluation.json"),
            }
        except (ValueError, RuntimeError, OSError) as exc:
            result = {**run, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
        result = {**result, "elapsed_seconds": time.perf_counter() - started}
        write_json(result_path, result)
        print(f"  {result['status']} seconds={result['elapsed_seconds']:.1f}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "report"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        if args.source is None:
            parser.error("prepare requires --source")
        prepare_experiment(args.source, args.root)
    elif args.command == "run":
        run_experiment(args.root)
    else:
        from .experiment_report import write_report

        write_report(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
