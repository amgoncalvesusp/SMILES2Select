"""Paired data-volume × Tiny-width experiment; prepare -> train -> evaluate.

All scientific settings come from the frozen study plan. No test-dependent
model choice, source replacement, or silent overwrite is permitted.
"""

import argparse
import json
import math
import os
import shutil
import traceback
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from s2s_decision.artifacts import file_hash, read_bundle, write_bundle, write_json
from s2s_decision.context import training_context
from s2s_decision.data import load_measurements
from s2s_decision.experiment import _extract_sources, curate_features, feasibility
from s2s_decision.features import chemistry_manifest, featurize, validate_chemistry_compatibility
from s2s_decision.schema import CONTEXT_NAMES, FeatureSet
from s2s_decision.splits import assign_splits


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def seal(path, payload):
    if Path(path).exists():
        require(load(path) == payload, f"Frozen artifact changed: {path}")
    else:
        write_json(path, payload)


def check_lock(path):
    payload = load(path)
    for source, digest in payload["files"].items():
        require(file_hash(source) == digest, f"Frozen file changed: {source}")
    return payload


def hashes(paths):
    return {str(Path(path).resolve()): file_hash(path) for path in sorted(paths)}


def runtime_versions():
    return {
        name: version(name)
        for name in (
            "numpy",
            "pandas",
            "rdkit",
            "torch",
            "scikit-learn",
            "onnx",
            "onnxruntime",
            "skl2onnx",
            "scipy",
            "threadpoolctl",
        )
    }


def write_library(path, held):
    expected = (
        held[["molecule_id", "original_smiles"]]
        .rename(columns={"molecule_id": "ID", "original_smiles": "SMILES"})
        .to_csv(index=False, lineterminator="\n")
        .encode("utf-8")
    )
    if path.exists():
        require(path.read_bytes() == expected, "Heldout library changed")
    else:
        path.write_bytes(expected)


def assert_unexposed(frame, identities):
    require(not set(frame.identity) & identities, "Overlap with historical identities")


def training_subset(frame, fraction, seed):
    require(0 < fraction <= 1, "Training fraction must be in (0, 1]")
    ordered = frame.sort_values("record_id").reset_index(drop=True)
    train = ordered.loc[ordered.split.eq("train")]
    require(set(train.y_active) == {0, 1}, "Training requires both classes")
    rng = np.random.default_rng(seed)
    chosen = []
    for label in (0, 1):
        ids = train.loc[train.y_active.eq(label), "record_id"].to_numpy()
        chosen.extend(rng.permutation(ids)[: max(1, math.floor(len(ids) * fraction))])
    return ordered.loc[~ordered.split.eq("train") | ordered.record_id.isin(chosen)].copy()


def settings(root):
    plan = load(root / "plan.json")
    files = [root / "plan.json", root / "excluded-previous-identities.json", Path(plan["source"])]
    require(file_hash(files[1]) == plan["excluded_identities_sha256"], "Exclusions changed")
    require(file_hash(files[2]) == plan["source_sha256"], "Source changed")
    seal(root / "plan-lock.json", {"files": hashes(files)})
    check_lock(root / "plan-lock.json")
    require(plan["train_fractions"] == [0.5, 1.0], "This protocol requires nested 50%/100% arms")
    require(plan["widths"] == [1, 2], "This protocol requires widths 1/2")
    return plan


def prepare(root, work):
    del work
    plan = settings(root)
    if (root / "prepared.json").exists():
        check_lock(root / "preparation-lock.json")
        return
    prior = set(load(root / "excluded-previous-identities.json")["identities"])
    _extract_sources(root, plan)
    statuses = []
    for task in plan["tasks"]:
        status_path = root / "preparation" / f"{task['id']}.json"
        if status_path.exists():
            status = load(status_path)
            for arm in status["arms"]:
                if arm["eligible"]:
                    require(
                        read_bundle(arm["dataset"]).manifest["records_sha256"]
                        == arm["records_sha256"],
                        "Dataset changed",
                    )
            statuses.append(status)
            continue
        print("PREPARE", task["id"], flush=True)
        measured = load_measurements(
            root / "sources" / f"{task['target_id']}.tsv",
            task["target_id"],
            task["endpoint"],
            plan["threshold"],
        )
        features, exclusions = curate_features(featurize(measured, plan["fingerprint_bits"]))
        repeated = features.records.identity.isin(prior)
        exclusions["historical_records"] = features.records.loc[
            repeated, ["record_id", "identity"]
        ].to_dict("records")
        accepted = features.records.loc[~repeated].copy().reset_index(drop=True)
        assert_unexposed(accepted, prior)
        split = assign_splits(accepted, method="scaffold", seed=plan["split_seed"])
        arms = []
        for fraction in plan["train_fractions"]:
            name = f"train{round(100 * fraction)}"
            selected = training_subset(split, fraction, plan["subset_seed"])
            support = feasibility(selected)
            destination = root / "datasets" / task["id"] / name
            if support["eligible"]:
                selected.attrs["threshold"] = plan["threshold"]
                context = training_context(selected, seed=plan["split_seed"])
                complete = pd.concat(
                    [selected.drop(columns=list(CONTEXT_NAMES), errors="ignore"), context], axis=1
                )
                provenance = {
                    **features.manifest,
                    **measured.attrs,
                    "target_id": task["target_id"],
                    "endpoint": task["endpoint"],
                    "threshold": plan["threshold"],
                    "full_source_sha256": plan["source_sha256"],
                    "context_method": "exact_tanimoto_scaffold_oof_v1",
                    "split_method": "scaffold",
                    "split_seed": plan["split_seed"],
                    "subset_seed": plan["subset_seed"],
                    "train_fraction": fraction,
                    "experiment_plan_sha256": file_hash(root / "plan.json"),
                }
                write_bundle(FeatureSet(complete, provenance), destination)
            arms.append(
                {
                    **support,
                    "name": name,
                    "fraction": fraction,
                    "dataset": str(destination),
                    "records_sha256": read_bundle(destination).manifest["records_sha256"]
                    if support["eligible"]
                    else None,
                    "record_ids_by_split": {
                        part: selected.loc[selected.split.eq(part), "record_id"].tolist()
                        for part in ("train", "validation", "calibration", "test")
                    },
                }
            )
        for part in ("validation", "calibration", "test"):
            require(
                arms[0]["record_ids_by_split"][part] == arms[1]["record_ids_by_split"][part],
                "Holdout mismatch",
            )
        require(
            set(arms[0]["record_ids_by_split"]["train"])
            <= set(arms[1]["record_ids_by_split"]["train"]),
            "Training arms not nested",
        )
        status = {
            "task_id": task["id"],
            "source_records": len(measured),
            "accepted_records": len(accepted),
            "exclusions": exclusions,
            "eligible": all(arm["eligible"] for arm in arms),
            "arms": arms,
        }
        write_json(status_path, status)
        statuses.append(status)
    write_json(
        root / "prepared.json", {"plan_sha256": file_hash(root / "plan.json"), "tasks": statuses}
    )
    paths = (
        [root / "plan-lock.json", root / "prepared.json"]
        + list((root / "preparation").glob("*.json"))
        + list((root / "datasets").rglob("*.json"))
        + list((root / "datasets").rglob("*.jsonl"))
    )
    seal(root / "preparation-lock.json", {"files": hashes(paths)})


def model_options(plan):
    for seed in plan["seeds"]:
        for width in plan["widths"]:
            yield {
                "name": f"tiny_w{width}_seed{seed}",
                "family": f"tiny_w{width}",
                "estimator": "tiny",
                "width": width,
                "seed": seed,
            }
        for estimator in ("logistic", "gradient_boosting"):
            yield {
                "name": f"{estimator}_seed{seed}",
                "family": estimator,
                "estimator": estimator,
                "seed": seed,
            }


def validate_model(path, dataset, option, plan):
    from s2s_decision.training import TrainingConfig

    manifest = load(path / "manifest.json")
    validate_chemistry_compatibility(dataset.manifest, chemistry_manifest(plan["fingerprint_bits"]))
    if option["estimator"] == "tiny":
        expected = asdict(
            TrainingConfig(**plan["tiny"], seed=option["seed"], width_multiplier=option["width"])
        )
        expected = {
            key: value for key, value in expected.items() if key not in ("epochs", "resume")
        }
        require(
            manifest.get("config") == expected and "estimator" not in manifest,
            "Tiny configuration mismatch",
        )
        require(0 < len(manifest["history"]) <= plan["tiny"]["epochs"], "Tiny epoch count mismatch")
    else:
        require(
            manifest.get("estimator") == option["estimator"]
            and manifest.get("input_layout") == "scalar_fingerprint"
            and manifest["config"]["seed"] == option["seed"],
            "Baseline configuration mismatch",
        )
    require(
        manifest.get("runtime_ready") and not manifest.get("heldout_test_evaluated"),
        "Model not ready or test already used",
    )
    require(file_hash(path / "model.onnx") == manifest["onnx_sha256"], "ONNX changed")
    require(
        manifest["provenance"] == dataset.manifest,
        "Model trained on another dataset",
    )
    refs = read_bundle(path / "references")
    require(
        set(refs.records.identity)
        == set(dataset.records.loc[dataset.records.split.eq("train"), "identity"]),
        "Reference scope mismatch",
    )
    require(
        refs.manifest["records_sha256"] == manifest["reference_records_sha256"],
        "Reference hash mismatch",
    )
    return manifest


def fit_one(root, work, task, arm, option, plan):
    from s2s_decision.training import TrainingConfig
    from s2s_decision.workflows import train_baseline_dataset, train_dataset

    destination = root / "models" / task["task_id"] / arm["name"] / option["name"]
    local = work / "models" / task["task_id"] / arm["name"] / option["name"]
    run_path = root / "training" / task["task_id"] / arm["name"] / f"{option['name']}.json"
    dataset = read_bundle(arm["dataset"])
    if run_path.exists():
        saved = check_lock(run_path)
        validate_model(destination, dataset, option, plan)
        return saved["model"]
    require(not destination.exists(), f"Unsealed destination exists: {destination}")
    timing = local.parent / f"{local.name}-timing.json"
    if not local.exists():
        print("TRAIN", task["task_id"], arm["name"], option["name"], flush=True)
        started = perf_counter()
        if option["estimator"] == "tiny":
            train_dataset(
                arm["dataset"],
                local,
                TrainingConfig(
                    **plan["tiny"], seed=option["seed"], width_multiplier=option["width"]
                ),
                threads=plan["threads"],
            )
        else:
            train_baseline_dataset(
                arm["dataset"],
                local,
                estimator=option["estimator"],
                seed=option["seed"],
                threads=plan["threads"],
                input_layout="scalar_fingerprint",
            )
        write_json(
            timing,
            {
                "training_export_seconds": perf_counter() - started,
                "plan_sha256": file_hash(root / "plan.json"),
                "execution_lock_sha256": file_hash(root / "execution-lock.json"),
            },
        )
    require(timing.exists(), f"Incomplete local fit requires explicit recovery: {local}")
    receipt = load(timing)
    require(
        receipt.get("plan_sha256") == file_hash(root / "plan.json")
        and receipt.get("execution_lock_sha256") == file_hash(root / "execution-lock.json"),
        "Local fit plan/execution receipt mismatch",
    )
    manifest = validate_model(local, dataset, option, plan)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(local, destination)
    validation_ap = manifest.get("validation_pr_auc")
    if option["estimator"] == "tiny":
        validation_ap = max(row["validation_pr_auc"] for row in manifest["history"])
    result = {
        **option,
        "model_path": str(destination),
        "validation_pr_auc": validation_ap,
        "parameter_count": manifest.get("parameter_count"),
        "epochs": len(manifest.get("history", [])) or None,
        "optimizer_updates": math.ceil(
            dataset.records.split.eq("train").sum() / plan["tiny"]["batch_size"]
        )
        * len(manifest["history"])
        if option["estimator"] == "tiny"
        else None,
        "onnx_bytes": (destination / "model.onnx").stat().st_size,
        "package_bytes": sum(
            path.stat().st_size for path in destination.rglob("*") if path.is_file()
        ),
        "onnx_parity_max_abs": manifest["onnx_parity_max_abs"],
        **load(timing),
    }
    seal(
        run_path,
        {
            "model": result,
            "files": hashes(path for path in destination.rglob("*") if path.is_file()),
        },
    )
    return result


def train(root, work):
    from s2s_decision.training import TrainingConfig

    plan = settings(root)
    check_lock(root / "preparation-lock.json")
    source_root = Path(__file__).resolve().parents[1] / "src"
    source_files = [Path(__file__)] + list(source_root.rglob("*.py"))
    for source in source_files:
        archived = root / "source-snapshot" / source.relative_to(source_root.parent)
        if archived.exists():
            require(file_hash(source) == file_hash(archived), "Source snapshot changed")
        else:
            archived.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, archived)
    resolved = {
        "tiny_configs": [
            asdict(
                TrainingConfig(
                    **plan["tiny"], seed=option["seed"], width_multiplier=option["width"]
                )
            )
            for option in model_options(plan)
            if option["estimator"] == "tiny"
        ],
        "context_folds": 5,
        "dropout": 0.1,
        "validation_choice_pool": "Unfiltered validation; primary test basket uses chemical filtering",
        "versions": runtime_versions(),
    }
    seal(
        root / "execution-lock.json",
        {
            "files": hashes(source_files + list((root / "source-snapshot").rglob("*.py"))),
            "resolved_settings": resolved,
        },
    )
    check_lock(root / "execution-lock.json")
    tasks = []
    for task in load(root / "prepared.json")["tasks"]:
        if not task["eligible"]:
            continue
        arms = []
        for arm in task["arms"]:
            options = [
                fit_one(root, work, task, arm, option, plan) for option in model_options(plan)
            ]
            means = [
                {
                    "arm": arm["name"],
                    "family": family,
                    "mean_validation_ap": float(
                        np.mean(
                            [row["validation_pr_auc"] for row in options if row["family"] == family]
                        )
                    ),
                }
                for family in dict.fromkeys(row["family"] for row in options)
            ]
            arms.append(
                {
                    "name": arm["name"],
                    "dataset": arm["dataset"],
                    "models": options,
                    "validation_condition_means": means,
                }
            )
        conditions = [condition for arm in arms for condition in arm["validation_condition_means"]]
        choice = max(conditions, key=lambda row: row["mean_validation_ap"])
        simple = max(
            [row for row in conditions if not row["family"].startswith("tiny")],
            key=lambda row: row["mean_validation_ap"],
        )
        tasks.append(
            {
                "task_id": task["task_id"],
                "arms": arms,
                "validation_choice": choice,
                "validation_simple_choice": simple,
            }
        )
    paths = [
        root / "plan-lock.json",
        root / "preparation-lock.json",
        root / "execution-lock.json",
    ] + list((root / "training").rglob("*.json"))
    seal(root / "validation-choice-freeze.json", {"files": hashes(paths), "tasks": tasks})
    print("ALL VALIDATION CHOICES FROZEN", flush=True)


def filtered_pool(root, task_id, held, plan):
    from s2s_decision.workflows import prepare_candidates
    from smiles2select.io.importer import ColumnMapping, SourceFile
    from smiles2select.pipeline.config import RunConfig
    from smiles2select.pipeline.runner import run

    destination = root / "evaluation" / task_id
    destination.mkdir(parents=True, exist_ok=True)
    library = destination / "heldout-library.csv"
    write_library(library, held)
    database = destination / "smiles2select-run.sqlite"
    if not database.exists():
        run(
            RunConfig(
                sources=(SourceFile(library, ColumnMapping("SMILES", "ID")),),
                profile_ids=tuple(plan["chemical_filter"]["profile_ids"]),
                compute_sa=plan["chemical_filter"]["compute_sa"],
                compute_np=plan["chemical_filter"]["compute_np"],
                n_jobs=1,
                database_path=database,
            )
        )
    candidates_path = destination / "candidates"
    if not candidates_path.exists():
        prepare_candidates(database, candidates_path)
    candidates = read_bundle(candidates_path).records
    require(set(candidates.identity) == set(held.identity), "Candidate identity mismatch")
    require(not candidates.identity.duplicated().any(), "Duplicate candidate identity")
    require(not {"y_active", "pactivity"} & set(candidates), "Candidate labels leaked")
    mapping = candidates.set_index("identity")
    require(
        mapping.loc[held.identity, "molecule_id"].tolist() == held.molecule_id.tolist(),
        "Candidate IDs changed",
    )
    seal(
        destination / "pool-lock.json",
        {
            "files": hashes(
                [
                    library,
                    database,
                    candidates_path / "manifest.json",
                    candidates_path / "records.jsonl",
                ]
            ),
            "eligible_identities": sorted(candidates.loc[candidates.eligible, "identity"]),
        },
    )
    return held.assign(eligible=held.identity.map(mapping.eligible).astype(bool))


def score_report(frame, scores, plan, destination, probability=True):
    from s2s_decision.metrics import evaluate_predictions
    from s2s_decision.selection import select_candidates

    require(np.isfinite(scores).all() and len(scores) == len(frame), "Incomplete prediction pool")
    if destination.exists():
        check_lock(destination / "score-lock.json")
        previous = pd.read_csv(destination / "predictions.csv")
        require(
            previous.record_id.tolist() == frame.record_id.tolist()
            and np.allclose(previous.priority_score, scores, rtol=0, atol=1e-14),
            "Repeated scoring changed",
        )
        return load(destination / "metrics.json")
    scored = frame.assign(priority_score=np.asarray(scores, dtype=float))
    selected = select_candidates(
        FeatureSet(scored, {}), n=plan["n"], max_per_scaffold=plan["max_per_scaffold"]
    )
    final = selected.records.loc[selected.records.is_final]
    eligible = scored.loc[scored.eligible]
    full_metrics = evaluate_predictions(scored.y_active, scored.priority_score, n=plan["n"])
    filtered_metrics = evaluate_predictions(eligible.y_active, eligible.priority_score, n=plan["n"])
    if not probability:
        for metrics in (full_metrics, filtered_metrics):
            metrics.update(brier=None, ece=None, score_kind="heuristic_not_probability")
    destination.mkdir(parents=True, exist_ok=False)
    columns = [
        "record_id",
        "molecule_id",
        "identity",
        "murcko_scaffold",
        "y_active",
        "pactivity",
        "eligible",
        "priority_score",
        "similarity_active_max",
        "novelty",
        "is_final",
        "selection_reason",
    ]
    selected.records[columns].to_csv(destination / "predictions.csv", index=False)
    final[columns].to_csv(destination / "selected.csv", index=False)
    positives = int(eligible.y_active.sum())
    precision = float(final.y_active.mean()) if len(final) else None
    prevalence = float(eligible.y_active.mean()) if len(eligible) else None
    result = {
        "before_filter": full_metrics,
        "after_filter": filtered_metrics,
        "basket_count": len(final),
        "basket_active": int(final.y_active.sum()),
        "basket_precision": precision,
        "basket_recall": float(final.y_active.sum() / positives) if positives else None,
        "basket_enrichment": precision / prevalence
        if precision is not None and prevalence
        else None,
        "scaffolds_covered": int(final.murcko_scaffold.nunique()),
        "final_record_ids": final.record_id.tolist(),
        "eligible_record_ids": eligible.record_id.tolist(),
        "selection": selected.manifest,
    }
    write_json(destination / "metrics.json", result)
    seal(
        destination / "score-lock.json",
        {
            "files": hashes(
                [
                    destination / "predictions.csv",
                    destination / "selected.csv",
                    destination / "metrics.json",
                ]
            )
        },
    )
    return result


def evaluate(root, work):
    del work
    from s2s_decision.inference import predict

    plan = settings(root)
    check_lock(root / "preparation-lock.json")
    execution = check_lock(root / "execution-lock.json")
    require(
        execution["resolved_settings"]["versions"] == runtime_versions(),
        "Dependency versions changed",
    )
    frozen = check_lock(root / "validation-choice-freeze.json")
    for path in (root / "training").rglob("*.json"):
        check_lock(path)
    prior = set(load(root / "excluded-previous-identities.json")["identities"])
    reports = []
    for task in frozen["tasks"]:
        result_path = root / "evaluation" / task["task_id"] / "result.json"
        if result_path.exists():
            check_lock(result_path.parent / "result-lock.json")
            reports.append(load(result_path))
            continue
        rows = []
        expected_ids = None
        for arm in task["arms"]:
            dataset = read_bundle(arm["dataset"])
            held = (
                dataset.records.loc[dataset.records.split.eq("test")]
                .sort_values("record_id")
                .copy()
            )
            assert_unexposed(held, prior)
            held = filtered_pool(root, task["task_id"], held, plan)
            ids = held.loc[held.eligible, "record_id"].tolist()
            require(expected_ids is None or expected_ids == ids, "Unequal eligible pools")
            expected_ids = ids
            for option in arm["models"]:
                model = Path(option["model_path"])
                validate_model(model, dataset, option, plan)
                print("EVALUATE", task["task_id"], arm["name"], option["name"], flush=True)
                start = perf_counter()
                prediction = predict(
                    held.drop(columns=["y_active", "pactivity", "raw_record"], errors="ignore"),
                    model,
                )
                seconds = perf_counter() - start
                require(
                    prediction.record_id.tolist() == held.record_id.tolist(),
                    "Prediction row mismatch",
                )
                destination = result_path.parent / arm["name"] / option["name"]
                report = score_report(
                    held, prediction.activity_probability.to_numpy(), plan, destination
                )
                rows.append(
                    {
                        **option,
                        "arm": arm["name"],
                        "inference_seconds": seconds,
                        "inference_timing_scope": "Single predict call; CPU ONNX session creation, preprocessing and forward; no reference-context generation, not warmed throughput",
                        **report,
                    }
                )
            similarity = score_report(
                held,
                held.similarity_active_max.to_numpy(),
                plan,
                result_path.parent / arm["name"] / "similarity",
                probability=False,
            )
            rows.append(
                {
                    "name": "similarity",
                    "family": "similarity",
                    "seed": None,
                    "arm": arm["name"],
                    **similarity,
                }
            )
        qed = score_report(
            held, held.qed.to_numpy(), plan, result_path.parent / "qed", probability=False
        )
        rows.append({"name": "qed", "family": "qed", "seed": None, "arm": "shared", **qed})
        result = {
            "task_id": task["task_id"],
            "heldout_count": len(held),
            "eligible_count": len(expected_ids),
            "models": rows,
            "validation_choice": task["validation_choice"],
            "validation_simple_choice": task["validation_simple_choice"],
        }
        write_json(result_path, result)
        seal(
            result_path.parent / "result-lock.json",
            {
                "files": hashes(
                    path
                    for path in result_path.parent.rglob("*")
                    if path.is_file() and path.name != "result-lock.json"
                )
            },
        )
        reports.append(result)
    outcome = {
        "plan_sha256": file_hash(root / "plan.json"),
        "choice_freeze_sha256": file_hash(root / "validation-choice-freeze.json"),
        "tasks": reports,
    }
    seal(root / "results.json", outcome)
    flat = [
        {
            "task_id": task["task_id"],
            "family": row["family"],
            "arm": row["arm"],
            "seed": row["seed"],
            "ap_full": row["before_filter"]["pr_auc"],
            "ap_filtered": row["after_filter"]["pr_auc"],
            **{
                key: row.get(key)
                for key in (
                    "basket_count",
                    "basket_active",
                    "basket_precision",
                    "basket_recall",
                    "basket_enrichment",
                    "scaffolds_covered",
                    "parameter_count",
                    "onnx_bytes",
                    "training_export_seconds",
                    "inference_seconds",
                )
            },
        }
        for task in reports
        for row in task["models"]
    ]
    pd.DataFrame(flat).to_csv(root / "benchmark-summary.csv", index=False)
    print("RESULTS", root / "results.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--work",
        type=Path,
        default=Path(os.environ.get("LOCALAPPDATA", ".")) / "S2SDecision/data-capacity-v1",
    )
    args = parser.parse_args()
    try:
        {"prepare": prepare, "train": train, "evaluate": evaluate}[args.phase](
            args.root.resolve(), args.work.resolve()
        )
    except Exception:
        args.root.mkdir(parents=True, exist_ok=True)
        with (args.root / "failures.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "utc": datetime.now(UTC).isoformat(),
                        "phase": args.phase,
                        "traceback": traceback.format_exc(),
                    }
                )
                + "\n"
            )
        raise
