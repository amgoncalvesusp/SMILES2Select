"""Frozen external ranking-transfer evaluation; no fitting or assay-label calibration.

Freeze inputs after preparation and control training. External labels are only
joined after ranking. Raw logits avoid reversals from source-domain calibration.
"""

import argparse
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from s2s_decision.artifacts import file_hash, read_bundle, write_json
from s2s_decision.context import build_context
from s2s_decision.features import validate_chemistry_compatibility
from s2s_decision.preprocessing import Preprocessor
from s2s_decision.schema import CONTEXT_NAMES, fingerprint_matrix

COHORTS = ("primary_identity_novel", "secondary_scaffold_novel")
ASSAYS = {"ESR1_ant": "P03372_WT_IC50", "PPARG": "P37231_WT_EC50"}
SEEDS = (42, 43, 44)
KS = (20, 50, 100, 500)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rank_scores(identities, scores):
    identities = pd.Series(identities, dtype=str).reset_index(drop=True)
    scores = np.asarray(scores, dtype=float)
    if identities.duplicated().any() or identities.str.strip().eq("").any():
        raise ValueError("Ranking requires unique nonempty identities")
    if scores.shape != (len(identities),) or not np.isfinite(scores).all():
        raise ValueError("Ranking requires aligned finite scores")
    return (pd.DataFrame({"identity": identities, "score": scores})
            .sort_values(["score", "identity"], ascending=[False, True], kind="stable")
            .reset_index(drop=True).assign(rank=lambda frame: np.arange(1, len(frame) + 1)))


def cohort(frame, name):
    if name not in COHORTS or not set(COHORTS) <= set(frame.columns):
        raise ValueError("Unknown or missing cohort")
    if any(frame[column].isna().any() or not pd.api.types.is_bool_dtype(frame[column])
           for column in COHORTS):
        raise ValueError("Cohort flags must be nonmissing Booleans")
    if (frame[COHORTS[1]] & ~frame[COHORTS[0]]).any():
        raise ValueError("Scaffold-novel cohort must be a subset of identity-novel cohort")
    return frame.loc[frame[name]].copy().reset_index(drop=True)


def ranking_metrics(truth, ranked, ks=KS):
    if truth.identity.duplicated().any() or ranked.identity.duplicated().any():
        raise ValueError("Metrics require unique identities")
    if set(truth.identity) != set(ranked.identity):
        raise ValueError("Every method must rank the same identities")
    ordered = ranked.merge(truth[["identity", "y_active", "murcko_scaffold"]],
                           on="identity", how="left", validate="one_to_one")
    y = ordered.y_active.to_numpy()
    if not np.isin(y, [0, 1]).all() or truth.murcko_scaffold.isna().any():
        raise ValueError("Metrics require observed binary labels and defined scaffolds")
    if any(type(k) is not int or k < 1 for k in ks):
        raise ValueError("Budgets must be positive integers")
    n, positive = len(y), int(y.sum())
    negative, prevalence = n - positive, positive / n if n else None
    both = positive > 0 and negative > 0
    positive_scaffolds = truth.loc[truth.y_active.eq(1), "murcko_scaffold"].nunique()
    scaffolds = truth.murcko_scaffold.nunique()
    selections = {}
    for k in ks:
        selected = ordered.iloc[:k]
        effective, hits = len(selected), int(selected.y_active.sum())
        precision = hits / effective if effective else None
        recovered = selected.loc[selected.y_active.eq(1), "murcko_scaffold"].nunique()
        selections[str(k)] = {
            "effective_k": effective, "hits": hits, "precision": precision,
            "recall": hits / positive if positive else None,
            "enrichment_factor": precision / prevalence if positive and effective else None,
            "positive_scaffolds_recovered": int(recovered),
            "positive_scaffold_recall": recovered / positive_scaffolds if positive_scaffolds else None,
            "selected_scaffolds": int(selected.murcko_scaffold.nunique()),
            "scaffold_coverage": selected.murcko_scaffold.nunique() / scaffolds if scaffolds else None,
        }
    return {"observed": n, "positive": positive, "negative": negative, "prevalence": prevalence,
            "status": "sufficient_support" if positive >= 5 and negative >= 100 else "insufficient_support",
            "average_precision": float(average_precision_score(y, ordered.score)) if both else None,
            "roc_auc": float(roc_auc_score(y, ordered.score)) if both else None,
            "positive_scaffolds": int(positive_scaffolds), "scaffolds": int(scaffolds),
            "selection": selections}


def random_control(truth, ks=KS, repetitions=1000, seed=0):
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("Random repetitions must be positive")
    # Exact hit expectation; seeded draws quantify finite-budget scaffold recovery.
    ordered = truth.sort_values("identity").reset_index(drop=True)
    n, positives = len(ordered), int(ordered.y_active.sum())
    draws = {str(k): {"hits": [], "selected_scaffolds": [], "positive_scaffolds_recovered": []}
             for k in ks}
    for repetition in range(repetitions):
        generator = np.random.default_rng(seed + repetition)
        sampled = ordered.iloc[generator.permutation(n)[:min(max(ks), n)]]
        for k in ks:
            selected = sampled.iloc[:k]
            values = draws[str(k)]
            values["hits"].append(int(selected.y_active.sum()))
            values["selected_scaffolds"].append(int(selected.murcko_scaffold.nunique()))
            values["positive_scaffolds_recovered"].append(
                int(selected.loc[selected.y_active.eq(1), "murcko_scaffold"].nunique()))
    selection = {}
    for k in ks:
        effective = min(k, n)
        selection[str(k)] = {
            "effective_k": effective, "expected_hits": effective * positives / n if n else 0.,
            "expected_precision": positives / n if n else None,
            "expected_recall": effective / n if n and positives else None,
            "expected_enrichment_factor": 1. if positives and effective else None,
            **{name: {"mean": float(np.mean(values)),
                      "p025": float(np.quantile(values, .025)),
                      "p975": float(np.quantile(values, .975))}
               for name, values in draws[str(k)].items()},
        }
    return {"repetitions": repetitions, "seeds": [seed, seed + repetitions - 1],
            "interval_scope": "random selection variability, not model confidence",
            "selection": selection}


def similarity_strata(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Training Tanimoto must be finite within [0, 1]")
    return np.select([values <= .4, values <= .6, values <= .8],
                     ["<=0.4", "(0.4,0.6]", "(0.6,0.8]"], default=">0.8")


def verify_lock(lock):
    if not lock.get("files"):
        raise ValueError("Frozen file manifest cannot be empty")
    for path, digest in lock["files"].items():
        if file_hash(path) != digest:
            raise ValueError(f"Frozen input changed: {path}")


def molecular_inputs(frame, preprocess):
    intrinsic = frame.assign(**dict.fromkeys(CONTEXT_NAMES, np.nan))
    properties, _ = Preprocessor.from_dict(read_json(preprocess)).transform(intrinsic)
    return np.concatenate((fingerprint_matrix(frame, 2048), properties), axis=1)


def predict_checkpoint(path, x, task):
    import torch

    from s2s_decision.multitask import S2Decision
    from s2s_decision.training import _require_safe_checkpoint_runtime

    _require_safe_checkpoint_runtime()
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = S2Decision(saved["inputs"], saved["tasks"], tuple(saved["hidden"]))
    model.load_state_dict(saved["state_dict"])
    model.eval()
    torch.set_num_threads(4)
    if x.shape[1] != saved["inputs"] or not 0 <= task < saved["tasks"]:
        raise ValueError("Checkpoint dimensions or task index mismatch")
    with torch.inference_mode():
        parts = [model(torch.from_numpy(x[start:start + 1024]))[:, task].numpy()
                 for start in range(0, len(x), 1024)]
    return np.concatenate(parts) if parts else np.empty(0)


def control_runner():
    path = Path(__file__).with_name("external_controls.py")
    spec = importlib.util.spec_from_file_location("external_controls", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_assay(frame, task_id, b09, controls):
    if "y_active" in frame:
        raise ValueError("External labels must be removed before prediction")
    tasks = read_json(b09 / "data/tasks.json")
    task_index = next(index for index, item in enumerate(tasks) if item["task_id"] == task_id)
    x = molecular_inputs(frame, b09 / "data/preprocess.json")
    scores = {"qed": frame.qed.to_numpy(dtype=float)}
    with np.load(b09 / "run/logistic/coefficients.npz", allow_pickle=False) as saved:
        scores["logistic"] = x @ saved["coefficients"][task_index] + saved["intercept"][task_index]
    task_folder = controls / task_id
    reference_bundle = read_bundle(task_folder / "tiny/seed-42/references")
    references = reference_bundle.records
    if "split" in references and not references.split.eq("train").all():
        raise ValueError("Context references must be train only")
    context = build_context(frame, references)
    scores["max_tanimoto_active"] = context[CONTEXT_NAMES[0]].to_numpy()
    maximum = 1. - context[CONTEXT_NAMES[4]].to_numpy()
    for seed in SEEDS:
        scores[f"s2_decision_seed{seed}"] = predict_checkpoint(
            b09 / f"run/s2_decision_seed{seed}/best.pt", x, task_index)
        scores[f"single_task_seed{seed}"] = predict_checkpoint(
            task_folder / f"matched/seed-{seed}/best.pt", x, 0)
        tiny = task_folder / f"tiny/seed-{seed}"
        if read_json(tiny / "manifest.json")["reference_records_sha256"] != reference_bundle.manifest["records_sha256"]:
            raise ValueError("Tiny models must share the frozen task-training references")
        scores[f"tiny_seed{seed}"] = control_runner().score_control(
            frame, tiny, "tiny", context_features=context)["raw_logits"] if len(frame) else np.empty(0)
    return scores, maximum


def verify_completion(root):
    manifest_path = root / "completion.json"
    completion = read_json(manifest_path)
    if not completion.get("files"):
        raise ValueError(f"Completed artifact manifest is empty: {root}")
    paths = [manifest_path]
    for relative, expected in completion["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != expected:
            raise ValueError(f"Completed artifact changed: {path}")
        paths.append(path)
    return paths


def verify_provenance(external, b09, controls):
    paths = [*verify_completion(b09 / "data"), *verify_completion(b09 / "run"),
             *verify_completion(controls)]
    data = external / "data"
    preparation = read_json(data / "preparation-manifest.json")
    preparation_code = Path(__file__).with_name("litpcba_external.py")
    if file_hash(preparation_code) != preparation["code_sha256"]:
        raise ValueError("External preparation executable changed")
    paths.append(preparation_code)
    for name, key in (("dataset.parquet", "dataset_sha256"), ("chemistry.json", "chemistry_sha256"),
                      ("source-manifest.json", "source_manifest_sha256")):
        path = data / name
        if file_hash(path) != preparation[key]:
            raise ValueError(f"Prepared external artifact changed: {path}")
        paths.append(path)
    source = read_json(data / "source-manifest.json")
    expected = {**preparation["exposure"]["source_files"], **source["smiles_files"],
                **{item["path"]: item["sha256"] for item in source["downloads"]}}
    for name, digest in expected.items():
        if file_hash(name) != digest:
            raise ValueError(f"Original external source or exposure changed: {name}")
        paths.append(Path(name))
    return paths


def freeze(external, b09, controls, plan):
    paths = {"external": str(external.resolve()), "b09": str(b09.resolve()),
             "controls": str(controls.resolve()), "plan": str(plan.resolve())}
    lock_path = external / "evaluation-lock.json"
    if lock_path.exists():
        raise ValueError("Evaluation lock already exists; frozen evidence is preserved")
    validate_plan(read_json(plan))
    required = [external / "data/dataset.parquet", external / "data/preparation-manifest.json",
                external / "data/chemistry.json", controls / "completion.json",
                b09 / "data/preprocess.json", b09 / "data/tasks.json", b09 / "data/chemistry.json",
                b09 / "run/logistic/coefficients.npz", plan]
    required += [b09 / f"run/s2_decision_seed{seed}/best.pt" for seed in SEEDS]
    for path in required:
        if not path.is_file():
            raise ValueError(f"Required completed input is missing: {path}")
    provenance_files = verify_provenance(external, b09, controls)
    chemistry = read_json(b09 / "data/chemistry.json")
    external_chemistry = read_json(external / "data/chemistry.json")
    validate_chemistry_compatibility(chemistry, external_chemistry)
    repo = Path(__file__).resolve().parents[1]
    files = [*required, *provenance_files,
             *[p for p in external.iterdir() if p.is_file() and p.suffix in {".json", ".md"}],
             *[p for p in (external / "data").rglob("*") if p.is_file()],
             *[p for p in controls.rglob("*") if p.is_file()],
             *(repo / "src").rglob("*.py"), Path(__file__).resolve(),
             Path(__file__).with_name("external_controls.py")]
    payload = {"schema": "s2-decision-external-evaluation/1", "paths": paths,
               "created_utc": datetime.now(UTC).isoformat(), "K": list(KS),
               "seeds": list(SEEDS), "random_repetitions": 1000, "random_seed": 0,
               "assays": ASSAYS, "cohorts": list(COHORTS),
               "score_contract": "raw logits; descending score, ascending identity ties",
               "scope": "ranking transfer from Papyrus thresholds to observed LIT-PCBA labels",
               "files": {str(path.resolve()): file_hash(path) for path in sorted(set(files))}}
    write_json(lock_path, payload)
    return payload


def validate_plan(plan):
    if plan["selection"]["K"] != list(KS) or plan["selection"]["primary_K"] != 50:
        raise ValueError("Frozen plan K differs from evaluator")
    if plan["selection"]["random_repetitions"] != 1000:
        raise ValueError("Frozen plan random repetitions differ from evaluator")
    if plan["selection"]["random_seeds"] != "0 through 999":
        raise ValueError("Frozen plan random seeds differ from evaluator")
    if plan["models"]["control_seeds"] != list(SEEDS):
        raise ValueError("Frozen model seeds differ from evaluator")
    if {item["assay"]: item["task_id"] for item in plan["assays"]} != ASSAYS:
        raise ValueError("Frozen assay-task mappings differ from evaluator")


def evaluate(external):
    lock = read_json(external / "evaluation-lock.json")
    if external.resolve() != Path(lock["paths"]["external"]).resolve():
        raise ValueError("Evaluation root differs from frozen input root")
    verify_lock(lock)
    validate_plan(read_json(lock["paths"]["plan"]))
    output = external / "evaluation"
    output.mkdir(parents=True, exist_ok=False)
    b09, controls = (Path(lock["paths"][key]) for key in ("b09", "controls"))
    dataset = pd.read_parquet(external / "data/dataset.parquet")
    reports = {}
    for assay, task_id in lock["assays"].items():
        frame = dataset.loc[dataset.assay.eq(assay)].copy()
        frame = cohort(frame, COHORTS[0])
        scores, maximum = score_assay(frame.drop(columns="y_active"), task_id, b09, controls)
        scored = frame[["identity", *COHORTS]].assign(maximum_training_tanimoto=maximum,
                                                    similarity_stratum=similarity_strata(maximum),
                                                    **scores)
        scored.to_parquet(output / f"{assay}-scores.parquet", index=False)
        frame[["identity", "y_active", "murcko_scaffold"]].to_parquet(
            output / f"{assay}-truth.parquet", index=False)
        reports[assay] = {"task_id": task_id, "cohorts": {}}
        for name in COHORTS:
            pool = cohort(frame, name)
            selected_scores = scored.loc[scored.identity.isin(pool.identity)]
            methods = {}
            for method in scores:
                ranked = rank_scores(selected_scores.identity, selected_scores[method])
                ranked.to_parquet(output / f"{assay}-{name}-{method}-ranking.parquet", index=False)
                for k in lock["K"]:
                    ranked.iloc[:k].to_csv(output / f"{assay}-{name}-{method}-K{k}.csv", index=False)
                methods[method] = ranking_metrics(pool, ranked, lock["K"])
            reports[assay]["cohorts"][name] = {
                "methods": methods,
                "random": random_control(pool, lock["K"], lock["random_repetitions"], lock["random_seed"]),
                "training_similarity_strata": {
                    str(stratum): {"molecules": len(group),
                                   "positive": int(pool.set_index("identity").loc[group.identity, "y_active"].sum())}
                    for stratum, group in selected_scores.groupby("similarity_stratum")},
            }
    verify_lock(lock)
    write_json(output / "metrics.json", reports)
    write_json(output / "completion.json", {
        "lock_sha256": file_hash(external / "evaluation-lock.json"),
        "files": {path.name: file_hash(path) for path in sorted(output.iterdir()) if path.is_file()},
        "completed_utc": datetime.now(UTC).isoformat(), "inputs_verified_before_and_after": True})
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "evaluate"))
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--b09", type=Path)
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        if any(value is None for value in (args.b09, args.controls, args.plan)):
            parser.error("freeze requires --b09, --controls and --plan")
        freeze(args.external, args.b09, args.controls, args.plan)
    else:
        evaluate(args.external)


if __name__ == "__main__":
    main()
