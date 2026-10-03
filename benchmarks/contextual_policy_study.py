"""B15 expanded frozen contextual policy benchmark. Research outcomes stay explicit."""

import argparse
import json
import shutil
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import contextual_policy_fit as fit
import numpy as np
import pandas as pd
from rdkit import DataStructs
from scipy.special import expit

from s2s_decision.artifacts import file_hash, write_json
from s2s_decision.contextual_model import (
    SCHEMA,
    chemistry_manifest,
    feature_matrix,
    feature_names,
    fit_spec,
    max_similarity,
    prediction_sets,
)
from s2s_decision.contextual_policy import (
    PolicyContext,
    PolicySettings,
    ScoreEvidence,
    apply_contextual_policy,
)
from s2s_decision.qex import QEXFitError, QEXModel, fit_qex, predict_qex, qed_properties
from s2s_decision.schema import DESCRIPTOR_NAMES
from s2s_decision.selection import select_candidates
from smiles2select.profiles.loader import BUILTIN_DIR, load_profile_file
from smiles2select.rules.evaluator import evaluate_profiles

BUDGETS = (20, 50, 100)
QUOTAS = (3, None)


def seal(root):
    write_json(root / "completion.json", {"created_utc": datetime.now(UTC).isoformat(),
        "files": {p.relative_to(root).as_posix(): file_hash(p)
                  for p in sorted(root.rglob("*")) if p.is_file() and p.name != "completion.json"}})


def load_training(root):
    root = Path(root)
    seal_data = json.loads((root / "completion.json").read_text(encoding="utf-8"))
    for name, digest in seal_data["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != digest:
            raise ValueError("Training package hash mismatch or unsafe path")
    return json.loads((root / "models.json").read_text(encoding="utf-8"))


def _frozen_settings(panel, protocol):
    here = Path(__file__).resolve()
    sources = [here, here.with_name("contextual_policy_fit.py"),
               here.parents[1] / "src/s2s_decision/contextual_model.py",
               here.parents[1] / "src/s2s_decision/qex.py"]
    return {"created_utc": datetime.now(UTC).isoformat(), "panel_sha256": file_hash(panel),
        "protocol_path": str(protocol.resolve()), "protocol_sha256": file_hash(protocol),
        "sources": {str(p): file_hash(p) for p in sources},
        "C_grid": fit.CS, "seeds": fit.SEEDS, "budgets": BUDGETS, "quotas": QUOTAS,
        "logistic_variants": fit.VARIANTS, "gradient_boosting": {
            "depth_grid": [2, 3], "trees": 100, "learning_rate": .05,
            "subsample": .8, "min_leaf": 10, "fingerprint": "MACCS"},
        "shared": "pooled Morgan plus supported chemistry, target deviations; context ablation",
        "support": fit.GATE, "alpha": .1, "domain_threshold": .3,
        "probability_calibration": "scaffold SHA256 partition by seed; monotone sigmoid",
        "conformal": "other calibration half; class-conditional finite-sample quantile",
        "stage": "configured hit/lead/fragment policy scenarios; no stage-labeled outcomes",
        "qex": "published equation independent implementation; train-active only",
        "evaluation": "B13/B14 outcomes already observed; exploratory retrospective"}


def _single_models(task, groups, output, grids):
    train = task.loc[task.split.eq("train")]
    valid = task.loc[task.split.eq("validation")]
    models = {}
    for variant in fit.VARIANTS:
        spec = fit_spec(train, fit.columns_for(variant, groups),
                        "maccs" if variant == "lr_maccs" else "morgan")
        model, grid = fit.select_logistic(feature_matrix(train, spec), train.y_active,
            feature_matrix(valid, spec), valid.y_active)
        models[variant] = {**model, "spec": spec}
        grids.extend({"task_id": task.task_id.iloc[0], "variant": variant, "seed": None, **row}
                     for row in grid)
    spec = fit_spec(train, fit.columns_for("boosting", groups), "maccs")
    x, xv = feature_matrix(train, spec), feature_matrix(valid, spec)
    for seed in fit.SEEDS:
        candidates, scores = [], []
        for depth in (2, 3):
            _, model = fit.fit_boosting(x, train.y_active, depth, seed)
            score = fit.ranking_metrics(valid.y_active, fit.predict_estimator(xv, model))["ap"]
            candidates.append(model)
            scores.append(score)
            grids.append({"task_id": task.task_id.iloc[0], "variant": "boosting", "seed": seed,
                          "depth": depth, "validation_ap": score})
        models[f"boosting:{seed}"] = {**candidates[int(np.argmax(scores))], "spec": spec}
    return models


def _bundle(task_id, task, model, support, seed):
    accession, species, endpoint = task_id.split("_")
    chemistry = chemistry_manifest(2048)
    names = feature_names(model["spec"])
    coefficients = dict(zip(names, model["weights"]))
    support = [{**row, "coefficient": coefficients.get(row["feature_id"]),
                "origin": "learned", "action": "warn", "evidence_endpoint": "activity",
                "source": "B15 ChEMBL37 training subset"} for row in support]
    return {**model, "schema": SCHEMA, "task_id": task_id, "stage": "hit",
            "source": "B15 ChEMBL37; research only; no prospective performance claim",
            "context": {"target": accession, "species": "Homo sapiens", "endpoint": endpoint,
                "stage": "hit_finding", "assay_context": "pooled_biochemical",
                "activity_threshold": 1000., "activity_unit": "nM", "activity_relation": "<=",
                "source_version": "ChEMBL37/B13", "model_version": f"B15-lr-full-seed{seed}",
                "chemistry_version": chemistry["chemistry_hash"]},
            "activity_threshold": {"value": 1000, "unit": "nM", "relation": "<="},
            "support": support, "chemistry": chemistry,
            "training_fingerprints": task.loc[task.split.eq("train"), "fingerprint_hex"].tolist(),
            "domain_threshold": .3, "domain_warning": "descriptive similarity cut; not validated guarantee",
            "supported_assay_contexts": sorted(task.loc[task.split.eq("train"), "assay_context"].unique())}


def train(panel, output, protocol):
    panel, output, protocol = map(Path, (panel, output, protocol))
    frame = pd.read_parquet(panel)
    fit.validate_panel(frame)
    if not protocol.is_file():
        raise ValueError("Frozen parent protocol required before fitting")
    output.mkdir(parents=True, exist_ok=False)
    settings = _frozen_settings(panel, protocol)
    write_json(output / "fit-freeze.json", settings)
    for source in settings["sources"]:
        destination = output / "execution-source" / Path(source).name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(source, destination)
    shutil.copy2(protocol, output / "PROTOCOL_FROZEN.md")
    started = perf_counter()
    grouped = {name: part.reset_index(drop=True) for name, part in frame.groupby("task_id")}
    support, groups = {}, {}
    for name, part in grouped.items():
        groups[name], support[name] = fit.support_features(part)
    write_json(output / "support.json", support)
    write_json(output / "feature-groups.json", groups)
    models, grids = {}, []
    for name, task in grouped.items():
        print(f"Fitting {name}: separate-target comparators", flush=True)
        models[name] = _single_models(task, groups[name], output, grids)
    tr, va = frame.loc[frame.split.eq("train")], frame.loc[frame.split.eq("validation")]
    shared = {}
    for use_context in (False, True):
        name = "shared_context" if use_context else "shared"
        print(f"Fitting {name}", flush=True)
        spec = fit.shared_spec(tr, groups, use_context)
        model, grid = fit.select_logistic(fit.shared_matrix(tr, spec), tr.y_active,
            fit.shared_matrix(va, spec), va.y_active,
            (tr.task_id.to_numpy(), va.task_id.to_numpy()))
        shared[name] = {**model, "shared_spec": spec}
        grids.extend({"task_id": "all", "variant": name, "seed": None, **row} for row in grid)
    for task_id, task in grouped.items():
        cal = task.loc[task.split.eq("calibration")]
        calibrated = {}
        for name, raw in {**models[task_id], **shared}.items():
            x = fit.shared_matrix(cal, raw["shared_spec"]) if name.startswith("shared") else feature_matrix(cal, raw["spec"])
            logits = fit.predict_estimator(x, raw)
            for seed in fit.SEEDS:
                if name.startswith("boosting:") and int(name.split(":")[1]) != seed:
                    continue
                key = f"{name.split(':')[0]}:{seed}"
                calibrated[key] = {**raw, **fit.calibrate_model(cal, logits, seed)}
                if name == "lr_full":
                    write_json(output / "bundles" / f"{task_id}-seed{seed}.json",
                               _bundle(task_id, task, calibrated[key], support[task_id], seed))
        models[task_id] = calibrated
    qex = {}
    for task_id, task in grouped.items():
        print(f"Fitting QEX {task_id}", flush=True)
        active = task.loc[task.split.eq("train") & task.y_active.eq(1)]
        try:
            qex[task_id] = {"status": "fitted", "model": fit_qex(qed_properties(active.model_smiles.tolist())).to_dict()}
        except QEXFitError as exc:
            qex[task_id] = {"status": "failed", "error": str(exc)}
    package = {"models": models, "qex": qex, "groups": groups,
        "supported_contexts": {name: sorted(task.loc[task.split.eq("train"), "assay_context"].unique())
                               for name, task in grouped.items()},
        "training_rows": {name: task.loc[task.split.eq("train"), ["identity", "fingerprint_hex", "y_active",
                              "murcko_scaffold"]].to_dict("records") for name, task in grouped.items()},
        "fit_identities": sorted(frame.loc[frame.split.isin(["train", "validation", "calibration"]), "identity"].unique()),
        "settings": settings}
    write_json(output / "models.json", package)
    pd.DataFrame(grids).to_parquet(output / "validation-grid.parquet", index=False)
    write_json(output / "training.json", {"seconds": perf_counter() - started, "validation_fits": len(grids),
        "task_count": len(grouped), "predictive_models": sum(len(v) for v in models.values()),
        "probability_calibrations": sum(len(v) for v in models.values()), "qex": {k: v["status"] for k, v in qex.items()}})
    seal(output)
    return package


def stage_masks(frame):
    result = {"hit": np.ones(len(frame), bool)}
    for stage, name in (("lead", "lead_like"), ("fragment_core", "rule_of_three_core"),
                        ("fragment_extended", "rule_of_three_extended")):
        profile = load_profile_file(BUILTIN_DIR / f"{name}.json")
        status = evaluate_profiles(frame, [profile]).status
        result[stage] = status[f"{profile.id}__passed"].to_numpy(bool)
    return result


def select_basket(frame, score, eligible, n, quota):
    ranks = pd.Series(score).rank(method="dense").to_numpy(float)
    ranks = ranks / max(1, ranks.max())
    needed = [name for name in ("record_id", "identity", "murcko_scaffold", "y_active", "mol_wt", "tpsa") if name in frame]
    pool = frame[needed].assign(priority_score=ranks, eligible=np.asarray(eligible, bool), valid=True)
    selected = select_candidates(pool, n, max_per_scaffold=quota)
    return _selection_metrics(frame, selected, eligible, n, quota), selected.records


def _selection_metrics(frame, selected, eligible, n, quota):
    final = selected.records.loc[selected.records.is_final]
    positives = int(frame.y_active.sum())
    hits = int(final.y_active.sum())
    metrics = {"requested_n": n, "max_per_scaffold": quota, "selected_n": len(final),
        "shortfall": selected.manifest["shortfall"], "hits": hits,
        "missed_positives": positives - hits, "total_positives": positives,
        "universe_n": len(frame), "eligible_n": int(np.sum(eligible)),
        "removed_positives": int(frame.loc[~np.asarray(eligible, bool), "y_active"].sum()),
        "precision": hits / len(final) if len(final) else None,
        "recall": hits / positives if positives else None,
        "enrichment_requested": (hits / n) / (positives / len(frame)) if positives else None,
        "scaffold_count": int(final.murcko_scaffold.nunique()),
        "acyclic_selected": int(final.murcko_scaffold.eq("").sum())}
    return metrics


def production_basket(frame, values, model, task, seed, similarity, n, quota):
    """Execute the same public policy engine and informational defaults as the GUI."""
    context = PolicyContext(target=task.split("_")[0], species="Homo sapiens",
        endpoint=task.split("_")[-1], stage="hit_finding", assay_context="pooled_biochemical",
        source_version="ChEMBL37/B13", model_version=f"B15-lr-full-seed{seed}",
        chemistry_version=chemistry_manifest(2048)["chemistry_hash"],
        activity_threshold=1000., activity_unit="nM", activity_relation="<=")
    cal = model["calibration"]
    evidence = ScoreEvidence(context=context, calibrated=True, calibration_method=cal["method"],
        calibration_n=cal["n"], positive_n=cal["positive_n"], negative_n=cal["negative_n"],
        source="B15 ChEMBL37; research only")
    names = np.asarray([name for name in frame if name.startswith("alert__")])
    matched = frame[names].to_numpy(bool)
    needed = ["record_id", "identity", "murcko_scaffold", "y_active", "model_smiles", *DESCRIPTOR_NAMES]
    pool = frame[needed].assign(eligible=True, valid=True, activity_score=values["probability"],
        in_domain=similarity >= .3, activity_prediction_set=prediction_sets(
            values["probability"], values["conformal"]["quantiles"]),
        alert_ids=[[name.removeprefix("alert__") for name in names[row]] for row in matched])
    contexts = sorted(frame.assay_context.unique())
    requested = replace(context, assay_context=contexts[0]) if len(contexts) == 1 else context
    result = apply_contextual_policy(pool, requested, PolicySettings(n=n, activity_evidence=evidence,
                                                                  max_per_scaffold=quota))
    metrics = _selection_metrics(frame, result, np.ones(len(frame), bool), n, quota)
    return {**metrics, "review_selected": int(result.records.loc[result.records.is_final, "review_required"].sum())}, result.records


def analogue_pairs(frame, threshold=.5):
    rows = []
    flags = sorted(name for name in frame if name.startswith("alert__") or
                   (name.startswith("rule__") and name.endswith("__violation")))
    for scaffold, group in frame.groupby("murcko_scaffold"):
        if not scaffold or len(group) < 2:
            continue
        records = list(group.to_dict("records"))
        fps = [DataStructs.CreateFromBinaryText(bytes.fromhex(row["fingerprint_hex"])) for row in records]
        for i, a in enumerate(records):
            similarities = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
            for j, similarity in enumerate(similarities, i + 1):
                if similarity < threshold:
                    continue
                b = records[j]
                rows.append({"identity_a": a["identity"], "identity_b": b["identity"],
                    "scaffold": scaffold, "similarity": similarity, "y_a": a["y_active"], "y_b": b["y_active"],
                    "context_a": a["assay_context"], "context_b": b["assay_context"],
                    "same_context": a["assay_context"] == b["assay_context"],
                    "changed_features": [name for name in flags if bool(a[name]) != bool(b[name])]})
    return pd.DataFrame(rows, columns=["identity_a", "identity_b", "scaffold", "similarity", "y_a", "y_b",
                                      "context_a", "context_b", "same_context", "changed_features"])


def _scores(frame, models):
    result = {}
    for name, model in models.items():
        x = fit.shared_matrix(frame, model["shared_spec"]) if name.startswith("shared") else feature_matrix(frame, model["spec"])
        raw = fit.predict_estimator(x, model)
        cal = model["calibration"]
        result[name] = {"raw": raw, "probability": expit(cal["slope"] * raw + cal["intercept"]),
                        "conformal": model["conformal"]}
    return result


def _arms(frame, scores, package, task, seed, similarity):
    all_rows = np.ones(len(frame), bool)
    base, full = scores[f"lr_base:{seed}"], scores[f"lr_full:{seed}"]
    rigid = (frame.lipinski_violations.eq(0) & frame.veber_violations.eq(0)
             & frame.pains_count.eq(0) & frame.brenk_count.eq(0)).to_numpy()
    arms = {"rigid": (base["raw"], rigid), "qed": (frame.qed.to_numpy(), all_rows),
            "random": (np.random.default_rng(seed).random(len(frame)), all_rows)}
    for tolerance in (1, 2):
        mask = (frame.lipinski_violations.le(tolerance) & frame.veber_violations.eq(0)).to_numpy()
        arms[f"tolerance{tolerance}"] = (base["raw"], mask)
    for name, score in scores.items():
        if name.endswith(f":{seed}"):
            arms[name.split(":")[0]] = (score["raw"], all_rows)
    active = [row["fingerprint_hex"] for row in package["training_rows"][task] if row["y_active"] == 1]
    arms["tanimoto_active"] = (max_similarity(frame, active), all_rows)
    qex = package["qex"][task]
    if qex["status"] == "fitted":
        arms["qex"] = (predict_qex(QEXModel.from_dict(qex["model"]), qed_properties(frame.model_smiles.tolist())), all_rows)
    sets = prediction_sets(full["probability"], full["conformal"]["quantiles"])
    # Explicit review strata; no molecule is excluded for uncertainty.
    review = np.array([s != [1] for s in sets]) | (similarity < .3)
    arms["review_priority_sensitivity"] = (full["probability"] + 2 * (~review), all_rows)
    arms["production_contextual"] = (full["probability"], all_rows)
    for stage, mask in stage_masks(frame).items():
        if stage != "hit":
            arms[f"{stage}_hard_profile_sensitivity"] = (full["raw"], mask)
    return arms


def evaluate_cohort(frame, package, task, name, output):
    output.mkdir(parents=True, exist_ok=False)
    frame = frame.reset_index(drop=True)
    if frame.identity.duplicated().any() or set(frame.identity) & set(package["fit_identities"]):
        raise ValueError("Evaluation identity overlap with train/validation/calibration")
    references = [row["fingerprint_hex"] for row in package["training_rows"][task]]
    similarity = max_similarity(frame, references)
    scores = _scores(frame, package["models"][task])
    context_supported = frame.assay_context.isin(package["supported_contexts"][task]).to_numpy()
    metric_rows, score_rows = [], []
    for key, values in scores.items():
        variant, seed = key.split(":")
        common = {"task_id": task, "cohort": name, "variant": variant, "seed": int(seed)}
        for context, index in [("all", np.ones(len(frame), bool)), *[
                (str(c), frame.assay_context.eq(c).to_numpy()) for c in sorted(frame.assay_context.unique())]]:
            metric_rows.append({**common, "assay_context": context,
                **fit.ranking_metrics(frame.y_active[index], values["raw"][index]),
                **fit.probability_metrics(frame.y_active[index], values["probability"][index], values["conformal"])})
        score_rows.append(frame[["identity", "record_id", "y_active", "murcko_scaffold", "assay_context"]].assign(
            **common, raw_score=values["raw"], probability=values["probability"], max_training_similarity=similarity,
            assay_context_supported=context_supported,
            context_status=np.where(context_supported, "observed_for_task", "unseen_for_task_review"),
            prediction_set=prediction_sets(values["probability"], values["conformal"]["quantiles"])))
    pd.concat(score_rows).to_parquet(output / "scores.parquet", index=False)
    pd.DataFrame(metric_rows).to_parquet(output / "probability-metrics.parquet", index=False)
    selections, summaries, rankings, pairs = [], [], [], []
    for seed in fit.SEEDS:
        print(f"Selecting {task}/{name}, seed {seed}", flush=True)
        baskets = {}
        for arm, (score, eligible) in _arms(frame, scores, package, task, seed, similarity).items():
            common = {"task_id": task, "cohort": name, "arm": arm, "seed": seed}
            if np.asarray(eligible).any():
                rankings.append({**common, **fit.ranking_metrics(frame.y_active[eligible], score[eligible]),
                                 "metric_universe": "eligible"})
            for n in BUDGETS:
                for quota in QUOTAS:
                    if arm == "production_contextual":
                        metrics, selected = production_basket(frame, scores[f"lr_full:{seed}"],
                            package["models"][task][f"lr_full:{seed}"], task, seed, similarity, n, quota)
                    else:
                        metrics, selected = select_basket(frame, score, eligible, n, quota)
                    chosen = selected.loc[selected.is_final]
                    baskets[(arm, n, quota)] = chosen
                    summaries.append({**common, **metrics,
                        "selected_mean_similarity": float(similarity[selected.is_final.to_numpy()].mean()) if len(chosen) else None,
                        "selected_mean_mw": float(chosen.mol_wt.mean()) if len(chosen) else None,
                        "selected_mean_tpsa": float(chosen.tpsa.mean()) if len(chosen) else None})
                    extra = [name for name in ("review_required", "policy_warnings", "policy_explanation") if name in selected]
                    selections.append(selected[["record_id", "identity", "y_active", "is_final", "selection_reason", *extra]].assign(
                        **common, requested_n=n, max_per_scaffold=quota))
        for (arm, n, quota), chosen in baskets.items():
            if arm == "lr_base":
                continue
            base = baskets[("lr_base", n, quota)]
            pairs.append({"task_id": task, "cohort": name, "seed": seed, "arm": arm,
                "requested_n": n, "max_per_scaffold": quota,
                "delta_hits": int(chosen.y_active.sum() - base.y_active.sum()),
                "rescued": int(chosen.loc[~chosen.identity.isin(base.identity), "y_active"].sum()),
                "displaced": int(base.loc[~base.identity.isin(chosen.identity), "y_active"].sum()),
                "changed": int((~chosen.identity.isin(base.identity)).sum())})
    pd.concat(selections).to_parquet(output / "selections.parquet", index=False)
    pd.DataFrame(summaries).to_parquet(output / "selection-metrics.parquet", index=False)
    pd.DataFrame(rankings).to_parquet(output / "ranking-metrics.parquet", index=False)
    pd.DataFrame(pairs).to_parquet(output / "paired-comparisons.parquet", index=False)
    analogue_pairs(frame).to_parquet(output / "analogue-pairs.parquet", index=False)
    write_json(output / "cohort.json", {"task": task, "cohort": name, "n": len(frame),
        "positive_n": int(frame.y_active.sum()), "baskets": len(summaries),
        "scope": "exploratory; observed B13/B14 cohorts are not untouched holdouts"})
    seal(output)


def evaluate(panel, training, output):
    output = Path(output)
    package = load_training(training)
    frame = pd.read_parquet(panel)
    fit.validate_panel(frame)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "evaluation-freeze.json", {"panel_sha256": file_hash(panel),
        "training_completion_sha256": file_hash(Path(training) / "completion.json"),
        "created_utc": datetime.now(UTC).isoformat(), "sources": {
            str(p): file_hash(p) for p in [Path(__file__), Path(__file__).parents[1] / "src/s2s_decision/contextual_policy.py"]}})
    shutil.copy2(__file__, output / "contextual_policy_study.py")
    for task, group in frame.groupby("task_id"):
        for split in ("test", "recent"):
            part = group.loc[group.split.eq(split)]
            if len(part):
                print(f"Evaluating {task}/{split}: {len(part)} molecules", flush=True)
                evaluate_cohort(part, package, task, split, output / f"{task}-{split}")
    seal(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    trainer = sub.add_parser("train")
    trainer.add_argument("--panel", type=Path, required=True)
    trainer.add_argument("--output", type=Path, required=True)
    trainer.add_argument("--protocol", type=Path, required=True)
    evaluator = sub.add_parser("evaluate")
    evaluator.add_argument("--panel", type=Path, required=True)
    evaluator.add_argument("--training", type=Path, required=True)
    evaluator.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "train":
        train(args.panel, args.output, args.protocol)
    else:
        evaluate(args.panel, args.training, args.output)


if __name__ == "__main__":
    main()
