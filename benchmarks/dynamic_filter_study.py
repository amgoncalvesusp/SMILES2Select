"""B14 research runner: cross-fitted contextual ranking, then sealed evaluation.

No production model is installed. Scores support ranking of one measured assay
cohort; they are neither validated activity probabilities nor P(advance).
"""

import argparse
import json
import warnings
from datetime import UTC, datetime
from importlib.metadata import version
from numbers import Integral
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from s2s_decision.artifacts import file_hash, write_json
from s2s_decision.preprocessing import Preprocessor
from s2s_decision.schema import CONTEXT_NAMES, PROPERTY_NAMES, fingerprint_matrix
from s2s_decision.selection import select_candidates

SEEDS = (42, 43, 44)
CS = (0.1, 1.0, 10.0)
HEADS = ("control", "counts", "rules", "alerts", "full")
BUDGETS = (20, 50, 100)
QUOTAS = (3, None)
GROUPS = {"counts", "rules", "alerts"}
COUNTS = PROPERTY_NAMES[-4:]
RESERVED = {
    "record_id", "identity", "murcko_scaffold", "fingerprint_hex", "split",
    "y_active", "label", "activity", "target", "target_id", "task_id",
    "endpoint", "year", "document_id", "assay_id", "priority_score", "eligible",
    "valid", "is_final", "scaffold_novel", "source", "source_id",
    *PROPERTY_NAMES[:16], *CONTEXT_NAMES,
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object")
    groups = config.get("feature_groups")
    if not isinstance(groups, dict) or set(groups) != GROUPS:
        raise ValueError("feature_groups must contain exactly counts, rules, alerts")
    names = []
    for group, columns in groups.items():
        if not isinstance(columns, list) or any(
            not isinstance(name, str) or not name.strip() for name in columns
        ):
            raise ValueError("Feature groups require lists of nonempty column names")
        if group == "counts" and not set(columns) <= set(COUNTS):
            raise ValueError("Count features must be original chemical counts")
        if any(name in RESERVED or any(token in name.lower() for token in (
            "label", "truth", "split", "prediction", "probability", "y_active",
            "document", "assay", "endpoint", "pchembl",
        )) for name in columns):
            raise ValueError("Outcome, split, identity and metadata features are forbidden")
        names.extend(columns)
    if len(names) != len(set(names)):
        raise ValueError("Feature columns must be unique across groups")
    return names


def validate_frame(frame, config, *, development=False):
    names = validate_config(config)
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Input must be a dataframe")
    required = {*PROPERTY_NAMES, *CONTEXT_NAMES, *names, "record_id", "identity",
                "murcko_scaffold", "fingerprint_hex", "y_active"}
    if development:
        required.add("split")
    if frame.empty or not frame.columns.is_unique or not required <= set(frame):
        raise ValueError("Input must be nonempty with unique columns and all required fields")
    if any(not isinstance(v, str) or not v.strip() for v in frame.identity):
        raise ValueError("Chemical identities must be nonempty strings")
    if frame.identity.duplicated().any() or frame.record_id.duplicated().any():
        raise ValueError("Chemical identities and record_id must be unique")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral)
           for v in frame.record_id):
        raise ValueError("record_id must contain integers")
    if any(not isinstance(v, str) for v in frame.murcko_scaffold):
        raise ValueError("Every molecule requires a known scaffold; empty means acyclic")
    labels = pd.to_numeric(frame.y_active, errors="raise").to_numpy(float)
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be observed binary values; unknown is not inactive")
    raw = frame[[*PROPERTY_NAMES, *CONTEXT_NAMES]].to_numpy(dtype=float)
    if np.isinf(raw).any():
        raise ValueError("Raw features must be finite or missing, never infinite")
    qed = frame.qed.to_numpy(dtype=float)
    if not np.isfinite(qed).all() or not ((qed >= 0) & (qed <= 1)).all():
        raise ValueError("QED must be finite in [0,1]")
    counts = frame[list(COUNTS)].to_numpy(dtype=float)
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts % 1 != 0).any():
        raise ValueError("Filter counts must be finite nonnegative integers")
    if names and not np.isfinite(frame[names].to_numpy(dtype=float)).all():
        raise ValueError("Configured chemical annotations must be finite numeric values")
    fingerprint_matrix(frame)
    if development:
        if set(frame.split) != {"train", "validation"}:
            raise ValueError("Development must contain train and validation only")
        if frame.groupby("murcko_scaffold").split.nunique().gt(1).any():
            raise ValueError("Development scaffold groups must not cross splits")
        for split in ("train", "validation"):
            if set(frame.loc[frame.split.eq(split), "y_active"]) != {0, 1}:
                raise ValueError(f"Both classes required in {split}")


def base_features(frame, preprocessor):
    properties, _ = preprocessor.transform(frame)
    continuous = np.concatenate((properties[:, :16], properties[:, 20:36]), axis=1)
    return np.concatenate((fingerprint_matrix(frame), continuous), axis=1)


def head_columns(config, head):
    if head == "control":
        return []
    if head == "full":
        return [name for group in ("counts", "rules", "alerts")
                for name in config["feature_groups"][group]]
    return list(config["feature_groups"][head])


def head_features(frame, base_logit, columns):
    return np.column_stack((base_logit, frame[columns].to_numpy(dtype=float)))


def fit_logistic(x, y, c):
    started = perf_counter()
    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=4):
        warnings.simplefilter("always", ConvergenceWarning)
        estimator = LogisticRegression(C=c, max_iter=2000, solver="lbfgs",
                                       random_state=0).fit(x, y)
    if any(issubclass(item.category, ConvergenceWarning) for item in captured):
        raise ValueError("Logistic fit did not converge; no silent incomplete model")
    return {"C": c, "weights": estimator.coef_[0].tolist(),
            "intercept": float(estimator.intercept_[0]),
            "iterations": int(estimator.n_iter_[0]), "fit_seconds": perf_counter() - started,
            "coefficient_count": int(x.shape[1])}


def linear_score(x, fitted):
    with threadpool_limits(limits=4):
        return x @ np.asarray(fitted["weights"]) + fitted["intercept"]


def select_regularization(x_train, y_train, x_validation, y_validation, name, seed):
    candidates, grid = [], []
    for c in CS:
        fitted = fit_logistic(x_train, y_train, c)
        ap = float(average_precision_score(y_validation, linear_score(x_validation, fitted)))
        grid.append({"model": name, "seed": seed, "C": c,
                     "average_precision": ap, "fit_seconds": fitted["fit_seconds"]})
        candidates.append((ap, fitted))
    # Iteration order is ascending C, so exact ties select smaller regularization C.
    best = max(range(len(candidates)), key=lambda index: candidates[index][0])
    return candidates[best][1], grid


def crossfit(frame, c, seed):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    if frame.murcko_scaffold.nunique() < 5:
        raise ValueError("Cross-fitting requires at least five training scaffolds")
    y = frame.y_active.to_numpy(int)
    oof, assignments, saved = np.full(len(frame), np.nan), np.full(len(frame), -1), {}
    for fold, (fit_rows, held_rows) in enumerate(splitter.split(frame, y, frame.murcko_scaffold)):
        if set(y[fit_rows]) != {0, 1}:
            raise ValueError("Every OOF training fold requires both classes; no reseeding")
        fit_frame, held_frame = frame.iloc[fit_rows], frame.iloc[held_rows]
        preprocessor = Preprocessor.fit(fit_frame)
        fitted = fit_logistic(base_features(fit_frame, preprocessor), y[fit_rows], c)
        oof[held_rows] = linear_score(base_features(held_frame, preprocessor), fitted)
        assignments[held_rows] = fold
        saved[f"{seed}:{fold}"] = {**fitted, "preprocessor": preprocessor.to_dict(),
                                 "train_rows": len(fit_rows), "held_rows": len(held_rows)}
    if not np.isfinite(oof).all() or (assignments < 0).any():
        raise ValueError("Cross-fitting did not assign every training molecule")
    table = frame[["record_id", "identity", "murcko_scaffold", "y_active"]].assign(
        seed=seed, fold=assignments, base_logit=oof)
    return oof, table, saved


def fit_heads(train_frame, validation, oof, validation_logits, config, seed):
    heads, grids = {}, []
    for name in HEADS:
        columns = head_columns(config, name)
        x = head_features(train_frame, oof, columns)
        xv = head_features(validation, validation_logits, columns)
        scaler = StandardScaler().fit(x)
        fitted, grid = select_regularization(
            scaler.transform(x), train_frame.y_active.to_numpy(int),
            scaler.transform(xv), validation.y_active.to_numpy(int), name, seed)
        if name == "control" and fitted["weights"][0] <= 0:
            raise ValueError("Control base-logit slope is not positive; no silent inversion")
        heads[f"{name}:{seed}"] = {**fitted, "columns": columns,
                                  "mean": scaler.mean_.tolist(),
                                  "scale": scaler.scale_.tolist(),
                                  "base_logit_slope_positive": fitted["weights"][0] > 0}
        grids.extend(grid)
    return heads, grids


def seal(root):
    files = {path.relative_to(root).as_posix(): file_hash(path)
             for path in sorted(root.rglob("*"))
             if path.is_file() and path.name != "completion.json"}
    write_json(root / "completion.json", {"created_utc": datetime.now(UTC).isoformat(),
                                          "files": files})


def train(development, config_path, output):
    development, config_path, output = map(Path, (development, config_path, output))
    frame, config = pd.read_parquet(development), read_json(config_path)
    validate_frame(frame, config, development=True)
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    code = [Path(__file__).resolve(), *sorted((root / "src").rglob("*.py"))]
    protocol = {
        "schema": "s2-decision-dynamic-filter/1", "created_utc": datetime.now(UTC).isoformat(),
        "config": config, "seeds": list(SEEDS), "C_grid": list(CS), "folds": 5,
        "selection_budgets": list(BUDGETS), "selection_quotas": list(QUOTAS),
        "primary": {"arm": "dynamic_full", "comparator": "nofilter_control",
                    "requested_n": 50, "max_per_scaffold": 3},
        "selection": "validation average precision; exact ties smaller C",
        "score_scope": "retrospective ranking; not validated probability or P(advance)",
        "external_usage": "separate evaluation after all models have been sealed",
        "input_hashes": {str(p.resolve()): file_hash(p) for p in (development, config_path)},
        "code_hashes": {str(p.resolve()): file_hash(p) for p in code},
        "versions": {name: version(name) for name in (
            "numpy", "pandas", "scipy", "scikit-learn", "rdkit", "threadpoolctl")},
    }
    write_json(output / "protocol.json", protocol)
    started = perf_counter()
    training = frame.loc[frame.split.eq("train")].reset_index(drop=True)
    validation = frame.loc[frame.split.eq("validation")].reset_index(drop=True)
    preprocessor = Preprocessor.fit(training)
    x, xv = (base_features(part, preprocessor) for part in (training, validation))
    baseline, grid = select_regularization(x, training.y_active, xv, validation.y_active,
                                          "baseline", -1)
    model = {"config": config, "baseline": baseline, "preprocessor": preprocessor.to_dict(),
             "development_identities": frame.identity.tolist(), "heads": {}, "folds": {}}
    oof_tables = []
    validation_logits = linear_score(xv, baseline)
    for seed in SEEDS:
        oof, table, folds = crossfit(training, baseline["C"], seed)
        heads, head_grid = fit_heads(training, validation, oof, validation_logits, config, seed)
        model = {**model, "heads": {**model["heads"], **heads},
                 "folds": {**model["folds"], **folds}}
        oof_tables.append(table)
        grid.extend(head_grid)
    write_json(output / "models.json", model)
    pd.concat(oof_tables, ignore_index=True).to_parquet(output / "oof.parquet", index=False)
    pd.DataFrame(grid).to_parquet(output / "validation-grid.parquet", index=False)
    np.savez_compressed(output / "baseline-validation.npz", logits=validation_logits,
                        identities=validation.identity.to_numpy(dtype=str))
    write_json(output / "training.json", {"fit_seconds": perf_counter() - started,
        "train_molecules": len(training), "validation_molecules": len(validation),
        "fits": 3 + len(SEEDS) * (5 + len(HEADS) * len(CS)),
        "baseline_C": baseline["C"], "promotion": "none; research only"})
    seal(output)
    return model


def load_training(root):
    root = Path(root).resolve()
    files = read_json(root / "completion.json")["files"]
    if not {"protocol.json", "models.json", "oof.parquet", "validation-grid.parquet"} <= set(files):
        raise ValueError("Training completion lacks required files")
    for name, digest in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or file_hash(path) != digest:
            raise ValueError(f"Sealed training artifact changed or unsafe: {name}")
    protocol = read_json(root / "protocol.json")
    for path, digest in {**protocol["input_hashes"], **protocol["code_hashes"]}.items():
        if file_hash(path) != digest:
            raise ValueError(f"Frozen source or input changed: {path}")
    return read_json(root / "models.json")


def predict(frame, model):
    baseline = linear_score(base_features(frame, Preprocessor.from_dict(model["preprocessor"])),
                            model["baseline"])
    predictions, contributions = {}, {}
    for name, head in model["heads"].items():
        raw = head_features(frame, baseline, head["columns"])
        centered = (raw - np.asarray(head["mean"])) / np.asarray(head["scale"])
        terms = centered * np.asarray(head["weights"])
        contributions[name] = pd.DataFrame(
            np.column_stack((np.full(len(frame), head["intercept"]), terms)),
            columns=["intercept", "base_logit", *head["columns"]])
        predictions[name] = linear_score(centered, head)
    return predictions, contributions


def ranking_metrics(y, scores):
    common = {"observed": len(y), "positives": int(np.asarray(y).sum())}
    if len(set(y)) < 2:
        return {"average_precision": None, "roc_auc": None, "status": "one_class", **common}
    return {"average_precision": float(average_precision_score(y, scores)),
            "roc_auc": float(roc_auc_score(y, scores)), "status": "both_classes", **common}


def arms(frame, predictions, seed):
    default = (frame.lipinski_violations <= 1) & (frame.veber_violations == 0)
    rigid = (frame.lipinski_violations == 0) & (frame.veber_violations == 0)
    rigid &= (frame.pains_count == 0) & (frame.brenk_count == 0)
    all_rows = np.ones(len(frame), dtype=bool)
    control = predictions[f"control:{seed}"]
    result = {"product_default": (frame.qed.to_numpy(float), default.to_numpy()),
              "default_activity": (control, default.to_numpy()),
              "rigid_activity": (control, rigid.to_numpy()),
              "nofilter_control": (control, all_rows),
              "qed_nofilter": (frame.qed.to_numpy(float), all_rows)}
    return {**result, **{f"dynamic_{name}": (predictions[f"{name}:{seed}"], all_rows)
                         for name in HEADS if name != "control"}}


def selection_result(frame, score, eligible, n, quota):
    # Monotonic rank scaling avoids saturated sigmoid ties in the native [0,1] API.
    priority = pd.Series(score).rank(method="dense").to_numpy(float)
    priority = priority / max(1.0, float(priority.max()))
    pool = frame.assign(priority_score=priority, eligible=eligible, valid=True)
    selected = select_candidates(pool, n, max_per_scaffold=quota)
    chosen = selected.records.loc[selected.records.is_final]
    hits = int(chosen.y_active.sum())
    total = int(frame.y_active.sum())
    prevalence = total / len(frame)
    precision = hits / len(chosen) if len(chosen) else None
    metrics = {"requested_n": n, "max_per_scaffold": quota,
               "final_count": len(chosen), "shortfall": selected.manifest["shortfall"],
               "excess": selected.manifest["excess"], "hits": hits,
               "hits_per_requested_n": hits / n,
               "precision_selected": precision,
               "recall_common_universe": hits / total if total else None,
               "universe_molecules": len(frame), "universe_prevalence": prevalence,
               "enrichment_selected": precision / prevalence
               if precision is not None and prevalence else None,
               "enrichment_requested_n": hits / n / prevalence if prevalence else None,
               "total_positives": total, "eligible_count": int(np.asarray(eligible).sum()),
               "filter_removed_positives": int(frame.loc[~eligible, "y_active"].sum()),
               "selected_scaffolds": chosen.murcko_scaffold.nunique(),
               "selection_status": selected.manifest["status"]}
    return metrics, selected


def paired_comparisons(baskets):
    rows = []
    for seed in SEEDS:
        for n in BUDGETS:
            for quota in QUOTAS:
                suffix = (seed, n, quota)
                full = baskets[("dynamic_full", *suffix)]
                for comparator in ("nofilter_control", "default_activity"):
                    other = baskets[(comparator, *suffix)]
                    rescued = full.loc[~full.identity.isin(other.identity)]
                    displaced = other.loc[~other.identity.isin(full.identity)]
                    rows.append({"seed": seed, "requested_n": n, "max_per_scaffold": quota,
                        "comparator": comparator,
                        "delta_hits": int(full.y_active.sum() - other.y_active.sum()),
                        "rescued_positives": int(rescued.y_active.sum()),
                        "displaced_positives": int(displaced.y_active.sum()),
                        "changed_selected": len(rescued),
                        "primary": n == 50 and quota == 3 and comparator == "nofilter_control"})
    return pd.DataFrame(rows)


def evaluate(training, external, output):
    training, external, output = map(Path, (training, external, output))
    model = load_training(training)
    frame = pd.read_parquet(external).reset_index(drop=True)
    validate_frame(frame, model["config"])
    if set(frame.identity) & set(model["development_identities"]):
        raise ValueError("External/development identity overlap is forbidden")
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    scores, terms = predict(frame, model)
    scoring_seconds = perf_counter() - started
    score_table = frame[["record_id", "identity", "murcko_scaffold", "y_active"]].copy()
    for name, values in scores.items():
        score_table[name] = values
        terms[name].assign(record_id=frame.record_id.to_numpy(), identity=frame.identity.to_numpy()
                          ).to_parquet(output / f"contributions-{name.replace(':', '-')}.parquet",
                                       index=False)
    score_table.to_parquet(output / "scores.parquet", index=False)
    rankings, metrics, ranking_reports, baskets, selections = [], [], [], {}, []
    for seed in SEEDS:
        for arm, (score, eligible) in arms(frame, scores, seed).items():
            ranked = frame[["record_id", "identity", "murcko_scaffold", "y_active"]].assign(
                seed=seed, arm=arm, raw_score=score, eligible=eligible
            ).sort_values(["raw_score", "record_id"], ascending=[False, True])
            rankings.append(ranked.assign(rank=np.arange(1, len(ranked) + 1)))
            ranking_reports.append({"seed": seed, "arm": arm,
                "universe": "eligible", **ranking_metrics(frame.y_active[eligible], score[eligible])})
            for n in BUDGETS:
                for quota in QUOTAS:
                    result, selected = selection_result(frame, score, eligible, n, quota)
                    metrics.append({"seed": seed, "arm": arm, **result})
                    chosen = selected.records.loc[selected.records.is_final,
                        ["record_id", "identity", "murcko_scaffold", "y_active"]]
                    baskets[(arm, seed, n, quota)] = chosen
                    selections.append(selected.records[["record_id", "identity", "is_final",
                        "selection_reason"]].assign(seed=seed, arm=arm, requested_n=n,
                                                    max_per_scaffold=quota))
    pd.concat(rankings, ignore_index=True).to_parquet(output / "rankings.parquet", index=False)
    pd.DataFrame(ranking_reports).to_parquet(output / "ranking-metrics.parquet", index=False)
    pd.DataFrame(metrics).to_parquet(output / "selection-metrics.parquet", index=False)
    pd.concat(selections, ignore_index=True).to_parquet(output / "selections.parquet", index=False)
    paired_comparisons(baskets).to_parquet(output / "paired-comparisons.parquet", index=False)
    write_json(output / "evaluation.json", {
        "created_utc": datetime.now(UTC).isoformat(), "molecules": len(frame),
        "positives": int(frame.y_active.sum()), "scoring_seconds": scoring_seconds,
        "total_seconds": perf_counter() - started,
        "training_completion_sha256": file_hash(training / "completion.json"),
        "external_path": str(external.resolve()), "external_sha256": file_hash(external),
        "scope": "retrospective assay ranking; no calibration or causal alert effects",
        "uncertainty": "seeds vary OOF assignments; not independent biological replicates",
        "metric_definitions": {
            "enrichment_selected": "precision_selected / common-universe prevalence",
            "enrichment_requested_n": "(hits / requested_n) / common-universe prevalence",
            "recall_common_universe": "hits / all positives before optional filters",
            "zero_positives": "recall and enrichment undefined (null)",
            "ranking_metrics": "AP and ROC use eligible rows; scores for all rows retained",
            "score_transform": "dense rank scaled into [0,1]; exact ties keep record_id order",
        },
    })
    seal(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--development", type=Path, required=True)
    fit.add_argument("--config", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--training", type=Path, required=True)
    evaluate_parser.add_argument("--external", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "train":
        train(args.development, args.config, args.output)
    else:
        evaluate(args.training, args.external, args.output)


if __name__ == "__main__":
    main()
