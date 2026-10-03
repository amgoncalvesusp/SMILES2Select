"""B15 train-only estimators and independent probability/conformal calibration."""

import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from s2s_decision.contextual_model import (
    feature_matrix,
    fit_spec,
    prediction_sets,
)
from s2s_decision.schema import DESCRIPTOR_NAMES, PROPERTY_NAMES

CS = (.1, 1., 10.)
SEEDS = (42, 43, 44)
GATE = {"support_n": 20, "positive_n": 5, "negative_n": 5,
        "scaffolds": 5, "documents": 3}
VARIANTS = ("lr_base", "lr_counts", "lr_rules", "lr_alerts", "lr_full", "lr_maccs")


def validate_panel(frame):
    needed = {"identity", "task_id", "split", "y_active", "murcko_scaffold",
              "fingerprint_hex", "model_smiles", "assay_context", *PROPERTY_NAMES}
    if not isinstance(frame, pd.DataFrame) or frame.empty or not needed <= set(frame):
        raise ValueError("Complete nonempty measured panel required")
    if frame.duplicated(["identity", "task_id"]).any():
        raise ValueError("Identity/task outcomes must be unique")
    if not frame.y_active.isin([0, 1]).all():
        raise ValueError("Unknown labels cannot become inactive")
    if not set(frame.split) <= {"train", "validation", "calibration", "test", "recent"}:
        raise ValueError("Unknown panel split")
    if frame.groupby("identity").split.nunique().gt(1).any():
        raise ValueError("Identity split leakage across targets")
    internal = frame.loc[frame.split.ne("recent")]
    if internal.groupby("murcko_scaffold").split.nunique().gt(1).any():
        raise ValueError("Scaffold split leakage across targets")
    for _, task in frame.groupby("task_id"):
        for split in ("train", "validation", "calibration"):
            if set(task.loc[task.split.eq(split), "y_active"]) != {0, 1}:
                raise ValueError(f"Both classes required in {split}")


def support_features(frame):
    train = frame.loc[frame.split.eq("train")]
    if train.empty or not train.y_active.isin([0, 1]).all():
        raise ValueError("Support needs observed binary training rows")
    groups = {"counts": list(PROPERTY_NAMES[-4:]), "rules": [], "alerts": []}
    rows = []
    candidates = sorted(name for name in frame if name.startswith("alert__") or
                        (name.startswith("rule__") and name.endswith("__violation")))
    for name in candidates:
        flagged = train.loc[train[name].astype(bool)]
        doc_column = "source_document_ids" if "source_document_ids" in flagged else "document_ids"
        documents = set()
        for value in flagged[doc_column]:
            documents.update(json.loads(value) if isinstance(value, str) else value)
        stats = {"feature_id": name, "support_n": len(flagged),
                 "positive_n": int(flagged.y_active.sum()),
                 "negative_n": int(flagged.y_active.eq(0).sum()),
                 "scaffolds": flagged.murcko_scaffold.replace("", np.nan).nunique(),
                 "documents": len(documents)}
        admitted = all(stats[key] >= lower for key, lower in GATE.items())
        rows.append({**stats, "admitted": admitted})
        if admitted and name.startswith("alert__"):
            groups["alerts"].append(name)
        elif admitted:
            groups["rules"].extend([name, name.replace("__violation", "__normalized_excess")])
    return groups, rows


def columns_for(variant, groups):
    chemical = []
    for group in ("counts", "rules", "alerts"):
        if variant in (f"lr_{group}", "lr_full", "boosting", "shared"):
            chemical.extend(groups[group])
    return [*DESCRIPTOR_NAMES, *chemical]


def calibration_partition(frame, seed):
    # Label-blind group hash; no retry until favorable class balance.
    return np.array(["probability" if int(hashlib.sha256(
        f"{seed}:{scaffold}".encode()).hexdigest()[:8], 16) % 2 == 0 else "conformal"
        for scaffold in frame.murcko_scaffold])


def _binary(y):
    values = np.asarray(y, float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
        raise ValueError("Observed binary labels required")
    return values.astype(int)


def fit_calibrator(logits, y):
    y, x = _binary(y), np.asarray(logits, float)
    if x.shape != y.shape or not np.isfinite(x).all():
        raise ValueError("Finite aligned calibration logits required")
    counts = [int((y == label).sum()) for label in (0, 1)]
    if min(counts) < 5:
        raise ValueError("Probability calibration requires at least five observations per class")
    def objective(params):
        z = params[0] * x + params[1]
        return np.mean(np.logaddexp(0, z) - y * z) + 1e-4 * params[0] ** 2
    result = minimize(objective, [1., 0.], method="L-BFGS-B", bounds=[(0., None), (None, None)])
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f"Sigmoid calibration failed: {result.message}")
    return {"slope": float(result.x[0]), "intercept": float(result.x[1]),
            "n": len(y), "negative_n": counts[0], "positive_n": counts[1],
            "status": "fitted", "method": "monotone sigmoid; L2 slope penalty 0.0001"}


def fit_conformal(y, probabilities, alpha=.1):
    y, p = _binary(y), np.asarray(probabilities, float)
    if p.shape != y.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or not 0 < alpha < 1:
        raise ValueError("Invalid conformal probabilities/alpha")
    quantiles, counts = [], []
    for label in (0, 1):
        errors = np.sort(p[y == label] if label == 0 else 1 - p[y == label])
        counts.append(len(errors))
        rank = int(np.ceil((len(errors) + 1) * (1 - alpha)))
        quantiles.append(float(errors[rank - 1]) if rank <= len(errors) else 1.)
    return {"alpha": alpha, "quantiles": quantiles, "counts": counts,
            "status": "fitted" if min(counts) else "insufficient_class_support",
            "assumption": "label-conditional exchangeability; shift coverage is empirical only"}


def probability_metrics(y, probabilities, conformal):
    y, p = _binary(y), np.asarray(probabilities, float)
    sets = prediction_sets(p, conformal["quantiles"])
    covered = np.array([label in values for label, values in zip(y, sets)])
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean()
              for b in range(10) if (bins == b).any())
    return {"brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "ece": float(ece), "coverage": float(covered.mean()),
            "coverage_negative": float(covered[y == 0].mean()) if (y == 0).any() else None,
            "coverage_positive": float(covered[y == 1].mean()) if (y == 1).any() else None,
            "mean_set_size": float(np.mean([len(s) for s in sets])),
            "empty_fraction": float(np.mean([not s for s in sets])),
            "singleton_fraction": float(np.mean([len(s) == 1 for s in sets])),
            "both_fraction": float(np.mean([len(s) == 2 for s in sets]))}


def ranking_metrics(y, score):
    y = _binary(y)
    return {"n": len(y), "positive_n": int(y.sum()),
            "ap": float(average_precision_score(y, score)) if len(set(y)) == 2 else None,
            "roc_auc": float(roc_auc_score(y, score)) if len(set(y)) == 2 else None}


def fit_logistic(x, y, c, sample_weight=None):
    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=4):
        warnings.simplefilter("always", ConvergenceWarning)
        fitted = LogisticRegression(C=c, max_iter=3000, solver="lbfgs", random_state=0).fit(
            x, _binary(y), sample_weight=sample_weight)
    if any(issubclass(w.category, ConvergenceWarning) for w in captured):
        raise ValueError("Logistic fit failed convergence")
    return {"family": "logistic", "C": c, "weights": fitted.coef_[0].tolist(),
            "intercept": float(fitted.intercept_[0]), "iterations": int(fitted.n_iter_[0])}


def fit_boosting(x, y, depth, seed):
    fitted = GradientBoostingClassifier(n_estimators=100, learning_rate=.05, max_depth=depth,
        min_samples_leaf=10, subsample=.8, random_state=seed).fit(x, _binary(y))
    trees = [{"left": tree.tree_.children_left.tolist(), "right": tree.tree_.children_right.tolist(),
              "feature": tree.tree_.feature.tolist(), "threshold": tree.tree_.threshold.tolist(),
              "value": tree.tree_.value[:, 0, 0].tolist()} for tree in fitted.estimators_[:, 0]]
    prior = fitted.init_.class_prior_[1]
    return fitted, {"family": "boosting", "depth": depth, "seed": seed, "trees": trees,
                    "learning_rate": .05, "intercept": float(np.log(prior / (1 - prior)))}


def predict_estimator(x, model):
    if model["family"] == "logistic":
        with threadpool_limits(limits=4):
            return x @ np.asarray(model["weights"]) + model["intercept"]
    if model["family"] != "boosting":
        raise ValueError("Unknown estimator family")
    # sklearn tree inference casts features to float32 before threshold comparisons.
    x = np.asarray(x, dtype=np.float32)
    scores = np.full(len(x), model["intercept"])
    for tree in model["trees"]:
        nodes = np.zeros(len(x), dtype=int)
        left, right = np.asarray(tree["left"]), np.asarray(tree["right"])
        features, thresholds = np.asarray(tree["feature"]), np.asarray(tree["threshold"])
        while True:
            active = np.flatnonzero(left[nodes] >= 0)
            if not len(active):
                break
            current = nodes[active]
            nodes[active] = np.where(x[active, features[current]] <= thresholds[current],
                                     left[current], right[current])
        scores += model["learning_rate"] * np.asarray(tree["value"])[nodes]
    return scores


def select_logistic(x, y, xv, yv, tasks=None):
    models, rows = [], []
    weight = None
    if tasks is not None:
        train_tasks, valid_tasks = tasks
        counts = pd.Series(train_tasks).value_counts()
        weight = np.array([len(train_tasks) / len(counts) / counts[t] for t in train_tasks])
    for c in CS:
        model = fit_logistic(x, y, c, weight)
        scores = predict_estimator(xv, model)
        ap = float(average_precision_score(yv, scores)) if tasks is None else float(np.mean([
            average_precision_score(np.asarray(yv)[valid_tasks == task], scores[valid_tasks == task])
            for task in sorted(set(valid_tasks))]))
        models.append(model)
        rows.append({"C": c, "validation_ap": ap})
    best = max(range(len(models)), key=lambda i: rows[i]["validation_ap"])
    return models[best], rows


def shared_spec(train, groups_by_task, context):
    tasks = sorted(groups_by_task)
    chemical = sorted({name for groups in groups_by_task.values()
                       for names in groups.values() for name in names})
    return {"numeric": fit_spec(train, [*DESCRIPTOR_NAMES, *chemical]), "chemical": chemical,
            "tasks": tasks, "groups": groups_by_task,
            "contexts": sorted(train.assay_context.unique()) if context else []}


def shared_matrix(frame, spec):
    base = feature_matrix(frame, spec["numeric"])
    chemical = frame[spec["chemical"]].to_numpy(float)
    # Train scale for interaction magnitudes; unsupported target patterns remain zero.
    offset = len(DESCRIPTOR_NAMES)
    scale = np.asarray(spec["numeric"]["scale"])[offset:]
    chem = chemical / scale
    blocks = [base]
    for task in spec["tasks"]:
        mask = frame.task_id.eq(task).to_numpy(float)[:, None]
        allowed = {name for group in spec["groups"][task].values() for name in group}
        supported = np.array([name in allowed for name in spec["chemical"]])
        blocks.extend((mask, chem * mask * supported))
    for context in spec["contexts"]:
        mask = frame.assay_context.eq(context).to_numpy(float)[:, None]
        blocks.extend((mask, chem * mask))
    return np.column_stack(blocks)


def calibrate_model(frame, logits, seed):
    parts = calibration_partition(frame, seed)
    prob = parts == "probability"
    calibration = fit_calibrator(np.asarray(logits)[prob], frame.y_active.to_numpy()[prob])
    predicted = expit(calibration["slope"] * np.asarray(logits)[~prob] + calibration["intercept"])
    conformal = fit_conformal(frame.y_active.to_numpy()[~prob], predicted)
    identities = {name: frame.loc[parts == name, "identity"].tolist()
                  for name in ("probability", "conformal")}
    return {"calibration": calibration, "conformal": conformal,
            "calibration_identities": identities, "calibration_seed": seed}
