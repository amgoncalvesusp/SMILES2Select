"""B15 measured-endpoint risk benchmark; unknown never means safe.

Each invocation fits one explicitly named assay outcome. Activity-evaluation
scaffolds can be reserved in their entirety for a separate cross-endpoint audit.
"""

import argparse
import hashlib
import json
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from s2s_decision.artifacts import file_hash, write_json
from s2s_decision.features import featurize
from s2s_decision.risk_model import predict_risk
from s2s_decision.schema import fingerprint_matrix

SPLIT_NAMES = ("train", "validation", "probability_calibration", "conformal_calibration", "test")
SPLIT_BOUNDS = (0.50, 0.65, 0.75, 0.85, 1.0)
CS = (0.1, 1.0, 10.0)


def split_for(scaffold):
    """Label-independent, deterministic scaffold assignment; no reseeding."""
    digest = hashlib.sha256(("B15-risk-split42:" + scaffold).encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    return SPLIT_NAMES[next(i for i, bound in enumerate(SPLIT_BOUNDS) if fraction < bound)]


def prepare(frame, endpoint):
    required = {"identity", "murcko_scaffold", "fingerprint_hex", "y_risk", "risk_endpoint"}
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("A named measured risk endpoint is required")
    if frame.empty or not frame.columns.is_unique or not required <= set(frame):
        raise ValueError("Risk input requires nonempty uniquely named evidence columns")
    if (frame.identity.duplicated().any() or any(
            not isinstance(v, str) or not v.strip() for v in frame.identity)):
        raise ValueError("Risk identity must be nonempty and unique")
    if any(not isinstance(v, str) for v in frame.murcko_scaffold):
        raise ValueError("Known scaffold required; empty scaffold is an acyclic group")
    if not frame.risk_endpoint.eq(endpoint).all():
        raise ValueError("Risk endpoint mismatch; distinct measurements cannot be pooled")
    labels = pd.to_numeric(frame.y_risk, errors="raise").to_numpy(dtype=float)
    if not np.all(np.isnan(labels) | np.isin(labels, [0, 1])):
        raise ValueError("Risk labels must be measured 0/1 or unknown NaN")
    fingerprint_matrix(frame)
    split = frame.murcko_scaffold.map(split_for)
    if "evaluation_only" in frame:
        if any(not isinstance(v, (bool, np.bool_)) for v in frame.evaluation_only):
            raise ValueError("evaluation_only must be Boolean")
        reserved = set(frame.loc[frame.evaluation_only, "murcko_scaffold"])
        split = split.where(~frame.murcko_scaffold.isin(reserved), "external_activity_overlap")
    if "split" in frame and not frame.split.eq(split).all():
        raise ValueError("Assigned split differs from fixed scaffold protocol")
    return frame.assign(y_risk=labels, split=split).sort_values("identity").reset_index(drop=True)


def _fit_lr(x, labels, c):
    with warnings.catch_warnings(), threadpool_limits(limits=4):
        warnings.simplefilter("error", ConvergenceWarning)
        model = LogisticRegression(C=c, solver="lbfgs", max_iter=2000, random_state=42)
        model.fit(x, labels)
    return {"C": c, "weights": model.coef_[0].tolist(), "intercept": float(model.intercept_[0])}


def _logits(x, fitted):
    with threadpool_limits(limits=4):
        return x @ np.asarray(fitted["weights"]) + fitted["intercept"]


def _calibrate(logits, labels):
    """Monotone sigmoid calibration; zero slope explicitly means no discrimination."""
    def objective(parameters):
        z = logits * parameters[0] + parameters[1]
        residual = expit(z) - labels
        loss = np.mean(np.logaddexp(0, z) - labels * z) + 1e-6 * parameters[0]**2
        gradient = np.array([np.mean(residual * logits) + 2e-6 * parameters[0],
                             np.mean(residual)])
        return loss, gradient

    fit = minimize(objective, [1.0, 0.0], jac=True, method="L-BFGS-B",
                   bounds=[(0, None), (None, None)])
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError(f"Risk probability calibration failed: {fit.message}")
    return {"method": "monotone_sigmoid", "slope": float(fit.x[0]),
            "intercept": float(fit.x[1]), "slope_ridge": 1e-6}


def conformal_thresholds(probabilities, labels, alpha=0.1):
    thresholds = []
    for label in (0, 1):
        selected = probabilities[labels == label]
        scores = selected if label == 0 else 1-selected
        rank = int(np.ceil((len(scores)+1)*(1-alpha)))
        thresholds.append(float(np.sort(scores)[rank-1]) if rank <= len(scores) else 1.0)
    return thresholds


def prediction_sets(probabilities, thresholds):
    return np.column_stack((probabilities <= thresholds[0], 1-probabilities <= thresholds[1]))


def fit(frame, endpoint):
    frame = prepare(frame, endpoint)
    x, y = fingerprint_matrix(frame), frame.y_risk.to_numpy(float)
    masks = {name: frame.split.eq(name).to_numpy() & np.isfinite(y) for name in SPLIT_NAMES}
    for name, minimum in (("train", 10), ("validation", 1), ("probability_calibration", 5),
                          ("conformal_calibration", 1)):
        counts = [int((y[masks[name]] == label).sum()) for label in (0, 1)]
        if min(counts) < minimum:
            raise ValueError(f"Insufficient measured class support in {name}: {counts}")
    grid, models = [], []
    for c in CS:
        model = _fit_lr(x[masks["train"]], y[masks["train"]], c)
        ap = float(average_precision_score(y[masks["validation"]],
                                         _logits(x[masks["validation"]], model)))
        models.append(model)
        grid.append({"C": c, "validation_ap": ap})
    best = max(range(len(grid)), key=lambda i: grid[i]["validation_ap"])
    predictor = models[best]
    logits = _logits(x, predictor)
    pcal = masks["probability_calibration"]
    calibrator = _calibrate(logits[pcal], y[pcal])
    probabilities = expit(logits*calibrator["slope"]+calibrator["intercept"])
    ccal = masks["conformal_calibration"]
    thresholds = conformal_thresholds(probabilities[ccal], y[ccal])
    names = (*SPLIT_NAMES, "external_activity_overlap")
    return {"schema": "s2-decision-risk-benchmark/1", "risk_scope": endpoint,
            "features": "Morgan radius2 2048bits no chirality; frozen source chemistry",
            "predictor": predictor, "calibrator": calibrator, "validation_grid": grid,
            "conformal_alpha": 0.1, "conformal_thresholds": thresholds,
            "calibration_class_support": [int((y[pcal] == label).sum()) for label in (0, 1)],
            "nominal_coverage_is_shift_guarantee": False,
            "partition_identities": {name: frame.loc[frame.split.eq(name), "identity"].tolist()
                                     for name in names},
            "conformal_class_support": [int((y[ccal] == label).sum()) for label in (0, 1)],
            "unknown_labels": int(np.isnan(y).sum()),
            "inference_scope": "Risk for this measured endpoint only; not general safety"}


def predict(frame, bundle):
    return predict_risk(frame, bundle)


def verify_input_chemistry(frame):
    """Recompute every identity/fingerprint before claiming runtime chemistry parity."""
    if "original_smiles" not in frame:
        return {"chemistry_hash": None, "status": "unverified_no_source_smiles"}
    computed = featurize(frame[["original_smiles"]].reset_index(drop=True))
    for field in ("identity", "fingerprint_hex", "murcko_scaffold"):
        if not np.array_equal(computed.records[field].to_numpy(), frame[field].to_numpy()):
            raise ValueError(f"Risk input chemistry parity failed: {field}")
    return {**computed.manifest, "status": "all_rows_recomputed", "verified_rows": len(frame)}


def metrics(labels, probabilities, thresholds):
    measured = np.isfinite(labels)
    y, p = labels[measured].astype(int), probabilities[measured]
    if not len(y):
        return {"measured": 0, "unknown": int((~measured).sum()), "status": "no_measured_outcomes"}
    sets = prediction_sets(p, thresholds)
    both = len(np.unique(y)) == 2
    bins = np.minimum((p*10).astype(int), 9)
    ece = sum(np.mean(bins == b)*abs(np.mean(p[bins == b])-np.mean(y[bins == b]))
              for b in range(10) if (bins == b).any())
    return {"measured": len(y), "unknown": int((~measured).sum()), "positive": int(y.sum()),
            "average_precision": float(average_precision_score(y, p)) if both else None,
            "roc_auc": float(roc_auc_score(y, p)) if both else None,
            "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "ece_10_equal_width": float(ece),
            "conformal_coverage": float(sets[np.arange(len(y)), y].mean()),
            "conformal_mean_set_size": float(sets.sum(axis=1).mean()),
            "conformal_empty": int((sets.sum(axis=1) == 0).sum()),
            "conformal_ambiguous": int((sets.sum(axis=1) == 2).sum()),
            "class_coverage": {str(c): float(sets[y == c, c].mean()) if (y == c).any() else None
                               for c in (0, 1)}}


def run(input_path, endpoint, output):
    input_path, output = Path(input_path), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = {"created_utc": datetime.now(UTC).isoformat(), "endpoint": endpoint,
              "input": str(input_path.resolve()), "input_sha256": file_hash(input_path),
              "code_sha256": file_hash(__file__), "C_grid": CS,
              "split_names": SPLIT_NAMES, "split_bounds": SPLIT_BOUNDS,
              "split_key": "B15-risk-split42:", "alpha": 0.1,
              "reserved": "Entire scaffold if any row evaluation_only=True"}
    write_json(output / "fit-freeze.json", config)
    frame = prepare(pd.read_parquet(input_path), endpoint)
    fitted = fit(frame, endpoint)
    chemistry = verify_input_chemistry(frame)
    bundle = {**fitted, "data_source": config["input"],
              "data_sha256": config["input_sha256"], "source_version": "B15 measured endpoint source",
              "chemistry_version": chemistry["chemistry_hash"], "chemistry": chemistry}
    write_json(output / "model.json", bundle)
    frame.to_parquet(output / "partitions.parquet", index=False)
    probabilities = predict(frame, bundle)
    sets = prediction_sets(probabilities, bundle["conformal_thresholds"])
    scores = frame[["identity", "murcko_scaffold", "y_risk", "risk_endpoint", "split"]].assign(
        risk_probability=probabilities, includes_no_risk=sets[:, 0], includes_risk=sets[:, 1])
    scores.to_parquet(output / "scores.parquet", index=False)
    for name in ("test", "external_activity_overlap"):
        subset = scores.loc[scores.split.eq(name)]
        result = metrics(subset.y_risk.to_numpy(float), subset.risk_probability.to_numpy(),
                         bundle["conformal_thresholds"])
        write_json(output / f"{name}-metrics.json", result)
    if file_hash(input_path) != config["input_sha256"]:
        raise ValueError("Risk input changed during fitting")
    completion = {"status": "complete", "scope": endpoint,
                  "files": {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()}}
    write_json(output / "completion.json", completion)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.endpoint, args.output)))


if __name__ == "__main__":
    main()
