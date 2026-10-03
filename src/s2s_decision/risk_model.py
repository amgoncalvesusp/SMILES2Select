"""Portable inference for explicitly measured risk endpoints, with JSON parameters."""

import json
from pathlib import Path

import numpy as np

from .schema import fingerprint_matrix

SCHEMA = "s2-decision-risk-benchmark/1"


def _parameters(bundle):
    if not isinstance(bundle, dict) or bundle.get("schema") != SCHEMA:
        raise ValueError("Unsupported risk bundle schema")
    if not isinstance(bundle.get("risk_scope"), str) or not bundle["risk_scope"].strip():
        raise ValueError("Measured risk endpoint is required")
    try:
        predictor, calibrator = bundle["predictor"], bundle["calibrator"]
        weights = np.asarray(predictor["weights"], dtype=float)
        scalars = np.asarray([predictor["intercept"], calibrator["slope"],
                              calibrator["intercept"]], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Malformed risk predictor parameters") from exc
    if weights.shape != (2048,) or not np.isfinite(weights).all() or not np.isfinite(scalars).all():
        raise ValueError("Malformed risk predictor parameters")
    if scalars[1] < 0:
        raise ValueError("Risk calibration must preserve ranking")
    return weights, scalars


def load_risk_bundle(path):
    """Load bounded data-only JSON; never deserialize executable model objects."""
    path = Path(path)
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Risk bundle exceeds 16 MiB")
    bundle = json.loads(path.read_text(encoding="utf-8"))
    _parameters(bundle)
    try:
        counts = np.asarray(bundle["calibration_class_support"], dtype=float)
        thresholds = np.asarray(bundle["conformal_thresholds"], dtype=float)
        alpha = float(bundle["conformal_alpha"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Missing risk calibration evidence") from exc
    if (counts.shape != (2,) or not np.isfinite(counts).all() or (counts < 5).any()
            or (counts % 1 != 0).any()):
        raise ValueError("Risk probability calibration requires five measured outcomes per class")
    if (thresholds.shape != (2,) or not np.isfinite(thresholds).all()
            or (thresholds < 0).any() or (thresholds > 1).any() or not 0 < alpha < 1):
        raise ValueError("Invalid risk conformal settings")
    return bundle


def predict_risk(frame, bundle):
    """Return endpoint-specific adverse-outcome scores, never general safety."""
    weights, scalars = _parameters(bundle)
    if "risk_endpoint" in frame and not frame.risk_endpoint.eq(bundle["risk_scope"]).all():
        raise ValueError("Prediction endpoint mismatch")
    logits = (fingerprint_matrix(frame) @ weights + scalars[0]) * scalars[1] + scalars[2]
    if not np.isfinite(logits).all():
        raise ValueError("Risk inference produced nonfinite logits")
    return np.exp(-np.logaddexp(0, -logits))


def risk_prediction_sets(probabilities, bundle):
    """Classes: 0 no adverse assay call; 1 adverse assay call for named endpoint."""
    values = np.asarray(probabilities, dtype=float)
    thresholds = np.asarray(bundle.get("conformal_thresholds"), dtype=float)
    if (values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any()
            or (values > 1).any() or thresholds.shape != (2,)
            or not np.isfinite(thresholds).all() or (thresholds < 0).any() or (thresholds > 1).any()):
        raise ValueError("Risk prediction sets require valid probabilities and thresholds")
    return [[c for c, keep in enumerate((p <= thresholds[0], 1-p <= thresholds[1])) if keep]
            for p in values]
