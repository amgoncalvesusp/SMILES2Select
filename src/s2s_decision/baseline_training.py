"""Fixed sklearn classifiers exported as portable ONNX, with train-only transforms."""

import platform
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from .artifacts import file_hash, write_json
from .metrics import evaluate_predictions
from .preprocessing import Preprocessor
from .schema import SCHEMA_VERSION, fingerprint_matrix


def _validate(frame):
    required = {"record_id", "identity", "split", "y_active"}
    if required - set(frame):
        raise ValueError(f"Missing training columns: {sorted(required - set(frame))}")
    if frame.empty or frame[["record_id", "identity"]].isna().any().any():
        raise ValueError("Training requires nonempty records and identities")
    if frame.record_id.duplicated().any() or frame.identity.duplicated().any():
        raise ValueError("Duplicate record or molecular identity; curate assay replicates first")
    if set(frame.split) != {"train", "validation", "calibration", "test"}:
        raise ValueError("All four splits must contain records")
    labels = pd.to_numeric(frame.y_active, errors="raise")
    if not labels.dropna().isin([0, 1]).all():
        raise ValueError("y_active must be binary or missing")
    for split in ("train", "validation"):
        if set(labels.loc[frame.split.eq(split)].dropna()) != {0, 1}:
            raise ValueError(f"{split} requires both activity classes")


def probability_logits(probabilities: np.ndarray) -> np.ndarray:
    """Finite logits use the same explicit clipping contract as ONNX inference."""
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(clipped) - np.log1p(-clipped)


def _calibrate(probabilities, labels):
    valid = np.isfinite(labels)
    logits, observed = probability_logits(probabilities)[valid], labels[valid]
    if len(observed) < 20 or min(np.sum(observed == 0), np.sum(observed == 1)) < 5:
        return {
            "slope": 1.0,
            "intercept": 0.0,
            "status": "uncalibrated_insufficient_data",
            "count": len(observed),
        }

    def objective(parameters):
        values = parameters[0] * logits + parameters[1]
        return np.mean(np.logaddexp(0, values) - observed * values)

    result = minimize(
        objective, [1.0, 0.0], bounds=((1e-6, 100.0), (-100.0, 100.0)), method="L-BFGS-B"
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Calibration failed: {result.message}")
    return {
        "slope": float(result.x[0]),
        "intercept": float(result.x[1]),
        "status": "fitted_held_out_not_prospectively_validated",
        "count": len(observed),
    }


def _export(estimator, inputs, destination):
    import onnxruntime as ort
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    graph = convert_sklearn(
        estimator,
        name="s2s_decision_classifier",
        initial_types=[("features", FloatTensorType([None, inputs.shape[1]]))],
        options={id(estimator): {"zipmap": False}},
        target_opset=17,
    )
    if isinstance(estimator, HistGradientBoostingClassifier):
        graph = _preserve_histogram_boundaries(graph, estimator)
    model = destination / "model.onnx"
    model.write_bytes(graph.SerializeToString())
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    output_name = session.get_outputs()[1].name
    # Export parity checks only development partitions, never held-out test rows.
    observed = session.run([output_name], {"features": inputs})[0][:, 1]
    expected = estimator.predict_proba(inputs)[:, 1]
    np.testing.assert_allclose(observed, expected, atol=1e-5, rtol=1e-5)
    return output_name, float(np.max(np.abs(observed - expected)))


def _preserve_histogram_boundaries(graph, estimator):
    """Preserve float32 x <= float64 threshold when ONNX stores float32 thresholds."""
    corrected = deepcopy(graph)
    trees = [node for node in corrected.graph.node if node.op_type == "TreeEnsembleClassifier"]
    if len(trees) != 1:
        raise ValueError("Unsupported histogram ONNX tree graph")
    attrs = {attribute.name: attribute for attribute in trees[0].attribute}
    thresholds = list(attrs["nodes_values"].floats)
    for index, mode in enumerate(attrs["nodes_modes"].strings):
        if mode == b"LEAF":
            continue
        if mode != b"BRANCH_LEQ":
            raise ValueError("Unsupported histogram ONNX branch mode")
        tree_id = attrs["nodes_treeids"].ints[index]
        node_id = attrs["nodes_nodeids"].ints[index]
        if not 0 <= tree_id < len(estimator._predictors) or not 0 <= node_id < len(
            estimator._predictors[tree_id][0].nodes
        ):
            raise ValueError("Histogram ONNX tree/node mapping differs from source")
        source = estimator._predictors[tree_id][0].nodes[node_id]
        if source["is_leaf"] or source["is_categorical"]:
            raise ValueError("Unsupported histogram source node")
        if (
            attrs["nodes_featureids"].ints[index] != source["feature_idx"]
            or attrs["nodes_truenodeids"].ints[index] != source["left"]
            or attrs["nodes_falsenodeids"].ints[index] != source["right"]
        ):
            raise ValueError("Histogram ONNX branch mapping differs from source")
        thresholds[index] = _float32_leq_threshold(float(source["num_threshold"]))
    del attrs["nodes_values"].floats[:]
    attrs["nodes_values"].floats.extend(thresholds)
    return corrected


def _float32_leq_threshold(threshold: float) -> float:
    # skl2onnx 1.20 disables this adjustment for HistGradientBoosting.
    # Python float conversion avoids NumPy scalar promotion during comparison.
    rounded = np.float32(threshold)
    if float(rounded) > threshold:
        rounded = np.nextafter(rounded, np.float32(-np.inf))
    return float(rounded)


def train_baseline(
    frame: pd.DataFrame,
    destination: Path,
    provenance: dict,
    estimator: str = "logistic",
    seed: int = 42,
    threads: int = 4,
    input_layout: str | None = None,
) -> dict:
    """Fit fixed parameters on train; validation reports never choose test results."""
    if estimator not in {"logistic", "gradient_boosting"}:
        raise ValueError("Unknown estimator")
    if threads < 1:
        raise ValueError("threads must be positive")
    layout = input_layout or ("scalar_fingerprint" if estimator == "logistic" else "scalar")
    if layout not in {"scalar", "scalar_fingerprint"}:
        raise ValueError("Unknown input layout")
    _validate(frame)
    train = frame.split.eq("train")
    # The classifier does not inspect test features or labels beyond schema validation.
    development = frame.loc[frame.split.ne("test")].copy()
    prep = Preprocessor.fit(frame.loc[train])
    properties, context = prep.transform(development)
    inputs = np.concatenate((properties, context), axis=1)
    if layout == "scalar_fingerprint":
        inputs = np.concatenate(
            (inputs, fingerprint_matrix(development, provenance["fingerprint_bits"])), axis=1
        )
    fit_rows = development.split.eq("train") & development.y_active.notna()
    classifier = (
        LogisticRegression(max_iter=1000, random_state=seed)
        if estimator == "logistic"
        else HistGradientBoostingClassifier(max_iter=100, random_state=seed)
    )
    with threadpool_limits(limits=threads):
        classifier.fit(inputs[fit_rows], development.loc[fit_rows, "y_active"])
        probabilities = classifier.predict_proba(inputs)[:, 1]
        calibrating = development.split.eq("calibration")
        calibrator = _calibrate(
            probabilities[calibrating], development.loc[calibrating, "y_active"].to_numpy(float)
        )
        validation = development.split.eq("validation")
        validation_metrics = evaluate_predictions(
            development.loc[validation, "y_active"], probabilities[validation]
        )
        output_name, parity = _export(classifier, inputs, destination)
    manifest = {
        "schema": SCHEMA_VERSION,
        "format_version": 1,
        "estimator": estimator,
        "input_layout": layout,
        "input_width": inputs.shape[1],
        "output_format": "binary_probabilities",
        "probability_output": output_name,
        "class_labels": [0, 1],
        "probability_clip": 1e-7,
        "fingerprint_bits": provenance["fingerprint_bits"],
        "provenance": provenance,
        "preprocessing": prep.to_dict(),
        "calibrator": calibrator,
        "regression_supported": False,
        "regression_mean": 0.0,
        "regression_scale": 1.0,
        "onnx_sha256": file_hash(destination / "model.onnx"),
        "onnx_parity_max_abs": parity,
        "export_threshold_policy": "float32_floor_preserves_leq_v1"
        if estimator == "gradient_boosting"
        else None,
        "validation_metrics": validation_metrics,
        "validation_pr_auc": validation_metrics["pr_auc"],
        "validation_score_kind": "uncalibrated_probability_fixed_estimator",
        "heldout_test_evaluated": False,
        "automatic_model_selection": False,
        "fit_count": int(fit_rows.sum()),
        "training_reference_count": int(train.sum()),
        "config": {"seed": seed, "estimator_parameters": classifier.get_params()},
        "versions": {
            "sklearn": sklearn_version,
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
        "meaning": "Activity for the target, endpoint and threshold in provenance; never P(advance).",
    }
    write_json(destination / "manifest.json", manifest)
    return manifest
