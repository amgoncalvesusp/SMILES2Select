"""Offline inference imports ONNX Runtime, never PyTorch."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .preprocessing import Preprocessor
from .schema import SCHEMA_VERSION, fingerprint_matrix


def validate_model_manifest(manifest: dict) -> Preprocessor:
    """Validate the shared runtime contract without loading ONNX or training libraries."""
    if manifest.get("schema") != SCHEMA_VERSION or manifest.get("format_version") != 1:
        raise ValueError("Incompatible model schema")
    prep = Preprocessor.from_dict(manifest["preprocessing"])
    estimator = manifest.get("estimator", "tiny")
    layout = manifest.get("input_layout", "tiny_branches")
    if estimator not in {"tiny", "logistic", "gradient_boosting"}:
        raise ValueError("Unsupported estimator")
    baseline = estimator != "tiny"
    if (baseline and layout not in {"scalar", "scalar_fingerprint"}) or (
        not baseline and layout != "tiny_branches"
    ):
        raise ValueError("Unsupported input layout")
    if baseline and (
        manifest.get("output_format") != "binary_probabilities"
        or manifest.get("class_labels") != [0, 1]
        or manifest.get("probability_clip") != 1e-7
    ):
        raise ValueError("Incompatible classifier output schema")
    if manifest.get("fingerprint_bits") not in (1024, 2048):
        raise ValueError("Unsupported fingerprint dimensions")
    if baseline:
        width = 54 + (manifest["fingerprint_bits"] if layout == "scalar_fingerprint" else 0)
        if manifest.get("input_width") != width:
            raise ValueError("Incompatible classifier input width")
        if manifest.get("regression_supported") is not False:
            raise ValueError("Classifier regression is unsupported")
        if (
            not isinstance(manifest.get("probability_output"), str)
            or not manifest["probability_output"]
        ):
            raise ValueError("Missing classifier output name")
    calibrator = manifest["calibrator"]
    slope, intercept = float(calibrator["slope"]), float(calibrator["intercept"])
    if not np.isfinite([slope, intercept]).all() or slope <= 0 or not calibrator.get("status"):
        raise ValueError("Invalid calibrator")
    if (
        not np.isfinite([manifest["regression_scale"], manifest["regression_mean"]]).all()
        or manifest["regression_scale"] <= 0
    ):
        raise ValueError("Invalid regression transform")
    return prep


def predict(frame: pd.DataFrame, model_dir: str | Path, batch_size: int = 1024) -> pd.DataFrame:
    import onnxruntime as ort

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    folder = Path(model_dir)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    prep = validate_model_manifest(manifest)
    model_path = folder / "model.onnx"
    model_bytes = model_path.read_bytes()
    if hashlib.sha256(model_bytes).hexdigest() != manifest["onnx_sha256"]:
        raise ValueError("Model checksum mismatch")
    if (
        "record_id" not in frame
        or frame.record_id.isna().any()
        or frame.record_id.duplicated().any()
    ):
        raise ValueError("Inference requires unique record_id")
    baseline = manifest.get("estimator", "tiny") != "tiny"
    layout = manifest.get("input_layout", "tiny_branches")
    session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    outputs = []
    for start in range(0, len(frame), batch_size):
        batch = frame.iloc[start : start + batch_size]
        properties, context = prep.transform(batch)
        if baseline:
            inputs = np.concatenate((properties, context), axis=1)
            if layout == "scalar_fingerprint":
                inputs = np.concatenate(
                    (inputs, fingerprint_matrix(batch, manifest["fingerprint_bits"])), axis=1
                )
            if inputs.shape[1] != manifest.get("input_width"):
                raise ValueError("Incompatible classifier input width")
            probabilities = session.run([manifest["probability_output"]], {"features": inputs})[0]
            if probabilities.shape != (len(batch), 2) or not np.isfinite(probabilities).all():
                raise ValueError("Invalid classifier output")
            if (probabilities < -1e-6).any() or (probabilities > 1 + 1e-6).any():
                raise ValueError("Classifier probabilities outside unit interval")
            clipped = np.clip(probabilities[:, 1].astype(np.float64), 1e-7, 1 - 1e-7)
            outputs.append(
                np.column_stack((np.log(clipped) - np.log1p(-clipped), np.zeros(len(batch))))
            )
            continue
        outputs.append(
            session.run(
                None,
                {
                    "properties": properties,
                    "context": context,
                    "fingerprint": fingerprint_matrix(batch, manifest["fingerprint_bits"]),
                },
            )[0]
        )
    raw = np.concatenate(outputs) if outputs else np.empty((0, 2), np.float32)
    if raw.shape != (len(frame), 2) or not np.isfinite(raw).all():
        raise ValueError("Invalid or nonfinite model output")
    calibrator = manifest["calibrator"]
    slope, intercept = float(calibrator["slope"]), float(calibrator["intercept"])
    logits = slope * raw[:, 0].astype(np.float64) + intercept
    probability = np.exp(-np.logaddexp(0.0, -logits))
    activity = raw[:, 1] * manifest["regression_scale"] + manifest["regression_mean"]
    if not manifest["regression_supported"]:
        activity[:] = np.nan
    return pd.DataFrame(
        {
            "record_id": frame.record_id.to_numpy(),
            "activity_probability": probability,
            "predicted_pactivity": activity,
            "priority_score": probability,
            "calibration_status": calibrator["status"],
        }
    )
