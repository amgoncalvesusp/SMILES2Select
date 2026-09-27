"""Executable classifier packages preserve held-out partitions and runtime isolation."""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from s2s_decision.artifacts import read_bundle, write_bundle
from s2s_decision.features import chemistry_manifest
from s2s_decision.schema import CONTEXT_NAMES, PROPERTY_NAMES, FeatureSet


def dataset(tmp_path):
    rng = np.random.default_rng(9)
    n = 100
    frame = pd.DataFrame(
        {name: rng.uniform(0.1, 1, n) for name in (*PROPERTY_NAMES, *CONTEXT_NAMES)}
    )
    frame = frame.assign(
        record_id=[f"r{i}" for i in range(n)],
        identity=[f"i{i}" for i in range(n)],
        fingerprint_hex=[
            rng.integers(0, 256, 128, dtype=np.uint8).tobytes().hex() for _ in range(n)
        ],
        murcko_scaffold=[f"s{i}" for i in range(n)],
        split=["train"] * 50 + ["validation"] * 12 + ["calibration"] * 26 + ["test"] * 12,
        y_active=np.arange(n) % 2,
        pactivity=5 + np.arange(n) % 2,
    )
    frame["mol_wt"] = frame.y_active * 100 + rng.uniform(100, 120, n)
    path = tmp_path / "dataset"
    write_bundle(
        FeatureSet(
            frame,
            {
                **chemistry_manifest(1024),
                "target_id": "fixture",
                "endpoint": "IC50",
                "threshold": 6.0,
                "context_method": "fixture",
            },
        ),
        path,
    )
    return frame, path


@pytest.mark.parametrize(
    "estimator,layout",
    [("logistic", None), ("gradient_boosting", None), ("gradient_boosting", "scalar_fingerprint")],
)
def test_classifier_package_parity_calibration_and_references(tmp_path, estimator, layout):
    from s2s_decision.inference import predict
    from s2s_decision.workflows import train_baseline_dataset

    frame, source = dataset(tmp_path)
    folder = tmp_path / "model"
    report = train_baseline_dataset(source, folder, estimator=estimator, input_layout=layout)
    manifest = json.loads((folder / "manifest.json").read_text())
    assert report["heldout_test_evaluated"] is False
    assert manifest["runtime_ready"] and manifest["estimator"] == estimator
    assert manifest["calibrator"]["count"] == 26
    assert manifest["onnx_parity_max_abs"] < 1e-5
    assert manifest["validation_metrics"]["count"] == 12
    refs = read_bundle(folder / "references")
    assert set(refs.records.record_id) == set(frame.loc[frame.split.eq("train"), "record_id"])
    assert refs.manifest["records_sha256"] == manifest["reference_records_sha256"]
    assert not list(folder.rglob("*.pkl")) and not list(folder.rglob("*.pt"))
    predicted = predict(frame.iloc[-12:], folder, batch_size=7)
    assert predicted.activity_probability.between(0, 1).all()
    assert predicted.predicted_pactivity.isna().all()
    np.testing.assert_allclose(
        predicted.priority_score,
        predict(frame.iloc[-12:], folder, 1).priority_score,
        atol=1e-5,
        rtol=1e-5,
    )
    assert predict(frame.iloc[:0], folder).empty
    with pytest.raises(ValueError, match="exists"):
        train_baseline_dataset(source, folder, estimator=estimator)
    if estimator == "logistic":
        code = f"from s2s_decision.inference import predict; from s2s_decision.artifacts import read_bundle; import sys; predict(read_bundle({str(source)!r}).records.iloc[-2:], {str(folder)!r}); assert 'torch' not in sys.modules and 'sklearn' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
        from s2s_decision.workflows import evaluate_dataset, score_candidates

        evaluation = evaluate_dataset(source, folder)
        assert evaluation["estimator"] == "logistic" and "tiny" not in evaluation
        assert evaluation["model"]["count"] == 12
        candidate_frame = frame.iloc[-3:].assign(valid=True, eligible=[True, True, False])
        candidates = tmp_path / "candidates"
        write_bundle(FeatureSet(candidate_frame, chemistry_manifest(1024)), candidates)
        score_candidates(candidates, folder, tmp_path / "scored")
        scores = read_bundle(tmp_path / "scored").records
        assert scores.reference_count.iloc[:2].tolist() == [50, 50]
        assert scores.reference_similarity_max.iloc[:2].between(0, 1).all()
        assert scores.activity_probability.iloc[2:].isna().all()
        from s2s_decision.inference import validate_model_manifest

        for replacement, message in [
            ({"calibrator": {**manifest["calibrator"], "slope": -1}}, "calibrator"),
            ({"class_labels": [1, 0]}, "schema"),
            ({"input_width": 1}, "width"),
            ({"regression_supported": True}, "regression"),
        ]:
            with pytest.raises(ValueError, match=message):
                validate_model_manifest({**manifest, **replacement})
        identity_calibration = {**manifest["calibrator"], "slope": 1.0, "intercept": 0.0}
        (folder / "manifest.json").write_text(
            json.dumps({**manifest, "calibrator": identity_calibration})
        )
        raw_probability = predict(frame.iloc[-4:], folder).activity_probability.to_numpy()
        (folder / "manifest.json").write_text(
            json.dumps(
                {
                    **manifest,
                    "calibrator": {**identity_calibration, "slope": 2.0, "intercept": -0.3},
                }
            )
        )
        logits = np.log(raw_probability) - np.log1p(-raw_probability)
        expected = np.exp(-np.logaddexp(0, -(2 * logits - 0.3)))
        np.testing.assert_allclose(predict(frame.iloc[-4:], folder).activity_probability, expected)
    manifest["input_layout"] = "unknown"
    (folder / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="layout"):
        predict(frame.iloc[-1:], folder)


def test_baseline_training_guards_and_no_test_label_dependency(tmp_path):
    from s2s_decision.workflows import train_baseline_dataset

    frame, source = dataset(tmp_path)
    with pytest.raises(ValueError, match="estimator"):
        train_baseline_dataset(source, tmp_path / "bad", estimator="mystery")
    with pytest.raises(ValueError, match="threads"):
        train_baseline_dataset(source, tmp_path / "bad", threads=0)
    with pytest.raises(ValueError, match="layout"):
        train_baseline_dataset(source, tmp_path / "bad", input_layout="unknown")
    original = train_baseline_dataset(source, tmp_path / "original")
    altered = frame.assign(
        y_active=frame.y_active.where(frame.split.ne("test"), 1 - frame.y_active)
    )
    other = tmp_path / "altered"
    write_bundle(FeatureSet(altered, read_bundle(source).manifest), other)
    changed = train_baseline_dataset(other, tmp_path / "changed")
    assert original["onnx_sha256"] == changed["onnx_sha256"]


def test_probability_clipping_is_finite_and_calibration_support_is_explicit():
    from s2s_decision.baseline_training import _calibrate, probability_logits

    logits = probability_logits([0.0, 0.5, 1.0])
    assert np.isfinite(logits).all() and np.all(np.diff(logits) > 0)
    assert logits[1] == 0
    assert (
        _calibrate(np.array([0.0, 1.0]), np.array([0.0, 1.0]))["status"]
        == "uncalibrated_insufficient_data"
    )


def test_histogram_export_preserves_adjacent_float32_branch_boundaries(tmp_path):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits

    from s2s_decision.baseline_training import _export

    left = np.nextafter(np.float32(1), np.float32(np.inf))
    right = np.nextafter(left, np.float32(np.inf))
    values = np.tile([left, right], 30).astype(np.float32).reshape(-1, 1)
    labels = np.tile([0, 1], 30)
    with threadpool_limits(limits=1):
        model = HistGradientBoostingClassifier(max_iter=2, min_samples_leaf=5).fit(values, labels)
        _, parity = _export(model, values, tmp_path)
    assert parity < 1e-5


@pytest.mark.parametrize(
    "threshold",
    [1.0, -1.0, 1.0000001788139343, 1.0000000596046448, -1.0000001788139343, -1.0000000596046448],
)
def test_float32_leq_threshold_round_up_down_exact_and_negative(threshold):
    from s2s_decision.baseline_training import _float32_leq_threshold

    rounded = np.float32(threshold)
    probes = np.array(
        [
            np.nextafter(rounded, np.float32(-np.inf)),
            rounded,
            np.nextafter(rounded, np.float32(np.inf)),
        ],
        dtype=np.float32,
    )
    corrected = _float32_leq_threshold(threshold)
    np.testing.assert_array_equal(probes.astype(np.float64) <= threshold, probes <= corrected)
