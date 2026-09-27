"""Test rows stay sealed until explicit evaluation, including feature inference."""

import numpy as np
import pandas as pd
import pytest
import torch

from s2s_decision import training
from s2s_decision.schema import CONTEXT_NAMES, PROPERTY_NAMES

from .test_learning import sample_frame


def test_training_never_transforms_test_and_is_invariant_to_test_contents(tmp_path, monkeypatch):
    pytest.importorskip("onnx")
    # Test first catches export samples accidentally drawn from the full input.
    source = sample_frame()
    frame = pd.concat((source.iloc[-4:], source.iloc[:-4]))
    poisoned = frame.astype({"pactivity": float}).copy(deep=True)
    heldout = poisoned.split.eq("test")
    poisoned.loc[heldout, list(PROPERTY_NAMES)] = np.nan
    poisoned.loc[heldout, list(CONTEXT_NAMES)] = 1e300
    poisoned.loc[heldout, "fingerprint_hex"] = "sealed-test-not-decoded"
    poisoned.loc[heldout, "y_active"] = 1 - poisoned.loc[heldout, "y_active"]
    poisoned.loc[heldout, "pactivity"] = 1e200

    real_transform = training.Preprocessor.transform
    real_fingerprints = training.fingerprint_matrix
    real_calibrate = training._calibrate
    real_export = training._export
    observed = {"transforms": [], "fingerprints": [], "calibration": 0, "export": 0}
    development_ids = frame.loc[~frame.split.eq("test"), "record_id"].tolist()
    development_count = len(development_ids)

    def transform(preprocessor, rows):
        assert not rows.split.eq("test").any(), "Test features reached preprocessing"
        observed["transforms"].append(rows.record_id.tolist())
        return real_transform(preprocessor, rows)

    def fingerprints(rows, bits):
        assert not rows.split.eq("test").any(), "Test fingerprints were decoded"
        observed["fingerprints"].append(rows.record_id.tolist())
        return real_fingerprints(rows, bits)

    def calibrate(logits, labels):
        expected = frame.loc[frame.split.eq("calibration"), "y_active"].to_numpy()
        np.testing.assert_array_equal(labels, expected)
        assert len(logits) == len(expected)
        observed["calibration"] += 1
        return real_calibrate(logits, labels)

    def export(model, arrays, folder):
        assert all(len(values) == development_count for values in arrays)
        observed["export"] += 1
        return real_export(model, arrays, folder)

    monkeypatch.setattr(training.Preprocessor, "transform", transform)
    monkeypatch.setattr(training, "fingerprint_matrix", fingerprints)
    monkeypatch.setattr(training, "_calibrate", calibrate)
    monkeypatch.setattr(training, "_export", export)
    config = training.TrainingConfig(epochs=2, batch_size=8, patience=2)
    original = training.train_model(frame, tmp_path / "original", config)
    changed = training.train_model(poisoned, tmp_path / "changed", config)

    assert observed == {
        "transforms": [development_ids, development_ids],
        "fingerprints": [development_ids, development_ids],
        "calibration": 2,
        "export": 2,
    }
    for field in (
        "preprocessing",
        "history",
        "calibrator",
        "regression_mean",
        "regression_scale",
        "onnx_sha256",
        "heldout_test_evaluated",
    ):
        assert original[field] == changed[field]
    weights = [
        torch.load(tmp_path / name / "checkpoint.pt", weights_only=True)["best_model"]
        for name in ("original", "changed")
    ]
    for name in weights[0]:
        torch.testing.assert_close(weights[0][name], weights[1][name], rtol=0, atol=0)
    # Provenance and resume still cover every source row, including sealed test.
    assert original["data_hash"] != changed["data_hash"]
    with pytest.raises(ValueError, match="Resume incompatible"):
        training.train_model(
            poisoned,
            tmp_path / "original",
            training.TrainingConfig(epochs=3, batch_size=8, patience=2, resume=True),
        )
