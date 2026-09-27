"""Behavioral checks use synthetic data, not evidence of predictive performance."""

import numpy as np
import pandas as pd
import pytest
import torch

from s2s_decision.metrics import evaluate_predictions
from s2s_decision.model import Tiny
from s2s_decision.preprocessing import Preprocessor
from s2s_decision.schema import CONTEXT_NAMES, PROPERTY_NAMES


def sample_frame(n=40):
    rng = np.random.default_rng(12)
    frame = pd.DataFrame(
        {name: rng.uniform(0.1, 4, n) for name in (*PROPERTY_NAMES, *CONTEXT_NAMES)}
    )
    frame["fingerprint_hex"] = [
        rng.integers(0, 256, 256, dtype=np.uint8).tobytes().hex() for _ in range(n)
    ]
    frame["record_id"] = [f"r{i}" for i in range(n)]
    frame["identity"] = frame["record_id"]
    frame["split"] = ["train"] * 20 + ["validation"] * 8 + ["calibration"] * 8 + ["test"] * 4
    frame["y_active"] = np.arange(n) % 2
    frame["pactivity"] = 5 + frame["y_active"]
    return frame


def test_architecture_and_preprocessing_train_only():
    assert sum(p.numel() for p in Tiny(2048).parameters()) == 281730
    assert sum(p.numel() for p in Tiny(1024).parameters()) == 150658
    frame = sample_frame()
    prep = Preprocessor.fit(frame[frame.split == "train"])
    props, context = prep.transform(frame)
    assert props.shape == (40, 40) and context.shape == (40, 14)
    missing = frame.copy()
    missing.loc[0, PROPERTY_NAMES[0]] = np.nan
    assert prep.transform(missing)[0][0, 20] == 1
    assert np.isfinite(prep.transform(missing)[0]).all()
    invalid = frame.copy()
    invalid[PROPERTY_NAMES[0]] = np.nan
    with pytest.raises(ValueError, match="missing"):
        Preprocessor.fit(invalid[invalid.split == "train"])
    with pytest.raises(ValueError, match="train"):
        Preprocessor.fit(frame)
    restored = Preprocessor.from_dict(prep.to_dict())
    np.testing.assert_array_equal(restored.transform(frame)[0], props)
    absent = frame.copy()
    absent[list(CONTEXT_NAMES)] = np.nan
    neutral = Preprocessor.fit(absent[absent.split == "train"])
    assert neutral.context_all_missing == tuple(CONTEXT_NAMES)
    assert np.isfinite(neutral.transform(absent)[1]).all()


def test_metrics_missing_classes_and_perfect_ranking():
    report = evaluate_predictions([1, 1], [0.9, 0.8], n=1)
    assert report["roc_auc"] is None
    assert report["precision_at_n"] == 1
    perfect = evaluate_predictions([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2], n=2)
    assert perfect["roc_auc"] == perfect["pr_auc"] == 1
    assert perfect["recall_at_n"] == perfect["precision_at_n"] == 1
    assert evaluate_predictions([], [], n=1)["roc_auc"] is None
    regression = evaluate_predictions([1, 0], [0.8, 0.2], [6.0, 4.0], [5.0, np.nan])
    assert regression["mae"] == regression["rmse"] == 1


def test_training_export_predict_and_resume_guards(tmp_path):
    pytest.importorskip("onnx")
    from s2s_decision.inference import predict
    from s2s_decision.training import TrainingConfig, train_model

    frame = sample_frame()
    result = train_model(
        frame, tmp_path / "model", TrainingConfig(epochs=2, batch_size=8, patience=2)
    )
    assert result["parameter_count"] == 281730
    assert result["heldout_test_evaluated"] is False
    assert (tmp_path / "model" / "model.onnx").exists()
    predictions = predict(frame.iloc[-4:], tmp_path / "model", batch_size=3)
    assert len(predictions) == 4
    assert predictions.activity_probability.between(0, 1).all()
    one = predict(frame.iloc[-4:], tmp_path / "model", batch_size=1)
    np.testing.assert_allclose(
        one.activity_probability, predictions.activity_probability, atol=1e-6
    )
    bad = frame.copy()
    bad.loc[0, "pactivity"] = 9
    with pytest.raises(ValueError, match="incompatible"):
        train_model(
            bad, tmp_path / "model", TrainingConfig(epochs=3, batch_size=8, patience=2, resume=True)
        )
    with pytest.raises(ValueError, match="exists"):
        train_model(frame, tmp_path / "model", TrainingConfig(epochs=2))
    assert torch.load(tmp_path / "model" / "checkpoint.pt", weights_only=True)["epoch"] == 1
    with pytest.raises(ValueError, match="All four splits"):
        train_model(
            frame[frame.split != "test"], tmp_path / "missing-test", TrainingConfig(epochs=1)
        )
    # Resume preserves RNG, optimizer, and scheduler, matching uninterrupted training.
    resumed = train_model(
        frame, tmp_path / "model", TrainingConfig(epochs=3, batch_size=8, patience=2, resume=True)
    )
    uninterrupted = train_model(
        frame, tmp_path / "full", TrainingConfig(epochs=3, batch_size=8, patience=2)
    )
    np.testing.assert_allclose(
        predict(frame.iloc[-4:], tmp_path / "model").priority_score,
        predict(frame.iloc[-4:], tmp_path / "full").priority_score,
        atol=1e-7,
    )
    assert resumed["history"] == uninterrupted["history"]
    assert predict(frame.iloc[:0], tmp_path / "model").empty
    model_file = tmp_path / "model" / "model.onnx"
    model_file.write_bytes(model_file.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        predict(frame.iloc[-4:], tmp_path / "model")


def test_calibration_masked_losses_and_baselines():
    from s2s_decision.training import _calibrate, _loss, _validate, benchmark_baselines

    outputs = torch.tensor([[0.0, 0.0], [1.0, 1.0]], requires_grad=True)
    loss = _loss(outputs, torch.tensor([1.0, float("nan")]), torch.tensor([float("nan"), 1.0]))
    loss.backward()
    assert outputs.grad[0, 1] == outputs.grad[1, 0] == 0
    fitted = _calibrate(np.tile([-1.0, 1.0], 20), np.tile([0.0, 1.0], 20))
    assert fitted["slope"] > 0 and fitted["status"].startswith("fitted")
    assert _calibrate(np.zeros(10), np.zeros(10))["status"].startswith("uncalibrated")
    frame = sample_frame()
    frame["qed"] = frame["qed"] / 4
    frame["similarity_active_max"] = frame["similarity_active_max"] / 4
    report = benchmark_baselines(frame, n=2)
    assert set(report) == {"logistic", "gradient_boosting", "similarity", "qed"}
    assert all(item["count"] == 4 for item in report.values())
    assert report["qed"]["brier"] is None and report["similarity"]["ece"] is None
    missing = frame.copy()
    missing.loc[len(frame) - 1, "similarity_active_max"] = np.nan
    common = benchmark_baselines(missing, n=2)
    assert all(item["count"] == 3 and item["full_test_count"] == 4 for item in common.values())
    duplicate = frame.copy()
    duplicate.loc[1, "identity"] = duplicate.loc[0, "identity"]
    with pytest.raises(ValueError, match="replicate"):
        _validate(duplicate)


@pytest.mark.parametrize(
    "version", ["2.9.1", "2.9.1+cpu", "2.10.0rc1", "2.11.0.dev20260101", "unknown"]
)
def test_resume_rejects_unsafe_torch_before_loading(tmp_path, monkeypatch, version):
    from s2s_decision.training import TrainingConfig, train_model

    (tmp_path / "checkpoint.pt").write_bytes(b"not-loaded")
    monkeypatch.setattr(torch, "__version__", version)

    def forbidden_load(*args, **kwargs):
        raise AssertionError("Checkpoint loader must not run on unsupported PyTorch")

    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="PyTorch >= 2.10.0 stable"):
        train_model(sample_frame(), tmp_path, TrainingConfig(epochs=1, resume=True))
