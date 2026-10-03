"""Behavioral checks for sparse multitask training and independent calibration."""

import json

import numpy as np
import pytest
import torch

from s2s_decision.multitask import (
    calibrate_multitask,
    evaluate_multitask,
    masked_task_loss,
    train_multitask,
)


def sample():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(100, 8)).astype(np.float32)
    y = np.column_stack((x[:, 0] > 0, x[:, 1] > 0)).astype(np.float32)
    y[::3, 1] = np.nan
    split = np.array(["train"] * 55 + ["validation"] * 15 + ["calibration"] * 15 + ["test"] * 15)
    return x, y, split


def test_missing_labels_have_zero_gradient_and_global_task_balance():
    logits = torch.tensor([[0., 2.], [1., 0.]], requires_grad=True)
    labels = torch.tensor([[1., float("nan")], [0., 1.]])
    loss = masked_task_loss(logits, labels, torch.tensor([2., 1.]), 2)
    expected = (torch.nn.functional.softplus(torch.tensor(0.)) / 2
                + torch.nn.functional.softplus(torch.tensor(1.)) / 2
                + torch.nn.functional.softplus(torch.tensor(0.))) / 2
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad[0, 1] == 0
    assert torch.isfinite(logits.grad).all()


def test_calibration_uses_only_calibration_rows_and_reports_fallback():
    _, y, split = sample()
    logits = np.zeros_like(y)
    probabilities, info = calibrate_multitask(logits, y, split)
    assert np.all(probabilities == .5)
    assert all(task["method"] == "identity_insufficient_support" for task in info)
    changed = y.copy()
    changed[split != "calibration"] = 1 - changed[split != "calibration"]
    actual, actual_info = calibrate_multitask(logits, changed, split)
    np.testing.assert_array_equal(actual, probabilities)
    assert actual_info == info


def test_metrics_ignore_unknowns_and_report_effective_k():
    y = np.array([[1., np.nan], [0., 1.], [1., 0.]])
    p = np.array([[.9, .9], [.2, .8], [.8, .2]])
    result = evaluate_multitask(y, p, ks=(1, 50))
    assert result["macro_average_precision"] == 1
    assert result["tasks"][1]["observed"] == 2
    assert result["tasks"][0]["selection"]["50"]["effective_k"] == 3
    assert result["tasks"][0]["selection"]["1"]["enrichment_factor"] == 1.5


def test_supported_calibration_fits_observed_calibration_only():
    y = np.tile([[0.], [1.]], (30, 1)).astype(np.float32)
    logits = (y * 2 - 1) * .1
    split = np.array(["train"] * 4 + ["validation"] * 4 + ["calibration"] * 48 + ["test"] * 4)
    original = y.copy()
    predicted, info = calibrate_multitask(logits, y, split)
    assert info[0]["method"] == "platt_calibration_only"
    assert info[0]["negative"] == info[0]["positive"] == 24
    assert predicted[y == 1].mean() > .9
    assert predicted[y == 0].mean() < .1
    np.testing.assert_array_equal(y, original)
    y[split != "calibration"] = 1 - y[split != "calibration"]
    changed, changed_info = calibrate_multitask(logits, y, split)
    np.testing.assert_array_equal(predicted, changed)
    assert changed_info == info


@pytest.mark.parametrize("fault", ["shape", "infinity", "label", "split", "no_train_class"])
def test_invalid_training_inputs_fail_before_writing(tmp_path, fault):
    x, y, split = sample()
    if fault == "shape":
        y = y[:-1]
    elif fault == "infinity":
        x[0, 0] = np.inf
    elif fault == "label":
        y[0, 0] = .5
    elif fault == "split":
        split[0] = "oops"
    else:
        y[split == "train", 0] = 1
    with pytest.raises(ValueError):
        train_multitask(x, y, split, tmp_path / "invalid", max_epochs=1)
    assert not (tmp_path / "invalid").exists()


def test_training_restores_best_exports_dynamic_onnx_and_never_selects_on_test(tmp_path):
    import onnxruntime as ort

    x, y, split = sample()
    first = train_multitask(x, y, split, tmp_path / "one", hidden=(8, 4), max_epochs=3)
    changed = y.copy()
    changed[split == "test"] = 1 - changed[split == "test"]
    second = train_multitask(x, changed, split, tmp_path / "two", hidden=(8, 4), max_epochs=3)
    assert first["history"] == second["history"]
    expected = np.load(tmp_path / "one" / "probabilities.npy")
    np.testing.assert_array_equal(expected, np.load(tmp_path / "two" / "probabilities.npy"))
    session = ort.InferenceSession(str(tmp_path / "one" / "model.onnx"))
    for size in (1, 7):
        actual = session.run(None, {"features": x[:size]})[0]
        assert actual.shape == (size, 2)
        np.testing.assert_allclose(actual, expected[:size], atol=1e-5, rtol=1e-5)
    assert (tmp_path / "one" / "best.pt").exists()
    assert json.loads((tmp_path / "one" / "summary.json").read_text())["name"] == "S2 Decision"
