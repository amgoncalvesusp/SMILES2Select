"""Controlled width changes preserve inference and checkpoint contracts."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from s2s_decision.model import Tiny
from s2s_decision.training import TrainingConfig, train_model

from .test_learning import sample_frame


@pytest.mark.parametrize("bits, expected", [(2048, 598274), (1024, 336130)])
def test_double_width_preserves_input_output_contract(bits, expected):
    model = Tiny(bits, width_multiplier=2)
    assert sum(p.numel() for p in model.parameters()) == expected
    for size in (0, 1, 7):
        output = model(torch.zeros(size, 40), torch.zeros(size, bits), torch.zeros(size, 14))
        assert output.shape == (size, 2)
        assert torch.isfinite(output).all()


def test_default_width_is_exact_original_architecture():
    torch.manual_seed(123)
    default = Tiny()
    torch.manual_seed(123)
    explicit = Tiny(width_multiplier=1)
    assert sum(p.numel() for p in default.parameters()) == 281730
    for name, weights in default.state_dict().items():
        torch.testing.assert_close(weights, explicit.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("width", [None, True, False, 0, -1, 3, 1.0, 2.0, "2", [], {}])
def test_width_requires_supported_integer(width):
    for constructor in (Tiny, TrainingConfig):
        with pytest.raises(ValueError, match="width_multiplier"):
            constructor(width_multiplier=width)


def test_wider_model_preserves_existing_configuration_guards(tmp_path):
    with pytest.raises(ValueError, match="fingerprint_bits"):
        Tiny(4096, width_multiplier=2)
    with pytest.raises(ValueError, match="positive"):
        TrainingConfig(epochs=0, width_multiplier=2)
    with pytest.raises(ValueError, match="learning rate"):
        TrainingConfig(learning_rate=float("nan"), width_multiplier=2)
    with pytest.raises(ValueError, match="checkpoint does not exist"):
        train_model(sample_frame(), tmp_path, TrainingConfig(width_multiplier=2, resume=True))


def test_double_width_training_export_resume_and_reject_width_change(tmp_path):
    from s2s_decision.inference import predict

    frame = sample_frame()
    folder = tmp_path / "wide"
    config = TrainingConfig(epochs=1, batch_size=8, patience=3, width_multiplier=2)
    result = train_model(frame, folder, config)
    assert result["config"]["width_multiplier"] == 2
    assert result["parameter_count"] == 598274
    assert result["heldout_test_evaluated"] is False
    assert result["onnx_parity_max_abs"] < 1e-5
    checkpoint = torch.load(folder / "checkpoint.pt", weights_only=True)
    assert checkpoint["contract"]["config"]["width_multiplier"] == 2
    predictions = predict(frame.iloc[-4:], folder)
    assert predictions.activity_probability.between(0, 1).all()
    with pytest.raises(ValueError, match="Resume incompatible"):
        train_model(frame, folder, replace(config, epochs=2, width_multiplier=1, resume=True))
    resumed = train_model(frame, folder, replace(config, epochs=2, resume=True))
    full = tmp_path / "full"
    complete = train_model(frame, full, replace(config, epochs=2))
    assert resumed["history"] == complete["history"]
    np.testing.assert_allclose(
        predict(frame.iloc[-4:], folder).activity_probability,
        predict(frame.iloc[-4:], full).activity_probability,
        atol=1e-7,
    )


def test_legacy_checkpoint_without_width_resumes_only_at_original_size(tmp_path):
    frame = sample_frame()
    config = TrainingConfig(epochs=1, batch_size=8, patience=3)
    train_model(frame, tmp_path, config)
    path = tmp_path / "checkpoint.pt"
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["contract"]["config"].pop("width_multiplier")
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="Resume incompatible"):
        train_model(frame, tmp_path, replace(config, epochs=2, width_multiplier=2, resume=True))
    report = train_model(frame, tmp_path, replace(config, epochs=2, resume=True))
    assert report["parameter_count"] == 281730
    assert report["config"]["width_multiplier"] == 1


def test_training_cli_propagates_width(tmp_path, monkeypatch):
    from s2s_decision import cli, workflows

    received = []

    def train(source, destination, config, threads):
        received.append(config)
        return {"parameter_count": 598274}

    monkeypatch.setattr(workflows, "train_dataset", train)
    arguments = ["train", "--input", "unused", "--output", str(tmp_path / "out")]
    cli._run(cli.parser().parse_args([*arguments, "--width-multiplier", "2"]))
    assert received[0].width_multiplier == 2
    assert cli.parser().parse_args(arguments).width_multiplier == 1
    with pytest.raises(SystemExit):
        cli.parser().parse_args([*arguments, "--width-multiplier", "3"])
