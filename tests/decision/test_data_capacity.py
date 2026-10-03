"""Controls that keep the capacity experiment paired and immutable."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def runner():
    path = Path(__file__).parents[2] / "benchmarks/data_capacity.py"
    spec = importlib.util.spec_from_file_location("data_capacity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nested_training_preserves_holdouts_and_row_order_independence():
    module = runner()
    frame = pd.DataFrame(
        {
            "record_id": range(12),
            "identity": [f"i{x}" for x in range(12)],
            "split": ["train"] * 8 + ["validation", "calibration", "test", "test"],
            "y_active": [0, 1] * 6,
        }
    )
    small = module.training_subset(frame, 0.5, 73)
    shuffled = module.training_subset(frame.sample(frac=1, random_state=9), 0.5, 73)
    assert small.record_id.tolist() == shuffled.record_id.tolist()
    assert small.loc[small.split.eq("train"), "y_active"].value_counts().to_dict() == {0: 2, 1: 2}
    assert small.loc[~small.split.eq("train"), "record_id"].tolist() == [8, 9, 10, 11]
    assert len(module.training_subset(frame, 1.0, 73)) == len(frame)
    with pytest.raises(ValueError, match="historical"):
        module.assert_unexposed(small, {small.identity.iloc[0]})


def test_freeze_required_and_changes_refused(tmp_path):
    module = runner()
    artifact = tmp_path / "data.txt"
    artifact.write_text("frozen")
    lock = tmp_path / "lock.json"
    with pytest.raises(FileNotFoundError):
        module.check_lock(lock)
    module.seal(lock, {"files": {str(artifact): module.file_hash(artifact)}})
    module.check_lock(lock)
    with pytest.raises(ValueError, match="changed"):
        module.seal(lock, {"files": {str(artifact): "different"}})
    artifact.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        module.check_lock(lock)


def test_library_newlines_are_stable_on_windows(tmp_path):
    module = runner()
    frame = pd.DataFrame({"molecule_id": ["a", "b"], "original_smiles": ["CC", "CCC"]})
    path = tmp_path / "library.csv"
    module.write_library(path, frame)
    module.write_library(path, frame)
    assert path.read_bytes() == b"ID,SMILES\na,CC\nb,CCC\n"
    with pytest.raises(ValueError, match="changed"):
        module.write_library(path, frame.iloc[:1])


def test_model_reuse_rejects_wrong_seed_or_width(tmp_path, monkeypatch):
    module = runner()
    from dataclasses import asdict

    from s2s_decision.training import TrainingConfig

    plan = {"tiny": {"epochs": 80, "batch_size": 256, "patience": 15}, "fingerprint_bits": 2048}
    config = asdict(TrainingConfig(**plan["tiny"], seed=11, width_multiplier=1))
    manifest = {
        "config": {key: value for key, value in config.items() if key not in ("epochs", "resume")}
    }
    monkeypatch.setattr(module, "load", lambda path: manifest)
    monkeypatch.setattr(module, "validate_chemistry_compatibility", lambda *args: None)
    monkeypatch.setattr(module, "chemistry_manifest", lambda bits: {})
    dataset = module.FeatureSet(pd.DataFrame(), {})
    for option in (
        {"estimator": "tiny", "seed": 11, "width": 2},
        {"estimator": "tiny", "seed": 42, "width": 1},
    ):
        with pytest.raises(ValueError, match="configuration mismatch"):
            module.validate_model(tmp_path, dataset, option, plan)


def test_real_tiny_and_baseline_packages_pass_runner_and_reuse(tmp_path):
    from .test_learning import sample_frame

    module = runner()
    root, work = tmp_path / "study", tmp_path / "scratch"
    root.mkdir()
    plan = {
        "tiny": {"epochs": 2, "batch_size": 8, "patience": 2},
        "fingerprint_bits": 2048,
        "threads": 1,
    }
    module.write_json(root / "plan.json", plan)
    module.write_json(root / "execution-lock.json", {"files": {}})
    features = module.FeatureSet(
        sample_frame(),
        {
            **module.chemistry_manifest(2048),
            "target_id": "synthetic",
            "endpoint": "IC50",
            "threshold": 6.0,
            "context_method": "test_only",
        },
    )
    dataset = root / "dataset"
    module.write_bundle(features, dataset)
    task = {"task_id": "synthetic"}
    arm = {"name": "train100", "dataset": str(dataset)}
    options = [
        {
            "name": "tiny_w2_seed11",
            "family": "tiny_w2",
            "estimator": "tiny",
            "width": 2,
            "seed": 11,
        },
        {"name": "logistic_seed11", "family": "logistic", "estimator": "logistic", "seed": 11},
    ]
    for option in options:
        first = module.fit_one(root, work, task, arm, option, plan)
        assert module.fit_one(root, work, task, arm, option, plan) == first
        assert first["onnx_parity_max_abs"] < 1e-5
    assert first["optimizer_updates"] is None
