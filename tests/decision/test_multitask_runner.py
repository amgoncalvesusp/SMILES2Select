"""Protocol locks and paired summaries for the multitask training experiment."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def runner():
    path = Path(__file__).parents[2] / "benchmarks/train_multitask.py"
    spec = importlib.util.spec_from_file_location("train_multitask", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_seals_inputs_before_fit_and_refuses_overwrite(tmp_path):
    module = runner()
    data = tmp_path / "data"
    data.mkdir()
    (data / "tasks.json").write_text('[{"task_id":"target_IC50"}]')
    (data / "preprocess.json").write_text("{}")
    np.savez(data / "arrays.npz", x=np.zeros((4, 2)))
    output = tmp_path / "run"
    protocol = module.freeze_protocol(data, output, [42, 43], 40)
    assert protocol["model_name"] == "S2 Decision"
    assert protocol["seeds"] == [42, 43]
    assert protocol["test_usage"] == "evaluation_only_after_all_fits"
    assert json.loads((output / "protocol.json").read_text()) == protocol
    module.verify_inputs(protocol)
    with pytest.raises(FileExistsError):
        module.freeze_protocol(data, output, [42], 40)
    (data / "tasks.json").write_text("[]")
    with pytest.raises(ValueError, match="changed"):
        module.verify_inputs(protocol)


def task(index, ap, ef):
    return {
        "task_index": index, "observed": 8, "positive": 4,
        "average_precision": ap, "roc_auc": ap, "brier": 0.2,
        "selection": {str(k): {"enrichment_factor": ef} for k in (50, 100)},
    }


def seal_data(root):
    from s2s_decision.artifacts import file_hash

    for name in ("preprocess.json", "chemistry.json"):
        if not (root / name).exists():
            (root / name).write_text("{}")
    manifest = {"files": {path.name: file_hash(path) for path in root.iterdir()
                          if path.is_file() and path.name != "completion.json"}}
    (root / "completion.json").write_text(json.dumps(manifest))


def test_paired_summary_matches_tasks_and_reports_defined_denominators():
    module = runner()
    neural = {"tasks": [task(0, 0.9, 1.5), task(1, None, None), task(2, 0.5, 1.0)]}
    baseline = {"tasks": [task(2, 0.4, 1.0), task(0, 0.7, 1.2), task(1, None, None)]}
    summary = module.paired_summary(neural, baseline)
    assert summary["average_precision"]["paired_tasks"] == 2
    assert summary["average_precision"]["mean_delta"] == pytest.approx(0.15)
    assert summary["ef_at_50"]["mean_delta"] == pytest.approx(0.15)
    assert summary["average_precision"]["wins"] == 2
    assert summary["total_tasks"] == 3
    assert module.macro_summary(neural)["average_precision"]["tasks"] == 2


def test_invalid_dataset_rejects_identity_leakage_and_unknown_split(tmp_path):
    module = runner()
    payload = {
        "x": np.zeros((4, 2)), "labels": np.zeros((4, 1)),
        "splits": np.array(["train", "validation", "calibration", "test"]),
        "identities": np.array(["a", "a", "b", "c"]),
    }
    np.savez(tmp_path / "arrays.npz", **payload)
    (tmp_path / "tasks.json").write_text('[{"task_id":"target_IC50"}]')
    seal_data(tmp_path)
    with pytest.raises(ValueError, match="identities"):
        module.load_dataset(tmp_path)
    payload = {**payload, "identities": np.array(["a", "b", "c", "d"]),
               "splits": np.array(["train", "validation", "calibration", "future"])}
    np.savez(tmp_path / "arrays.npz", **payload)
    seal_data(tmp_path)
    with pytest.raises(ValueError, match="split"):
        module.load_dataset(tmp_path)


def sample_data(root):
    root.mkdir()
    labels = np.array([[0], [1]] * 8, dtype=np.float32)
    arrays = {
        "x": np.arange(32, dtype=np.float32).reshape(16, 2) / 32,
        "labels": labels,
        "splits": np.repeat(["train", "validation", "calibration", "test"], 4),
        "identities": np.array([str(i) for i in range(16)]),
    }
    np.savez(root / "arrays.npz", **arrays)
    tasks = [{"task_id": "target_IC50"}]
    (root / "tasks.json").write_text(json.dumps(tasks))
    (root / "preprocess.json").write_text("{}")
    seal_data(root)
    return arrays, tasks


def test_preparation_hashes_checked_before_dataset_loading(tmp_path):
    module = runner()
    root = tmp_path / "data"
    sample_data(root)
    (root / "tasks.json").write_text('[{"task_id":"changed"}]')
    with pytest.raises(ValueError, match="Prepared data changed"):
        module.load_dataset(root)


def test_runner_evaluates_only_after_all_fits_and_exports_finite_metrics(tmp_path, monkeypatch):
    from s2s_decision import multitask

    module = runner()
    data, output = tmp_path / "data", tmp_path / "run"
    arrays, _ = sample_data(data)
    fits = []

    def fake_train(x, labels, splits, output_dir, seed, **config):
        assert (output / "protocol.json").is_file()
        fits.append(seed)
        output_dir.mkdir()
        for name in ("raw_probabilities", "probabilities"):
            np.save(output_dir / f"{name}.npy", np.full_like(labels, 0.5))
        return {"seed": seed}

    evaluate = multitask.evaluate_multitask

    def checked_evaluate(labels, probabilities, ks):
        assert fits == [42, 43]
        assert (output / "fits-complete.json").is_file()
        return evaluate(labels, probabilities, ks)

    monkeypatch.setattr(multitask, "train_multitask", fake_train)
    monkeypatch.setattr(multitask, "evaluate_multitask", checked_evaluate)
    result = module.run(data, output, [42, 43], 1)
    assert result["test_molecules"] == 4
    assert len(result["methods"]) == 6
    assert len(result["paired"]) == 4
    json.dumps(result, allow_nan=False)
    completion = json.loads((output / "completion.json").read_text())
    assert completion["files"]["metrics.json"] == module.file_hash(output / "metrics.json")
    coefficients = np.load(output / "logistic/coefficients.npz", allow_pickle=False)
    assert coefficients["coefficients"].shape == (1, arrays["x"].shape[1])


def test_baseline_coefficients_do_not_depend_on_holdout_labels(tmp_path):
    module = runner()
    arrays, tasks = sample_data(tmp_path / "data")
    config = {"C": 1, "max_iter": 1000, "solver": "lbfgs"}
    labels = arrays["labels"].copy()
    labels[arrays["splits"] != "train"] = 1 - labels[arrays["splits"] != "train"]
    for suffix, targets in (("a", arrays["labels"]), ("b", labels)):
        module.fit_baseline(arrays["x"], targets, arrays["splits"], tmp_path / suffix, config, tasks)
    with np.load(tmp_path / "a/coefficients.npz") as first, np.load(tmp_path / "b/coefficients.npz") as second:
        np.testing.assert_array_equal(first["coefficients"], second["coefficients"])
        np.testing.assert_array_equal(first["intercept"], second["intercept"])
