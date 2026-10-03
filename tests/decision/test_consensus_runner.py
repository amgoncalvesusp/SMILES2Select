"""Frozen train-only curation and paired ranking comparisons."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def runner():
    folder = Path(__file__).parents[2] / "benchmarks"
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location("train_consensus", folder / "train_consensus.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def payload():
    labels = np.array([[0], [1]] * 8, dtype=np.float32)
    return {"x": np.arange(32, dtype=np.float32).reshape(16, 2) / 32,
            "labels": labels, "splits": np.repeat(
                ["train", "validation", "calibration", "test"], 4),
            "identities": np.array([f"id-{i:02}" for i in range(16)])}


def test_train_only_mask_cannot_change_truth_or_features():
    module = runner()
    original = payload()
    clean = {**original, "labels": original["labels"].copy()}
    clean["labels"][0, 0] = np.nan
    mask = np.zeros_like(original["labels"], dtype=bool)
    mask[0, 0] = True
    module.validate_pair(original, clean, mask)
    changed = {**clean, "labels": clean["labels"].copy()}
    changed["labels"][4, 0] = 1
    with pytest.raises(ValueError, match="held-out"):
        module.validate_pair(original, changed, mask)
    with pytest.raises(ValueError, match="features"):
        module.validate_pair(original, {**clean, "x": clean["x"] + 1}, mask)
    with pytest.raises(ValueError, match="mask"):
        module.validate_pair(original, clean, np.ones((1, 1), dtype=bool))
    changed["labels"] = original["labels"].copy()
    changed["labels"][0, 0] = 1
    with pytest.raises(ValueError, match="train"):
        module.validate_pair(original, changed, mask)


def test_rank_metrics_use_logits_and_identity_ties_separate_brier():
    module = runner()
    truth = np.array([1, 0, 1, np.nan])
    result = module.task_metrics(truth, np.array([100, 100, 99, 0.]),
                                 np.array([1., 1., 1., .5]),
                                 np.array([.8, .7, .6, .5]),
                                 np.array(["b", "a", "c", "d"]), [1, 50])
    assert result["selection"]["1"]["hits"] == 0
    assert result["selection"]["50"]["effective_k"] == 3
    assert result["selection"]["50"]["hits"] == 2
    assert result["average_precision"] == pytest.approx(7 / 12)
    assert result["brier_raw"] == pytest.approx(1 / 3)
    assert result["brier_calibrated"] == pytest.approx(.23)
    empty = module.task_metrics(np.array([np.nan]), np.array([1.]),
                                np.array([.5]), np.array([.5]), np.array(["a"]), [50])
    assert empty["average_precision"] is None
    assert empty["selection"]["50"]["effective_k"] == 0


def test_candidate_gate_uses_validation_only_and_rejects_ties():
    module = runner()
    def report(ap, ef):
        return {"tasks": [{"task_id": "t", "average_precision": ap,
                           "selection": {"50": {"enrichment_factor": ef}}}]}
    methods = {f"baseline_seed{s}": report(.8, 2.) for s in module.SEEDS}
    methods.update({f"consensus_seed{s}": report(.8, 2.1) for s in module.SEEDS})
    assert module.candidate_gate(methods)["selected"] == "consensus"
    methods["consensus_seed42"] = report(.79, 2.1)
    assert module.candidate_gate(methods)["selected"] == "baseline"
    methods.update({f"consensus_seed{s}": report(.8, 2.) for s in module.SEEDS})
    assert module.candidate_gate(methods)["selected"] == "baseline"


def prepare(root, arrays):
    from s2s_decision.artifacts import file_hash
    root.mkdir()
    np.savez(root / "arrays.npz", **arrays)
    for name, value in (("tasks.json", [{"task_id": "target_IC50"}]),
                        ("preprocess.json", {}), ("chemistry.json", {})):
        (root / name).write_text(json.dumps(value))
    np.savez(root / "consensus-mask.npz", mask=np.zeros_like(arrays["labels"], dtype=bool))
    (root / "completion.json").write_text(json.dumps({"files": {
        path.name: file_hash(path) for path in root.iterdir() if path.is_file()}}))


def add_lineage(root, original, module):
    (root / "source-lineage.json").write_text(json.dumps({
        "completion_sha256": module.file_hash(original / "completion.json"),
        "preparation_code_sha256": module.file_hash(Path(module.__file__).with_name("prepare_consensus.py"))}))
    (root / "mask-audit.parquet").write_bytes(b"Synthetic audit placeholder; production comes from preparer")
    (root / "completion.json").write_text(json.dumps({"files": {
        path.name: module.file_hash(path) for path in root.iterdir()
        if path.is_file() and path.name != "completion.json"}}))


def test_run_freezes_first_evaluates_after_all_fits_and_preserves_baseline(tmp_path, monkeypatch):
    from s2s_decision import multitask
    module = runner()
    arrays = payload()
    for name in ("original", "clean"):
        prepare(tmp_path / name, arrays)
    add_lineage(tmp_path / "clean", tmp_path / "original", module)
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    def outputs(folder):
        folder.mkdir()
        for name, values in (("logits", arrays["labels"] * 2 - 1),
                             ("raw_probabilities", .25 + .5 * arrays["labels"]),
                             ("probabilities", .3 + .4 * arrays["labels"])):
            np.save(folder / f"{name}.npy", values)
        for name in ("best.pt", "model.onnx"):
            (folder / name).write_bytes(b"frozen-test-placeholder")
    for seed in module.SEEDS:
        outputs(baseline / f"s2_decision_seed{seed}")
    outputs(baseline / "logistic")
    np.savez(baseline / "logistic/coefficients.npz", coefficients=np.ones((1, 2)), intercept=[0.])
    (baseline / "protocol.json").write_text(json.dumps({
        "data": str(tmp_path / "original"), "baseline": module.LOGISTIC,
        "training": module.TRAINING, "seeds": module.SEEDS,
        "data_hashes": {str(path): module.file_hash(path)
                        for path in (tmp_path / "original").iterdir() if path.is_file()}}))
    (baseline / "completion.json").write_text(json.dumps({"files": {
        str(path.relative_to(baseline)): module.file_hash(path)
        for path in baseline.rglob("*") if path.is_file()}}))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"curation": {"expected_tasks": 1, "expected_training_masked": 0,
                                            "expected_training_remaining": 4,
                                            "minimum_training_support": 1},
                                "training": {"seeds": module.SEEDS, **module.TRAINING},
                                "comparison": {"K": module.KS, "primary_K": 50,
                                               "candidate_rule": "EF50 up and AP not lower"}}))
    target = tmp_path / "run"
    calls = []
    def fake_train(x, labels, splits, output_dir, seed, **config):
        assert (target / "protocol.json").exists()
        calls.append(seed)
        outputs(output_dir)
        return {"seed": seed}
    def fake_logistic(x, labels, splits, output, config, tasks):
        calls.append("logistic")
        outputs(output)
        return {"method": "logistic"}
    original_evaluate = module.evaluate
    def checked_evaluate(*args):
        assert calls == [42, 43, 44, "logistic"]
        assert (target / "fits-complete.json").exists()
        return original_evaluate(*args)
    monkeypatch.setattr(multitask, "train_multitask", fake_train)
    monkeypatch.setattr(module, "fit_baseline", fake_logistic)
    monkeypatch.setattr(module, "evaluate", checked_evaluate)
    result = module.run(tmp_path / "clean", tmp_path / "original", baseline, target, plan)
    assert result["selection"]["selected"] == "baseline"
    assert result["selection"]["test_used_for_selection"] is False
    assert len(result["splits"]["test"]["full"]) == 8
    json.dumps(result, allow_nan=False)
    protocol = json.loads((target / "protocol.json").read_text())
    module.verify_inputs(protocol)
    assert (target / "completion.json").exists()
    with pytest.raises(FileExistsError):
        module.run(tmp_path / "clean", tmp_path / "original", baseline, target, plan)
    (baseline / "s2_decision_seed42/best.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        module.verify_inputs(protocol)
    (baseline / "s2_decision_seed42/best.pt").write_bytes(b"frozen-test-placeholder")
    np.savez(tmp_path / "original/arrays.npz", **{**arrays, "x": arrays["x"] + 1})
    with pytest.raises(ValueError, match="prediction provenance"):
        module.freeze(tmp_path / "clean", tmp_path / "original", baseline, tmp_path / "wrong", plan)
    assert not (tmp_path / "wrong").exists()


def test_preparation_contract_rejects_unsealed_mask_and_modified_preprocessor(tmp_path):
    module = runner()
    arrays = payload()
    for name in ("original", "clean"):
        prepare(tmp_path / name, arrays)
    data, original = tmp_path / "clean", tmp_path / "original"
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"curation": {"expected_tasks": 1, "expected_training_masked": 0,
                                            "expected_training_remaining": 4,
                                            "minimum_training_support": 1}}))
    mask = np.zeros_like(arrays["labels"], dtype=bool)
    with pytest.raises(ValueError, match="must seal"):
        module.validate_prepared(data, original, plan, arrays, mask)
    add_lineage(data, original, module)
    module.validate_prepared(data, original, plan, arrays, mask)
    (data / "preprocess.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="contract"):
        module.validate_prepared(data, original, plan, arrays, mask)


def test_metrics_reject_invalid_predictions_and_dtype_changes():
    module = runner()
    original = payload()
    with pytest.raises(ValueError, match="features"):
        module.validate_pair(original, {**original, "x": original["x"].astype(np.float64)},
                             np.zeros_like(original["labels"], dtype=bool))
    for raw, error in ((np.array([np.nan]), "finite"), (np.array([1.1]), "Probability")):
        with pytest.raises(ValueError, match=error):
            module.task_metrics(np.array([1.]), np.array([1.]), raw,
                                np.array([.5]), np.array(["a"]), [50])
