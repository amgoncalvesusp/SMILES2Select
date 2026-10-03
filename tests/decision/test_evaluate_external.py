"""External ranking is label-blind, paired, deterministic and frozen."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def runner():
    path = Path(__file__).parents[2] / "benchmarks/evaluate_external.py"
    spec = importlib.util.spec_from_file_location("evaluate_external", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pool():
    return pd.DataFrame({"identity": ["d", "b", "a", "c"], "y_active": [1, 0, 1, 0],
                         "murcko_scaffold": ["s1", "s2", "s1", "s3"],
                         "primary_identity_novel": [True, True, True, False],
                         "secondary_scaffold_novel": [False, True, True, False]})


def test_ties_and_labels_cannot_change_selection():
    module = runner()
    frame = pool()
    ranked = module.rank_scores(frame.identity, np.array([.5, .5, .5, .1]))
    assert ranked.identity.tolist() == ["a", "b", "d", "c"]
    permutation = frame.assign(y_active=1 - frame.y_active)
    assert module.rank_scores(permutation.identity, [.5, .5, .5, .1]).equals(ranked)
    assert "y_active" not in ranked.columns
    with pytest.raises(ValueError, match="finite"):
        module.rank_scores(frame.identity, [.5, np.nan, .5, .1])
    with pytest.raises(ValueError, match="unique"):
        module.rank_scores(["a", "a"], [1, 2])


def test_cohorts_and_metrics_keep_every_method_on_identical_pool():
    module = runner()
    frame = pool()
    primary = module.cohort(frame, "primary_identity_novel")
    assert set(primary.identity) == {"a", "b", "d"}
    ranks = module.rank_scores(primary.identity, [3, 2, 1])
    result = module.ranking_metrics(primary, ranks, [2, 50])
    assert result["positive"] == 2
    assert result["selection"]["2"]["hits"] == 1
    assert result["selection"]["50"]["effective_k"] == 3
    assert result["selection"]["50"]["positive_scaffolds_recovered"] == 1
    assert result["status"] == "insufficient_support"
    with pytest.raises(ValueError, match="same identities"):
        module.ranking_metrics(frame, ranks, [2])
    invalid = frame.assign(secondary_scaffold_novel=[True, True, True, True])
    with pytest.raises(ValueError, match="subset"):
        module.cohort(invalid, "secondary_scaffold_novel")


def test_random_expected_hits_and_empty_pool():
    module = runner()
    summary = module.random_control(pool(), [2, 20], repetitions=20, seed=4)
    assert summary["selection"]["2"]["expected_hits"] == 1
    assert summary["selection"]["20"]["expected_hits"] == 2
    empty = pool().iloc[:0]
    ranks = module.rank_scores(empty.identity, [])
    result = module.ranking_metrics(empty, ranks, [20])
    assert result["average_precision"] is None
    assert result["selection"]["20"]["effective_k"] == 0


def test_lock_requires_hashes_and_rejects_modified_input(tmp_path):
    module = runner()
    source = tmp_path / "source.txt"
    source.write_text("original")
    lock = {"files": {str(source): module.file_hash(source)}}
    module.verify_lock(lock)
    source.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        module.verify_lock(lock)
    with pytest.raises(ValueError, match="empty"):
        module.verify_lock({"files": {}})


def test_similarity_strata_boundaries():
    module = runner()
    assert module.similarity_strata([0, .4, .40001, .6, .8, 1]).tolist() == [
        "<=0.4", "<=0.4", "(0.4,0.6]", "(0.4,0.6]", "(0.6,0.8]", ">0.8"]


def test_completed_artifact_preserves_original_checksums(tmp_path):
    module = runner()
    item = tmp_path / "model.bin"
    item.write_bytes(b"original model")
    module.write_json(tmp_path / "completion.json", {"files": {item.name: module.file_hash(item)}})
    assert item in module.verify_completion(tmp_path)
    item.write_bytes(b"replacement model")
    with pytest.raises(ValueError, match="changed"):
        module.verify_completion(tmp_path)
    module.write_json(tmp_path / "completion.json", {"files": {}})
    with pytest.raises(ValueError, match="empty"):
        module.verify_completion(tmp_path)


def test_checkpoint_prediction_loads_raw_logits_and_handles_empty(tmp_path):
    import torch

    from s2s_decision.multitask import S2Decision

    module = runner()
    model = S2Decision(3, 2, (4,)).eval()
    path = tmp_path / "best.pt"
    torch.save({"inputs": 3, "tasks": 2, "hidden": (4,), "state_dict": model.state_dict()}, path)
    x = np.ones((3, 3), dtype=np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).numpy()[:, 1]
    np.testing.assert_allclose(module.predict_checkpoint(path, x, 1), expected)
    assert module.predict_checkpoint(path, x[:0], 1).shape == (0,)
    with pytest.raises(ValueError, match="dimensions"):
        module.predict_checkpoint(path, x[:, :2], 1)
    with pytest.raises(ValueError, match="labels"):
        module.score_assay(pool(), "unused", tmp_path, tmp_path)


def test_freeze_and_synthetic_evaluation_keep_truth_outside_predictor(tmp_path, monkeypatch):
    module = runner()
    external, b09, controls = (tmp_path / name for name in ("external", "b09", "controls"))
    plan = {"selection": {"K": list(module.KS), "primary_K": 50, "random_repetitions": 1000,
                          "random_seeds": "0 through 999"},
            "models": {"control_seeds": list(module.SEEDS)},
            "assays": [{"assay": key, "task_id": value} for key, value in module.ASSAYS.items()]}
    plan_path = tmp_path / "plan.json"
    module.write_json(plan_path, plan)
    paths = [external / "data/preparation-manifest.json", external / "data/chemistry.json",
             controls / "completion.json", b09 / "data/preprocess.json", b09 / "data/tasks.json",
             b09 / "data/chemistry.json", b09 / "run/logistic/coefficients.npz",
             *[b09 / f"run/s2_decision_seed{seed}/best.pt" for seed in module.SEEDS]]
    for path in paths:
        module.write_json(path, {})
    pd.concat([pool().assign(assay=assay) for assay in module.ASSAYS], ignore_index=True).to_parquet(
        external / "data/dataset.parquet", index=False)
    module.write_json(external / "data/source-manifest.json", {"smiles_files": {}, "downloads": []})
    module.write_json(external / "data/preparation-manifest.json", {
        "dataset_sha256": module.file_hash(external / "data/dataset.parquet"),
        "chemistry_sha256": module.file_hash(external / "data/chemistry.json"),
        "source_manifest_sha256": module.file_hash(external / "data/source-manifest.json"),
        "code_sha256": module.file_hash(Path(module.__file__).with_name("litpcba_external.py")),
        "exposure": {"source_files": {}},
    })
    module.write_json(controls / "fixture.json", {"complete": True})
    for root in (b09 / "data", b09 / "run", controls):
        files = {str(p.relative_to(root)): module.file_hash(p) for p in root.rglob("*")
                 if p.is_file() and p.name != "completion.json"}
        module.write_json(root / "completion.json", {"files": files})
    monkeypatch.setattr(module, "validate_chemistry_compatibility", lambda *args: None)
    locked = module.freeze(external, b09, controls, plan_path)
    assert locked["random_seed"] == 0
    relocated = tmp_path / "different_root"
    module.write_json(relocated / "evaluation-lock.json", locked)
    with pytest.raises(ValueError, match="root differs"):
        module.evaluate(relocated)
    with pytest.raises(ValueError, match="already exists"):
        module.freeze(external, b09, controls, plan_path)

    def synthetic_predict(features, *args):
        assert "y_active" not in features
        return {"fixture": np.arange(len(features), dtype=float)}, np.zeros(len(features))

    monkeypatch.setattr(module, "score_assay", synthetic_predict)
    report = module.evaluate(external)
    assert report["ESR1_ant"]["cohorts"]["primary_identity_novel"]["methods"]["fixture"]["positive"] == 2
    ranking = pd.read_parquet(external / "evaluation/ESR1_ant-primary_identity_novel-fixture-ranking.parquet")
    assert "y_active" not in ranking
    assert (external / "evaluation/completion.json").exists()
    with pytest.raises(FileExistsError):
        module.evaluate(external)


def test_score_assay_uses_only_frozen_inputs_and_shared_train_references(tmp_path, monkeypatch):
    from types import SimpleNamespace

    module = runner()
    task = "P03372_WT_IC50"
    frame = pool().drop(columns="y_active").assign(qed=.5)
    references = pool().assign(split="train")
    module.write_json(tmp_path / "data/tasks.json", [{"task_id": task}])
    (tmp_path / "run/logistic").mkdir(parents=True)
    np.savez(tmp_path / "run/logistic/coefficients.npz", coefficients=np.ones((1, 3)), intercept=[0.])
    monkeypatch.setattr(module, "molecular_inputs", lambda *args: np.ones((4, 3), dtype=np.float32))
    monkeypatch.setattr(module, "read_bundle", lambda *args: SimpleNamespace(
        records=references, manifest={"records_sha256": "same"}))
    monkeypatch.setattr(module, "build_context", lambda *args: pd.DataFrame(
        .5, index=frame.index, columns=module.CONTEXT_NAMES))
    monkeypatch.setattr(module, "predict_checkpoint", lambda *args: np.arange(4))
    monkeypatch.setattr(module, "control_runner", lambda: SimpleNamespace(
        score_control=lambda *args, **kwargs: {"raw_logits": np.arange(4)}))
    for seed in module.SEEDS:
        module.write_json(tmp_path / task / f"tiny/seed-{seed}/manifest.json",
                          {"reference_records_sha256": "same"})
    scores, maximum = module.score_assay(frame, task, tmp_path, tmp_path)
    assert len(scores) == 12
    np.testing.assert_equal(maximum, [.5] * 4)
    np.testing.assert_equal(scores["logistic"], [3.] * 4)
    module.write_json(tmp_path / task / "tiny/seed-44/manifest.json", {"reference_records_sha256": "other"})
    with pytest.raises(ValueError, match="share"):
        module.score_assay(frame, task, tmp_path, tmp_path)


@pytest.mark.parametrize("mutation", [
    ("selection", "K", [50]), ("selection", "primary_K", 20),
    ("selection", "random_repetitions", 10), ("selection", "random_seeds", "1 through 1000"),
    ("models", "control_seeds", [42]),
])
def test_plan_cannot_silently_change_predeclared_budgets(mutation):
    module = runner()
    plan = {"selection": {"K": list(module.KS), "primary_K": 50, "random_repetitions": 1000,
                          "random_seeds": "0 through 999"},
            "models": {"control_seeds": list(module.SEEDS)},
            "assays": [{"assay": key, "task_id": value} for key, value in module.ASSAYS.items()]}
    section, key, value = mutation
    plan[section][key] = value
    with pytest.raises(ValueError, match="differ"):
        module.validate_plan(plan)
