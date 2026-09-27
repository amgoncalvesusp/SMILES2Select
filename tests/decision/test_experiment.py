import pandas as pd
import pytest

from s2s_decision.experiment import curate_features, feasibility, make_plan
from s2s_decision.schema import FeatureSet


def test_curating_excludes_all_duplicate_identities_without_averaging():
    frame = pd.DataFrame(
        {
            "record_id": [1, 2, 3, 4, 5],
            "valid": [True, True, True, False, True],
            "identity": ["same", "same", "unique", "", "unknown"],
            "y_active": [1.0, 0.0, 1.0, 1.0, None],
            "pactivity": [7.0, 5.0, 8.0, 7.0, None],
        }
    )
    source = FeatureSet(frame, {"source": "test"})
    result, exclusions = curate_features(source)
    assert result.records.record_id.tolist() == [3]
    assert exclusions["duplicate_identity_records"] == [1, 2]
    assert exclusions["invalid_records"] == [4]
    assert exclusions["unlabeled_records"] == [5]
    assert len(source.records) == 5


def test_feasibility_no_seed_retry_and_support_requirements():
    frame = pd.DataFrame(
        {
            "split": ["train"] * 200 + ["validation"] * 30 + ["calibration"] * 20 + ["test"] * 50,
            "y_active": [0, 1] * 150,
        }
    )
    assert feasibility(frame)["eligible"]
    bad = frame.copy()
    bad.loc[bad.split.eq("calibration"), "y_active"] = 0
    check = feasibility(bad)
    assert not check["eligible"]
    assert "calibration" in check["reason"]


def test_plan_excludes_exposed_pilot_and_freezes_all_seeds(tmp_path):
    source = tmp_path / "source.tsv"
    source.write_text("test", encoding="utf-8")
    plan = make_plan(source)
    assert len(plan["tasks"]) == 10
    assert plan["seeds"] == [11, 23, 42, 71, 101]
    assert all(t["target_id"] != "P22303_WT" for t in plan["tasks"])
    assert plan["split_methods"] == ["scaffold", "temporal"]
    assert plan["training"]["epochs"] == 80
    with pytest.raises(FileNotFoundError):
        make_plan(tmp_path / "missing")


def test_freeze_refuses_changed_dependency_versions(tmp_path, monkeypatch):
    from s2s_decision import experiment

    monkeypatch.setattr(experiment, "version", lambda _: "1.0")
    experiment.freeze_runtime(tmp_path)
    experiment.freeze_runtime(tmp_path)
    monkeypatch.setattr(experiment, "version", lambda _: "2.0")
    with pytest.raises(ValueError, match="changed"):
        experiment.freeze_runtime(tmp_path)


def test_run_records_failures_continues_other_seeds_and_checks_completed_hash(
    tmp_path, monkeypatch
):
    from s2s_decision import benchmark_evaluation, experiment, workflows
    from s2s_decision.artifacts import file_hash, write_json

    plan = {
        "training": {"epochs": 1},
        "threads": 1,
        "fingerprint_bits": 2048,
        "n": 50,
        "max_per_scaffold": 3,
    }
    write_json(tmp_path / "plan.json", plan)
    runs = [
        {
            "task_id": "task",
            "split_method": "scaffold",
            "seed": seed,
            "eligible": True,
            "dataset": "fake",
        }
        for seed in [11, 23]
    ]
    write_json(
        tmp_path / "prepared.json", {"plan_sha256": file_hash(tmp_path / "plan.json"), "runs": runs}
    )
    monkeypatch.setattr(experiment, "freeze_runtime", lambda _: None)
    monkeypatch.setattr(experiment, "_validate_prepared", lambda *args: None)
    calls = []

    def train(source, model, config, threads):
        calls.append(config.seed)
        if config.seed == 11:
            raise ValueError("unsupported fixture")
        return {"epochs": 1}

    monkeypatch.setattr(workflows, "train_dataset", train)
    monkeypatch.setattr(
        experiment, "read_bundle", lambda _: FeatureSet(pd.DataFrame(), {"records_sha256": "hash"})
    )
    monkeypatch.setattr(
        benchmark_evaluation, "evaluate_benchmark", lambda *args, **kwargs: {"checked": True}
    )
    experiment.run_experiment(tmp_path)
    assert calls == [11, 23]
    experiment.run_experiment(tmp_path)
    assert calls == [11, 23]  # no silent retries after failure or completion
    evaluation = tmp_path / "runs/task/scaffold/23/evaluation.json"
    evaluation.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        experiment.run_experiment(tmp_path)


def test_cached_partition_rejects_changed_plan_before_reuse(tmp_path):
    from s2s_decision import experiment
    from s2s_decision.artifacts import write_json

    write_json(tmp_path / "plan.json", {"changed": True})
    write_json(
        tmp_path / "datasets/task/scaffold/11-feasibility.json",
        {"eligible": False, "plan_sha256": "old-plan", "reason": "insufficient support"},
    )
    with pytest.raises(ValueError, match="another experiment plan"):
        experiment._prepare_partition(
            tmp_path, {"id": "task"}, None, "scaffold", 11, {"seeds": [11]}
        )
