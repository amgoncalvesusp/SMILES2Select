"""Reports preserve task-level replication and disclose incomplete runs."""

import json

import pytest

from s2s_decision.artifacts import file_hash, write_json
from s2s_decision.experiment_report import summarize_experiment, write_report

SEEDS = [11, 23, 42, 71, 101]
METHODS = ["tiny", "logistic", "gradient_boosting", "similarity", "qed"]


def write_evaluation(path, payload):
    write_json(path, payload)
    result_path = path.parent / "result.json"
    result = json.loads(result_path.read_text())
    write_json(result_path, {**result, "evaluation_sha256": file_hash(path)})


def plan(root, tasks=("a", "b", "c")):
    write_json(
        root / "plan.json",
        {
            "tasks": [
                {"id": task, "target_id": task.upper(), "endpoint": "IC50"} for task in tasks
            ],
            "seeds": SEEDS,
            "split_methods": ["scaffold", "temporal"],
        },
    )
    write_json(root / "prepared.json", {"plan_sha256": file_hash(root / "plan.json")})


def run(root, task, seed, score, *, split="scaffold", status="completed"):
    folder = root / "runs" / task / split / str(seed)
    write_json(
        folder / "result.json",
        {
            "task_id": task,
            "split_method": split,
            "seed": seed,
            "status": status,
            "elapsed_seconds": 2,
            "reason": None if status == "completed" else "insufficient data",
        },
    )
    if status == "completed":
        write_evaluation(
            folder / "evaluation.json",
            {
                "seed": seed,
                "n_requested": 50,
                "max_per_scaffold": 3,
                "pool": {"common_count": 100},
                "methods": {
                    method: {
                        "raw": {
                            "pr_auc": score if method == "tiny" else 0.5,
                            "precision_at_n": score if method == "tiny" else 0.5,
                            "roc_auc": score,
                            "brier": None,
                            "ece": None,
                        },
                        "selection": {
                            "precision": score if method == "tiny" else 0.5,
                            "recall": 0.1,
                            "final_count": 50,
                            "scaffolds_covered": 25,
                            "shortfall": 0,
                            "warnings": [],
                        },
                    }
                    for method in METHODS
                },
            },
        )


def test_only_complete_tasks_enter_primary_and_seeds_are_not_independent(tmp_path):
    plan(tmp_path)
    for seed in SEEDS:
        run(tmp_path, "a", seed, 0.8)
        run(tmp_path, "b", seed, 0.2)
    for seed in SEEDS[:4]:
        run(tmp_path, "c", seed, 1.0)
    result = summarize_experiment(tmp_path)
    scaffold = result["protocols"]["scaffold"]
    assert scaffold["supported_tasks"] == 2
    assert scaffold["partial_tasks"] == ["c"]
    assert scaffold["methods"]["tiny"]["pr_auc"]["mean"] == pytest.approx(0.5)
    paired = scaffold["paired"]["gradient_boosting"]["pr_auc"]
    assert paired["task_count"] == 2
    assert paired["mean_delta"] == pytest.approx(0)
    assert paired["ci95"] == pytest.approx([-0.3, 0.3])
    assert paired["interpretation"] == "inconclusive"
    assert result["status_counts"] == {"completed": 14, "pending": 16}
    assert result == summarize_experiment(tmp_path)


def test_failed_excluded_missing_evaluation_and_protocols_remain_visible(tmp_path):
    plan(tmp_path, ("a",))
    for seed in SEEDS:
        run(tmp_path, "a", seed, 0.9, split="temporal")
    run(tmp_path, "a", 11, 0.8, status="failed")
    run(tmp_path, "a", 23, 0.8, status="excluded")
    run(tmp_path, "a", 42, 0.8)
    (tmp_path / "runs/a/scaffold/42/evaluation.json").unlink()
    result = summarize_experiment(tmp_path)
    assert result["protocols"]["scaffold"]["supported_tasks"] == 0
    paired = result["protocols"]["temporal"]["paired"]["logistic"]["pr_auc"]
    assert paired["task_count"] == 1
    assert paired["ci95"] is None
    assert paired["interpretation"] == "insufficient_tasks"
    assert result["status_counts"] == {"completed": 5, "failed": 2, "excluded": 1, "pending": 2}
    paths = write_report(tmp_path)
    report = paths["report"].read_text(encoding="utf-8")
    assert "failed" in report and "excluded" in report and "pending" in report
    assert "retrospectiv" in report
    assert json.loads(paths["summary"].read_text(encoding="utf-8"))["expected_runs"] == 10


def test_mismatched_seed_cannot_masquerade_as_completed(tmp_path):
    plan(tmp_path, ("a",))
    run(tmp_path, "a", 11, 0.8)
    path = tmp_path / "runs/a/scaffold/11/evaluation.json"
    payload = json.loads(path.read_text())
    write_evaluation(path, {**payload, "seed": 42})
    result = summarize_experiment(tmp_path)
    assert result["status_counts"] == {"failed": 1, "pending": 9}


def test_null_metrics_keep_their_own_task_denominator(tmp_path):
    plan(tmp_path, ("a", "b"))
    for task in ("a", "b"):
        for seed in SEEDS:
            run(tmp_path, task, seed, 0.7)
    path = tmp_path / "runs/b/scaffold/11/evaluation.json"
    payload = json.loads(path.read_text())
    payload["methods"]["tiny"]["raw"]["pr_auc"] = None
    write_evaluation(path, payload)
    result = summarize_experiment(tmp_path)["protocols"]["scaffold"]
    assert result["supported_tasks"] == 2
    assert result["methods"]["tiny"]["pr_auc"]["task_count"] == 1
    assert result["paired"]["logistic"]["pr_auc"]["task_count"] == 1


def test_missing_baseline_prevents_completed_status(tmp_path):
    plan(tmp_path, ("a",))
    run(tmp_path, "a", 11, 0.8)
    path = tmp_path / "runs/a/scaffold/11/evaluation.json"
    payload = json.loads(path.read_text())
    del payload["methods"]["qed"]
    write_evaluation(path, payload)
    result = summarize_experiment(tmp_path)
    assert result["status_counts"] == {"failed": 1, "pending": 9}


@pytest.mark.parametrize(
    "field,value", [("seeds", [11, 11]), ("split_methods", ["random"]), ("seeds", [-1])]
)
def test_invalid_plan_rejected(tmp_path, field, value):
    plan(tmp_path, ("a",))
    path = tmp_path / "plan.json"
    payload = json.loads(path.read_text())
    write_json(path, {**payload, field: value})
    write_json(tmp_path / "prepared.json", {"plan_sha256": file_hash(path)})
    with pytest.raises(ValueError):
        summarize_experiment(tmp_path)


def test_changed_evaluation_is_reported_as_failed(tmp_path):
    plan(tmp_path, ("a",))
    run(tmp_path, "a", 11, 0.8)
    path = tmp_path / "runs/a/scaffold/11/evaluation.json"
    payload = json.loads(path.read_text())
    write_json(path, {**payload, "n_requested": 10})
    result = summarize_experiment(tmp_path)
    assert result["status_counts"] == {"failed": 1, "pending": 9}
    assert "checksum" in result["runs"][0]["reason"]


def test_changed_plan_cannot_change_reporting_universe(tmp_path):
    plan(tmp_path, ("a",))
    path = tmp_path / "plan.json"
    payload = json.loads(path.read_text())
    write_json(path, {**payload, "seeds": [11]})
    with pytest.raises(ValueError, match="plan checksum"):
        summarize_experiment(tmp_path)


def test_missing_preparation_manifest_rejected(tmp_path):
    plan(tmp_path, ("a",))
    (tmp_path / "prepared.json").unlink()
    with pytest.raises(FileNotFoundError):
        summarize_experiment(tmp_path)
