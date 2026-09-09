"""Reproducibility and label isolation for the cached-selection benchmark."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_selection.py"
SPEC = importlib.util.spec_from_file_location("benchmark_selection", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def library():
    return pd.DataFrame(
        {
            "record_id": range(1, 7),
            "approved": [1, 1, 1, 1, 1, 0],
            "qed": [0.9, 0.8, 0.7, 0.6, 0.5, 1.0],
            "murcko_scaffold": ["A", "A", "B", "C", "D", "E"],
            "active": [1, None, 0, 1, None, 1],
        }
    )


def test_reproducible_same_pool_quotas_and_unknown_activity():
    frame = benchmark.validate_frame(library(), activity_column="active")
    config = benchmark.BenchmarkConfig(
        count=3, seed=19, repeats=2, thresholds=(0.0, 0.65), max_per_scaffold=1
    )
    report = benchmark.run_benchmark(frame, config, source="test", source_hash="abc")
    again = benchmark.run_benchmark(frame, config, source="test", source_hash="abc")

    def clean(rows):
        return [{k: v for k, v in row.items() if k != "seconds"} for row in rows]

    assert clean(report["runs"]) == clean(again["runs"])
    ranked = report["runs"][0]
    assert ranked["eligible_count"] == 5
    assert ranked["selected_count"] == 3
    assert ranked["scaffolds_selected"] == 3
    assert ranked["known_active_selected"] == 2
    assert ranked["activity_unknown_selected"] == 0
    assert ranked["active_recovery"] == 1.0
    assert report["runs"][4]["eligible_count"] == 3
    assert report["runs"][4]["selected_count"] == 2
    assert report["stability"][0]["scenario_count"] == 2


def test_labels_cannot_influence_selection_and_missing_is_not_inactive():
    first = library()
    changed = first.assign(active=[0, 1, 1, 0, 1, 0])
    config = benchmark.BenchmarkConfig(count=4)
    reports = [
        benchmark.run_benchmark(
            benchmark.validate_frame(frame, "active"), config, source="test", source_hash="same"
        )
        for frame in (first, changed)
    ]
    assert [r["selection_sha256"] for r in reports[0]["runs"]] == [
        r["selection_sha256"] for r in reports[1]["runs"]
    ]
    assert reports[0]["runs"][0]["activity_unknown_selected"] == 1
    unlabeled = benchmark.validate_frame(first.drop(columns="active"))
    result = benchmark.run_benchmark(unlabeled, config, source="test", source_hash="same")
    assert result["runs"][0]["active_recovery"] is None
    assert result["runs"][0]["known_inactive_selected"] is None


def test_random_priority_is_stable_between_thresholds():
    frame = benchmark.synthetic_frame(200, 7)
    config = benchmark.BenchmarkConfig(count=20, repeats=1, thresholds=(0.0, 0.5))
    all_pool = frame.loc[frame.approved]
    tight_pool = all_pool.loc[all_pool.qed.ge(0.5)]
    _, before = benchmark._one_run(all_pool, config, "seeded_random", 0, 0.0, frame.index)
    _, after = benchmark._one_run(tight_pool, config, "seeded_random", 0, 0.5, frame.index)
    assert before.intersection(tight_pool.index).issubset(after)


@pytest.mark.parametrize(
    "column,values",
    [
        ("record_id", [1, 1, 2, 3, 4, 5]),
        ("approved", [1, 0, 1, 1, 1, None]),
        ("qed", [0.1, 0.2, 0.3, 0.4, 0.5, float("inf")]),
        ("active", [0, 1, 0, 1, 2, None]),
    ],
)
def test_invalid_input_rejected(column, values):
    with pytest.raises(ValueError):
        benchmark.validate_frame(library().assign(**{column: values}), "active")


def test_cli_real_csv_json_and_csv_outputs(tmp_path):
    source = tmp_path / "cached.csv"
    library().to_csv(source, index=False)
    output = tmp_path / "report.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(source),
            "--activity-column",
            "active",
            "--count",
            "3",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(output.read_text())
    assert json.loads(completed.stdout) == report
    assert report["source"]["kind"] == "cached_csv"
    assert len(report["source"]["sha256"]) == 64
    assert len(pd.read_csv(output.with_suffix(".csv"))) == len(report["runs"])
    assert "not measured" in report["limitations"][0]


def test_synthetic_main_and_output_guards(tmp_path, monkeypatch, capsys):
    output = tmp_path / "smoke.json"
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "--synthetic", "100", "--count", "10", "--output", str(output)]
    )
    assert benchmark.main() == 0
    assert json.loads(capsys.readouterr().out)["source"]["kind"] == "synthetic_smoke"
    assert json.loads(output.read_text())["memory"]["process_rss_peak_sampled_mib"] > 0
    for arguments in (
        ["--synthetic", "1", "--output", str(output.with_suffix(".csv"))],
        ["--input", str(output.with_suffix(".csv")), "--output", str(output)],
        ["--synthetic", "0", "--output", str(output)],
        ["--synthetic", "10", "--activity-column", "active", "--output", str(output)],
    ):
        monkeypatch.setattr(sys, "argv", [str(SCRIPT), *arguments])
        with pytest.raises(SystemExit) as error:
            benchmark.main()
        assert error.value.code == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"count": 0},
        {"repeats": 0},
        {"seed": -1},
        {"thresholds": ()},
        {"thresholds": (0.0, 0.0)},
        {"thresholds": (float("nan"),)},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        benchmark.BenchmarkConfig(**kwargs)
