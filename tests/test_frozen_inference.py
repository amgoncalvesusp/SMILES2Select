"""CI-only smoke: bundled worker runs a real ONNX model without system Python."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from s2s_decision.artifacts import read_bundle, write_bundle
from s2s_decision.features import chemistry_manifest, featurize
from s2s_decision.schema import FeatureSet


def test_frozen_worker_scores_real_onnx_model(tmp_path: Path) -> None:
    bundle_root = os.environ.get("S2S_FROZEN_BUNDLE")
    if not bundle_root:
        pytest.skip("Set S2S_FROZEN_BUNDLE when testing a PyInstaller bundle")
    bundle = Path(bundle_root).resolve()
    executable = bundle / ("SMILES2Select.exe" if sys.platform == "win32" else "SMILES2Select")
    assert executable.is_file()
    bundled_names = {part.casefold() for path in bundle.rglob("*") for part in path.parts}
    assert not bundled_names.intersection({"torch", "sklearn", "onnx", "onnxscript", "skl2onnx"})

    from decision.test_baseline_packages import dataset

    from s2s_decision.workflows import train_baseline_dataset

    frame, training_data = dataset(tmp_path)
    model = tmp_path / "model"
    train_baseline_dataset(training_data, model, estimator="logistic")
    candidates = tmp_path / "candidates"
    write_bundle(
        FeatureSet(
            frame.iloc[-3:].assign(valid=True, eligible=[True, True, False]),
            chemistry_manifest(1024),
        ),
        candidates,
    )
    output = tmp_path / "scored"
    result = subprocess.run(
        [
            str(executable),
            "--s2s-worker",
            "-m",
            "s2s_decision",
            "predict",
            "--input",
            str(candidates),
            "--output",
            str(output),
            "--model",
            str(model),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    scores = read_bundle(output).records
    assert scores.priority_score.notna().sum() == 2
    assert scores.activity_probability.iloc[:2].between(0, 1).all()


@pytest.mark.parametrize("model_name", (
    "Q72547_WT_IC50", "P0DMS8_WT_Ki", "Q07869_WT_EC50",
))
def test_frozen_bundle_scores_its_distributed_model(tmp_path: Path, model_name: str) -> None:
    bundle_root = os.environ.get("S2S_FROZEN_BUNDLE")
    if not bundle_root:
        pytest.skip("Set S2S_FROZEN_BUNDLE when testing a PyInstaller bundle")
    bundle = Path(bundle_root).resolve()
    executable = bundle / ("SMILES2Select.exe" if sys.platform == "win32" else "SMILES2Select")
    model = bundle / "_internal" / "s2s_decision" / "bundled_models" / model_name
    assert all((model / name).is_file() for name in (
        "manifest.json", "model.onnx", "MODEL_CARD.md",
        "references/manifest.json", "references/records.jsonl",
    ))
    candidates = tmp_path / "candidates"
    write_bundle(
        featurize(pd.DataFrame({
            "record_id": [1, 2, 3],
            "original_smiles": ["CCO", "CCN", "CC(=O)O"],
            "eligible": [True, True, False],
        })),
        candidates,
    )
    output = tmp_path / "scored"
    result = subprocess.run(
        [str(executable), "--s2s-worker", "-m", "s2s_decision", "predict",
         "--input", str(candidates), "--output", str(output), "--model", str(model)],
        cwd=tmp_path, capture_output=True, text=True, timeout=120, check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    scores = read_bundle(output).records
    assert scores.priority_score.notna().sum() == 2
    assert scores.activity_probability.iloc[:2].between(0, 1).all()
