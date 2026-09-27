"""Public CLI regression: synthetic labels test software, never scientific performance."""

import json
import shutil

import pandas as pd
import pytest

from s2s_decision.artifacts import file_hash, read_bundle
from s2s_decision.cli import main


def invoke(command, source, destination, *options):
    return main([command, "--input", str(source), "--output", str(destination), *map(str, options)])


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def local_workflow(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_cli_workflow")
    rings = (
        "c1ccccc1",
        "c1ccncc1",
        "c1ccoc1",
        "c1ccsc1",
        "C1CCCCC1",
        "C1CCCC1",
        "C1CCNCC1",
        "C1CCOCC1",
    )
    # Eight distinct scaffolds, each with five unique molecules and both classes.
    rows = [
        {
            "molecule_id": f"fixture-{ring_id}-{length}",
            "original_smiles": "C" * length + ring,
            "target_id": "SYNTHETIC_FIXTURE",
            "endpoint": "Ki",
            "pactivity": 5.0 if length % 2 else 7.0,
            "relation": "=",
            "quality": "high",
            "source": "synthetic_software_test",
        }
        for ring_id, ring in enumerate(rings)
        for length in range(1, 6)
    ]
    pd.DataFrame(rows).to_csv(root / "measurements.csv", index=False)
    assert (
        invoke(
            "dataset",
            root / "measurements.csv",
            root / "dataset",
            "--target",
            "SYNTHETIC_FIXTURE",
            "--endpoint",
            "Ki",
            "--fingerprint-bits",
            1024,
        )
        == 0
    )
    assert (
        invoke(
            "train",
            root / "dataset",
            root / "model",
            "--epochs",
            1,
            "--patience",
            1,
            "--threads",
            1,
        )
        == 0
    )
    candidates = pd.DataFrame(
        {
            "ID": ["NA", "NULL", "001", "upstream-excluded", "invalid"],
            "Original_SMILES": [
                "CCCCCCc1ccccc1",
                "CCCCCCc1ccncc1",
                "CCCCCCc1ccoc1",
                "CCCCCCc1ccsc1",
                "not-a-SMILES",
            ],
            "eligible": [True, True, True, False, True],
            "Source_File": ["upstream_µ.csv"] * 5,
            "Source_Row": [3, 7, 9, 14, 17],
        }
    )
    candidates.to_csv(root / "candidates.csv", index=False)
    assert (
        invoke("prepare", root / "candidates.csv", root / "prepared", "--fingerprint-bits", 1024)
        == 0
    )
    assert (
        invoke(
            "predict",
            root / "prepared",
            root / "scored",
            "--model",
            root / "model",
            "--batch-size",
            2,
        )
        == 0
    )
    return root


def test_cli_training_to_final_export_preserves_task_and_source_evidence(local_workflow):
    root = local_workflow
    dataset = read_bundle(root / "dataset")
    model = load_json(root / "model" / "manifest.json")
    refs = read_bundle(root / "model" / "references")
    scored = read_bundle(root / "scored")
    assert len(dataset.records) == dataset.records.identity.nunique() == 40
    assert dataset.records.groupby("murcko_scaffold").split.nunique().eq(1).all()
    assert set(dataset.records.split) == {"train", "validation", "calibration", "test"}
    assert set(refs.records.identity) == set(
        dataset.records.loc[dataset.records.split.eq("train"), "identity"]
    )
    assert set(refs.records.split) == {"train"}
    assert model["reference_records_sha256"] == refs.manifest["records_sha256"]
    assert model["provenance"]["records_sha256"] == dataset.manifest["records_sha256"]
    assert model["heldout_test_evaluated"] is False
    assert scored.manifest["task"] == {
        "target_id": "SYNTHETIC_FIXTURE",
        "endpoint": "Ki",
        "threshold": 6.0,
    }
    assert scored.manifest["reference_records_sha256"] == model["reference_records_sha256"]
    assert scored.records.loc[scored.records.eligible, "priority_score"].between(0, 1).all()
    assert scored.records.loc[~scored.records.eligible, "priority_score"].isna().all()
    assert scored.manifest["scored_count"] == 3
    assert "y_active" not in scored.records and "pactivity" not in scored.records
    assert (
        invoke(
            "select",
            root / "scored",
            root / "selected",
            "--n",
            2,
            "--max-per-scaffold",
            1,
            "--min-scaffolds",
            2,
        )
        == 0
    )
    final = pd.read_csv(root / "selected" / "final.csv", dtype=str, keep_default_na=False)
    selection = load_json(root / "selected" / "selection.json")
    assert len(final) == final.murcko_scaffold.nunique() == 2
    assert set(final.molecule_id).issubset({"NA", "NULL", "001"})
    assert final["source__ID"].tolist() == final.molecule_id.tolist()
    assert set(final.source_file) == {"upstream_µ.csv"}
    assert set(final.source_row).issubset({"3", "7", "9"})
    assert selection["final_csv_sha256"] == file_hash(root / "selected" / "final.csv")
    assert selection["input_provenance"]["model_sha256"] == model["onnx_sha256"]


def test_cli_explicit_heldout_evaluation_and_baselines(local_workflow):
    root = local_workflow
    assert (
        invoke(
            "evaluate",
            root / "dataset",
            root / "evaluation.json",
            "--model",
            root / "model",
            "--n",
            2,
        )
        == 0
    )
    report = load_json(root / "evaluation.json")
    assert report["split"] == "test"
    assert (
        report["dataset_records_sha256"] == read_bundle(root / "dataset").manifest["records_sha256"]
    )
    # Runtime only; numeric scores from synthetic labels are never acceptance evidence.
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        assert invoke("benchmark", root / "dataset", root / "benchmark.json", "--n", 2) == 0
    benchmark = load_json(root / "benchmark.json")
    assert set(benchmark["baselines"]) == {"logistic", "gradient_boosting", "similarity", "qed"}
    assert benchmark["dataset_sha256"] == report["dataset_records_sha256"]


@pytest.mark.parametrize("guard", ["chemistry", "reference", "reference_records"])
def test_cli_rejects_incompatible_or_tampered_runtime_packages(
    local_workflow, tmp_path, capsys, guard
):
    root = local_workflow
    model_path = tmp_path / "model"
    prepared_path = tmp_path / "prepared"
    shutil.copytree(root / "model", model_path)
    shutil.copytree(root / "prepared", prepared_path)
    if guard == "reference_records":
        records = model_path / "references" / "records.jsonl"
        records.write_bytes(records.read_bytes() + b"\n")
    else:
        path = (
            prepared_path / "manifest.json"
            if guard == "chemistry"
            else model_path / "manifest.json"
        )
        manifest = load_json(path)
        manifest["chemistry_hash" if guard == "chemistry" else "reference_records_sha256"] = (
            "tampered"
        )
        path.write_text(json.dumps(manifest), encoding="utf-8")
    assert invoke("predict", prepared_path, tmp_path / "scored", "--model", model_path) == 2
    error = capsys.readouterr().err
    assert "chemistry" in error if guard == "chemistry" else "checksum" in error
    assert not (tmp_path / "scored").exists()


def test_cli_resume_rejects_changed_contract_without_overwriting_model(
    local_workflow, tmp_path, capsys
):
    root = local_workflow
    model_path = tmp_path / "model"
    shutil.copytree(root / "model", model_path)
    original = {
        name: file_hash(model_path / name)
        for name in ("model.onnx", "manifest.json", "checkpoint.pt")
    }
    assert (
        invoke(
            "train",
            root / "dataset",
            model_path,
            "--resume",
            "--epochs",
            2,
            "--patience",
            1,
            "--threads",
            1,
            "--learning-rate",
            0.5,
        )
        == 2
    )
    assert "Resume incompatible" in capsys.readouterr().err
    assert {name: file_hash(model_path / name) for name in original} == original
