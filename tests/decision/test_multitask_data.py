"""Multitask preparation preserves missingness and chemistry-wide partitions."""

import importlib.util
import json
import sys
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def runner():
    path = Path(__file__).parents[2] / "benchmarks/prepare_multitask.py"
    spec = importlib.util.spec_from_file_location("prepare_multitask", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_conflicts_remove_only_identity_task_pair_and_preserve_evidence():
    module = runner()
    rows = pd.DataFrame({
        "identity": ["a", "a", "a", "b", "b"],
        "task_id": ["t1", "t1", "t2", "t1", "t1"],
        "y_active": [0, 1, 1, 0, 0],
        "source_row": [1, 2, 3, 4, 5],
        "measurement_id": ["m1", "m2", "m3", "m4", "m5"],
    })
    clean, conflicts = module.collapse_observations(rows)
    assert clean[["identity", "task_id"]].values.tolist() == [["a", "t2"], ["b", "t1"]]
    assert clean.iloc[1].source_rows == [4, 5]
    assert conflicts[["identity", "task_id"]].values.tolist() == [["a", "t1"]]
    labels = module.make_labels(["a", "b"], ["t1", "t2"], clean)
    np.testing.assert_equal(labels, [[np.nan, 1], [0, np.nan]])


def test_support_gate_uses_train_and_validation_only():
    module = runner()
    observations = pd.DataFrame({
        "task_id": ["a"] * 5 + ["b"] * 4,
        "split": ["train", "train", "validation", "validation", "test"]
        + ["train", "train", "validation", "test"],
        "y_active": [0, 1, 0, 1, 0, 0, 1, 1, 0],
    })
    chosen, _ = module.select_tasks(observations, train_min=1, validation_min=1)
    assert chosen == ["a"]
    changed_test = observations.loc[~observations.split.eq("test")]
    assert module.select_tasks(changed_test, 1, 1)[0] == chosen


def test_global_scaffold_split_reused_for_all_tasks():
    module = runner()
    molecules = pd.DataFrame({
        "identity": [f"m{i}" for i in range(40)],
        "murcko_scaffold": [f"s{i // 2}" for i in range(40)],
    })
    partitioned = module.assign_splits(molecules, seed=42)
    assert partitioned.groupby("murcko_scaffold").split.nunique().max() == 1
    observations = pd.DataFrame({"identity": ["m0", "m0", "m1"], "task_id": ["a", "b", "a"]})
    attached = observations.merge(partitioned[["identity", "split"]], validate="many_to_one")
    assert attached.split.nunique() == 1


def test_label_matrix_rejects_invalid_observations():
    module = runner()
    clean = pd.DataFrame({"identity": ["a"], "task_id": ["t"], "y_active": [1]})
    with pytest.raises(ValueError, match="Duplicate"):
        module.make_labels(["a"], ["t"], pd.concat([clean, clean]))
    with pytest.raises(ValueError, match="binary"):
        module.make_labels(["a"], ["t"], clean.assign(y_active=2))
    with pytest.raises(ValueError, match="absent"):
        module.make_labels(["b"], ["t"], clean)


def test_sealed_settings_and_source_filter(tmp_path, monkeypatch):
    module = runner()
    module.seal(tmp_path / "settings.json", {"threshold": 6})
    module.seal(tmp_path / "settings.json", {"threshold": 6})
    with pytest.raises(ValueError, match="changed"):
        module.seal(tmp_path / "settings.json", {"threshold": 5})
    row = {"target_id": "T_WT", "endpoint": "Ki", "quality": "high", "relation": "=",
           "mixed_endpoint": False, "y_active": 1.0, "original_smiles": "CCO"}
    rejected = [dict(row, quality="low"), dict(row, relation="<"), dict(row, mixed_endpoint=True),
                dict(row, y_active=np.nan), dict(row, target_id="OTHER_WT")]
    monkeypatch.setattr(module, "_normalized", lambda *args: [[row, *rejected]])
    result = module.extract_source(tmp_path / "fake.tsv", {("T_WT", "Ki")}, tmp_path)
    assert len(result) == 1
    assert result.task_id.tolist() == ["T_WT_Ki"]
    assert len(module.extract_source(tmp_path / "fake.tsv", set(), tmp_path)) == 1


def test_finish_writes_aligned_arrays_and_train_only_preprocessor(tmp_path, monkeypatch):
    module = runner()
    from s2s_decision.schema import PROPERTY_NAMES

    n = 80
    features = pd.DataFrame({
        "original_smiles": [f"source{i}" for i in range(n)],
        "identity": [f"i{i:02d}" for i in range(n)],
        "murcko_scaffold": [f"s{i // 2}" for i in range(n)],
        "eligible": True,
        "fingerprint_hex": ["01" * 256] * n,
        **{name: np.arange(n, dtype=float) + 1 for name in PROPERTY_NAMES},
    })
    raw = features[["original_smiles"]].assign(
        source_row=np.arange(n), measurement_id=[f"row{i}" for i in range(n)],
        task_id="T_WT_Ki", target_id="T_WT", endpoint="Ki", y_active=np.arange(n) % 2,
    )
    original_gate = module.select_tasks
    monkeypatch.setattr(module, "select_tasks", lambda observations: original_gate(observations, 1, 1))
    module.finish(raw, features, tmp_path, tmp_path)
    arrays = np.load(tmp_path / "arrays.npz")
    assert arrays["x"].shape == (n, 2088)
    assert arrays["labels"].shape == (n, 1)
    records = pd.read_parquet(tmp_path / "molecules.parquet")
    assert records.identity.tolist() == arrays["identities"].tolist()
    restored = module.Preprocessor.from_dict(json.loads(
        (tmp_path / "preprocess.json").read_text(encoding="utf-8")))
    fitted = module.Preprocessor.fit(records.loc[records.split.eq("train")])
    assert restored == fitted
    assert (tmp_path / "completion.json").is_file()


def test_shared_chemistry_chunk_preserves_validity_and_fingerprints(tmp_path):
    module = runner()
    frame = pd.DataFrame({"record_id": [1, 2], "original_smiles": ["CCO", "not_smiles"]})
    path, count, _ = module.feature_chunk(frame, tmp_path / "chunk.parquet")
    records = pd.read_parquet(path)
    assert count == 2
    assert records.eligible.tolist() == [True, False]
    assert len(records.fingerprint_hex.iloc[0]) == 512
    assert pd.isna(records.identity.iloc[1])


def test_exposure_includes_previous_dataset_jsonl(tmp_path):
    module = runner()
    folder = tmp_path / "artifacts/data-capacity-v1/datasets/T/train100"
    folder.mkdir(parents=True)
    pd.DataFrame({"identity": ["seen"]}).to_json(folder / "records.jsonl", orient="records", lines=True)
    molecules = pd.DataFrame({"identity": ["seen", "new"], "split": ["test", "train"]})
    audit = module.exposure_audit(tmp_path, molecules)
    assert audit["historically_exposed_identities"] == 1
    assert audit["overlap_by_split"] == {"test": 1, "train": 0}


def test_feature_checkpoints_resume_with_identical_rows(tmp_path, monkeypatch):
    module = runner()

    class ImmediateExecutor:
        def __init__(self, max_workers):
            assert max_workers == 1

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, func, *args):
            future = Future()
            future.set_result(func(*args))
            return future

    monkeypatch.setattr(module, "ProcessPoolExecutor", ImmediateExecutor)
    raw = pd.DataFrame({"original_smiles": ["CCO", "CC", "CCO"]})
    first = module.generate_features(raw, tmp_path, 1)
    assert first.original_smiles.tolist() == ["CC", "CCO"]
    pd.testing.assert_frame_equal(module.generate_features(raw, tmp_path, 1), first)
    (tmp_path / "features-before-identity-collapse.parquet").unlink()
    pd.testing.assert_frame_equal(module.generate_features(raw, tmp_path, 1), first)


def test_cli_freezes_source_and_refuses_changed_completed_files(tmp_path, monkeypatch):
    module = runner()
    audit_dir = tmp_path / "artifacts/papyrus-audit"
    audit_dir.mkdir(parents=True)
    audit = {"source_sha256": module.SOURCE_HASH, "threshold": 6.0,
             "target_endpoint_counts": [{"target_id": "T_WT", "endpoint": "Ki", "rows": 500,
                                          "active_records": 250, "inactive_records": 250}]}
    module.write_json(audit_dir / "audit.json", audit)
    monkeypatch.setattr(sys, "argv", ["prepare_multitask", "--study-root", str(tmp_path), "--workers", "1"])
    real_hash = module.file_hash
    monkeypatch.setattr(module, "file_hash", lambda path: module.SOURCE_HASH
                        if str(path).endswith(".tsv.xz") else real_hash(path))
    monkeypatch.setattr(module, "extract_source", lambda source, candidates, output: pd.DataFrame())
    monkeypatch.setattr(module, "generate_features", lambda *args: pd.DataFrame())

    def fake_finish(raw, features, study, output):
        module.write_json(output / "audit.json", {"test": True})
        module.write_json(output / "completion.json", {"files": {"audit.json": real_hash(output / "audit.json")}})

    monkeypatch.setattr(module, "finish", fake_finish)
    module.main()
    module.main()
    output = tmp_path / "artifacts/s2-decision-multitask-v1/data"
    module.write_json(output / "audit.json", {"changed": True})
    with pytest.raises(ValueError, match="artifact changed"):
        module.main()
    monkeypatch.setattr(module, "file_hash", lambda path: "wrong")
    with pytest.raises(ValueError, match="checksum"):
        module.main()
