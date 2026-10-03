"""Training-label consensus preserves frozen heldouts and exact source lineage."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def module():
    path = Path(__file__).parents[2] / "benchmarks/prepare_consensus.py"
    spec = importlib.util.spec_from_file_location("prepare_consensus", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def data():
    identities = np.array(list("abcdefgh"))
    labels = np.array([[0], [1], [1], [0], [1], [1], [1], [np.nan]], np.float32)
    splits = np.array(["train"] * 4 + ["validation", "calibration", "test", "train"])
    arrays = {
        "x": np.zeros((8, 3), np.float32),
        "labels": labels,
        "identities": identities,
        "splits": splits,
    }
    values = ["5", "7", "5;7", "5;6", "5;7", "5;7", "5;7"]
    raw = pd.DataFrame(
        [
            {
                "source_row": i,
                "task_id": "T_IC50",
                "original_smiles": f"C{i}",
                "measurement_id": f"m{i}",
                "pchembl_value": v,
                "pchembl_value_N": str(len(v.split(";"))),
                "pchembl_value_Mean": str(np.mean([float(x) for x in v.split(";")])),
                "pactivity": float(np.mean([float(x) for x in v.split(";")])),
                "y_active": int(i not in (0, 3)),
            }
            for i, v in enumerate(values)
        ]
    )
    mapping = raw[["source_row", "task_id", "original_smiles"]].assign(
        identity=identities[:7]
    )
    observations = pd.DataFrame(
        [
            {
                "identity": identities[i],
                "task_id": "T_IC50",
                "source_rows": [i],
                "measurement_ids": [f"m{i}"],
                "split": splits[i],
                "y_active": labels[i, 0],
            }
            for i in range(7)
        ]
    )
    return arrays, [{"task_id": "T_IC50"}], observations, raw, mapping


def test_mask_only_train_and_keep_unknown_rows_and_all_heldouts():
    original = data()
    labels, audit, summary = module().consensus_labels(*original, min_train_per_class=1)
    assert np.isnan(labels[2:4]).all()
    np.testing.assert_array_equal(labels[4:], original[0]["labels"][4:])
    np.testing.assert_array_equal(original[0]["labels"][:4, 0], [0, 1, 1, 0])
    assert summary["masked_train_labels"] == 2
    assert summary["crossing_observations"] == 5
    assert summary["train_rows_without_labels"] == 3
    assert audit.loc[audit.identity == "c", "constituent_min"].item() == 5
    assert audit.loc[audit.identity == "c", "constituent_max"].item() == 7
    assert audit.loc[audit.identity == "e", "status"].item() == "heldout_unchanged"


@pytest.mark.parametrize(
    "column,value",
    [
        ("pchembl_value", ""),
        ("pchembl_value", "5;nan"),
        ("pchembl_value", None),
        ("pchembl_value_N", "0"),
        ("pchembl_value_N", "1.5"),
        ("pchembl_value_N", "2"),
        ("pchembl_value_Mean", "4"),
        ("pactivity", 5.5),
        ("y_active", 1),
        ("task_id", "wrong"),
        ("original_smiles", "wrong"),
        ("measurement_id", "wrong"),
    ],
)
def test_malformed_source_fails_closed(column, value):
    arrays, tasks, observations, raw, mapping = data()
    raw.loc[0, column] = value
    with pytest.raises(ValueError):
        module().consensus_labels(arrays, tasks, observations, raw, mapping, 1)


def test_missing_duplicate_and_wrong_identity_provenance_rejected():
    arrays, tasks, observations, raw, mapping = data()
    broken = mapping.copy()
    broken.loc[0, "identity"] = "other"
    for source, mapped in [
        (raw.iloc[1:], mapping),
        (pd.concat([raw, raw.iloc[:1]]), mapping),
        (raw, broken),
    ]:
        with pytest.raises(ValueError):
            module().consensus_labels(arrays, tasks, observations, source, mapped, 1)
    with pytest.raises(ValueError, match="support"):
        module().consensus_labels(arrays, tasks, observations, raw, mapping, 2)


def test_incomplete_and_duplicate_observations_rejected():
    arrays, tasks, observations, raw, mapping = data()
    for broken in [
        observations.iloc[1:],
        pd.concat([observations, observations.iloc[:1]]),
    ]:
        with pytest.raises(ValueError):
            module().consensus_labels(arrays, tasks, broken, raw, mapping, 1)


def test_prepare_compatible_dataset_hashes_and_refuses_overwrite(tmp_path):
    from s2s_decision.artifacts import file_hash

    arrays, tasks, observations, raw, mapping = data()
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    np.savez_compressed(source / "arrays.npz", **arrays)
    (source / "tasks.json").write_text(json.dumps(tasks))
    for name in ["preprocess.json", "chemistry.json"]:
        (source / name).write_text("{}")
    for name, frame in [
        ("observations.parquet", observations),
        ("raw-selected-measurements.parquet", raw),
        ("source-identity-map.parquet", mapping),
        ("molecules.parquet", mapping),
    ]:
        frame.to_parquet(source / name, index=False)
    (source / "completion.json").write_text(
        json.dumps({"files": {p.name: file_hash(p) for p in source.iterdir()}})
    )
    result = module().prepare(source, output, min_train_per_class=1)
    prepared, prepared_tasks = module().load_dataset(output)
    assert result["masked_train_labels"] == 2
    assert prepared_tasks == tasks
    for key in ["x", "splits", "identities"]:
        np.testing.assert_array_equal(prepared[key], arrays[key])
    assert file_hash(output / "tasks.json") == file_hash(source / "tasks.json")
    with pytest.raises(FileExistsError):
        module().prepare(source, output, min_train_per_class=1)
    with (source / "observations.parquet").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        module().prepare(source, tmp_path / "bad", min_train_per_class=1)
