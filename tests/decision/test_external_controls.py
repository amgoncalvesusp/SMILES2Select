"""Paired control lineage and exact source supervision contracts."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/external_controls.py"
spec = importlib.util.spec_from_file_location("external_controls", SCRIPT)
controls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controls)


def source_fixture():
    molecules = pd.DataFrame({"identity": list("abcde"), "split": [
        "train", "train", "validation", "calibration", "test"],
        "record_id": range(5), "murcko_scaffold": list("abcde")})
    arrays = {"x": np.arange(15, dtype=np.float32).reshape(5, 3),
              "labels": np.array([[0], [np.nan], [1], [0], [1]], dtype=np.float32),
              "identities": molecules.identity.to_numpy(), "splits": molecules.split.to_numpy()}
    observations = pd.DataFrame({"identity": list("acde"), "task_id": ["T"] * 4,
                                 "y_active": [0, 1, 0, 1], "source_rows": [[1, 2], [3], [4], [5]],
                                 "split": ["train", "validation", "calibration", "test"]})
    mapping = pd.DataFrame({"source_row": range(1, 6), "identity": list("aacde"), "task_id": ["T"] * 5})
    raw = pd.DataFrame({"source_row": range(1, 6), "task_id": ["T"] * 5,
                        "pactivity": [4., 5., 7., 5., 8.], "relation": ["="] * 5})
    return molecules, arrays, observations, mapping, raw


def test_subset_retains_exact_global_features_splits_and_excludes_unknowns():
    molecules, arrays, *_ = source_fixture()
    subset, frame = controls.observed_subset(arrays, molecules, 0)
    np.testing.assert_array_equal(subset["x"], arrays["x"][[0, 2, 3, 4]])
    assert frame.identity.tolist() == list("acde")
    assert subset["splits"].tolist() == ["train", "validation", "calibration", "test"]
    assert subset["labels"].shape == (4, 1)


@pytest.mark.parametrize("field", ["identities", "splits"])
def test_subset_rejects_misaligned_lineage(field):
    molecules, arrays, *_ = source_fixture()
    arrays[field] = arrays[field][::-1]
    with pytest.raises(ValueError, match="align"):
        controls.observed_subset(arrays, molecules, 0)


def test_subset_rejects_invalid_labels_and_shared_scaffolds():
    molecules, arrays, *_ = source_fixture()
    with pytest.raises(ValueError, match="binary"):
        controls.observed_subset({**arrays, "labels": arrays["labels"] + 2}, molecules, 0)
    with pytest.raises(ValueError, match="scaffold"):
        controls.observed_subset(arrays, molecules.assign(murcko_scaffold="same"), 0)


def test_exact_activity_join_averages_only_declared_source_rows():
    molecules, arrays, observations, mapping, raw = source_fixture()
    subset, frame = controls.observed_subset(arrays, molecules, 0)
    joined = controls.join_activity(frame, subset["labels"], observations, mapping, raw, "T")
    assert joined.pactivity.tolist() == [4.5, 7., 5., 8.]
    assert joined.y_active.tolist() == [0., 1., 0., 1.]
    assert joined.split.tolist() == frame.split.tolist()
    assert "pactivity" not in frame


def test_exact_activity_join_uses_median_of_source_means():
    molecules, arrays, observations, mapping, raw = source_fixture()
    subset, frame = controls.observed_subset(arrays, molecules, 0)
    observations = observations.copy()
    observations.at[0, "source_rows"] = [1, 2, 6]
    mapping = pd.concat([mapping, pd.DataFrame({"source_row": [6], "identity": ["a"], "task_id": ["T"]})])
    raw = pd.concat([raw, pd.DataFrame({"source_row": [6], "task_id": ["T"], "pactivity": [5.5], "relation": ["="]})])
    joined = controls.join_activity(frame, subset["labels"], observations, mapping, raw, "T")
    assert joined.pactivity.iloc[0] == 5.


def test_control_inference_exposes_raw_logits_and_verified_tiny_references(tmp_path):
    from s2s_decision.artifacts import write_bundle
    from s2s_decision.context import training_context
    from s2s_decision.features import featurize
    from s2s_decision.multitask import train_multitask
    from s2s_decision.preprocessing import Preprocessor
    from s2s_decision.schema import CONTEXT_NAMES, FeatureSet, fingerprint_matrix
    from s2s_decision.training import TrainingConfig
    from s2s_decision.workflows import train_dataset

    features = featurize(pd.DataFrame({"original_smiles": ["C" * n for n in range(2, 18)]}))
    frame = features.records.assign(split=np.repeat(["train", "validation", "calibration", "test"], 4),
                                    y_active=[0., 1.] * 8, pactivity=[5., 7.] * 8,
                                    murcko_scaffold=[str(i) for i in range(16)])
    frame = pd.concat([frame, training_context(frame)], axis=1)
    prep = Preprocessor.fit(frame.loc[frame.split.eq("train")])
    properties, _ = prep.transform(frame)
    x = np.concatenate([fingerprint_matrix(frame, 2048), properties], axis=1)
    train_multitask(x, frame.y_active.to_numpy()[:, None], frame.split.to_numpy(),
                    tmp_path / "matched", hidden=(8,), max_epochs=1)
    actual = controls.score_control(frame, tmp_path / "matched", "matched", prep.to_dict())
    expected = np.load(tmp_path / "matched/logits.npy")[:, 0]
    np.testing.assert_allclose(actual["raw_logits"], expected, atol=1e-6)
    assert set(actual) == {"raw_logits", "raw_probabilities", "probabilities"}
    dataset = tmp_path / "dataset"
    write_bundle(FeatureSet(frame, {**features.manifest, "target_id": "T", "endpoint": "IC50",
                                    "threshold": 6., "context_method": "scaffold OOF"}), dataset)
    train_dataset(dataset, tmp_path / "tiny", TrainingConfig(epochs=1, batch_size=4))
    result = controls.score_control(frame.drop(columns=list(CONTEXT_NAMES)), tmp_path / "tiny", "tiny")
    assert np.isfinite(result["raw_logits"]).all()
    assert result["raw_logits"].shape == (16,)
    from s2s_decision.artifacts import read_bundle
    from s2s_decision.context import build_context

    context = build_context(frame, read_bundle(tmp_path / "tiny/references").records)
    reused = controls.score_control(frame, tmp_path / "tiny", "tiny", context_features=context)
    np.testing.assert_array_equal(reused["raw_logits"], result["raw_logits"])
    with pytest.raises(ValueError, match="Context"):
        controls.score_control(frame, tmp_path / "tiny", "tiny", context_features=context.iloc[::-1])
    with pytest.raises(ValueError, match="Unknown"):
        controls.score_control(frame, tmp_path, "unknown")
    path = tmp_path / "tiny/references/records.jsonl"
    path.write_text("corrupt")
    with pytest.raises(ValueError, match="checksum"):
        controls.score_control(frame, tmp_path / "tiny", "tiny")


@pytest.mark.parametrize("fault", ["missing", "identity", "relation", "label", "duplicate", "split"])
def test_exact_activity_join_rejects_bad_provenance(fault):
    molecules, arrays, observations, mapping, raw = source_fixture()
    subset, frame = controls.observed_subset(arrays, molecules, 0)
    if fault == "missing":
        raw = raw.iloc[1:]
    elif fault == "identity":
        mapping = mapping.assign(identity="wrong")
    elif fault == "relation":
        raw = raw.assign(relation=">")
    elif fault == "label":
        raw = raw.assign(pactivity=8.)
    elif fault == "duplicate":
        raw = pd.concat([raw, raw.iloc[:1]])
    else:
        observations = observations.assign(split="test")
    with pytest.raises(ValueError):
        controls.join_activity(frame, subset["labels"], observations, mapping, raw, "T")


def test_runner_freezes_before_fitting_and_preserves_full_lineage(tmp_path, monkeypatch):
    from s2s_decision import context, multitask, workflows
    from s2s_decision.schema import CONTEXT_NAMES

    molecules, arrays, observations, mapping, raw = source_fixture()
    molecules = molecules.assign(**dict.fromkeys(CONTEXT_NAMES, np.nan))
    data = tmp_path / "data"
    data.mkdir()
    for name, frame in [("molecules", molecules), ("observations", observations),
                        ("source-identity-map", mapping), ("raw-selected-measurements", raw)]:
        frame.to_parquet(data / f"{name}.parquet")
    (data / "chemistry.json").write_text('{"chemistry_hash": "fixture", "fingerprint_bits": 2048}')
    (tmp_path / "plan.json").write_text('{}')
    output = tmp_path / "controls"
    monkeypatch.setattr(controls, "TASKS", ("T",))
    monkeypatch.setattr(controls, "SEEDS", (42,))
    monkeypatch.setitem(sys.modules, "train_multitask", SimpleNamespace(load_dataset=lambda p: (
        arrays, [{"task_id": "T", "target_id": "TARGET", "endpoint": "IC50"}])))
    monkeypatch.setattr(context, "training_context", lambda frame, seed: pd.DataFrame(
        np.nan, index=frame.index, columns=CONTEXT_NAMES))
    calls = []

    def matched(x, labels, splits, folder, seed):
        assert (output / "protocol.json").is_file()
        np.testing.assert_array_equal(x, arrays["x"][[0, 2, 3, 4]])
        assert splits.tolist() == ["train", "validation", "calibration", "test"]
        assert labels[:, 0].tolist() == [0., 1., 0., 1.]
        calls.append(("matched", seed))
        folder.mkdir(parents=True)
        return {"history": [{}]}

    def tiny(dataset, folder, config):
        assert config.epochs == 40 and config.patience == 6 and config.batch_size == 512
        assert (dataset / "manifest.json").is_file()
        folder.mkdir(parents=True)
        calls.append(("tiny", config.seed))
        return {"epochs": 1}

    monkeypatch.setattr(multitask, "train_multitask", matched)
    monkeypatch.setattr(workflows, "train_dataset", tiny)
    monkeypatch.setattr(controls, "score_control", lambda frame, folder, kind: {
        "raw_logits": np.zeros(len(frame)), "raw_probabilities": np.full(len(frame), .5),
        "probabilities": np.full(len(frame), .5)})
    controls.run(data, output)
    assert calls == [("matched", 42), ("tiny", 42)]
    assert (output / "completion.json").is_file()
    lineage = pd.read_csv(output / "T/lineage.csv")
    assert lineage.identity.tolist() == list("acde")
    assert lineage.pactivity.tolist() == [4.5, 7., 5., 8.]
    with pytest.raises(FileExistsError):
        controls.freeze_protocol(data, output)
