"""Target-panel safeguards; small fixtures are software checks, not biology."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def runner():
    folder = Path(__file__).parents[2] / "benchmarks"
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location(
        "train_target_panel", folder / "train_target_panel.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_observed_subset_preserves_unknowns_and_split_order():
    module = runner()
    arrays = {
        "x": np.arange(12).reshape(6, 2),
        "labels": np.array([[0, 1], [np.nan, 0], [1, np.nan], [0, 1], [1, 0], [0, 1]]),
        "splits": np.array(
            ["train", "train", "validation", "calibration", "test", "test"]
        ),
        "identities": np.array(list("abcdef")),
    }
    subset = module.single_task(arrays, 0)
    assert subset["identities"].tolist() == list("acdef")
    assert subset["labels"].shape == (5, 1)
    assert np.isnan(arrays["labels"][1, 0])
    assert subset["splits"].tolist() == [
        "train",
        "validation",
        "calibration",
        "test",
        "test",
    ]


def test_external_membership_and_dates_fail_closed():
    module = runner()
    dev = pd.DataFrame({"identity": ["d"], "murcko_scaffold": ["S"]})
    ext = pd.DataFrame(
        {"identity": ["x", "y"], "murcko_scaffold": ["S", "N"], "year": [2024, 2025]}
    )
    exposure = {"identities": ["old"], "scaffolds": ["O"]}
    assert module.external_novelty(dev, ext, exposure).tolist() == [False, True]
    for bad in (ext.assign(identity=["old", "y"]), ext.assign(identity=["d", "y"])):
        with pytest.raises(ValueError, match="exposed"):
            module.external_novelty(dev, bad, exposure)
    with pytest.raises(ValueError, match="2024"):
        module.external_novelty(dev, ext.assign(year=[2023, 2025]), exposure)


def test_metrics_ignore_unknown_and_do_not_call_scores_probabilities():
    module = runner()
    frame = pd.DataFrame(
        {"identity": ["b", "a", "c"], "murcko_scaffold": ["B", "A", "C"]}
    )
    report, ranked = module.observed_metrics(
        frame, np.array([1.0, 0.0, np.nan]), np.array([3.0, 3.0, 99.0]), None, None
    )
    assert report["observed"] == 2
    assert ranked.identity.tolist() == ["a", "b"]
    assert report["brier_raw"] is None
    assert report["selection"]["50"]["effective_k"] == 2


def test_calibration_is_applied_without_access_to_external_labels():
    module = runner()
    probability = module.apply_calibration(
        np.array([0.0, 1.0]), {"slope": 2.0, "intercept": -1.0}
    )
    np.testing.assert_allclose(probability, [0.268941421, 0.731058579])
    with pytest.raises(ValueError, match="finite"):
        module.apply_calibration(np.array([np.nan]), {"slope": 1.0, "intercept": 0.0})


def test_all_seeds_and_head_fits_receive_only_development(tmp_path, monkeypatch):
    module = runner()
    labels = np.tile([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], (4, 1))
    arrays = {
        "x": np.zeros((8, 2), dtype=np.float32),
        "labels": labels,
        "splits": np.repeat(["train", "validation", "calibration", "test"], 2),
        "identities": np.array([str(i) for i in range(8)]),
    }
    calls = []
    monkeypatch.setattr(
        module, "fit_baseline", lambda *args: calls.append(("lr", args[1].shape))
    )
    monkeypatch.setattr(
        module,
        "train_multitask",
        lambda x, y, split, path, **kw: calls.append((kw["seed"], y.shape)),
    )
    methods = module.fit_models(
        arrays, [{"task_id": str(i)} for i in range(3)], tmp_path
    )
    assert calls[0] == ("lr", (8, 3))
    assert len(calls) == 13
    assert len(methods) == 13
    for seed in (42, 43, 44):
        assert calls.count((seed, (8, 3))) == 1
        assert calls.count((seed, (8, 1))) == 3


def prepare(tmp_path, module):
    data = tmp_path / "data"
    data.mkdir()
    rows = 320
    identities = np.array([f"id{i}" for i in range(rows)])
    x = np.zeros((rows, 2088), dtype=np.float32)
    labels = np.tile([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], (rows // 2, 1))
    splits = np.repeat(
        ["train", "validation", "calibration", "test"], [200, 40, 40, 40]
    )
    np.savez(
        data / "arrays.npz", x=x, labels=labels, splits=splits, identities=identities
    )
    frame = pd.DataFrame(
        {
            "identity": identities,
            "murcko_scaffold": identities,
            "fingerprint_hex": "00" * 256,
            "qed": 0.5,
        }
    )
    frame.to_parquet(data / "molecules.parquet")
    ext = frame.iloc[:3].assign(
        identity=["e1", "e2", "e3"],
        murcko_scaffold=["new", "id1", "new2"],
        year=2024,
        scaffold_novel=[True, False, True],
    )
    ext.to_parquet(data / "external-molecules.parquet")
    np.savez(
        data / "external-arrays.npz",
        x=x[:3],
        labels=labels[:3],
        identities=ext.identity.to_numpy(dtype=str),
        scaffold_novel=ext.scaffold_novel.to_numpy(),
    )
    tasks = [{"task_id": f"target{i}"} for i in range(3)]
    for name, value in (("tasks", tasks), ("preprocess", {}), ("chemistry", {})):
        module.write_json(data / f"{name}.json", value)
    module.write_json(
        data / "development-exposure.json",
        {"identities": identities.tolist(), "scaffolds": identities.tolist()},
    )
    seal(data, module)
    exposure = tmp_path / "exposure.json"
    module.write_json(
        exposure, {"identities": ["historical"], "scaffolds": ["historical-scaffold"]}
    )
    plan = tmp_path / "plan.json"
    module.write_json(
        plan,
        {
            "training": {**module.TRAINING, "seeds": module.SEEDS},
            "comparison": {"K": module.KS, "primary_K": 50, "random_repetitions": 1000},
            "task_ids": [task["task_id"] for task in tasks],
        },
    )
    return data, exposure, plan


def seal(data, module):
    module.write_json(
        data / "completion.json",
        {
            "files": {
                p.name: module.file_hash(p)
                for p in data.iterdir()
                if p.name != "completion.json"
            }
        },
    )


def test_run_freezes_before_fit_and_scores_only_after_all_fits(tmp_path, monkeypatch):
    module = runner()
    data, exposure, plan = prepare(tmp_path, module)
    output = tmp_path / "run"
    fits = []

    def fit(x, y, splits, folder, *args, **kwargs):
        assert (output / "protocol.json").exists()
        assert x.shape[0] == 320
        assert set(splits) == {"train", "validation", "calibration", "test"}
        folder.mkdir()
        coefficients = [
            {"slope": 1.0, "intercept": 0.0, "method": "platt_calibration_only"}
        ] * y.shape[1]
        module.write_json(
            folder / "calibration.json",
            {"tasks": coefficients} if args else coefficients,
        )
        if args:
            np.savez(
                folder / "coefficients.npz",
                coefficients=np.zeros((3, 2088)),
                intercept=np.zeros(3),
            )
        fits.append(folder.name)

    monkeypatch.setattr(module, "fit_baseline", fit)
    monkeypatch.setattr(module, "train_multitask", fit)

    def predict(path, x, task):
        assert len(fits) == 13
        assert (output / "fits-complete.json").exists()
        return np.zeros(len(x))

    monkeypatch.setattr(module, "predict_checkpoint", predict)
    random_calls = []

    def random(truth, **kwargs):
        random_calls.append(kwargs)
        return {"repetitions": kwargs["repetitions"]}

    monkeypatch.setattr(module, "random_control", random)
    module.run(data, exposure, output, plan)
    result = json.loads((output / "evaluation.json").read_text())
    assert len(result["records"]) == 120
    assert len(random_calls) == 12
    assert all(call["repetitions"] == 1000 for call in random_calls)
    assert len(list(output.glob("*-scores.parquet"))) == 108
    assert (output / "completion.json").exists()
    with pytest.raises(FileExistsError):
        module.run(data, exposure, output, plan)


def test_load_rejects_alignment_insufficient_support_and_changed_inputs(tmp_path):
    module = runner()
    data, exposure, plan = prepare(tmp_path, module)
    arrays, ext, tasks, frames, secondary = module.load_panel(
        data, module.read_json(exposure)
    )
    assert secondary.tolist() == [True, False, True]
    with pytest.raises(ValueError, match="identity order"):
        module.validate_frame(frames[0].iloc[::-1], arrays)
    with pytest.raises(ValueError, match="QED"):
        module.validate_frame(frames[0].assign(qed=2), arrays)
    with pytest.raises(ValueError, match="Fingerprint"):
        module.validate_frame(frames[0].assign(fingerprint_hex="ff" * 256), arrays)
    np.savez(
        data / "arrays.npz",
        **{
            **arrays,
            "labels": np.where(arrays["labels"] == 0, np.nan, arrays["labels"]),
        },
    )
    seal(data, module)
    with pytest.raises(ValueError, match="Insufficient train"):
        module.load_panel(data, module.read_json(exposure))
    (data / "chemistry.json").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        module.load_panel(data, module.read_json(exposure))


@pytest.mark.parametrize("field", ["training", "comparison", "task_ids"])
def test_freeze_rejects_plan_drift(tmp_path, field):
    module = runner()
    data, exposure, plan = prepare(tmp_path, module)
    payload = module.read_json(plan)
    payload[field] = (
        [] if field == "task_ids" else {**payload[field], "seeds": [1], "K": [1]}
    )
    module.write_json(plan, payload)
    tasks = module.read_json(data / "tasks.json")
    with pytest.raises(ValueError):
        module.freeze(data, exposure, tmp_path / "run", plan, tasks)


@pytest.mark.parametrize("artifact", ["arrays", "molecules"])
def test_saved_external_novelty_must_match_recomputed_exposure(tmp_path, artifact):
    module = runner()
    data, exposure, _ = prepare(tmp_path, module)
    if artifact == "arrays":
        path = data / "external-arrays.npz"
        with np.load(path, allow_pickle=False) as saved:
            values = {key: saved[key] for key in saved.files}
        np.savez(path, **{**values, "scaffold_novel": np.array([True, True, True])})
    else:
        path = data / "external-molecules.parquet"
        pd.read_parquet(path).assign(scaffold_novel=True).to_parquet(path)
    seal(data, module)
    with pytest.raises(ValueError, match="Saved scaffold novelty"):
        module.load_panel(data, module.read_json(exposure))
