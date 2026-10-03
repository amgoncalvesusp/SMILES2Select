"""B15 fitting/evaluation contracts: no truth reuse, inspectable estimators."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import contextual_policy_fit as fit
import contextual_policy_study as study

from s2s_decision.contextual_model import annotate_chemistry
from s2s_decision.features import featurize


def test_calibration_scaffold_partition_is_label_blind_and_disjoint():
    frame = pd.DataFrame({"identity": [f"i{i}" for i in range(100)],
                          "murcko_scaffold": [f"s{i//2}" for i in range(100)],
                          "y_active": np.arange(100) % 2})
    a = fit.calibration_partition(frame, 42)
    b = fit.calibration_partition(frame.assign(y_active=1 - frame.y_active), 42)
    assert np.array_equal(a, b)
    assert set(a) == {"probability", "conformal"}
    assert frame.assign(part=a).groupby("murcko_scaffold").part.nunique().max() == 1


def test_conformal_finite_sample_quantile_and_class_coverage():
    y = np.array([0] * 20 + [1] * 20)
    p = np.r_[np.linspace(0, .5, 20), np.linspace(.5, 1, 20)]
    saved = fit.fit_conformal(y, p)
    assert saved["counts"] == [20, 20]
    assert saved["quantiles"][0] == sorted(p[:20])[18]
    report = fit.probability_metrics(y, p, saved)
    assert report["coverage"] >= .9
    assert report["brier"] < .1
    assert report["ece"] >= 0
    assert fit.fit_conformal(np.array([0]), np.array([.2]))["quantiles"] == [1., 1.]


def test_platt_never_inverts_ranking_and_insufficient_support_explicit():
    x = np.tile([-2., 2.], 30)
    y = np.tile([0, 1], 30)
    cal = fit.fit_calibrator(x, y)
    assert cal["slope"] >= 0
    assert cal["status"] == "fitted"
    with pytest.raises(ValueError):
        fit.fit_calibrator(x, np.ones(60))
    with pytest.raises(ValueError):
        fit.fit_calibrator([np.inf], [0])


def test_exported_gradient_trees_match_estimator():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(100, 6))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    estimator, saved = fit.fit_boosting(x, y, 2, 42)
    assert np.allclose(fit.predict_estimator(x, saved), estimator.decision_function(x), atol=1e-10)
    with pytest.raises(ValueError):
        fit.predict_estimator(x, {"family": "unknown"})


def test_exported_gradient_threshold_boundaries_match_float32_sklearn():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(100, 6))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    estimator, saved = fit.fit_boosting(x, y, 2, 42)
    probes = []
    for tree in estimator.estimators_[:, 0]:
        for feature, threshold in zip(tree.tree_.feature, tree.tree_.threshold):
            if feature >= 0:
                for eps in (-1e-12, 0., 1e-12):
                    row = x[0].copy()
                    row[feature] = threshold + eps
                    probes.append(row)
    probes = np.asarray(probes)
    assert np.allclose(fit.predict_estimator(probes, saved), estimator.decision_function(probes), atol=1e-10)


def test_support_admission_train_only_and_unknown_not_negative():
    base = pd.DataFrame({"identity": [f"i{i}" for i in range(40)],
        "split": ["train"] * 20 + ["test"] * 20, "y_active": [0, 1] * 20,
        "murcko_scaffold": [f"s{i}" for i in range(40)],
        "document_ids": [[f"d{i%5}"] for i in range(40)],
        "rule__r__violation": [1] * 40, "rule__r__normalized_excess": [.1] * 40,
        "alert__rare": [0] * 20 + [1] * 20})
    groups, support = fit.support_features(base)
    assert groups["alerts"] == []
    assert groups["rules"] == ["rule__r__violation", "rule__r__normalized_excess"]
    assert support[0]["support_n"] <= 20
    with pytest.raises(ValueError):
        fit.support_features(base.assign(y_active=np.nan))


def test_selection_universe_shortfall_and_common_denominator():
    f = pd.DataFrame({"record_id": [1, 2, 3], "identity": ["a", "b", "c"],
        "murcko_scaffold": ["s", "s", "x"], "y_active": [1, 1, 0]})
    metric, rows = study.select_basket(f, np.array([.9, .8, .7]),
                                      np.array([True, True, False]), 3, 1)
    assert metric["selected_n"] == 1
    assert metric["shortfall"] == 2
    assert metric["recall"] == .5
    assert rows.is_final.sum() == 1


def test_stage_profiles_have_explicit_scenario_not_relabeling():
    frame = featurize(pd.DataFrame({"record_id": [0, 1], "original_smiles": [
        "CCO", "CCCCCCCCCCCCCCCCCCCCCCCCCCCC"]})).records
    masks = study.stage_masks(frame)
    assert set(masks) == {"hit", "lead", "fragment_core", "fragment_extended"}
    assert masks["hit"].all()
    assert not masks["fragment_core"][1]


def test_production_policy_retains_alerts_and_matches_default_activity_ranking():
    frame = synthetic_panel().query("task_id == 'T1_HUMAN_IC50' and split == 'test'").reset_index(drop=True)
    probability = np.linspace(.05, .95, len(frame))
    values = {"probability": probability, "conformal": {"quantiles": [.6, .6]}}
    model = {"calibration": {"method": "monotone sigmoid", "n": 20, "positive_n": 10, "negative_n": 10}}
    metric, result = study.production_basket(frame, values, model, frame.task_id.iloc[0], 42,
                                            np.ones(len(frame)), 10, 3)
    _, control = study.select_basket(frame, probability, np.ones(len(frame), bool), 10, 3)
    assert set(result.loc[result.is_final, "identity"]) == set(control.loc[control.is_final, "identity"])
    assert metric["review_selected"] == metric["selected_n"]  # Risk remains unmeasured.
    assert result.policy_warnings.map(lambda items: "risk_unknown" in items).all()
    _, transferred = study.production_basket(frame.assign(assay_context="new_fluorescence"),
        values, model, frame.task_id.iloc[0], 42, np.ones(len(frame)), 10, 3)
    assert transferred.policy_warnings.map(lambda items: "context_mismatch" in items).all()
    assert transferred.policy_warnings.map(lambda items: "activity_uncalibrated" in items).all()
    assert set(transferred.loc[transferred.is_final, "identity"]) == set(result.loc[result.is_final, "identity"])


def test_analogue_pairs_never_cross_scaffolds_or_repeat():
    raw = pd.DataFrame({"record_id": [0, 1, 2], "original_smiles": [
        "Cc1ccccc1", "Oc1ccccc1", "CCO"]})
    frame = annotate_chemistry(featurize(raw).records).assign(y_active=[0, 1, 0],
        assay_context=["a", "a", "b"], identity=["a", "b", "c"])
    pairs = study.analogue_pairs(frame, .1)
    assert len(pairs) == 1
    assert set(pairs.iloc[0][["identity_a", "identity_b"]]) == {"a", "b"}


def test_group_leakage_and_unknown_labels_rejected():
    with pytest.raises(ValueError):
        fit.validate_panel(pd.DataFrame())


def synthetic_panel():
    base = annotate_chemistry(featurize(pd.DataFrame({"record_id": [0, 1],
        "original_smiles": ["CCO", "c1ccccc1"]})).records)
    rows = []
    for task_id in ("T1_HUMAN_IC50", "T2_HUMAN_Ki"):
        offset = 0
        for split, count in (("train", 40), ("validation", 20), ("calibration", 80), ("test", 20), ("recent", 20)):
            for i in range(count):
                row = base.iloc[i % 2].to_dict()
                rows.append({**row, "record_id": offset + i, "identity": f"{split}-{i}",
                    "task_id": task_id, "split": split, "y_active": i % 2,
                    "murcko_scaffold": f"{split}-{i//2}", "assay_context": ["color", "fluo"][i%2],
                    "source_document_ids": '["d1","d2","d3"]'})
            offset += count
    return pd.DataFrame(rows)


def test_complete_train_calibrate_predict_select_export_and_integrity(tmp_path, monkeypatch):
    panel = synthetic_panel()
    path = tmp_path / "panel.parquet"
    panel.to_parquet(path, index=False)
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Frozen synthetic integration protocol", encoding="utf-8")
    def unavailable(_):
        raise study.QEXFitError("constant synthetic properties")
    monkeypatch.setattr(study, "fit_qex", unavailable)
    out = tmp_path / "train"
    package = study.train(path, out, protocol)
    assert package["qex"]["T1_HUMAN_IC50"]["status"] == "failed"
    assert len(list((out / "bundles").glob("*.json"))) == 6
    assert all(not (set(model["calibration_identities"]["probability"]) &
                       set(model["calibration_identities"]["conformal"]))
               for task in package["models"].values() for model in task.values())
    study.evaluate(path, out, tmp_path / "evaluation")
    metrics = pd.read_parquet(tmp_path / "evaluation/T1_HUMAN_IC50-test/selection-metrics.parquet")
    assert metrics.shortfall.max() > 0
    assert {"shared", "shared_context", "production_contextual", "review_priority_sensitivity", "lr_maccs", "boosting", "tolerance2"} <= set(metrics.arm)
    assert (metrics.hits <= metrics.selected_n).all()
    assert study.load_training(out)["settings"]["panel_sha256"] == study.file_hash(path)
    with pytest.raises(FileExistsError):
        study.train(path, out, protocol)
    (out / "models.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        study.load_training(out)


@pytest.mark.parametrize("mutation", ["unknown", "duplicate", "identity_split", "scaffold_split", "bad_split", "one_class"])
def test_validation_edge_cases(mutation):
    panel = synthetic_panel()
    if mutation == "unknown":
        panel.loc[0, "y_active"] = np.nan
    elif mutation == "duplicate":
        panel = pd.concat([panel, panel.iloc[:1]])
    elif mutation == "identity_split":
        panel.loc[1, "identity"] = "test-0"
    elif mutation == "scaffold_split":
        panel.loc[1, "murcko_scaffold"] = "test-0"
    elif mutation == "bad_split":
        panel.loc[0, "split"] = "invalid"
    else:
        panel.loc[panel.split.eq("validation"), "y_active"] = 1
    with pytest.raises(ValueError):
        fit.validate_panel(panel)


def test_unknown_shared_context_falls_back_without_new_column():
    frame = synthetic_panel()
    train = frame.loc[frame.split.eq("train")]
    groups = {task: fit.support_features(rows)[0] for task, rows in frame.groupby("task_id")}
    spec = fit.shared_spec(train, groups, True)
    x = fit.shared_matrix(train, spec)
    new = fit.shared_matrix(train.assign(assay_context="unseen"), spec)
    assert x.shape == new.shape
    context_columns = len(spec["contexts"]) * (1 + len(spec["chemical"]))
    assert not new[:, -context_columns:].any()


@pytest.mark.parametrize("labels,probabilities,alpha", [
    ([0], [np.nan], .1), ([0], [2.], .1), ([0], [.5], 0), ([np.nan], [.1], .1),
])
def test_conformal_invalid_data_fails(labels, probabilities, alpha):
    with pytest.raises(ValueError):
        fit.fit_conformal(labels, probabilities, alpha)
