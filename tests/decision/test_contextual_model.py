"""Behavioral contract for inspectable contextual activity packages."""

import copy

import numpy as np
import pandas as pd
import pytest

from s2s_decision import contextual_model as model
from s2s_decision.artifacts import write_json
from s2s_decision.features import featurize


def molecules():
    return featurize(pd.DataFrame({"record_id": [0, 1, 2], "original_smiles": [
        "CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]})).records


def bundle(frame):
    spec = model.fit_spec(frame, ["mol_wt"], "morgan")
    n = model.feature_matrix(frame, spec).shape[1]
    return {"schema": model.SCHEMA, "family": "logistic", "task_id": "T_IC50",
            "stage": "hit", "source": "training only", "spec": spec,
            "weights": [0.0] * n, "intercept": 0.0,
            "context": {"target": "T", "species": "Homo sapiens", "endpoint": "IC50", "stage": "hit_finding",
                        "chemistry_version": model.chemistry_manifest(2048)["chemistry_hash"],
                        "activity_threshold": 1000., "activity_unit": "nM", "activity_relation": "<="},
            "activity_threshold": {"value": 1000., "unit": "nM", "relation": "<="},
            "calibration": {"slope": 1., "intercept": 0., "n": 30, "positive_n": 15, "negative_n": 15},
            "conformal": {"alpha": .1, "quantiles": [.6, .6], "counts": [10, 10]},
            "training_fingerprints": frame.fingerprint_hex.tolist(),
            "domain_threshold": .3, "chemistry": model.chemistry_manifest(2048)}


def test_annotations_match_canonical_counts_and_normalized_rules():
    frame = molecules()
    got = model.annotate_chemistry(frame)
    assert frame.columns.tolist() == molecules().columns.tolist()
    assert np.array_equal(got.filter(regex="^alert__pains:").sum(axis=1), frame.pains_count)
    assert np.array_equal(got.filter(regex="^alert__brenk:").sum(axis=1), frame.brenk_count)
    assert np.allclose(got.rule__lip_mw_max__normalized_excess,
                       np.maximum(frame.mol_wt - 500, 0) / 500)
    assert all(isinstance(ids, list) for ids in got.alert_ids)
    assert all(isinstance(details, list) for details in got.alert_details)
    stale = model.annotate_chemistry(frame.assign(alert__fake=True))
    assert "alert__fake" not in stale


def test_preprocessing_train_only_missing_and_alternate_fingerprint():
    frame = molecules().assign(mol_wt=[1., np.nan, 3.])
    for fp, bits in (("morgan", 2048), ("maccs", 167), ("none", 0)):
        spec = model.fit_spec(frame, ["mol_wt"], fp)
        assert spec["median"] == [2.]
        x = model.feature_matrix(frame, spec)
        assert x.shape == (3, bits + 2)
        assert np.isfinite(x).all()
        assert x[1, -1] == 1
        assert model.feature_matrix(frame.assign(mol_wt=999), spec)[0, -2] > 100


def test_portable_predictions_contributions_and_hash_roundtrip(tmp_path):
    frame = molecules()
    package = bundle(frame)
    package["weights"][-2] = 2
    path = tmp_path / "model.json"
    write_json(path, package)
    loaded = model.load_bundle(path)
    out = model.predict_portable(frame, loaded)
    assert len(out["probabilities"]) == len(frame)
    assert np.isfinite(out["probabilities"]).all()
    assert out["in_domain"].all()
    assert all(isinstance(s, list) for s in out["prediction_sets"])
    assert np.allclose(out["contributions"].sum(axis=1), out["logits"])
    scored = model.score_frame(frame, loaded)
    assert np.array_equal(scored.record_id, frame.record_id)
    assert all(isinstance(value, dict) for value in scored.activity_contributions)
    assert np.allclose([sum(value.values()) for value in scored.activity_contributions], scored.activity_logit)
    assert "activity_score" not in frame


@pytest.mark.parametrize("change", [
    {"schema": "bad"}, {"weights": [1.]}, {"intercept": float("nan")},
    {"domain_threshold": 2}, {"training_fingerprints": []},
    {"calibration": {"slope": -1., "intercept": 0}},
    {"conformal": {"alpha": .1, "quantiles": [float("nan"), .1]}},
])
def test_malformed_bundle_fails(change):
    frame = molecules()
    with pytest.raises(ValueError):
        model.predict_portable(frame, {**bundle(frame), **change})


def test_feature_input_errors():
    frame = molecules()
    spec = model.fit_spec(frame, ["mol_wt"], "morgan")
    for bad in (frame.drop(columns="mol_wt"), frame.assign(mol_wt=np.inf)):
        with pytest.raises(ValueError):
            model.feature_matrix(bad, spec)
    for columns in (["y_active"], ["mol_wt", "mol_wt"]):
        with pytest.raises(ValueError):
            model.fit_spec(frame, columns, "morgan")
    with pytest.raises(ValueError):
        model.fit_spec(frame, ["mol_wt"], "other")
    assert model.feature_matrix(frame.iloc[:0], spec).shape == (0, 2050)


def test_invalid_smiles_and_mismatched_chemistry():
    frame = molecules()
    with pytest.raises(ValueError):
        model.annotate_chemistry(frame.assign(model_smiles="invalid?"))
    package = bundle(frame)
    changed = copy.deepcopy(package)
    changed["chemistry"]["rdkit_version"] = "0.0.0"
    with pytest.raises(ValueError):
        model.validate_bundle(changed)


def test_prediction_sets_include_boundaries_and_empty_sets():
    assert model.prediction_sets(np.array([0., .5, 1.]), [.2, .2]) == [[0], [], [1]]
    assert model.prediction_sets(np.array([.2, .8]), [.2, .2]) == [[0], [1]]
    assert model.prediction_sets(np.array([.5]), [1., 1.]) == [[0, 1]]


def test_loaded_feature_recipe_cannot_read_outcome_or_metadata():
    frame = molecules()
    package = bundle(frame)
    package["spec"] = {"fingerprint": "none", "columns": ["y_active"],
                       "median": [0.], "mean": [0.], "scale": [1.]}
    package["weights"] = [10., 0.]
    with pytest.raises(ValueError):
        model.validate_bundle(package)


@pytest.mark.parametrize("field,value", [("target", "OTHER"), ("endpoint", "Ki"),
                                          ("chemistry_version", "changed"), ("activity_threshold", 100.),
                                          ("activity_unit", "uM"), ("activity_relation", ">=")])
def test_bundle_context_must_match_model_recipe(field, value):
    package = bundle(molecules())
    package["context"][field] = value
    with pytest.raises(ValueError):
        model.validate_bundle(package)


def test_extreme_logit_and_large_batch_are_finite():
    frame = molecules()
    package = bundle(frame)
    for intercept in (-1e6, 1e6):
        package["intercept"] = intercept
        result = model.predict_portable(frame, package)
        assert np.isfinite(result["probabilities"]).all()
        assert np.array_equal(result["probabilities"], np.full(3, float(intercept > 0)))
    large = pd.concat([frame] * 3334, ignore_index=True)
    assert model.feature_matrix(large, package["spec"]).shape == (10002, 2050)


@pytest.mark.parametrize("kind", ["scale", "mean_length", "calibration_counts", "calibration_size",
                                   "conformal_alpha", "conformal_counts", "stage", "species", "threshold"])
def test_untrusted_package_metadata_rejected(kind):
    package = bundle(molecules())
    if kind == "scale":
        package["spec"]["scale"] = [0.]
    elif kind == "mean_length":
        package["spec"]["mean"] = [0., 1.]
    elif kind == "calibration_counts":
        package["calibration"]["positive_n"] = -1
    elif kind == "calibration_size":
        package["calibration"]["n"] = 100
    elif kind == "conformal_alpha":
        package["conformal"]["alpha"] = 1.
    elif kind == "conformal_counts":
        package["conformal"]["counts"] = [-1, 4]
    elif kind == "threshold":
        package["activity_threshold"]["value"] = 1.
    else:
        package["context"][kind] = "other"
    with pytest.raises(ValueError):
        model.validate_bundle(package)


def test_empty_and_missing_numeric_recipe():
    frame = molecules()
    with pytest.raises(ValueError):
        model.fit_spec(frame.iloc[:0], ["mol_wt"])
    with pytest.raises(ValueError):
        model.fit_spec(frame, [])
    with pytest.raises(ValueError):
        model.feature_matrix(frame.assign(mol_wt="bad"), bundle(frame)["spec"])
    assert model.fit_spec(frame.assign(mol_wt=np.nan), ["mol_wt"])["median"] == [0.]


def test_score_batches_preserve_input_order_and_contributions(monkeypatch):
    small = molecules()
    frame = pd.concat([small, small, small.iloc[:1]])
    frame.index = [9, 2, 5, 9, 1, 0, -1]
    package = bundle(small)
    package["weights"][-2] = 2.3
    expected = model.predict_portable(frame, package)
    original = model.predict_portable
    sizes = []
    def watched(part, saved):
        sizes.append(len(part))
        return original(part, saved)
    monkeypatch.setattr(model, "predict_portable", watched)
    result = model.score_frame(frame, package, batch_size=2)
    assert sizes == [2, 2, 2, 1]
    assert result.index.tolist() == frame.index.tolist()
    assert result.record_id.tolist() == frame.record_id.tolist()
    assert np.array_equal(result.activity_score, expected["probabilities"])
    assert np.allclose([sum(v.values()) for v in result.activity_contributions], expected["logits"])
    empty = model.score_frame(small.iloc[:0], package, batch_size=2)
    assert empty.empty
    assert {"activity_score", "activity_contributions", "in_domain"} <= set(empty)


@pytest.mark.parametrize("batch_size", [0, -1, None, True, 1.5])
def test_invalid_batch_size_rejected(batch_size):
    frame = molecules()
    with pytest.raises(ValueError, match="batch_size"):
        model.score_frame(frame, bundle(frame), batch_size=batch_size)
