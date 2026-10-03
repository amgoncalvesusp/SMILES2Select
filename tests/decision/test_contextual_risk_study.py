"""Behavioral checks for measured-endpoint risk training, not inferred safety."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def module():
    path = Path(__file__).resolve().parents[2] / "benchmarks/contextual_risk_study.py"
    spec = importlib.util.spec_from_file_location("contextual_risk_study", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def risk():
    return module()


def sample(n=400):
    rows = []
    for i in range(n):
        bits = np.zeros(2048, dtype=np.uint8)
        bits[i % 11] = 1
        bits[100 + i % 31] = 1
        rows.append({"identity": f"id-{i}", "murcko_scaffold": f"scaffold-{i // 2}",
                     "fingerprint_hex": np.packbits(bits, bitorder="little").tobytes().hex(),
                     "y_risk": i % 2, "risk_endpoint": "measured_viability"})
    return pd.DataFrame(rows)


def test_split_is_global_by_scaffold_and_order_independent(risk):
    frame = sample()
    first = risk.prepare(frame, "measured_viability")
    second = risk.prepare(frame.sample(frac=1, random_state=4), "measured_viability")
    pd.testing.assert_frame_equal(first, second)
    assert first.groupby("murcko_scaffold").split.nunique().max() == 1
    assert set(first.split) == set(risk.SPLIT_NAMES)


@pytest.mark.parametrize("field,value", [("y_risk", 2), ("y_risk", np.inf),
    ("risk_endpoint", "other"), ("identity", ""), ("murcko_scaffold", None),
    ("fingerprint_hex", "bad")])
def test_reject_bad_evidence(risk, field, value):
    frame = sample().assign(y_risk=lambda d: d.y_risk.astype(float))
    frame.loc[0, field] = value
    with pytest.raises(ValueError):
        risk.prepare(frame, "measured_viability")


def test_missing_risk_remains_unknown(risk):
    frame = sample()
    frame.loc[0, "y_risk"] = np.nan
    output = risk.prepare(frame, "measured_viability")
    assert np.isnan(output.loc[output.identity.eq("id-0"), "y_risk"]).all()
    assert frame.loc[0, "y_risk"] is np.nan or np.isnan(frame.loc[0, "y_risk"])


def test_duplicate_identity_rejected(risk):
    with pytest.raises(ValueError, match="unique"):
        risk.prepare(pd.concat([sample(), sample().iloc[:1]]), "measured_viability")


def test_quantile_small_or_missing_class_is_conservative(risk):
    thresholds = risk.conformal_thresholds(np.array([0.1, 0.2]), np.array([0, 0]))
    assert thresholds == [1.0, 1.0]
    assert risk.prediction_sets(np.array([0.2]), thresholds).tolist() == [[True, True]]


def test_calibration_and_conformal_subsets_are_distinct(risk):
    frame = risk.prepare(sample(), "measured_viability")
    bundle = risk.fit(frame, "measured_viability")
    ids = bundle["partition_identities"]
    assert not set(ids["probability_calibration"]) & set(ids["conformal_calibration"])
    assert not set(ids["test"]) & set(ids["train"])
    assert bundle["risk_scope"] == "measured_viability"
    assert bundle["nominal_coverage_is_shift_guarantee"] is False


def test_test_labels_cannot_change_fitted_parameters(risk):
    frame = risk.prepare(sample(), "measured_viability")
    original = risk.fit(frame, "measured_viability")
    changed = frame.assign(y_risk=np.where(frame.split.eq("test"), 1-frame.y_risk, frame.y_risk))
    refit = risk.fit(changed, "measured_viability")
    for key in ["predictor", "calibrator", "conformal_thresholds", "validation_grid"]:
        assert original[key] == refit[key]


def test_serialization_predicts_same_scores(risk):
    frame = risk.prepare(sample(), "measured_viability")
    bundle = risk.fit(frame, "measured_viability")
    p = risk.predict(frame, bundle)
    np.testing.assert_array_equal(p, risk.predict(frame, json.loads(json.dumps(bundle))))
    assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()


def test_insufficient_support_does_not_invent_safe_model(risk):
    frame = risk.prepare(sample().assign(y_risk=0), "measured_viability")
    with pytest.raises(ValueError, match="class support"):
        risk.fit(frame, "measured_viability")


def test_metrics_count_measured_unknown_and_class_coverage(risk):
    result = risk.metrics(np.array([0, 1, np.nan]), np.array([0.1, 0.8, 0.99]), [0.3, 0.3])
    assert result["measured"] == 2 and result["unknown"] == 1
    assert result["brier"] == pytest.approx(0.025)
    assert result["conformal_coverage"] == 1.0
    assert result["conformal_mean_set_size"] == 1.0


def test_run_writes_new_artifacts_and_refuses_overwrite(risk, tmp_path):
    data = tmp_path / "input.parquet"
    sample().to_parquet(data, index=False)
    out = tmp_path / "run"
    result = risk.run(data, "measured_viability", out)
    assert result["status"] == "complete"
    assert (out / "test-metrics.json").exists()
    assert (out / "scores.parquet").exists()
    with pytest.raises(FileExistsError):
        risk.run(data, "measured_viability", out)


def test_preassigned_split_tampering_rejected(risk):
    frame = risk.prepare(sample(), "measured_viability")
    frame.loc[0, "split"] = "test" if frame.loc[0, "split"] != "test" else "train"
    with pytest.raises(ValueError, match="split"):
        risk.fit(frame, "measured_viability")


def test_evaluation_reservation_excludes_entire_scaffold(risk):
    source = sample().assign(evaluation_only=False)
    source.loc[0, "evaluation_only"] = True
    frame = risk.prepare(source, "measured_viability")
    rows = frame.loc[frame.murcko_scaffold.eq("scaffold-0")]
    assert len(rows) == 2 and rows.split.eq("external_activity_overlap").all()
    model = risk.fit(frame, "measured_viability")
    assert set(rows.identity).isdisjoint(model["partition_identities"]["train"])


@pytest.mark.parametrize("change", ["schema", "weights", "negative_slope", "endpoint"])
def test_inference_rejects_malformed_or_wrong_scope_bundle(risk, change):
    frame = sample()
    bundle = {"schema": "s2-decision-risk-benchmark/1", "risk_scope": "measured_viability",
              "predictor": {"weights": [0.0]*2048, "intercept": 0.0},
              "calibrator": {"slope": 1.0, "intercept": 0.0}}
    if change == "schema":
        bundle["schema"] = "invalid"
    elif change == "weights":
        bundle["predictor"]["weights"] = [0.0]
    elif change == "negative_slope":
        bundle["calibrator"]["slope"] = -1
    else:
        bundle["risk_scope"] = "different_biology"
    with pytest.raises(ValueError):
        risk.predict(frame, bundle)


def test_metrics_without_measurements_are_unavailable(risk):
    result = risk.metrics(np.array([np.nan]), np.array([0.5]), [0.8, 0.8])
    assert result["status"] == "no_measured_outcomes" and result["unknown"] == 1


def test_runtime_loads_inspectable_model_without_benchmark_import(risk, tmp_path):
    from s2s_decision.risk_model import load_risk_bundle, predict_risk

    frame = risk.prepare(sample(), "measured_viability")
    bundle = risk.fit(frame, "measured_viability")
    path = tmp_path / "risk.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    restored = load_risk_bundle(path)
    np.testing.assert_array_equal(predict_risk(frame, restored), risk.predict(frame, bundle))


def test_runtime_rejects_missing_calibration_evidence(tmp_path):
    from s2s_decision.risk_model import load_risk_bundle

    path = tmp_path / "risk.json"
    path.write_text('{}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_risk_bundle(path)


@pytest.mark.parametrize("change", ["missing", "counts", "threshold", "scope", "parameters"])
def test_runtime_rejects_invalid_calibration_contract(tmp_path, change):
    from s2s_decision.risk_model import load_risk_bundle

    bundle = {"schema": "s2-decision-risk-benchmark/1", "risk_scope": "viability",
              "predictor": {"weights": [0.0]*2048, "intercept": 0.0},
              "calibrator": {"slope": 1.0, "intercept": 0.0},
              "calibration_class_support": [10, 10], "conformal_thresholds": [.4, .4],
              "conformal_alpha": .1}
    if change == "missing":
        bundle.pop("calibration_class_support")
    elif change == "counts":
        bundle["calibration_class_support"] = [0, 10]
    elif change == "threshold":
        bundle["conformal_thresholds"] = [-1, .4]
    elif change == "scope":
        bundle["risk_scope"] = ""
    else:
        bundle["predictor"].pop("intercept")
    path = tmp_path / "risk.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    with pytest.raises(ValueError):
        load_risk_bundle(path)


def test_runtime_prediction_sets_preserve_empty_and_ambiguous():
    from s2s_decision.risk_model import risk_prediction_sets

    assert risk_prediction_sets([.1, .5, .9], {"conformal_thresholds": [.3, .3]}) == [[0], [], [1]]
    assert risk_prediction_sets([.5], {"conformal_thresholds": [.8, .8]}) == [[0, 1]]
    with pytest.raises(ValueError):
        risk_prediction_sets([np.nan], {"conformal_thresholds": [.8, .8]})


def test_chemistry_contract_requires_recomputed_identity_and_fingerprint(risk):
    from s2s_decision.features import featurize

    frame = featurize(pd.DataFrame({"original_smiles": ["CCO", "CCN"]})).records
    contract = risk.verify_input_chemistry(frame)
    assert len(contract["chemistry_hash"]) == 64
    changed = frame.assign(fingerprint_hex=frame.fingerprint_hex.iloc[0])
    with pytest.raises(ValueError, match="chemistry parity"):
        risk.verify_input_chemistry(changed)
