"""Paired ranking and constrained selection use one explicit held-out pool."""

import json

import numpy as np
import pandas as pd
import pytest

from s2s_decision.schema import CONTEXT_NAMES, PROPERTY_NAMES


def prepared_frame():
    rng = np.random.default_rng(23)
    frame = pd.DataFrame(
        {name: rng.uniform(0.1, 0.9, 16) for name in (*PROPERTY_NAMES, *CONTEXT_NAMES)}
    )
    return frame.assign(
        record_id=np.arange(16),
        identity=[f"mol{i}" for i in range(16)],
        split=["train"] * 6 + ["validation"] * 2 + ["calibration"] * 2 + ["test"] * 6,
        y_active=[0, 1] * 8,
        pactivity=[5.0, 7.0] * 8,
        fingerprint_hex=[
            rng.integers(0, 256, 256, dtype=np.uint8).tobytes().hex() for _ in range(16)
        ],
        valid=True,
        eligible=False,
        pinned=True,
        murcko_scaffold=["A"] * 13 + ["B"] * 3,
    )


def fake_predictions(frame, model_dir):
    return pd.DataFrame({"record_id": frame.record_id.iloc[::-1], "activity_probability": 0.5})


def test_same_pool_id_alignment_ties_and_actual_quota(monkeypatch):
    from s2s_decision import benchmark_evaluation as module

    frame = prepared_frame().sample(frac=1, random_state=9)
    frame.loc[frame.record_id.eq(15), "similarity_active_max"] = np.nan
    monkeypatch.setattr(module, "predict", fake_predictions)
    report = module.evaluate_benchmark(frame, "unused", n=4, max_per_scaffold=1)
    assert report["pool"]["record_ids"] == [10, 11, 12, 13, 14]
    assert report["pool"]["common_count"] == 5
    assert report["pool"]["excluded_missing_score"] == 1
    for name, result in report["methods"].items():
        assert result["record_ids"] == report["pool"]["record_ids"]
        assert result["raw"]["count"] == 5
        assert result["selection"]["final_count"] == 2
        assert result["selection"]["shortfall"] == 2
        assert result["selection"]["scaffolds_covered"] == 2
        assert result["selection"]["warnings"]
        assert result["selection"]["pinned_ids"] == []
        if name in ("qed", "similarity"):
            assert result["raw"]["brier"] is result["raw"]["ece"] is None
    tiny = report["methods"]["tiny"]
    assert tiny["selection"]["final_ids"] == [10, 13]
    assert tiny["raw"]["precision_at_n"] == 0.5
    assert tiny["selection"]["precision"] == 0.5
    assert tiny["selection"]["recall"] == 0.5
    json.dumps(report, allow_nan=False)


def test_baseline_fitting_never_uses_heldout_labels(monkeypatch):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    from s2s_decision.training import baseline_predictions

    fitted = []
    for estimator in (HistGradientBoostingClassifier, LogisticRegression):
        original = estimator.fit

        def spy(self, inputs, labels, original=original):
            fitted.append(list(labels.index))
            return original(self, inputs, labels)

        monkeypatch.setattr(estimator, "fit", spy)
    frame = prepared_frame()
    first = baseline_predictions(frame, seed=11)
    changed = frame.assign(
        y_active=np.where(frame.split.eq("train"), frame.y_active, 1 - frame.y_active)
    )
    second = baseline_predictions(changed, seed=11)
    pd.testing.assert_frame_equal(first, second)
    assert fitted == [list(range(6))] * 4


def test_reject_missing_or_misaligned_ids_and_leakage(monkeypatch):
    from s2s_decision import benchmark_evaluation as module

    frame = prepared_frame()
    monkeypatch.setattr(module, "predict", lambda frame, _: fake_predictions(frame.iloc[1:], _))
    with pytest.raises(ValueError, match="IDs"):
        module.evaluate_benchmark(frame, "unused")
    leaked = frame.copy()
    leaked.loc[10, "identity"] = leaked.loc[0, "identity"]
    with pytest.raises(ValueError, match="leakage"):
        module.evaluate_benchmark(leaked, "unused")
    with pytest.raises(ValueError, match="positive integer"):
        module.evaluate_benchmark(frame, "unused", n=True)


def test_missing_labels_and_invalid_molecules_not_selectable(monkeypatch):
    from s2s_decision import benchmark_evaluation as module

    frame = prepared_frame()
    frame.loc[10, "valid"] = False
    frame.loc[11, "y_active"] = np.nan
    monkeypatch.setattr(module, "predict", fake_predictions)
    result = module.evaluate_benchmark(frame, "unused", n=2)
    assert result["pool"]["record_ids"] == [12, 13, 14, 15]
    assert result["pool"]["excluded_invalid_or_unlabelled"] == 2
