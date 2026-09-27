"""Paired held-out ranking and the actual SMILES2Select diversity-constrained selection."""

from numbers import Integral

import numpy as np

from .inference import predict
from .metrics import evaluate_predictions
from .selection import select_candidates
from .training import _validate, baseline_predictions

METHODS = ("tiny", "logistic", "gradient_boosting", "similarity", "qed")


def _validate_benchmark(frame, n, max_per_scaffold, seed):
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer")
    if type(max_per_scaffold) is not int or max_per_scaffold < 1:
        raise ValueError("max_per_scaffold must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if not frame.columns.is_unique:
        raise ValueError("Benchmark columns must be unique")
    _validate(frame)
    if {"valid", "murcko_scaffold"} - set(frame):
        raise ValueError("Benchmark requires valid and murcko_scaffold columns")
    if any(not isinstance(value, (bool, np.bool_)) for value in frame.valid):
        raise ValueError("valid must contain booleans")
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
        for value in frame.record_id
    ):
        raise ValueError("Benchmark record_id values must be integers")


def _aligned(predictions, expected_ids, columns):
    if (
        {"record_id", *columns} - set(predictions)
        or predictions.record_id.duplicated().any()
        or set(predictions.record_id) != set(expected_ids)
    ):
        raise ValueError("Prediction IDs must exactly match requested test IDs")
    return predictions.set_index("record_id").loc[expected_ids, columns]


def evaluate_benchmark(frame, model_dir, fingerprint_bits=2048, seed=42, n=50, max_per_scaffold=3):
    """Evaluate frozen Tiny and train-only baselines; test rows never fit any estimator.

    This assay benchmark uses every valid labelled test molecule, independent of
    upstream eligibility or pins. It is not a simulation of an upstream-filtered campaign.
    """
    _validate_benchmark(frame, n, max_per_scaffold, seed)
    ordered = frame.sort_values("record_id", kind="stable").reset_index(drop=True)
    test = ordered.loc[ordered.split.eq("test")]
    valid = test.valid & test.y_active.notna()
    candidates = test.loc[valid].copy()
    tiny = _aligned(predict(candidates, model_dir), candidates.record_id, ["activity_probability"])
    baselines = baseline_predictions(ordered, fingerprint_bits, seed)
    scores = _aligned(baselines, test.record_id, list(METHODS[1:])).loc[candidates.record_id]
    scores = scores.assign(tiny=tiny.activity_probability)
    finite = np.isfinite(scores[list(METHODS)].to_numpy(float)).all(axis=1)
    common = candidates.loc[finite].copy()
    scores = scores.loc[finite]
    if not common.empty and ((scores < 0) | (scores > 1)).any().any():
        raise ValueError("Benchmark scores must be in [0, 1]")
    report = {
        "format_version": 1,
        "seed": seed,
        "n_requested": n,
        "max_per_scaffold": max_per_scaffold,
        "pool": {
            "definition": "Valid labelled held-out assay molecules; no upstream eligibility filter or pins",
            "full_test_count": len(test),
            "valid_test_count": len(candidates),
            "common_count": len(common),
            "excluded_invalid_or_unlabelled": len(test) - len(candidates),
            "excluded_missing_score": len(candidates) - len(common),
            "record_ids": common.record_id.astype(int).tolist(),
            "y_active": common.y_active.astype(int).tolist(),
        },
        "methods": {},
    }
    for name in METHODS:
        values = scores[name].to_numpy(float)
        raw = evaluate_predictions(common.y_active, values, n=n)
        heuristic = name in ("similarity", "qed")
        if heuristic:
            raw.update(brier=None, ece=None)
        pool = common[["record_id", "identity", "murcko_scaffold", "valid", "y_active"]].assign(
            eligible=True,
            priority_score=values,
        )
        selected = select_candidates(pool, n=n, max_per_scaffold=max_per_scaffold)
        final = selected.records.loc[selected.records.is_final]
        positives = int(common.y_active.sum())
        selection = {
            **selected.manifest,
            "precision": float(final.y_active.mean()) if len(final) else None,
            "recall": float(final.y_active.sum() / positives) if positives else None,
            "active_count": int(final.y_active.sum()),
        }
        report["methods"][name] = {
            "score_kind": "heuristic_ranking_score" if heuristic else "model_activity_probability",
            "record_ids": report["pool"]["record_ids"].copy(),
            "scores": values.tolist(),
            "raw": raw,
            "selection": selection,
        }
    return report
