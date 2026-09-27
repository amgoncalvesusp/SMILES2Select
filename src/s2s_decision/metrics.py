"""Ranking and calibration metrics with explicit undefined values."""

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def evaluate_predictions(
    y, probability, predicted_pactivity=None, pactivity=None, n=10, alpha=20.0
):
    if n < 1 or alpha <= 0:
        raise ValueError("n and BEDROC alpha must be positive")
    labels, scores = np.asarray(y, float), np.asarray(probability, float)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("Labels and predictions must be equal one-dimensional arrays")
    valid = np.isfinite(labels)
    labels, scores = labels[valid], scores[valid]
    if (
        not np.isin(labels, (0, 1)).all()
        or not np.isfinite(scores).all()
        or ((scores < 0) | (scores > 1)).any()
    ):
        raise ValueError("Invalid binary labels or activity probabilities")
    count, positives = len(labels), int(labels.sum())
    ranked = labels[np.argsort(-scores, kind="stable")]
    selected = min(n, count)
    report = {
        "count": count,
        "positives": positives,
        "n_requested": n,
        "n_selected": selected,
        "precision_at_n": float(ranked[:selected].mean()) if selected else None,
        "recall_at_n": float(ranked[:selected].sum() / positives) if positives else None,
        "roc_auc": None,
        "pr_auc": None,
        "ef1": None,
        "ef5": None,
        "bedroc": None,
        "bedroc_alpha": alpha,
        "brier": None,
        "ece": None,
        "mae": None,
        "rmse": None,
    }
    if count:
        report["brier"] = float(brier_score_loss(labels, scores))
        bins = np.minimum((scores * 10).astype(int), 9)
        report["ece"] = float(
            sum(
                np.mean(bins == b) * abs(labels[bins == b].mean() - scores[bins == b].mean())
                for b in range(10)
                if (bins == b).any()
            )
        )
    if 0 < positives < count:
        report["roc_auc"] = float(roc_auc_score(labels, scores))
        report["pr_auc"] = float(average_precision_score(labels, scores))
        weights = np.exp(-alpha * np.arange(1, count + 1) / count)
        best, worst = weights[:positives].sum(), weights[-positives:].sum()
        report["bedroc"] = float(((weights * ranked).sum() - worst) / (best - worst))
    if positives:
        for fraction, key in ((0.01, "ef1"), (0.05, "ef5")):
            k = max(1, int(np.ceil(count * fraction)))
            report[key] = float(ranked[:k].mean() / (positives / count))
    if predicted_pactivity is not None and pactivity is not None:
        expected, predicted = np.asarray(pactivity, float), np.asarray(predicted_pactivity, float)
        if expected.shape != predicted.shape or expected.shape != valid.shape:
            raise ValueError("pActivity arrays must match input labels")
        observed = np.isfinite(expected)
        if not np.isfinite(predicted[observed]).all():
            raise ValueError("Nonfinite pActivity prediction")
        if observed.any():
            errors = predicted[observed] - expected[observed]
            report["mae"], report["rmse"] = (
                float(np.abs(errors).mean()),
                float(np.sqrt((errors**2).mean())),
            )
    return report
