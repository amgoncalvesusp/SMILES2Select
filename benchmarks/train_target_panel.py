"""B13: frozen three-task comparison; external truth never enters fitting.

This runner is research-only. All seeds and controls are retained. Shared and
single-task networks have the same hidden widths, not equal total parameters.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_external import (
    predict_checkpoint,
    random_control,
    rank_scores,
    ranking_metrics,
)
from scipy.special import expit
from train_multitask import fit_baseline, load_dataset, verify_inputs

from s2s_decision.artifacts import file_hash, write_json
from s2s_decision.context import build_context
from s2s_decision.multitask import train_multitask
from s2s_decision.schema import fingerprint_matrix

SEEDS = [42, 43, 44]
KS = [20, 50, 100]
TRAINING = {
    "max_epochs": 40,
    "patience": 6,
    "batch_size": 512,
    "hidden": [512, 256],
    "learning_rate": 0.001,
}
LOGISTIC = {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def single_task(arrays, task):
    """Single-task training uses measured rows only, with unchanged split labels."""
    observed = np.isfinite(arrays["labels"][:, task])
    return {
        **{key: value[observed] for key, value in arrays.items() if key != "labels"},
        "labels": arrays["labels"][observed, task : task + 1],
    }


def external_novelty(development, external, exposure):
    ids = set(development.identity) | set(exposure["identities"])
    if set(external.identity) & ids:
        raise ValueError(
            "External identities must exclude ALL historically exposed and development molecules"
        )
    years = pd.to_numeric(external.year, errors="raise")
    if not np.isfinite(years).all() or (years < 2024).any():
        raise ValueError(
            "External observations must all have source document year >=2024"
        )
    scaffolds = set(development.murcko_scaffold) | set(exposure["scaffolds"])
    return ~external.murcko_scaffold.isin(scaffolds).to_numpy()


def validate_frame(frame, arrays):
    required = {"identity", "murcko_scaffold", "fingerprint_hex", "qed"}
    if not required <= set(frame) or frame[list(required)].isna().any().any():
        raise ValueError(
            "Molecular metadata must contain complete identity/scaffold/fingerprint/QED"
        )
    if not np.array_equal(frame.identity.to_numpy(), arrays["identities"]):
        raise ValueError(
            "Molecular metadata identity order must align exactly with arrays"
        )
    if (
        frame.identity.duplicated().any()
        or frame.identity.astype(str).str.strip().eq("").any()
    ):
        raise ValueError("Molecular identities must be unique and nonempty")
    qed = frame.qed.to_numpy(dtype=float)
    if not np.isfinite(qed).all() or ((qed < 0) | (qed > 1)).any():
        raise ValueError("QED must be finite and in [0,1]")
    if not np.array_equal(fingerprint_matrix(frame), arrays["x"][:, :2048]):
        raise ValueError("Fingerprint metadata must agree with the model inputs")


def load_panel(data, exposure):
    arrays, tasks = load_dataset(data)
    if arrays["x"].shape[1] != 2088 or arrays["labels"].shape[1] != 3:
        raise ValueError("Panel requires 2088 input features and exactly three tasks")
    completion = read_json(data / "completion.json")["files"]
    required = {
        "molecules.parquet",
        "external-arrays.npz",
        "external-molecules.parquet",
        "development-exposure.json",
    }
    if not required <= set(completion):
        raise ValueError("Completion manifest must seal all panel and external inputs")
    with np.load(data / "external-arrays.npz", allow_pickle=False) as saved:
        if "scaffold_novel" not in saved:
            raise ValueError("Saved scaffold novelty is required in external arrays")
        external = {
            key: saved[key] for key in ("x", "labels", "identities", "scaffold_novel")
        }
    x, y, identities = (external[key] for key in ("x", "labels", "identities"))
    if x.shape != (len(identities), 2088) or y.shape != (len(identities), 3):
        raise ValueError(
            "External arrays must align with three tasks and 2088 features"
        )
    if not np.isfinite(x).all() or not np.all(np.isnan(y) | (y == 0) | (y == 1)):
        raise ValueError("External features must be finite; labels binary or unknown")
    frames = [
        pd.read_parquet(data / name)
        for name in ("molecules.parquet", "external-molecules.parquet")
    ]
    for frame, values in zip(frames, (arrays, external)):
        validate_frame(frame, values)
    if (
        frames[0]
        .assign(split=arrays["splits"])
        .groupby("murcko_scaffold")
        .split.nunique()
        .gt(1)
        .any()
    ):
        raise ValueError("Development scaffolds must not cross splits")
    for split, minimum in (
        ("train", 100),
        ("validation", 20),
        ("calibration", 20),
        ("test", 20),
    ):
        selected = arrays["labels"][arrays["splits"] == split]
        if any(np.any((selected == label).sum(axis=0) < minimum) for label in (0, 1)):
            raise ValueError(
                f"Insufficient {split} support: each task needs {minimum} per class; no resplitting"
            )
    development_exposure = read_json(data / "development-exposure.json")
    combined = {
        key: list(set(exposure[key]) | set(development_exposure[key]))
        for key in ("identities", "scaffolds")
    }
    secondary = external_novelty(*frames, combined)
    if "scaffold_novel" not in frames[1]:
        raise ValueError("Saved scaffold novelty is required in external molecules")
    for saved_novelty in (
        external["scaffold_novel"],
        frames[1].scaffold_novel.to_numpy(),
    ):
        if saved_novelty.dtype != bool or not np.array_equal(saved_novelty, secondary):
            raise ValueError(
                "Saved scaffold novelty disagrees with recomputed exposure"
            )
    return arrays, external, tasks, frames, secondary


def freeze(data, exposure_path, output, plan_path, tasks):
    plan = read_json(plan_path)
    if plan["training"] != {**TRAINING, "seeds": SEEDS}:
        raise ValueError("Training settings differ from fixed B13 configuration")
    if (
        plan["comparison"]["K"] != KS
        or plan["comparison"]["primary_K"] != 50
        or plan["comparison"]["random_repetitions"] != 1000
    ):
        raise ValueError("Comparison settings differ from fixed B13 configuration")
    if plan["task_ids"] != [task["task_id"] for task in tasks]:
        raise ValueError("Task order differs from frozen plan")
    source = Path(__file__).resolve().parents[1]
    code = [
        *sorted((source / "src").rglob("*.py")),
        *sorted((source / "benchmarks").glob("*.py")),
    ]
    inputs = [
        *sorted(path for path in data.rglob("*") if path.is_file()),
        exposure_path,
        plan_path,
    ]
    exposure = read_json(exposure_path)
    for name, digest in exposure.get("source_files", {}).items():
        if file_hash(name) != digest:
            raise ValueError(f"Historical exposure source changed: {name}")
        inputs.append(Path(name))
    protocol = {
        "schema": "s2-decision-target-panel-run/1",
        "created_utc": datetime.now(UTC).isoformat(),
        "training": TRAINING,
        "seeds": SEEDS,
        "logistic": LOGISTIC,
        "K": KS,
        "calibration_min_per_class": 20,
        "random_repetitions": 1000,
        "output_resolved": str(output.resolve()),
        "promotion": "none; research comparison only",
        "external_usage": "evaluation only after all fits; no calibration or selection",
        "single_task_policy": "observed rows only; shared hidden widths; unequal total parameter budget",
        "data_hashes": {str(path.resolve()): file_hash(path) for path in inputs},
        "code_hashes": {str(path.resolve()): file_hash(path) for path in code},
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", protocol)
    return protocol


def fit_models(arrays, tasks, output):
    fit_baseline(
        arrays["x"],
        arrays["labels"],
        arrays["splits"],
        output / "logistic",
        LOGISTIC,
        tasks,
    )
    methods = [{"name": "logistic", "task": None}]
    for seed in SEEDS:
        name = f"s2_decision_seed{seed}"
        train_multitask(
            arrays["x"],
            arrays["labels"],
            arrays["splits"],
            output / name,
            seed=seed,
            **TRAINING,
        )
        methods.append({"name": name, "task": None})
        for task in range(len(tasks)):
            subset = single_task(arrays, task)
            name = f"single_task{task}_seed{seed}"
            train_multitask(
                subset["x"],
                subset["labels"],
                subset["splits"],
                output / name,
                seed=seed,
                **TRAINING,
            )
            methods.append({"name": name, "task": task})
    return methods


def apply_calibration(logits, coefficient):
    if (
        not np.isfinite(logits).all()
        or not np.isfinite([coefficient["slope"], coefficient["intercept"]]).all()
    ):
        raise ValueError("Calibration inputs must be finite")
    return expit(logits * coefficient["slope"] + coefficient["intercept"])


def observed_metrics(frame, labels, scores, raw, calibrated):
    mask = np.isfinite(labels)
    truth = frame.loc[mask, ["identity", "murcko_scaffold"]].assign(
        y_active=labels[mask]
    )
    ranked = rank_scores(truth.identity, scores[mask])
    result = ranking_metrics(truth, ranked, ks=KS)
    for name, probabilities in (("raw", raw), ("calibrated", calibrated)):
        result[f"brier_{name}"] = (
            float(np.mean((probabilities[mask] - labels[mask]) ** 2))
            if probabilities is not None and mask.any()
            else None
        )
    return result, ranked


def model_scores(output, method, x, task):
    folder = output / method["name"]
    index = task if method["task"] is None else 0
    coefficients = read_json(folder / "calibration.json")
    if method["name"] == "logistic":
        with np.load(folder / "coefficients.npz", allow_pickle=False) as saved:
            score = x @ saved["coefficients"][task] + saved["intercept"][task]
        coefficients = coefficients["tasks"]
    else:
        score = predict_checkpoint(folder / "best.pt", x, index)
    return (
        score,
        expit(score),
        apply_calibration(score, coefficients[index]),
        coefficients[index],
    )


def evaluate(arrays, external, tasks, frames, secondary, output, methods):
    development, external_frame = frames
    pools = [
        (
            split,
            arrays["x"][arrays["splits"] == split],
            arrays["labels"][arrays["splits"] == split],
            development.loc[arrays["splits"] == split].reset_index(drop=True),
        )
        for split in ("validation", "test")
    ]
    pools.extend(
        [
            (
                name,
                external["x"][mask],
                external["labels"][mask],
                external_frame.loc[mask].reset_index(drop=True),
            )
            for name, mask in (
                ("external_identity_novel", np.ones(len(external_frame), dtype=bool)),
                ("external_scaffold_novel", secondary),
            )
        ]
    )
    reports = []
    for task, definition in enumerate(tasks):
        active = (arrays["splits"] == "train") & (arrays["labels"][:, task] == 1)
        references = development.loc[active].assign(y_active=1)
        for cohort, x, labels, frame in pools:
            similarity = (
                build_context(frame, references).similarity_active_max.to_numpy()
                if len(frame)
                else np.empty(0)
            )
            scores = [
                ("qed", frame.qed.to_numpy(), None, None, None),
                ("max_tanimoto_active", similarity, None, None, None),
            ]
            scores.extend(
                (method["name"], *model_scores(output, method, x, task))
                for method in methods
                if method["task"] in (None, task)
            )
            for name, score, raw, calibrated, calibration in scores:
                report, ranked = observed_metrics(
                    frame, labels[:, task], score, raw, calibrated
                )
                prefix = f"task{task}-{cohort}-{name}"
                ranked.to_parquet(output / f"{prefix}-ranking.parquet", index=False)
                pd.DataFrame(
                    {
                        "identity": frame.identity,
                        "score": score,
                        "raw_probability": raw,
                        "calibrated_probability": calibrated,
                    }
                ).to_parquet(output / f"{prefix}-scores.parquet", index=False)
                reports.append(
                    {
                        "task_id": definition["task_id"],
                        "cohort": cohort,
                        "method": name,
                        "calibration": calibration,
                        **report,
                    }
                )
            observed = np.isfinite(labels[:, task])
            truth = frame.loc[observed].assign(y_active=labels[observed, task])
            reports.append(
                {
                    "task_id": definition["task_id"],
                    "cohort": cohort,
                    "method": "random",
                    **random_control(truth, ks=KS, repetitions=1000, seed=0),
                }
            )
    write_json(output / "evaluation.json", {"promotion": "none", "records": reports})
    return reports


def run(data, exposure_path, output, plan_path):
    arrays, external, tasks, frames, secondary = load_panel(
        data, read_json(exposure_path)
    )
    protocol = freeze(data, exposure_path, output, plan_path, tasks)
    verify_inputs(protocol)
    methods = fit_models(arrays, tasks, output)
    write_json(output / "fits-complete.json", {"methods": methods})
    verify_inputs(protocol)
    evaluate(arrays, external, tasks, frames, secondary, output, methods)
    verify_inputs(protocol)
    write_json(
        output / "completion.json",
        {
            "files": {
                path.relative_to(output).as_posix(): file_hash(path)
                for path in sorted(output.rglob("*"))
                if path.is_file()
            }
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "exposure", "output", "plan"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.data.resolve(),
        args.exposure.resolve(),
        args.output.resolve(),
        args.plan.resolve(),
    )


if __name__ == "__main__":
    main()
