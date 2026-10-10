"""Prespecified exploratory Laya head pilot; equal-evidence numerical controls."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Thread
from time import monotonic, perf_counter

import contextual_policy_fit as fit
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import brier_score_loss, log_loss

from s2s_decision.artifacts import file_hash, write_json
from s2s_decision.schema import PROPERTY_NAMES
from s2s_decision.selection import select_candidates

QUESTION = {
    "instructions": "Predict whether this compound inhibits human acetylcholinesterase "
                    "with measured IC50 <= 1000 nM, using only the supplied chemical evidence.",
    "positive": "Active: IC50 <= 1000 nM.",
    "negative": "Inactive: IC50 > 1000 nM.",
}
CONFIG = {
    "seeds": [42, 43, 44], "logistic_C": [.1, 1., 10.], "boosting_depth": [1, 2],
    "boosting_trees": 100, "boosting_learning_rate": .05, "boosting_subsample": .8,
    "boosting_min_leaf": 10, "laya_learning_rate": .0001, "laya_epochs": [1, 3],
    "budgets": [20, 50, 100], "caps": [3, None], "primary": {"n": 50, "cap": 3},
    "gate": {"ap_gain": .02, "hit_gain": 2, "brier_worsening": .02,
             "log_loss_worsening": .02, "required_seeds": 2},
    "resource": {"peak_rss_bytes": 8 * 1024**3, "seconds_per_1000": 1800,
                 "training_seconds_per_seed": 7200, "torch_threads": 4, "batch_size": 4,
                 "measurement": "warm resident pipeline from frozen Smiles2Select features to selection; cold load separate",
                 "engineering_pool": "first1000 source-panel unique identities by record_id, excluding full AChE train; query labels removed"},
    "question": QUESTION, "probability_calibration": "entire reserved calibration split",
    "evaluation": "previously observed historical test; recent secondary; exploratory only",
}


def validate_records(frame, columns):
    required = {"record_id", "identity", "murcko_scaffold", "split", "y_active", "state_text"}
    if not isinstance(frame, pd.DataFrame) or frame.empty or not required <= set(frame):
        raise ValueError("Complete nonempty pilot records required")
    forbidden = required | {"row_id", "model_smiles", "fingerprint_hex", "molecule_id"}
    if not columns or len(set(columns)) != len(columns) or set(columns) & forbidden:
        raise ValueError("Feature columns must be unique chemical evidence, never outcomes/identities")
    if not set(columns) <= set(frame):
        raise ValueError("Feature columns missing from records")
    if not frame.y_active.isin([0, 1]).all():
        raise ValueError("Observed binary outcomes required")
    if frame.identity.duplicated().any() or frame.record_id.duplicated().any():
        raise ValueError("Identity and record_id must be unique")
    if not set(frame.split) <= {"train", "validation", "calibration", "test", "recent"}:
        raise ValueError("Unknown split")
    internal = frame.loc[frame.split.ne("recent") & frame.murcko_scaffold.ne("")]
    if internal.groupby("murcko_scaffold").split.nunique().gt(1).any():
        raise ValueError("Scaffold leakage across development/holdout splits")
    for split in ("train", "validation", "calibration"):
        if set(frame.loc[frame.split.eq(split), "y_active"]) != {0, 1}:
            raise ValueError(f"Both observed classes required in {split}")
    if not frame.state_text.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("Nonempty serialized states required")
    values = frame[columns].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Feature evidence cannot contain infinity")


def fit_numeric_spec(frame, columns):
    train = frame.loc[frame.split.eq("train"), columns].to_numpy(dtype=float)
    if not len(train) or np.isinf(train).any():
        raise ValueError("Finite training evidence required")
    median = np.array([np.nanmedian(x) if np.isfinite(x).any() else 0 for x in train.T])
    imputed = np.where(np.isnan(train), median, train)
    scale = imputed.std(axis=0)
    return {"columns": list(columns), "median": median.tolist(),
            "mean": imputed.mean(axis=0).tolist(), "scale": np.where(scale > 0, scale, 1).tolist()}


def numeric_matrix(frame, spec):
    values = frame[spec["columns"]].to_numpy(dtype=float)
    return (np.where(np.isnan(values), spec["median"], values) - spec["mean"]) / spec["scale"]


def fit_numeric(frame, columns, family, seed):
    spec = fit_numeric_spec(frame, columns)
    x = numeric_matrix(frame, spec)
    train, valid = frame.split.eq("train").to_numpy(), frame.split.eq("validation").to_numpy()
    candidates, grid = [], []
    if family not in {"logistic", "boosting"}:
        raise ValueError("Unknown numeric family")
    for value in CONFIG["logistic_C" if family == "logistic" else "boosting_depth"]:
        model = (fit.fit_logistic(x[train], frame.y_active.to_numpy()[train], value)
                 if family == "logistic" else
                 fit.fit_boosting(x[train], frame.y_active.to_numpy()[train], value, seed)[1])
        ap = fit.ranking_metrics(frame.y_active.to_numpy()[valid],
                                 fit.predict_estimator(x[valid], model))["ap"]
        candidates.append(model)
        grid.append({"model": family, "seed": seed, "value": value, "validation_ap": ap})
    chosen = int(np.argmax([row["validation_ap"] for row in grid]))
    return {"spec": spec, "estimator": candidates[chosen],
            "validation_ap": grid[chosen]["validation_ap"]}, grid


def binary_logit(option_logits):
    values = np.asarray(option_logits, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite nonempty [N,2] option logits required")
    return values[:, 0] - values[:, 1]


def calibrate(frame, logits):
    values = np.asarray(logits, dtype=float)
    if values.shape != (len(frame),) or not np.isfinite(values).all():
        raise ValueError("Finite aligned logits required")
    mask = frame.split.eq("calibration").to_numpy()
    saved = fit.fit_calibrator(values[mask], frame.y_active.to_numpy()[mask])
    z = saved["slope"] * values + saved["intercept"]
    return np.exp(-np.logaddexp(0, -z)), {
        **saved, "identities": frame.loc[mask, "identity"].tolist(),
        "split": "calibration", "no_conformal_claim": True,
    }


def score_metrics(y, raw, probabilities):
    labels, values, p = np.asarray(y), np.asarray(raw, float), np.asarray(probabilities, float)
    if (labels.ndim != 1 or not len(labels) or values.shape != labels.shape or
            p.shape != labels.shape or not np.isin(labels, [0, 1]).all() or
            not np.isfinite(values).all() or not np.isfinite(p).all() or
            np.any((p < 0) | (p > 1))):
        raise ValueError("Aligned binary labels, finite scores and probabilities required")
    a, b = fit.ranking_metrics(labels, values), fit.ranking_metrics(labels, p)
    return {"n": len(labels), "positive_n": int(labels.sum()),
            "ap_raw": a["ap"], "roc_auc_raw": a["roc_auc"],
            "ap_calibrated": b["ap"], "roc_auc_calibrated": b["roc_auc"],
            "brier": float(brier_score_loss(labels, p)),
            "log_loss": float(log_loss(labels, p, labels=[0, 1]))}


def selection_metrics(frame, probabilities, n, cap):
    eligible = frame.assign(valid=frame.get("valid", True), eligible=frame.get("eligible", True),
                            priority_score=probabilities)
    chosen = select_candidates(eligible, int(n), max_per_scaffold=cap)
    selected = chosen.records.loc[chosen.records.is_final]
    hits, actual, positives = int(selected.y_active.sum()), len(selected), int(frame.y_active.sum())
    return {"requested_n": n, "cap": cap, "n_actual": actual, "hits": hits,
            "hits_per_actual": hits / actual if actual else None,
            "hits_per_requested": hits / n, "pool_n": len(frame),
            "positive_pool_n": positives, "recall_positive_pool": hits / positives if positives else None,
            "scaffolds": int(selected.murcko_scaffold.nunique()),
            "shortfall": chosen.manifest["shortfall"]}, sorted(map(int, selected.record_id))


def evaluate_model(frame, raw, probabilities, model, seed):
    predictions, metrics, baskets = [], [], []
    for split in ("test", "recent"):
        mask = frame.split.eq(split).to_numpy()
        pool, r, p = frame.loc[mask], np.asarray(raw)[mask], np.asarray(probabilities)[mask]
        if pool.empty:
            continue
        keys = {"model": model, "seed": seed, "split": split}
        metric = {**keys, **score_metrics(pool.y_active, r, p)}
        for n in CONFIG["budgets"]:
            for cap in CONFIG["caps"]:
                row, ids = selection_metrics(pool, p, n, cap)
                baskets.append({**keys, **row, "selected_record_ids": ids})
                if n == 50 and cap == 3:
                    metric["primary_hits"] = row["hits"]
                    metric["primary_n_actual"] = row["n_actual"]
        metrics.append(metric)
        predictions.extend(pool[["record_id", "identity", "murcko_scaffold", "split", "y_active"]]
                           .assign(raw_logit=r, probability=p, model=model, seed=seed)
                           .to_dict("records"))
    return predictions, metrics, baskets


def assess_gates(metrics, resources):
    rows = metrics.loc[metrics.split.eq("test") & metrics.model.isin(["laya", "numeric_selected"])]
    if (len(rows) != 6 or rows.duplicated(["seed", "model"]).any() or
            set(rows.seed) != set(CONFIG["seeds"])):
        raise ValueError("Gate needs six unique primary rows from all three genuine seeds")
    gates = []
    for seed in CONFIG["seeds"]:
        part = rows.loc[rows.seed.eq(seed)].set_index("model")
        laya, base = part.loc["laya"], part.loc["numeric_selected"]
        delta = {name: float(laya[name] - base[name]) for name in
                 ("ap_calibrated", "primary_hits", "brier", "log_loss")}
        passed = (all(np.isfinite(v) for v in delta.values()) and
                  delta["ap_calibrated"] >= .02 - 1e-12 and delta["primary_hits"] >= 2 and
                  delta["brier"] <= .02 + 1e-12 and delta["log_loss"] <= .02 + 1e-12)
        gates.append({"seed": seed, "delta": delta if all(np.isfinite(v) for v in delta.values()) else None,
                      "pass": bool(passed)})
    n, seconds, peak = (resources.get(k, 0) for k in ("n", "seconds", "peak_rss_bytes"))
    resource_pass = (n >= 1000 and np.isfinite(seconds) and 0 < seconds <= 1800 * n / 1000
                     and 0 < peak <= 8 * 1024**3)
    passing = [g["seed"] for g in gates if g["pass"]]
    performance_pass = len(passing) >= 2
    return {"seeds": gates, "passing_seeds": passing, "performance_pass": performance_pass,
            "resource_pass": bool(resource_pass), "independent_validation": False,
            "decision": "eligible_for_independent_validation" if performance_pass and resource_pass
                        else "do_not_advance_this_pilot", "resources": resources}


def freeze_run(output, inputs, protocol):
    output, protocol = Path(output), Path(protocol)
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    sources = [Path(__file__), Path(fit.__file__)]
    import s2s_decision.context as context
    import s2s_decision.contextual_model as chemistry
    import s2s_decision.schema as schema
    import s2s_decision.selection as selection
    import smiles2select.selection_intelligence.constrained_selection as selector
    sources += [Path(module.__file__) for module in (selector, chemistry, selection, context, schema)]
    sources += [Path(__file__).with_name(name) for name in ("laya_data.py", "laya_backend.py")
                if Path(__file__).with_name(name).is_file()]
    paths = [*map(Path, inputs), protocol, *sources]
    hashes = {str(p.resolve()): file_hash(p) for p in paths}
    output.mkdir(parents=True)
    versions = {}
    for package in ("numpy", "pandas", "scikit-learn", "torch", "transformers", "rdkit"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    frozen = {"created_utc": datetime.now(UTC).isoformat(), "config": CONFIG,
              "input_source_hashes": hashes, "package_versions": versions,
              "host": {"platform": platform.platform(), "processor": platform.processor(),
                       "python": platform.python_version()}}
    for source in sources:
        destination = output / "execution-source" / source.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, destination)
    shutil.copyfile(protocol, output / protocol.name)
    for path in paths:
        if path.suffix.lower() == ".md" and path != protocol:
            shutil.copyfile(path, output / path.name)
    write_json(output / "freeze.json", frozen)
    return frozen


def verify_frozen(frozen):
    for path, digest in frozen["input_source_hashes"].items():
        if file_hash(path) != digest:
            raise ValueError(f"Frozen input/source changed: {path}")


def measure(call):
    process, stop = psutil.Process(), Event()
    peak = [process.memory_info().rss]
    def sample():
        while not stop.wait(.02):
            peak[0] = max(peak[0], process.memory_info().rss)
    worker = Thread(target=sample, daemon=True)
    worker.start()
    started = perf_counter()
    try:
        result = call()
    finally:
        stop.set()
        worker.join()
        peak[0] = max(peak[0], process.memory_info().rss)
    return result, {"seconds": perf_counter() - started, "peak_rss_bytes": peak[0]}


def _load_backend(runtime_manifest):
    import torch
    from laya_backend import BinaryQuestion, LayaBackend
    torch.set_num_threads(CONFIG["resource"]["torch_threads"])
    return LayaBackend.load(Path(runtime_manifest)), BinaryQuestion(**QUESTION)


def _seal(output):
    write_json(output / "completion.json", {
        "created_utc": datetime.now(UTC).isoformat(),
        "files": {p.relative_to(output).as_posix(): file_hash(p) for p in sorted(output.rglob("*"))
                  if p.is_file() and p.name != "completion.json"},
    })


def _laya_fit(backend, cache, frame, output, seed):
    train = np.flatnonzero(frame.split.eq("train"))
    valid = np.flatnonzero(frame.split.eq("validation"))
    rows, best, previous = [], None, 0
    started = perf_counter()
    deadline = monotonic() + CONFIG["resource"]["training_seconds_per_seed"]
    labels = np.where(frame.split.eq("train"), frame.y_active.to_numpy(), 0)
    for epoch in CONFIG["laya_epochs"]:
        print(f"Training Laya seed={seed}, checkpoint={epoch}", flush=True)
        history = backend.train_head(cache, labels, indices=train,
            epochs=epoch - previous, learning_rate=CONFIG["laya_learning_rate"],
            seed=seed, batch_size=4, reset=previous == 0, deadline_monotonic=deadline)
        elapsed = perf_counter() - started
        if elapsed > CONFIG["resource"]["training_seconds_per_seed"]:
            raise TimeoutError(f"Laya seed {seed} exceeded two-hour training ceiling")
        logits = binary_logit(backend.predict_cached(cache, indices=valid))
        ap = fit.ranking_metrics(frame.y_active.to_numpy()[valid], logits)["ap"]
        checkpoint = output / "heads" / f"seed-{seed}-epoch-{epoch}"
        checkpoint.parent.mkdir(exist_ok=True)
        saved = backend.save_head(checkpoint)
        row = {"model": "laya", "seed": seed, "epoch": epoch, "validation_ap": ap,
               "training_seconds_cumulative": elapsed, "history": history,
               "checkpoint": str(checkpoint), "saved": saved}
        rows.append(row)
        write_json(output / f"laya-seed-{seed}-epoch-{epoch}.json", row)
        if best is None or ap > best["validation_ap"]:
            best = row
        previous = epoch
    backend.load_head(Path(best["checkpoint"]))
    return best, rows


def diagnostic(data_root, runtime_manifest, protocol, output, n=16):
    data_root, output = Path(data_root), Path(output)
    frame, _ = load_prepared(data_root)
    frozen = freeze_run(output, [data_root / "records.parquet", data_root / "manifest.json",
                                 runtime_manifest, *Path(protocol).parent.glob("*AMENDMENT*.md")], protocol)
    backend_pair, load = measure(lambda: _load_backend(runtime_manifest))
    backend, question = backend_pair
    states = frame.loc[frame.split.eq("train"), "state_text"].iloc[:n].tolist()
    if not states:
        raise ValueError("Diagnostic requires training states")
    backend.predict_logits(states[:1], question)
    logits, inference = measure(lambda: backend.predict_logits(states, question))
    binary_logit(logits)
    resources = {**inference, "n": len(states), "seconds_per_1000_extrapolated":
                 inference["seconds"] * 1000 / len(states), "cold_load": load,
                 "scope": "unlabeled training-state diagnostic; not measured 1000-state throughput"}
    resources["feasible_to_attempt"] = (resources["seconds_per_1000_extrapolated"] <= 1800
                                         and max(load["peak_rss_bytes"], inference["peak_rss_bytes"]) <= 8 * 1024**3)
    write_json(output / "diagnostic.json", resources)
    verify_frozen(frozen)
    _seal(output)
    return resources


def load_prepared(data_root):
    from laya_data import validate_prepared
    root = Path(data_root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if set(manifest.get("output_sha256", {})) != {"records.parquet", "states.jsonl", "auxiliary_models.json"}:
        raise ValueError("Prepared data inventory mismatch")
    for name, digest in manifest["output_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or file_hash(path) != digest:
            raise ValueError("Prepared data output checksum mismatch")
    frame = pd.read_parquet(root / "records.parquet").reset_index(drop=True)
    validate_prepared(frame, manifest)
    validate_records(frame, manifest["numeric_columns"])
    return frame, manifest


def run_pilot(data_root, runtime_manifest, protocol, output, diagnostic_path):
    data_root, output = Path(data_root), Path(output)
    engineering = json.loads(Path(diagnostic_path).read_text(encoding="utf-8"))
    if engineering.get("feasible_to_attempt") is not True:
        raise ValueError("A passing unlabeled CPU diagnostic is required before full pilot")
    manifest_path = data_root / "manifest.json"
    frame, manifest = load_prepared(data_root)
    parent_completion = Path(manifest["input_path"]).parents[2] / "completion.json"
    if file_hash(parent_completion) != "7352c79d4b3f934431e97f76b11e1d7acbefa2c10d1fbc5f340fc01917276d52":
        raise ValueError("Parent scientific completion manifest changed")
    inputs = [data_root / "records.parquet", manifest_path, runtime_manifest, diagnostic_path]
    inputs.extend([Path(manifest["input_path"]), parent_completion,
                   data_root / "states.jsonl", data_root / "auxiliary_models.json"])
    amendments = sorted(Path(protocol).parent.glob("*AMENDMENT*.md"))
    frozen = freeze_run(output, [*inputs, *amendments], protocol)
    columns = manifest["numeric_columns"]
    try:
        return _run_models(frame, columns, manifest, runtime_manifest, output, frozen, data_root)
    except Exception as error:
        write_json(output / "failure.json", {"type": type(error).__name__, "message": str(error),
                   "created_utc": datetime.now(UTC).isoformat(), "outcome": "incomplete experiment"})
        raise


def verify_cache(backend, cache, states, question, row_ids):
    expected = hashlib.sha256(json.dumps({"assets": backend.assets["signature"],
        "states": states, "question": question.internal(), "row_ids": row_ids},
        sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if cache.signature != expected or list(cache.row_ids) != row_ids:
        raise ValueError("Cache does not represent the current states/question/rows/assets")


def engineering_query(donors):
    """Borrow structures for timing, never their original biological context or outcomes."""
    required = {"record_id", "identity", "murcko_scaffold", "fingerprint_hex", *PROPERTY_NAMES}
    if not required <= set(donors):
        raise ValueError("Complete chemical feature input required for engineering workload")
    columns = ["record_id", "molecule_id", "identity", "model_smiles", "original_smiles",
               "murcko_scaffold", "fingerprint_hex", "valid", "eligible", *PROPERTY_NAMES]
    return donors[[name for name in columns if name in donors]].copy()


def full_inference_resource(backend, question, manifest, data_root, calibration, output, cold_load,
                            numeric_model, numeric_calibration, prior_workload=None):
    from laya_data import inference_states
    source = Path(manifest["input_path"])
    if file_hash(source) != manifest["input_sha256"]:
        raise ValueError("Engineering source panel changed")
    raw = pd.read_parquet(source)
    references = raw.loc[raw.task_id.eq(manifest["task_id"]) & raw.split.eq("train")]
    if prior_workload is None:
        donors = raw.loc[~raw.identity.isin(references.identity)].sort_values("record_id")
        donors = donors.drop_duplicates("identity").iloc[:1000]
    else:
        donors = pd.read_parquet(prior_workload)
    pool = engineering_query(donors)
    if pool.identity.duplicated().any() or set(pool.identity) & set(references.identity):
        raise ValueError("Engineering workload must be distinct from training references")
    workload = output / "engineering-workload.parquet"
    pool.to_parquet(workload, index=False)
    write_json(output / "engineering-workload.json", {"source_sha256": file_hash(source),
        "workload_sha256": file_hash(workload), "identities": pool.identity.tolist(),
        "record_ids": pool.record_id.astype(int).tolist(),
        "requested_context": {"task_id": manifest["task_id"],
            **{name: references[name].dropna().unique().tolist() for name in ("target_id", "endpoint")
               if name in references}},
        "donor_context_excluded": [name for name in donors if name not in pool],
        "prior_workload_sha256": file_hash(prior_workload) if prior_workload else None,
        "evaluation": "engineering only; no activity outcomes"})
    def execute():
        started = perf_counter()
        candidate = pd.read_parquet(workload)
        auxiliary = json.loads((data_root / "auxiliary_models.json").read_text(encoding="utf-8"))
        prepared = inference_states(candidate, references, auxiliary)
        preparation_seconds = perf_counter() - started
        raw_logits = binary_logit(backend.predict_logits(prepared.state_text.tolist(), question, batch_size=4))
        z = calibration["slope"] * raw_logits + calibration["intercept"]
        p = np.exp(-np.logaddexp(0, -z))
        result = select_candidates(prepared.assign(priority_score=p,
            valid=prepared.get("valid", True), eligible=prepared.get("eligible", True)), 50, max_per_scaffold=3)
        return result.manifest["final_ids"], prepared, preparation_seconds
    (ids, prepared, preparation_seconds), timing = measure(execute)
    def numeric_inference():
        logits = fit.predict_estimator(numeric_matrix(prepared, numeric_model["spec"]), numeric_model["estimator"])
        z = numeric_calibration["slope"] * logits + numeric_calibration["intercept"]
        return select_candidates(prepared.assign(priority_score=np.exp(-np.logaddexp(0, -z)),
            valid=prepared.get("valid", True), eligible=prepared.get("eligible", True)), 50, max_per_scaffold=3)
    _, numeric_timing = measure(numeric_inference)
    return {**timing, "n": len(pool), "selected_ids": ids, "cold_load": cold_load,
            "warm_plus_cold_seconds": timing["seconds"] + cold_load["seconds"],
            "peak_rss_bytes": max(timing["peak_rss_bytes"], cold_load["peak_rss_bytes"]),
            "scope": CONFIG["resource"]["measurement"], "torch_threads": 4, "batch_size": 4,
            "common_preparation_seconds": preparation_seconds,
            "numeric_inference_selection_seconds": numeric_timing["seconds"],
            "numeric_pipeline_seconds": preparation_seconds + numeric_timing["seconds"]}


def verify_preserved(parent, receipt_path):
    parent = Path(parent).resolve()
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    if Path(receipt["parent_run"]).resolve() != parent or not receipt.get("files"):
        raise ValueError("Preserved parent directory mismatch")
    for name, digest in receipt["files"].items():
        path = (parent / name).resolve()
        if not path.is_relative_to(parent) or file_hash(path) != digest:
            raise ValueError(f"Preserved parent changed: {name}")
    if (parent / "freeze.json").is_file():
        frozen = json.loads((parent / "freeze.json").read_text(encoding="utf-8"))
        for name, digest in frozen["input_source_hashes"].items():
            original = Path(name)
            path = parent / "execution-source" / original.name if original.suffix == ".py" else original
            if file_hash(path) != digest:
                raise ValueError(f"Preserved source snapshot or original input changed: {name}")
    return receipt


def reconstruct_evaluation(frame, predictions, models):
    """Replay only saved held-out scores; never refit, recalibrate or select checkpoints."""
    held = frame.loc[frame.split.isin(["test", "recent"])].set_index("record_id", drop=False)
    expected = {(name, seed) for seed in CONFIG["seeds"] for name in
                ("logistic", "boosting", "molecular", "laya", "numeric_selected")}
    if set(zip(predictions.model, predictions.seed)) != expected:
        raise ValueError("Complete saved model/seed prediction inventory required")
    metrics, baskets = [], []
    for (name, seed), rows in predictions.groupby(["model", "seed"], sort=True):
        if rows.record_id.duplicated().any() or set(rows.record_id) != set(held.record_id):
            raise ValueError("Complete unique held-out identity coverage required")
        aligned = rows.set_index("record_id").loc[held.index]
        for column in ("identity", "split", "y_active", "murcko_scaffold"):
            if not np.array_equal(aligned[column].to_numpy(), held[column].to_numpy()):
                raise ValueError("Saved prediction identity/outcome binding mismatch")
        cal = models[f"{name}:{seed}"]["calibration"]
        if cal["identities"] != frame.loc[frame.split.eq("calibration"), "identity"].tolist():
            raise ValueError("Saved calibration membership mismatch")
        raw = aligned.raw_logit.to_numpy(float)
        p = aligned.probability.to_numpy(float)
        expected_p = np.exp(-np.logaddexp(0, -(cal["slope"] * raw + cal["intercept"])))
        if not np.allclose(p, expected_p, rtol=0, atol=1e-15):
            raise ValueError("Saved probability differs from original calibration")
        _, mt, ba = evaluate_model(held.reset_index(drop=True), raw, p, name, int(seed))
        metrics.extend(mt)
        baskets.extend(ba)
    return metrics, baskets


def replay_run(data_root, runtime_manifest, protocol, output, parent, preservation):
    data_root, output, parent = Path(data_root), Path(output), Path(parent).resolve()
    verify_preserved(parent, preservation)
    frame, manifest = load_prepared(data_root)
    inputs = [preservation, runtime_manifest, data_root / "manifest.json", data_root / "records.parquet",
              data_root / "auxiliary_models.json", parent / "freeze.json", Path(manifest["input_path"]),
              *Path(protocol).parent.glob("*AMENDMENT*.md")]
    frozen = freeze_run(output, inputs, protocol)
    try:
        models = {}
        for seed in CONFIG["seeds"]:
            models.update(json.loads((parent / f"seed-{seed}-models.json").read_text(encoding="utf-8")))
        predictions = pd.read_parquet(parent / "predictions-through-seed-44.parquet")
        metrics, baskets = reconstruct_evaluation(frame, predictions, models)
        write_json(output / "models.json", models)
        write_json(output / "metrics.json", {"rows": metrics})
        write_json(output / "baskets.json", {"rows": baskets})
        predictions.to_parquet(output / "predictions.parquet", index=False)
        write_json(output / "recovery.json", {"parent": str(parent), "preservation_sha256": file_hash(preservation),
            "scope": "resource-context repair and evaluation replay; no fit/calibration/model selection",
            "numeric_rejected_candidate_grid": "not persisted before original failure; separate deterministic audit required",
            "laya_candidates": [json.loads(p.read_text(encoding="utf-8"))
                                for p in sorted(parent.glob("laya-seed-*-epoch-*.json"))]})
        backend_pair, load = measure(lambda: _load_backend(runtime_manifest))
        backend, question = backend_pair
        selected = models["laya:42"]["model"]
        checkpoint = parent / "heads" / Path(selected["checkpoint"]).name
        _, restore = measure(lambda: backend.load_head(checkpoint))
        load = {"seconds": load["seconds"] + restore["seconds"],
                "peak_rss_bytes": max(load["peak_rss_bytes"], restore["peak_rss_bytes"]),
                "checkpoint_restore_seconds": restore["seconds"]}
        backend.predict_logits(frame.state_text.iloc[:1].tolist(), question)
        winner = models["numeric_selected:42"]["numeric_selected_family"]
        resources = full_inference_resource(backend, question, manifest, data_root,
            models["laya:42"]["calibration"], output, load, models[f"{winner}:42"]["model"],
            models[f"{winner}:42"]["calibration"], parent / "engineering-workload.parquet")
        resources.update(model_seed=42, checkpoint=str(checkpoint), numeric_family=winner)
        report = assess_gates(pd.DataFrame(metrics), resources)
        write_json(output / "resources.json", resources)
        write_json(output / "gates.json", report)
        verify_frozen(frozen)
        verify_preserved(parent, preservation)
        _seal(output)
        return report
    except Exception as error:
        write_json(output / "failure.json", {"type": type(error).__name__, "message": str(error),
                   "created_utc": datetime.now(UTC).isoformat(), "outcome": "incomplete recovery"})
        raise


def _run_models(frame, columns, manifest, runtime_manifest, output, frozen, data_root):
    backend_pair, load = measure(lambda: _load_backend(runtime_manifest))
    backend, question = backend_pair
    states = frame.state_text.tolist()
    backend.predict_logits(states[:1], question)
    count = min(16, len(states))
    _, inference = measure(lambda: backend.predict_logits(states[:count], question))
    resources = {**inference, "n": count, "cold_load": load,
                 "peak_rss_bytes": max(inference["peak_rss_bytes"], load["peak_rss_bytes"]),
                 "scope": "preliminary transformer-only real-state timing; extrapolation"}
    write_json(output / "preliminary-resources.json", resources)
    if (resources["peak_rss_bytes"] > 8 * 1024**3 or
            resources["seconds"] > 1800 * count / 1000):
        raise RuntimeError("Full inference exceeds prespecified resource gate")
    cache, encoding = measure(lambda: backend.encode_cache(states, question, output / "encoder-cache",
        row_ids=frame.identity.tolist(), batch_size=4))
    verify_cache(backend, cache, states, question, frame.identity.tolist())
    write_json(output / "encoding.json", encoding)
    print(f"Cached {len(states)} real states in {encoding['seconds']:.1f}s", flush=True)
    predictions, metrics, baskets, grids, models = [], [], [], [], {}
    for seed in CONFIG["seeds"]:
        numeric = {}
        for family in ("logistic", "boosting"):
            numeric[family], grid = fit_numeric(frame, columns, family, seed)
            grids.extend(grid)
        winner = max(numeric, key=lambda key: numeric[key]["validation_ap"])
        head, grid = _laya_fit(backend, cache, frame, output, seed)
        grids.extend(grid)
        verify_frozen(frozen)
        raw = {family: fit.predict_estimator(numeric_matrix(frame, model["spec"]), model["estimator"])
               for family, model in numeric.items()}
        auxiliary = manifest.get("raw_auxiliary_logit_column", "auxiliary_molecular_logit_raw")
        raw.update(molecular=frame[auxiliary].to_numpy(float),
                   laya=binary_logit(backend.predict_cached(cache)))
        raw["numeric_selected"] = raw[winner]
        for name, values in raw.items():
            p, calibration = calibrate(frame, values)
            models[f"{name}:{seed}"] = {"calibration": calibration,
                "model": numeric.get(name, head if name == "laya" else {"reference": name}),
                "numeric_selected_family": winner}
            pr, mt, ba = evaluate_model(frame, values, p, name, seed)
            predictions.extend(pr)
            metrics.extend(mt)
            baskets.extend(ba)
        write_json(output / f"seed-{seed}-models.json", {k: v for k, v in models.items() if k.endswith(f":{seed}")})
        pd.DataFrame(predictions).to_parquet(output / f"predictions-through-seed-{seed}.parquet", index=False)
    # Persist scientific results before the separate engineering stage can fail.
    write_json(output / "models.json", models)
    write_json(output / "validation-grid.json", {"candidates": grids})
    write_json(output / "metrics.json", {"rows": metrics})
    write_json(output / "baskets.json", {"rows": baskets})
    pd.DataFrame(predictions).to_parquet(output / "predictions.parquet", index=False)
    backend.load_head(Path(models["laya:42"]["model"]["checkpoint"]))
    winner = models["numeric_selected:42"]["numeric_selected_family"]
    resources = full_inference_resource(backend, question, manifest, data_root,
        models["laya:42"]["calibration"], output, load, models[f"{winner}:42"]["model"],
        models[f"{winner}:42"]["calibration"])
    resources.update(model_seed=42, checkpoint=models["laya:42"]["model"]["checkpoint"],
                     numeric_family=winner)
    write_json(output / "resources.json", resources)
    report = assess_gates(pd.DataFrame(metrics), resources)
    write_json(output / "gates.json", report)
    verify_frozen(frozen)
    _seal(output)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["diagnostic", "run", "replay"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path)
    parser.add_argument("--parent-run", type=Path)
    parser.add_argument("--preservation", type=Path)
    args = parser.parse_args(argv)
    if args.command == "diagnostic":
        result = diagnostic(args.data, args.runtime, args.protocol, args.output)
    elif args.command == "replay":
        if args.parent_run is None or args.preservation is None:
            parser.error("replay requires --parent-run and --preservation")
        result = replay_run(args.data, args.runtime, args.protocol, args.output,
                            args.parent_run, args.preservation)
    else:
        if args.diagnostic is None:
            parser.error("run requires --diagnostic")
        result = run_pilot(args.data, args.runtime, args.protocol, args.output, args.diagnostic)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
