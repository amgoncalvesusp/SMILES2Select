"""Research pilot safeguards: matched evidence, untouched holdout, explicit gates."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import laya_study as study


def panel():
    n = 120
    return pd.DataFrame({
        "record_id": range(n), "row_id": [str(i) for i in range(n)],
        "identity": [f"i{i}" for i in range(n)],
        "murcko_scaffold": [f"s{i}" for i in range(n)],
        "split": ["train"] * 40 + ["validation"] * 20 + ["calibration"] * 20
                 + ["test"] * 20 + ["recent"] * 20,
        "y_active": np.arange(n) % 2,
        "state_text": [f"value {(i % 2) * 2 - 1}" for i in range(n)],
        "value": (np.arange(n) % 2) * 2. - 1.,
        "missing_value": np.zeros(n),
        "auxiliary_molecular_logit_raw": (np.arange(n) % 2) * 2. - 1.,
    })


def test_panel_requires_disjoint_measured_identity_and_scaffold_splits():
    frame = panel()
    study.validate_records(frame, ["value", "missing_value"])
    for column, value in [("identity", "i0"), ("murcko_scaffold", "s0")]:
        bad = frame.copy()
        bad.loc[50, column] = value
        with pytest.raises(ValueError, match="[Ii]dentity|[Ss]caffold"):
            study.validate_records(bad, ["value"])
    bad = frame.assign(y_active=np.nan)
    with pytest.raises(ValueError, match="binary"):
        study.validate_records(bad, ["value"])
    with pytest.raises(ValueError, match="[Ff]eature"):
        study.validate_records(frame, ["y_active"])


@pytest.mark.parametrize("value", [None, pd.DataFrame(), "frame"])
def test_invalid_input_fails_clearly(value):
    with pytest.raises(ValueError):
        study.validate_records(value, ["value"])


def test_training_transform_does_not_read_holdout_values():
    frame = panel()
    spec = study.fit_numeric_spec(frame, ["value", "missing_value"])
    changed = frame.copy()
    changed.loc[changed.split.ne("train"), "value"] = 1e12
    assert spec == study.fit_numeric_spec(changed, ["value", "missing_value"])
    x = study.numeric_matrix(frame, spec)
    assert x.shape == (120, 2) and np.isfinite(x).all()
    assert np.allclose(x[:40].mean(axis=0), 0)


def test_calibration_reads_only_reserved_calibration_labels():
    frame = panel()
    logits = frame.value.to_numpy()
    probabilities, saved = study.calibrate(frame, logits)
    changed = frame.copy()
    changed.loc[changed.split.ne("calibration"), "y_active"] ^= 1
    other, other_saved = study.calibrate(changed, logits)
    assert saved == other_saved
    assert np.array_equal(probabilities, other)
    assert saved["n"] == 20
    assert saved["identities"] == frame.loc[frame.split.eq("calibration"), "identity"].tolist()


def test_option_zero_is_positive_and_common_logit_offsets_cancel():
    scores = np.array([[2., -1.], [-3., 5.]])
    assert np.array_equal(study.binary_logit(scores), [3., -8.])
    assert np.array_equal(study.binary_logit(scores + 100), [3., -8.])
    for bad in ([], [[1]], [[np.inf, 0]], [1, 2]):
        with pytest.raises(ValueError):
            study.binary_logit(bad)


def test_native_basket_reports_actual_and_requested_denominators():
    frame = panel().iloc[:10].assign(murcko_scaffold="same")
    metric, selected = study.selection_metrics(frame, np.linspace(0, 1, 10), 5, 3)
    assert metric["n_actual"] == 3 and metric["shortfall"] == 2
    assert metric["hits"] == 2
    assert metric["hits_per_actual"] == pytest.approx(2 / 3)
    assert metric["hits_per_requested"] == .4
    assert metric["recall_positive_pool"] == .4
    assert selected == [7, 8, 9]
    assert metric["scaffolds"] == 1


def test_probability_and_raw_ranking_are_both_reported():
    metric = study.score_metrics([0, 1, 0, 1], [-2, 2, -1, 1], [.5] * 4)
    assert metric["ap_raw"] == 1
    assert metric["ap_calibrated"] == .5
    assert metric["brier"] == .25
    assert metric["log_loss"] == pytest.approx(np.log(2))
    assert study.score_metrics([1, 1], [0, 1], [.5, .8])["ap_raw"] is None
    with pytest.raises(ValueError):
        study.score_metrics([0, 1], [0, 1], [.5, 1.1])


def comparisons():
    return pd.DataFrame([
        {"seed": seed, "split": "test", "model": model,
         "ap_calibrated": .52 if model == "laya" else .5,
         "brier": .21 if model == "laya" else .2,
         "log_loss": .42 if model == "laya" else .4,
         "primary_hits": 22 if model == "laya" else 20}
        for seed in (42, 43, 44) for model in ("laya", "numeric_selected")
    ])


def test_gate_requires_two_real_seeds_and_all_primary_criteria():
    resource = {"n": 1000, "seconds": 1700, "peak_rss_bytes": 7 * 1024**3}
    result = study.assess_gates(comparisons(), resource)
    assert result["performance_pass"] and result["resource_pass"]
    assert result["passing_seeds"] == [42, 43, 44]
    assert not result["independent_validation"]
    assert result["decision"] == "eligible_for_independent_validation"
    rows = comparisons()
    rows.loc[(rows.model == "laya") & rows.seed.isin([43, 44]), "primary_hits"] = 21
    failed = study.assess_gates(rows, resource)
    assert not failed["performance_pass"]
    # A secondary cohort must not rescue a failed primary gate.
    recent = comparisons().assign(split="recent", ap_calibrated=.99)
    assert not study.assess_gates(pd.concat([rows, recent]), resource)["performance_pass"]


def test_gate_is_incomplete_with_missing_duplicate_or_nonfinite_primary_rows():
    resource = {"n": 1000, "seconds": 10, "peak_rss_bytes": 10}
    for frame in (comparisons().iloc[:4], pd.concat([comparisons(), comparisons().iloc[:1]])):
        with pytest.raises(ValueError, match="seed|duplicate|six"):
            study.assess_gates(frame, resource)
    bad = comparisons().assign(ap_calibrated=np.nan)
    assert not study.assess_gates(bad, resource)["performance_pass"]
    for resource in ({"n": 0, "seconds": 0, "peak_rss_bytes": 10},
                     {"n": 1000, "seconds": 1801, "peak_rss_bytes": 10},
                     {"n": 1000, "seconds": 1, "peak_rss_bytes": 9 * 1024**3}):
        assert not study.assess_gates(comparisons(), resource)["resource_pass"]


def test_numeric_model_selection_never_reads_test_labels():
    frame = panel()
    model, grid = study.fit_numeric(frame, ["value", "missing_value"], "logistic", 42)
    changed = frame.copy()
    changed.loc[changed.split.isin(["test", "recent", "calibration"]), "y_active"] ^= 1
    other, other_grid = study.fit_numeric(changed, ["value", "missing_value"], "logistic", 42)
    assert model == other and grid == other_grid
    assert len(grid) == 3
    assert all(row["validation_ap"] == 1 for row in grid)
    assert model["estimator"]["C"] == .1
    _, grid = study.fit_numeric(frame, ["value"], "boosting", 43)
    assert len(grid) == 2


def test_freeze_prevents_overwrite_and_detects_modified_input(tmp_path):
    source = tmp_path / "input.txt"
    source.write_text("frozen inputs", encoding="utf-8")
    protocol = tmp_path / "PROTOCOL.md"
    protocol.write_text("prespecified", encoding="utf-8")
    output = tmp_path / "run"
    frozen = study.freeze_run(output, [source], protocol)
    study.verify_frozen(frozen)
    assert json.loads((output / "freeze.json").read_text())["config"]["seeds"] == [42, 43, 44]
    with pytest.raises(ValueError, match="exists"):
        study.freeze_run(output, [source], protocol)
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        study.verify_frozen(frozen)


def test_model_evaluation_covers_all_budgets_and_common_pool():
    frame = panel()
    raw = frame.value.to_numpy()
    p = 1 / (1 + np.exp(-raw))
    predictions, metrics, baskets = study.evaluate_model(frame, raw, p, "test_model", 42)
    assert len(predictions) == 40
    assert {row["split"] for row in predictions} == {"test", "recent"}
    assert len(metrics) == 2 and len(baskets) == 12
    assert all(row["pool_n"] == 20 for row in baskets)
    assert all(row["positive_pool_n"] == 10 for row in baskets)
    assert all(row["primary_hits"] == 10 for row in metrics)


def test_gate_uses_calibrated_ap_even_if_raw_ranking_wins():
    rows = comparisons().assign(ap_raw=1.)
    rows.loc[rows.model.eq("laya"), "ap_calibrated"] = .5
    assert not study.assess_gates(rows, {"n": 1000, "seconds": 1,
                                       "peak_rss_bytes": 1})["performance_pass"]


def test_native_basket_preserves_explicit_invalid_and_ineligible_rows():
    frame = panel().iloc[:4].assign(valid=[True, True, False, True],
                                   eligible=[True, False, False, True])
    result, ids = study.selection_metrics(frame, [.1, .9, 1., .8], 4, None)
    assert ids == [0, 3]
    assert result["shortfall"] == 2


def test_measure_records_memory_and_propagates_failure():
    value, measurement = study.measure(lambda: 42)
    assert value == 42 and measurement["seconds"] > 0 and measurement["peak_rss_bytes"] > 0
    with pytest.raises(RuntimeError, match="boom"):
        study.measure(lambda: (_ for _ in ()).throw(RuntimeError("boom")))


class FakeBackend:
    """Cheap orchestration fixture only; never used for biological conclusions."""

    def __init__(self):
        self.epoch = 0
        self.training = []
        self.cache = None

    def predict_logits(self, states, question, **kwargs):
        x = np.array([float(state.split()[-1]) for state in states])
        return np.column_stack([x, -x])

    def encode_cache(self, states, question, output, row_ids, **kwargs):
        self.cache = {"states": states, "row_ids": row_ids}
        return self.cache

    def predict_cached(self, cache, indices=None, **kwargs):
        ids = np.arange(len(cache["states"])) if indices is None else indices
        return self.predict_logits([cache["states"][i] for i in ids], None)

    def train_head(self, cache, labels, indices, epochs, seed, reset, **kwargs):
        assert np.array_equal(indices, np.arange(40))
        assert np.array_equal(labels[40:], np.zeros(len(labels) - 40))
        self.training.append((seed, epochs, reset))
        self.epoch = epochs if reset else self.epoch + epochs
        return [{"epoch": self.epoch, "train_loss": .1}]

    def save_head(self, path):
        path.write_text(str(self.epoch))
        return {"epoch": self.epoch}

    def load_head(self, path):
        self.epoch = int(path.read_text())


def test_full_orchestration_freezes_selection_and_uses_real_seeds_contract(tmp_path, monkeypatch):
    frame, backend = panel(), FakeBackend()
    monkeypatch.setattr(study, "_load_backend", lambda path: (backend, None))
    monkeypatch.setattr(study, "verify_cache", lambda *args: None)
    monkeypatch.setattr(study, "full_inference_resource", lambda *args: {
        "n": 1000, "seconds": 1., "peak_rss_bytes": 1000})
    output = tmp_path / "run"
    output.mkdir()
    manifest = {"raw_auxiliary_logit_column": "auxiliary_molecular_logit_raw"}
    report = study._run_models(frame, ["value", "missing_value"], manifest,
                               tmp_path / "assets.json", output,
                               {"input_source_hashes": {}}, tmp_path)
    assert report["resource_pass"] and not report["performance_pass"]
    assert backend.training == [(seed, epochs, reset) for seed in (42, 43, 44)
                                for epochs, reset in ((1, True), (2, False))]
    models = json.loads((output / "models.json").read_text())
    assert len(models) == 15
    assert all(model["calibration"]["n"] == 20 for model in models.values())
    assert all(model["numeric_selected_family"] == "logistic" for model in models.values())
    assert (output / "completion.json").is_file()


def prepared_fixture(root):
    import laya_data
    root.mkdir()
    frame = panel()
    for name in laya_data.NUMERIC_COLUMNS:
        frame[name] = 0.
    frame["state_text"] = [laya_data.render_state(np.zeros(43))] * len(frame)
    frame.to_parquet(root / "records.parquet", index=False)
    (root / "states.jsonl").write_text("{}\n")
    (root / "auxiliary_models.json").write_text("{}")
    manifest = {"numeric_columns": list(laya_data.NUMERIC_COLUMNS),
                "state_schema": laya_data.STATE_SCHEMA, "task_id": laya_data.DEFAULT_TASK,
                "output_sha256": {name: study.file_hash(root / name) for name in
                    ("records.parquet", "states.jsonl", "auxiliary_models.json")}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return frame, manifest


def test_prepared_loader_rejects_tampering_and_numeric_text_mismatch(tmp_path):
    root = tmp_path / "data"
    frame, manifest = prepared_fixture(root)
    loaded, _ = study.load_prepared(root)
    assert len(loaded) == len(frame)
    (root / "states.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        study.load_prepared(root)
    manifest["output_sha256"]["states.jsonl"] = study.file_hash(root / "states.jsonl")
    frame.loc[0, manifest["numeric_columns"][0]] = 123.
    frame.to_parquet(root / "records.parquet", index=False)
    manifest["output_sha256"]["records.parquet"] = study.file_hash(root / "records.parquet")
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="evidence mismatch"):
        study.load_prepared(root)
    manifest["output_sha256"] = {}
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="inventory"):
        study.load_prepared(root)


def test_diagnostic_is_unlabeled_small_batch_and_does_not_claim_full_throughput(tmp_path, monkeypatch):
    root = tmp_path / "data"
    prepared_fixture(root)
    runtime = tmp_path / "runtime.json"
    protocol = tmp_path / "protocol.md"
    runtime.write_text("{}")
    protocol.write_text("protocol")
    backend = SimpleNamespace(predict_logits=lambda states, question: np.zeros((len(states), 2)))
    monkeypatch.setattr(study, "_load_backend", lambda path: (backend, None))
    result = study.diagnostic(root, runtime, protocol, tmp_path / "diagnostic")
    assert result["n"] == 16 and result["feasible_to_attempt"]
    assert "not measured" in result["scope"]
    assert (tmp_path / "diagnostic/completion.json").exists()


def test_run_refuses_failed_diagnostic_before_reading_data(tmp_path):
    diagnostic = tmp_path / "diagnostic.json"
    diagnostic.write_text('{"feasible_to_attempt": false}')
    with pytest.raises(ValueError, match="CPU diagnostic"):
        study.run_pilot(tmp_path / "missing", "runtime", "protocol", tmp_path / "out", diagnostic)


def test_cache_signature_must_bind_current_input_evidence():
    backend = SimpleNamespace(assets={"signature": "official"})
    question = SimpleNamespace(internal=lambda: {"A": "positive"})
    payload = {"assets": "official", "states": ["old"],
               "question": question.internal(), "row_ids": ["i1"]}
    digest = study.hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    cache = SimpleNamespace(signature=digest, row_ids=["i1"])
    study.verify_cache(backend, cache, ["old"], question, ["i1"])
    with pytest.raises(ValueError, match="Cache"):
        study.verify_cache(backend, cache, ["changed"], question, ["i1"])


def test_resource_pipeline_drops_query_labels_and_includes_native_selection(tmp_path, monkeypatch):
    import laya_data
    frame = panel().assign(task_id=laya_data.DEFAULT_TASK, fingerprint_hex="00", model_smiles="CC")
    for name in laya_data.PROPERTY_NAMES:
        frame[name] = frame.value
    source = tmp_path / "source.parquet"
    frame.to_parquet(source, index=False)
    (tmp_path / "auxiliary_models.json").write_text("{}")
    def evidence(query, references, model):
        assert "y_active" not in query
        assert len(references) == 40
        assert not set(query.identity) & set(references.identity)
        assert "target_id" not in query and "task_id" not in query
        return query.assign(state_text=[f"value {value}" for value in query[laya_data.PROPERTY_NAMES[0]]])
    monkeypatch.setattr(laya_data, "inference_states", evidence)
    numeric, _ = study.fit_numeric(frame, [laya_data.PROPERTY_NAMES[0]], "logistic", 42)
    manifest = {"input_path": str(source), "input_sha256": study.file_hash(source),
                "task_id": laya_data.DEFAULT_TASK}
    output = tmp_path / "engineering"
    output.mkdir()
    result = study.full_inference_resource(FakeBackend(), None, manifest, tmp_path,
        {"slope": 1., "intercept": 0.}, output, {"seconds": .01, "peak_rss_bytes": 1},
        numeric, {"slope": 1., "intercept": 0.})
    assert result["n"] == 80 and len(result["selected_ids"]) == 50
    assert result["seconds"] > result["common_preparation_seconds"]
    assert result["numeric_pipeline_seconds"] >= result["numeric_inference_selection_seconds"]
    assert "y_active" not in pd.read_parquet(output / "engineering-workload.parquet")


def test_cli_requires_diagnostic_for_actual_run():
    with pytest.raises(SystemExit) as exc:
        study.main(["run", "--data", "x", "--runtime", "x", "--protocol", "x", "--output", "x"])
    assert exc.value.code == 2


def test_engineering_query_projects_donor_targets_to_chemical_evidence_only():
    from s2s_decision.schema import PROPERTY_NAMES
    donor = panel().iloc[:2].assign(target_id=["CA2", "BACE1"], endpoint=["Ki", "IC50"],
        task_id="donor", assay_context="old assay", fingerprint_hex="00", model_smiles="CC")
    for name in PROPERTY_NAMES:
        donor[name] = [1., 2.]
    projected = study.engineering_query(donor)
    assert not {"target_id", "endpoint", "task_id", "assay_context", "y_active", "state_text"} & set(projected)
    assert projected.identity.tolist() == donor.identity.tolist()
    pd.testing.assert_frame_equal(projected[list(PROPERTY_NAMES)], donor[list(PROPERTY_NAMES)])
    with pytest.raises(ValueError, match="chemical"):
        study.engineering_query(donor.drop(columns=["fingerprint_hex"]))


def test_evaluation_replay_uses_saved_probabilities_without_fitting_or_calibration(monkeypatch):
    frame = panel()
    logits = frame.value.to_numpy()
    probabilities, calibration = study.calibrate(frame, logits)
    predictions, models = [], {}
    for seed in study.CONFIG["seeds"]:
        for model in ("logistic", "boosting", "molecular", "laya", "numeric_selected"):
            rows, _, _ = study.evaluate_model(frame, logits, probabilities, model, seed)
            predictions.extend(rows)
            models[f"{model}:{seed}"] = {"calibration": calibration}
    def forbidden(*args, **kwargs):
        raise AssertionError("Recovery must never fit or recalibrate")
    monkeypatch.setattr(study, "calibrate", forbidden)
    monkeypatch.setattr(study.fit, "fit_logistic", forbidden)
    metrics, baskets = study.reconstruct_evaluation(frame, pd.DataFrame(predictions), models)
    assert len(metrics) == 30 and len(baskets) == 180
    bad = pd.DataFrame(predictions)
    bad.loc[0, "probability"] = .123
    with pytest.raises(ValueError, match="calibrat"):
        study.reconstruct_evaluation(frame, bad, models)
    with pytest.raises(ValueError, match="complete|identity"):
        study.reconstruct_evaluation(frame, pd.DataFrame(predictions).iloc[1:], models)


def test_preserved_parent_receipt_checks_file_hash_and_path_boundary(tmp_path):
    parent = tmp_path / "old"
    parent.mkdir()
    (parent / "prediction.txt").write_text("saved")
    receipt = tmp_path / "preserved.json"
    receipt.write_text(json.dumps({"parent_run": str(parent),
        "files": {"prediction.txt": study.file_hash(parent / "prediction.txt")}}))
    study.verify_preserved(parent, receipt)
    (parent / "prediction.txt").write_text("tampered")
    with pytest.raises(ValueError, match="preserved|Preserved"):
        study.verify_preserved(parent, receipt)


def test_recovery_orchestration_restores_fixed_head_and_never_retrains(tmp_path, monkeypatch):
    data = tmp_path / "data"
    frame, manifest = prepared_fixture(data)
    source = tmp_path / "source.parquet"
    frame.to_parquet(source)
    manifest["input_path"] = str(source)
    (data / "manifest.json").write_text(json.dumps(manifest))
    parent = tmp_path / "parent"
    (parent / "heads").mkdir(parents=True)
    (parent / "freeze.json").write_text("{}")
    (parent / "heads/seed-42-epoch-3").write_text("3")
    (parent / "engineering-workload.parquet").write_bytes(b"original workload")
    logits = frame.value.to_numpy()
    probabilities, calibration = study.calibrate(frame, logits)
    predictions = []
    for seed in study.CONFIG["seeds"]:
        models = {}
        for name in ("logistic", "boosting", "molecular", "laya", "numeric_selected"):
            rows, _, _ = study.evaluate_model(frame, logits, probabilities, name, seed)
            predictions.extend(rows)
            models[f"{name}:{seed}"] = {"calibration": calibration,
                "model": {"checkpoint": "old/path/seed-42-epoch-3"},
                "numeric_selected_family": "logistic"}
        (parent / f"seed-{seed}-models.json").write_text(json.dumps(models))
    pd.DataFrame(predictions).to_parquet(parent / "predictions-through-seed-44.parquet")
    preservation, protocol, runtime = [tmp_path / name for name in ("preserved.json", "protocol.md", "assets.json")]
    for path in (preservation, protocol, runtime):
        path.write_text("{}")
    backend = FakeBackend()
    backend.predict_logits = lambda states, question: np.zeros((len(states), 2))
    monkeypatch.setattr(study, "verify_preserved", lambda *args: {})
    monkeypatch.setattr(study, "_load_backend", lambda path: (backend, None))
    def resource(*args):
        assert args[-1] == parent.resolve() / "engineering-workload.parquet"
        assert backend.epoch == 3
        return {"n": 1000, "seconds": 1., "peak_rss_bytes": 100}
    monkeypatch.setattr(study, "full_inference_resource", resource)
    def no_fit(*args, **kwargs):
        raise AssertionError("Recovery cannot train or recalibrate")
    monkeypatch.setattr(study, "calibrate", no_fit)
    monkeypatch.setattr(study.fit, "fit_logistic", no_fit)
    result = study.replay_run(data, runtime, protocol, tmp_path / "recovery", parent, preservation)
    assert result["resource_pass"]
    assert not backend.training
    assert (tmp_path / "recovery/completion.json").is_file()
