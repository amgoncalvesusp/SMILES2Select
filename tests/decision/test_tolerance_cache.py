"""Optional durable chemistry caching preserves exact annotations and source fields."""

import copy
import hashlib
import json
import sqlite3
import subprocess
import sys
from contextlib import closing, contextmanager

import numpy as np
import pandas as pd
import pytest

from s2s_decision import tolerance_features as features


@contextmanager
def database(path):
    with closing(sqlite3.connect(path)) as connection, connection:
        yield connection


def records():
    return pd.DataFrame({"record_id": [7, 9, 11], "model_smiles": ["CCO", "CC[N+](=O)[O-]", "invalid"],
                         "identity": ["source-a", "source-b", "source-c"],
                         "document_id": ["private-source-a", "private-source-b", None],
                         "task_id": "T", "pactivity": [7., 4., np.nan], "mol_wt": [600., np.nan, 50.]},
                        index=[10, 20, 30])


def test_cold_warm_reopened_cache_parity_and_source_immutability(tmp_path, monkeypatch):
    source = records()
    source.attrs["source_note"] = {"labels": ["unchanged"]}
    before = copy.deepcopy(source)
    expected, schema = features.annotate_tolerance(source, catalogs=("brenk",))
    path = tmp_path / "chemistry.sqlite"
    cold, cold_schema = features.annotate_tolerance(source, schema=schema, cache_path=path)
    pd.testing.assert_frame_equal(cold, expected)
    assert cold_schema == schema
    monkeypatch.setattr(features, "_criterion_annotations", lambda *args: pytest.fail("warm cache recomputed chemistry"))
    warm, warm_schema = features.annotate_tolerance(source, schema=schema, cache_path=path)
    pd.testing.assert_frame_equal(warm, expected)
    pd.testing.assert_frame_equal(source, before)
    assert warm.attrs == expected.attrs and source.attrs == before.attrs
    assert warm_schema == schema
    alert_columns = [meta["violation_column"] for meta in schema["criterion_columns"].values() if meta["kind"] == "alert"]
    assert all(isinstance(warm[name].dtype, pd.SparseDtype) for name in alert_columns)
    warm.tolerance_alert_details.iloc[1][0]["alert_name"] = "Changed output only"
    again, _ = features.annotate_tolerance(source, schema=schema, cache_path=path)
    pd.testing.assert_frame_equal(again, expected)


def test_fresh_python_process_reuses_durable_annotations(tmp_path):
    source = records()
    path = tmp_path / "durable.sqlite"
    expected, _ = features.annotate_tolerance(source, catalogs=("brenk",), cache_path=path)
    input_path, output_path = tmp_path / "input.json", tmp_path / "warm.json"
    input_path.write_text(source.to_json(orient="split"), encoding="utf-8")
    script = """
import io, pathlib, sys
import pandas as pd
from s2s_decision import tolerance_features as features
source = pd.read_json(io.StringIO(pathlib.Path(sys.argv[1]).read_text()), orient='split')
def forbidden(*args):
    raise AssertionError('A reopened durable cache must skip native criterion annotation')
features._criterion_annotations = forbidden
warm, _ = features.annotate_tolerance(source, catalogs=('brenk',), cache_path=sys.argv[2])
pathlib.Path(sys.argv[3]).write_text(warm.to_json(orient='split', double_precision=15))
"""
    run = subprocess.run([sys.executable, "-c", script, str(input_path), str(path), str(output_path)],
                         capture_output=True, text=True, timeout=120, check=False)
    assert run.returncode == 0, run.stderr
    assert output_path.read_text() == expected.to_json(orient="split", double_precision=15)


def test_aliases_source_labels_order_and_supplied_identities_are_not_cached(tmp_path, monkeypatch):
    source = records().iloc[:2]
    path = tmp_path / "aliases.sqlite"
    cold, schema = features.annotate_tolerance(source, catalogs=("brenk",), cache_path=path)
    descriptor_source = cold[[*source.columns, *[name for name in schema["descriptor_columns"] if name not in source]]]
    changed = descriptor_source.iloc[::-1].copy().assign(record_id=[99, 98], identity=["other-b", "other-a"],
                document_id="new-source", task_id="new-task", pactivity=1., murcko_scaffold="supplied")
    changed.loc[10, "model_smiles"] = "OCC"
    expected, _ = features.annotate_tolerance(changed, schema=schema)
    monkeypatch.setattr(features, "_criterion_annotations", lambda *args: pytest.fail("same canonical chemistry missed cache"))
    warm, _ = features.annotate_tolerance(changed, schema=schema, cache_path=path)
    pd.testing.assert_frame_equal(warm, expected)
    assert warm.identity.tolist() == ["other-b", "other-a"]
    assert warm.murcko_scaffold.tolist() == ["supplied", "supplied"]
    with database(path) as conn:
        bodies = [json.loads(row[0]) for row in conn.execute("SELECT payload FROM tolerance_annotation_cache")]
    for body in bodies:
        text = json.dumps(body)
        assert "private-source" not in text
        assert not {"record_id", "task_id", "pactivity", "document_id"} & set(body["annotation"])


def test_supplied_observations_and_full_recipe_changes_invalidate_entries(tmp_path, monkeypatch):
    source = records().iloc[:1]
    path = tmp_path / "invalidate.sqlite"
    features.annotate_tolerance(source, catalogs=("brenk",), cache_path=path)
    original, calls = features._criterion_annotations, []

    def watched(*args):
        calls.append(len(args[0]))
        return original(*args)

    monkeypatch.setattr(features, "_criterion_annotations", watched)
    altered, schema = features.annotate_tolerance(source.assign(mol_wt=400.), catalogs=("brenk",), cache_path=path)
    column = schema["criterion_columns"]["rule:lipinski:lip_mw_max"]["violation_column"]
    assert altered[column].iloc[0] == 0 and calls == [1]
    alternate, recipe = features.annotate_tolerance(source, catalogs=(), cache_path=path)
    assert calls == [1, 1] and recipe["chemistry_hash"] != schema["chemistry_hash"]
    assert not any(name.startswith("tolerance_alert__") for name in alternate)


@pytest.mark.parametrize("corruption", ["checksum", "json", "duplicate", "nonfinite", "shape", "bound_key"])
def test_corrupt_entries_are_logged_recomputed_and_replaced(tmp_path, monkeypatch, caplog, corruption):
    source = records().iloc[:1]
    path = tmp_path / "corrupt.sqlite"
    expected, schema = features.annotate_tolerance(source, catalogs=("brenk",), cache_path=path)
    with database(path) as conn:
        key, text = conn.execute("SELECT cache_key,payload FROM tolerance_annotation_cache").fetchone()
        body = json.loads(text)
        if corruption == "json":
            text = "broken JSON"
        elif corruption == "duplicate":
            text = '{"schema":"one","schema":"two"}'
        elif corruption == "nonfinite":
            text = '{"bad":NaN}'
        elif corruption == "shape":
            body["annotation"] = {}
            text = json.dumps(body)
        elif corruption == "bound_key":
            body["cache_key"] = "0" * 64
            text = json.dumps(body)
        digest = "0" * 64 if corruption == "checksum" else hashlib.sha256(text.encode()).hexdigest()
        conn.execute("UPDATE tolerance_annotation_cache SET payload=?,payload_sha256=? WHERE cache_key=?", (text, digest, key))
    original, calls = features._criterion_annotations, []

    def watched(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(features, "_criterion_annotations", watched)
    recovered, _ = features.annotate_tolerance(source, schema=schema, cache_path=path)
    pd.testing.assert_frame_equal(recovered, expected)
    assert calls == [True] and "recomputing" in caplog.text.lower()
    features.annotate_tolerance(source, schema=schema, cache_path=path)
    assert calls == [True]


def test_cache_coexists_with_descriptor_tables_and_has_clear_failure_behavior(tmp_path):
    path = tmp_path / "shared.sqlite"
    with database(path) as conn:
        conn.execute("CREATE TABLE descriptor_cache (receipt TEXT)")
        conn.execute("INSERT INTO descriptor_cache VALUES (?)", ("preserved",))
    features.annotate_tolerance(records().iloc[:1], cache_path=path)
    with database(path) as conn:
        assert conn.execute("SELECT receipt FROM descriptor_cache").fetchone() == ("preserved",)
    broken = tmp_path / "not_database.sqlite"
    broken.write_text("not a SQLite database")
    with pytest.raises(ValueError, match="tolerance annotation cache"):
        features.annotate_tolerance(records().iloc[:1], cache_path=broken)
    with pytest.raises(ValueError, match="tolerance annotation cache"):
        features.annotate_tolerance(records().iloc[:1], cache_path=tmp_path)


def test_profile_predictions_and_hash_are_exact_with_optional_cache(tmp_path, monkeypatch):
    from s2s_decision.tolerance_profile import fit_tolerance, predict_tolerance
    from s2s_decision.tolerance_schema import profile_hash

    training = pd.DataFrame({"record_id": [1, 2, 3], "model_smiles": ["CCN", "CCCN", "CCCCN"],
                             "pactivity": 7., "document_id": ["d1", "d2", "d3"]})
    profile = fit_tolerance(training, support_settings={"min_identities": 1, "min_scaffolds": 1,
                                                       "min_documents": 0})
    source = records()
    signature = profile_hash(profile)
    expected = predict_tolerance(profile, source, baseline_eligible=[True, False, False])
    cold = predict_tolerance(profile, source, baseline_eligible=[True, False, False],
                             cache_path=tmp_path / "prediction.sqlite")
    pd.testing.assert_frame_equal(cold, expected)
    monkeypatch.setattr(features, "_criterion_annotations", lambda *args: pytest.fail("cached prediction recomputed"))
    warm = predict_tolerance(profile, source, baseline_eligible=[True, False, False],
                             cache_path=tmp_path / "prediction.sqlite")
    pd.testing.assert_frame_equal(warm, expected)
    assert profile_hash(profile) == signature


def test_preview_and_replay_use_configured_cache_without_changing_selection(tmp_path, monkeypatch):
    from s2s_decision.artifacts import read_bundle
    from s2s_decision.tolerance_decision import (
        preview_tolerance_decision,
        tolerance_snapshot_from_workspace,
        verify_tolerance_decision,
    )
    from s2s_decision.tolerance_profile import fit_tolerance, write_tolerance_package
    from smiles2select.selection_intelligence.constrained_selection import SelectionConstraints
    from tests.decision.test_tolerance_decision import workspace

    training = pd.DataFrame({"record_id": [1, 2, 3], "model_smiles": ["CCN", "CCCN", "CCCCN"],
                             "pactivity": 7., "document_id": ["d1", "d2", "d3"]})
    package = tmp_path / "package"
    write_tolerance_package(fit_tolerance(training), package)
    result, candidates, basket = workspace()
    snapshot = tolerance_snapshot_from_workspace(result, candidates, basket,
        SelectionConstraints(2), revisited_profiles=("lipinski",))
    uncached = preview_tolerance_decision(snapshot, tmp_path / "plain", package)
    cache_path = tmp_path / "preview.sqlite"
    cached = preview_tolerance_decision(snapshot, tmp_path / "cached", package, cache_path=cache_path)
    assert cached == uncached
    pd.testing.assert_frame_equal(read_bundle(tmp_path / "plain" / "scored").records,
                                  read_bundle(tmp_path / "cached" / "scored").records)
    monkeypatch.setattr(features, "_criterion_annotations", lambda *args: pytest.fail("cached replay recomputed"))
    summary, proposed = verify_tolerance_decision(snapshot, tmp_path / "cached", package, cache_path=cache_path)
    assert summary == uncached and proposed.manifest["final_ids"] == cached["proposed_ids"]


def test_output_attrs_are_independent_of_input_and_other_cache_calls(tmp_path):
    source = records().iloc[:1]
    source.attrs["note"] = {"nested": ["original"]}
    cold, schema = features.annotate_tolerance(source, catalogs=(), cache_path=tmp_path / "attrs.sqlite")
    cold.attrs["note"]["nested"].append("output-only")
    assert source.attrs["note"] == {"nested": ["original"]}
    warm, _ = features.annotate_tolerance(source, schema=schema, cache_path=tmp_path / "attrs.sqlite")
    assert warm.attrs["note"] == {"nested": ["original"]}


def test_batched_repeated_chemistry_duplicate_indices_and_empty_records(tmp_path, monkeypatch):
    source = pd.DataFrame({"record_id": range(401), "model_smiles": "CCO"}, index=[1] * 401)
    expected, schema = features.annotate_tolerance(source, catalogs=())
    original, calls = features._criterion_annotations, []

    def watched(*args):
        calls.append(len(args[0]))
        return original(*args)

    monkeypatch.setattr(features, "_criterion_annotations", watched)
    path = tmp_path / "batches.sqlite"
    cold, _ = features.annotate_tolerance(source, schema=schema, cache_path=path)
    pd.testing.assert_frame_equal(cold, expected)
    assert calls == [1]
    with database(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM tolerance_annotation_cache").fetchone() == (1,)
    empty = source.iloc[:0]
    empty_expected, _ = features.annotate_tolerance(empty, schema=schema)
    empty_cached, _ = features.annotate_tolerance(empty, schema=schema, cache_path=path)
    pd.testing.assert_frame_equal(empty_cached, empty_expected)


@pytest.mark.parametrize("bad", [float("inf"), object()])
def test_unsupported_observation_values_fail_clearly_without_creating_entries(tmp_path, bad):
    source = records().iloc[:1].assign(mol_wt=bad)
    with pytest.raises(ValueError, match="cache observation"):
        features.annotate_tolerance(source, catalogs=(), cache_path=tmp_path / "bad.sqlite")


def test_cache_storage_strict_json_and_database_errors(tmp_path, monkeypatch):
    from s2s_decision import tolerance_cache as cache_module

    path = tmp_path / "strict.sqlite"
    def validate(_):
        return None
    with cache_module.ToleranceAnnotationCache(path, {"recipe": "fixture"}) as cache:
        # Quotes are data even in the lower-level repository interface.
        key = "'; DROP TABLE tolerance_annotation_cache; --"
        cache.store({key: {"nested": [None, True, "text", 12.]}}, validate)
        assert cache.fetch([key], validate)[key] == {"nested": [None, True, "text", 12.]}
        with pytest.raises(ValueError, match="only JSON"):
            cache.store({"unsupported": {"object": object()}}, validate)
        with pytest.raises(ValueError, match="finite"):
            cache.store({"nonfinite": {"number": float("nan")}}, validate)
        monkeypatch.setattr(cache_module, "MAX_ENTRY_BYTES", 100)
        with pytest.raises(ValueError, match="size limit"):
            cache.store({"large": {"value": "x" * 200}}, validate)
        monkeypatch.setattr(cache_module, "MAX_ENTRY_BYTES", 2 * 1024 * 1024)
        cache.connection.execute("DROP TABLE tolerance_annotation_cache")
        with pytest.raises(ValueError, match="read tolerance annotation cache"):
            cache.fetch(["missing"], validate)
        with pytest.raises(ValueError, match="write tolerance annotation cache"):
            cache.store({"missing": {}}, validate)


def test_oversized_corruption_is_bounded_before_python_receives_payload(tmp_path, monkeypatch, caplog):
    from s2s_decision import tolerance_cache as cache_module

    with cache_module.ToleranceAnnotationCache(tmp_path / "size.sqlite", {}) as cache:
        cache.connection.execute("INSERT INTO tolerance_annotation_cache VALUES (?,?,?)", ("bad", "x" * 1000, "digest"))
        monkeypatch.setattr(cache_module, "MAX_ENTRY_BYTES", 100)
        decode = cache._decode

        def bounded(key, payload, digest, validate):
            assert payload is None, "Oversized database text must not be materialized in Python"
            return decode(key, payload, digest, validate)

        monkeypatch.setattr(cache, "_decode", bounded)
        assert cache.fetch(["bad"], lambda _: pytest.fail("Oversized JSON cannot be validated")) == {}
        assert "recomputing" in caplog.text


@pytest.mark.parametrize("corruption", ["order", "integer_bool", "float_index"])
def test_native_hit_order_and_types_are_verified_after_valid_checksum(tmp_path, caplog, corruption):
    source = pd.DataFrame({"record_id": [1], "model_smiles": ["O=[N+]([O-])c1ccccc1[N+](=O)[O-]"]})
    path = tmp_path / "native.sqlite"
    expected, recipe = features.annotate_tolerance(source, catalogs=("brenk",), cache_path=path)
    with database(path) as conn:
        key, payload = conn.execute("SELECT cache_key,payload FROM tolerance_annotation_cache").fetchone()
        body = json.loads(payload)
        hits = body["annotation"]["tolerance_alert_details"]
        assert len(hits) >= 2
        if corruption == "order":
            hits.reverse()
        elif corruption == "integer_bool":
            hits[0]["entry_hit"] = 1
        else:
            hits[0]["entry_index"] = float(hits[0]["entry_index"])
        altered = json.dumps(body)
        conn.execute("UPDATE tolerance_annotation_cache SET payload=?,payload_sha256=? WHERE cache_key=?",
                     (altered, hashlib.sha256(altered.encode()).hexdigest(), key))
    restored, _ = features.annotate_tolerance(source, schema=recipe, cache_path=path)
    pd.testing.assert_frame_equal(restored, expected)
    assert "recomputing" in caplog.text


def test_closed_cache_handles_fail_clearly(tmp_path):
    from s2s_decision.tolerance_cache import ToleranceAnnotationCache

    cache = ToleranceAnnotationCache(tmp_path / "closed.sqlite", {})
    cache.close()
    cache.close()
    with pytest.raises(ValueError, match="cache is closed"):
        cache.fetch([], lambda _: None)
    with pytest.raises(ValueError, match="cache is closed"):
        cache.store({}, lambda _: None)
