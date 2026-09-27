"""Preparation checks use real source IO, curation, partitions and bundle hashes."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from s2s_decision import context, experiment, features
from s2s_decision.artifacts import file_hash, read_bundle, write_bundle, write_json
from s2s_decision.schema import CONTEXT_NAMES, FeatureSet


def _plan(tmp_path, source):
    plan = experiment.make_plan(source)
    plan["tasks"] = [{"id": "TARGET_IC50", "target_id": "TARGET", "endpoint": "IC50"}]
    write_json(tmp_path / "plan.json", plan)
    return plan


def test_extracted_source_keeps_global_ordinals_across_chunks_and_rejects_tamper(tmp_path):
    source = tmp_path / "full.tsv"
    frame = pd.DataFrame({"target_id": ["OTHER"] * 10003, "SMILES": ["CCO"] * 10003})
    frame.loc[[1, 9999, 10002], "target_id"] = "TARGET"
    frame.to_csv(source, sep="\t", index=False)
    plan = _plan(tmp_path, source)
    experiment._extract_sources(tmp_path, plan)
    extracted = tmp_path / "sources/TARGET.tsv"
    rows = pd.read_csv(extracted, sep="\t")
    assert rows.papyrus_source_row.tolist() == [2, 10000, 10003]
    marker = json.loads((tmp_path / "source-extraction.json").read_text())
    assert marker["source_rows"] == 10003
    assert marker["source_sha256"] == file_hash(source)
    original = extracted.read_bytes()
    experiment._extract_sources(tmp_path, plan)
    assert extracted.read_bytes() == original
    extracted.write_bytes(original + b"tampered\n")
    with pytest.raises(ValueError, match="checksum"):
        experiment._extract_sources(tmp_path, plan)


def test_incomplete_extraction_refuses_to_append_or_overwrite(tmp_path):
    source = tmp_path / "full.tsv"
    source.write_text("target_id\tSMILES\nTARGET\tCCO\n")
    plan = _plan(tmp_path, source)
    (tmp_path / "sources").mkdir()
    partial = tmp_path / "sources/TARGET.tsv"
    partial.write_text("preserve partial")
    with pytest.raises(ValueError, match="incomplete source extraction"):
        experiment._extract_sources(tmp_path, plan)
    assert partial.read_text() == "preserve partial"


def test_task_features_uses_real_labels_excludes_all_duplicates_and_checks_cache(
    tmp_path, monkeypatch
):
    source = tmp_path / "full.tsv"
    pd.DataFrame(
        {
            "target_id": ["OTHER", "TARGET", "TARGET", "TARGET", "TARGET"],
            "endpoint": ["IC50", "IC50", "IC50", "IC50", "Ki"],
            "pactivity": [8, 7, 5, 8, 9],
            "original_smiles": ["C", "CC", "CC", "CCC", "CCCC"],
        }
    ).to_csv(source, sep="\t", index=False)
    plan = _plan(tmp_path, source)
    experiment._extract_sources(tmp_path, plan)
    calls = []

    def featurize(measured, bits):
        calls.append((len(measured), bits))
        return FeatureSet(
            measured.assign(valid=True, identity=measured.original_smiles),
            {"fingerprint_bits": bits},
        )

    monkeypatch.setattr(features, "featurize", featurize)
    task = plan["tasks"][0]
    result = experiment._task_features(tmp_path, task, plan)
    assert calls == [(3, 2048)]
    assert result.records.original_smiles.tolist() == ["CCC"]
    assert result.records.y_active.tolist() == [1.0]
    assert result.records.papyrus_source_row.tolist() == ["4"]
    assert json.loads(result.records.raw_record_json.iloc[0])["papyrus_source_row"] == "4"
    assert result.manifest["excluded_records"]["duplicate_identity_records"] == [1, 2]
    assert result.manifest["source_sha256"] == file_hash(tmp_path / "sources/TARGET.tsv")
    assert result.manifest["full_source_sha256"] == file_hash(source)
    assert result.manifest["target_id"] == "TARGET"
    assert result.manifest["endpoint"] == "IC50"
    cached = experiment._task_features(tmp_path, task, plan)
    assert cached.records.equals(result.records)
    assert len(calls) == 1
    records = tmp_path / "features/TARGET_IC50/records.jsonl"
    records.write_text("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        experiment._task_features(tmp_path, task, plan)


@pytest.mark.parametrize("method", ["scaffold", "temporal"])
def test_real_partition_is_disjoint_hash_bound_and_temporal_seed_independent(
    tmp_path, monkeypatch, method
):
    source = tmp_path / "full.tsv"
    source.write_text("fixture")
    plan = _plan(tmp_path, source)
    frame = pd.DataFrame(
        {
            "record_id": np.arange(1000),
            "identity": [f"m{i}" for i in range(1000)],
            "murcko_scaffold": [f"s{i}" for i in range(1000)],
            "year": 2000 + np.arange(1000) // 50,
            "y_active": np.arange(1000) % 2,
        }
    )
    calls = []

    def deterministic_context(records, seed, temporal):
        calls.append((seed, temporal))
        return pd.DataFrame(0.0, index=records.index, columns=CONTEXT_NAMES)

    monkeypatch.setattr(context, "training_context", deterministic_context)
    feature_set = FeatureSet(frame, {"source_sha256": file_hash(source)})
    task = plan["tasks"][0]
    result = experiment._prepare_partition(tmp_path, task, feature_set, method, 23, plan)
    assert result["eligible"]
    expected_seed = 11 if method == "temporal" else 23
    assert calls == [(expected_seed, method == "temporal")]
    dataset = read_bundle(result["dataset"])
    assert len(dataset.records) == 1000
    assert dataset.records.groupby("identity").split.nunique().eq(1).all()
    assert dataset.manifest["experiment_plan_sha256"] == file_hash(tmp_path / "plan.json")
    assert set(CONTEXT_NAMES).issubset(dataset.records.columns)
    if method == "temporal":
        dates = dataset.records.groupby("split").year.agg(["min", "max"])
        assert dates.loc["train", "max"] < dates.loc["validation", "min"]
        assert dates.loc["validation", "max"] < dates.loc["calibration", "min"]
        assert dates.loc["calibration", "max"] < dates.loc["test", "min"]
    reused_seed = 71 if method == "temporal" else 23
    assert (
        experiment._prepare_partition(tmp_path, task, feature_set, method, reused_seed, plan)
        == result
    )
    assert len(calls) == 1
    manifest_path = Path(result["dataset"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    write_json(manifest_path, {**manifest, "experiment_plan_sha256": "different"})
    with pytest.raises(ValueError, match="another experiment plan"):
        experiment._prepare_partition(tmp_path, task, feature_set, method, reused_seed, plan)


def test_infeasible_temporal_partition_records_reason_and_never_falls_back(tmp_path, monkeypatch):
    source = tmp_path / "full.tsv"
    source.write_text("fixture")
    plan = _plan(tmp_path, source)
    frame = pd.DataFrame(
        {"identity": ["a", "b", "c", "d"], "year": [2020] * 4, "y_active": [0, 1, 0, 1]}
    )
    monkeypatch.setattr(
        context, "training_context", lambda *args, **kwargs: pytest.fail("No fallback")
    )
    result = experiment._prepare_partition(
        tmp_path, plan["tasks"][0], FeatureSet(frame, {}), "temporal", 23, plan
    )
    assert not result["eligible"]
    assert "four distinct date/year groups" in result["reason"]
    assert result["split_seed"] == 11
    assert not (tmp_path / "datasets/TARGET_IC50/temporal/11").exists()
    saved = json.loads((tmp_path / "datasets/TARGET_IC50/temporal/11-feasibility.json").read_text())
    assert saved == result


def test_prepare_rejects_changed_full_source_before_reusing_prepared_marker(tmp_path):
    source = tmp_path / "full.tsv"
    source.write_text("initial source")
    _plan(tmp_path, source)
    write_json(tmp_path / "prepared.json", {"runs": []})
    source.write_text("changed source")
    with pytest.raises(ValueError, match="source differs"):
        experiment.prepare_experiment(source, tmp_path)


@pytest.mark.parametrize(
    "changed_field,changed_value",
    [
        ("target_id", "WRONG"),
        ("endpoint", "Ki"),
        ("threshold", 7.0),
        ("fingerprint_bits", 1024),
        ("full_source_sha256", "different"),
        ("source_sha256", "different"),
    ],
)
def test_feature_cache_rejects_mismatched_task_provenance(tmp_path, changed_field, changed_value):
    source = tmp_path / "full.tsv"
    source.write_text("fixture")
    plan = _plan(tmp_path, source)
    subset = tmp_path / "sources/TARGET.tsv"
    subset.parent.mkdir()
    subset.write_text("subset")
    manifest = {
        "target_id": "TARGET",
        "endpoint": "IC50",
        "threshold": 6.0,
        "fingerprint_bits": 2048,
        "full_source_sha256": file_hash(source),
        "source_sha256": file_hash(subset),
        changed_field: changed_value,
    }
    write_bundle(
        FeatureSet(pd.DataFrame({"record_id": [1]}), manifest), tmp_path / "features/TARGET_IC50"
    )
    with pytest.raises(ValueError):
        experiment._task_features(tmp_path, plan["tasks"][0], plan)


def test_prepare_rejects_prepared_marker_with_different_plan_hash(tmp_path):
    source = tmp_path / "full.tsv"
    source.write_text("initial source")
    _plan(tmp_path, source)
    write_json(tmp_path / "prepared.json", {"plan_sha256": "different", "runs": []})
    with pytest.raises(ValueError):
        experiment.prepare_experiment(source, tmp_path)


@pytest.mark.parametrize("tamper", ["records", "contract", "runs"])
def test_prepared_reuse_checks_dataset_integrity_contract_and_run_inventory(tmp_path, tamper):
    source = tmp_path / "full.tsv"
    source.write_text("source fixture")
    plan = _plan(tmp_path, source)
    plan = {**plan, "seeds": [11], "split_methods": ["scaffold"]}
    write_json(tmp_path / "plan.json", plan)
    digest = file_hash(tmp_path / "plan.json")
    destination = tmp_path / "dataset"
    manifest = {
        "experiment_plan_sha256": digest,
        "target_id": "TARGET",
        "endpoint": "IC50",
        "split_method": "scaffold",
        "split_seed": 11,
    }
    write_bundle(FeatureSet(pd.DataFrame({"record_id": [1]}), manifest), destination)
    prepared = {
        "plan_sha256": digest,
        "runs": [
            {
                "task_id": "TARGET_IC50",
                "split_method": "scaffold",
                "seed": 11,
                "eligible": True,
                "dataset": str(destination),
            }
        ],
    }
    write_json(tmp_path / "prepared.json", prepared)
    assert experiment.prepare_experiment(source, tmp_path) == prepared
    if tamper == "records":
        (destination / "records.jsonl").write_text("{}\n")
    elif tamper == "contract":
        current = json.loads((destination / "manifest.json").read_text())
        write_json(destination / "manifest.json", {**current, "endpoint": "Ki"})
    else:
        write_json(tmp_path / "prepared.json", {**prepared, "runs": []})
    with pytest.raises(ValueError):
        experiment.prepare_experiment(source, tmp_path)
