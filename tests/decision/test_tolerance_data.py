"""Data-contract checks for the chemical-tolerance protocol."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import tolerance_data as tolerance_module
from tolerance_data import (
    assign_roles,
    collapse_positive_pairs,
    molecule_identity,
    normalize_record,
    seal,
    select_pilot_tasks,
    write_json,
)


def record(**updates):
    return {"target_id": "P12345_WT", "Quality": "High", "SMILES": "CCO",
            "type_IC50": "1", "type_EC50": "0", "type_KD": "0", "type_Ki": "0",
            "type_other": "0", "relation": "=", "pchembl_value_Mean": "6.5",
            "pchembl_value": "6.5", "pchembl_value_N": "1", "AID": "A1",
            "doc_id": "D1", "all_doc_ids": "D1", "Year": "2020.0", **updates}


def test_threshold_and_censored_are_distinct():
    assert normalize_record(record(), 1, {})["primary_positive"]
    assert not normalize_record(record(pchembl_value_Mean="6.499"), 2, {})["primary_positive"]
    censored = normalize_record(record(relation="<", pchembl_value_Mean="7"), 3, {})
    assert not censored["primary_positive"]
    assert censored["censored_positive"]
    assert np.isnan(censored["pActivity"])


def test_unknown_and_conflicting_constituents_not_positive():
    unknown = normalize_record(record(pchembl_value_Mean="", pchembl_value=""), 1, {})
    assert unknown["activity_status"] == "unknown"
    crossing = normalize_record(record(pchembl_value="6.2;7.2", pchembl_value_Mean="6.7",
                                      pchembl_value_N="2"), 2, {})
    assert crossing["activity_status"] == "conflicting_constituents"
    assert not crossing["primary_positive"]
    assert crossing["positive_evidence"]


def test_provenance_lists_are_never_paired_by_position():
    row = normalize_record(record(AID="A1;A2", all_doc_ids="D1;D2"), 1, {})
    assert row["document_id"] == ""
    assert row["assay_id"] == ""
    assert json.loads(row["source_document_ids"]) == ["D1", "D2"]
    assert row["provenance_linkage"] == "unpaired_aggregate_sets"
    assert row["species"] == "unknown"


def test_global_roles_keep_tasks_and_scaffolds_together_and_exposure_out_of_test():
    rows = pd.DataFrame({"identity": ["i1", "i2", "i1", "i3"],
                         "murcko_scaffold": ["s1", "s1", "s1", "s2"],
                         "historical_exposed": [False, True, False, False]})
    result = assign_roles(rows, seed=42)
    assert result.groupby("identity").split.nunique().max() == 1
    assert result.groupby("murcko_scaffold").split.nunique().max() == 1
    assert result.loc[result.murcko_scaffold.eq("s1"), "split"].ne("test").all()
    assert result.equals(assign_roles(rows.sample(frac=1, random_state=2), seed=42)
                         .loc[result.index])


def test_duplicate_pairs_with_conflict_quarantined():
    frame = pd.DataFrame({"identity": ["a", "a", "b"], "task_id": ["t", "t", "t"],
                          "pActivity": [6.8, 6.2, 7.0], "primary_positive": [True, False, True],
                          "activity_status": ["positive", "below_threshold", "positive"],
                          "record_id": [1, 2, 3], "source_document_ids": ['["d"]'] * 3,
                          "source_assay_ids": ['["a"]'] * 3})
    clean, conflicts = collapse_positive_pairs(frame)
    assert clean.identity.tolist() == ["b"]
    assert conflicts.identity.tolist() == ["a", "a"]


def test_pilot_selection_uses_fit_support_not_test_values():
    rows = []
    for task, family in [("t1", "enzyme"), ("t2", "receptor"), ("t3", "channel")]:
        for index in range(205):
            rows.append({"identity": f"{task}:{index}", "task_id": task, "split": "fit",
                         "murcko_scaffold": f"s{index % 30}", "document_id": f"d{index % 8}",
                         "target_family": family})
    frame = pd.DataFrame(rows)
    chosen, support = select_pilot_tasks(frame, limit=3)
    assert set(chosen) == {"t1", "t2", "t3"}
    assert support["t1"]["fit"]["identities"] == 205
    changed = pd.concat([frame, frame.iloc[:2].assign(split="test", pActivity=99)])
    assert select_pilot_tasks(changed, limit=3)[0] == chosen


def test_invalid_roles_and_nonfinite_activity_rejected():
    with pytest.raises(ValueError, match="finite"):
        normalize_record(record(pchembl_value_Mean="inf"), 1, {})
    with pytest.raises(ValueError, match="identity"):
        assign_roles(pd.DataFrame({"identity": [""], "murcko_scaffold": [""]}))


def test_acyclic_identity_has_own_group_and_inconsistent_scaffold_rejected():
    frame = pd.DataFrame({"identity": ["i1", "i2"], "murcko_scaffold": ["", ""]})
    result = assign_roles(frame)
    assert result.split_group.nunique() == 2
    bad = pd.DataFrame({"identity": ["same", "same"], "murcko_scaffold": ["s1", "s2"]})
    # The selected seed assigns these fixtures to different global roles.
    with pytest.raises(ValueError, match="inconsistent"):
        assign_roles(bad, seed=2)
    with pytest.raises(ValueError, match="seed"):
        assign_roles(frame, seed=True)
    with pytest.raises(ValueError, match="scaffold"):
        assign_roles(frame.assign(murcko_scaffold=None))


def test_identity_reuses_salt_and_stereo_policy():
    first = molecule_identity("C[C@H](O)c1ccccc1.Cl")
    second = molecule_identity("C[C@@H](O)c1ccccc1")
    assert first["identity"] == second["identity"]
    assert first["murcko_scaffold"] == "c1ccccc1"
    assert not molecule_identity("invalid smiles")["valid"]


def test_protocol_seal_and_atomic_json_validation(tmp_path):
    path = tmp_path / "nested/protocol.json"
    seal(path, {"threshold": 6.5})
    seal(path, {"threshold": 6.5})
    with pytest.raises(ValueError, match="Frozen"):
        seal(path, {"threshold": 6.0})
    with pytest.raises(ValueError):
        write_json(tmp_path / "invalid.json", {"score": np.inf})
    assert json.loads(path.read_text())["threshold"] == 6.5


def test_complete_preparation_preserves_unknown_reference_and_integrity(tmp_path, monkeypatch):
    study = tmp_path / "study"
    source = study / "data/papyrus/05.7++_combined_set_without_stereochemistry.tsv.xz"
    source.parent.mkdir(parents=True)
    structures = ["CCO", "CCN", "CCC", "c1ccccc1", "C1CCCCC1"]
    records = [record(SMILES=s, Activity_ID=f"a{i}", pchembl_value_Mean=value,
                      pchembl_value=value, doc_id=f"D{i}", all_doc_ids=f"D{i}")
               for i, (s, value) in enumerate(zip(structures, ["7", "6.5", "6.2", "", "7.1"]))]
    pd.DataFrame(records).to_csv(source, sep="\t", index=False, compression="xz")
    monkeypatch.setattr(tolerance_module, "SOURCE_HASH", tolerance_module.file_hash(source))
    proteins = study / "artifacts/s2-decision-tolerance-v1/protocol/sources/05.7_combined_set_protein_targets.tsv.xz"
    proteins.parent.mkdir(parents=True)
    pd.DataFrame([{"target_id": "P12345_WT", "Organism": "Homo sapiens (Human)",
                   "Classification": "Enzyme->Other"}]).to_csv(proteins, sep="\t", index=False,
                                                               compression="xz")
    monkeypatch.setattr(tolerance_module, "PROTEIN_HASH", tolerance_module.file_hash(proteins))
    cache = study / "artifacts/s2-decision-multitask-v1/data"
    cache.mkdir(parents=True)
    pd.DataFrame([molecule_identity(s) for s in structures]).to_parquet(
        cache / "features-before-identity-collapse.parquet", index=False)
    from dataclasses import asdict
    write_json(cache / "chemistry.json", {"rdkit_version": tolerance_module.rdkit_version(),
                                         "standardization": asdict(tolerance_module.STANDARDIZATION)})
    manifest = tolerance_module.prepare(study, workers=1)
    data = study / "artifacts/s2-decision-tolerance-v1/data"
    positive = pd.read_parquet(data / "positive-records.parquet")
    reference = pd.read_parquet(data / "reference-records.parquet")
    assert len(positive) == 3
    assert len(reference) == 2
    assert reference.y_active.isna().all() and reference.pActivity.isna().all()
    assert not set(positive.identity) & set(reference.identity)
    assert manifest["summary"]["test_exposed_groups"] == 0
    assert tolerance_module.prepare(study, workers=1) == manifest
    altered = data / "reference-records.parquet"
    altered.write_bytes(b"altered")
    with pytest.raises(ValueError, match="checksum"):
        tolerance_module.prepare(study, workers=1)


def test_historical_union_includes_all_roles_and_unpaired_provenance(tmp_path):
    study = tmp_path / "study"
    output = study / "artifacts/s2-decision-tolerance-v1"
    prior = study / "artifacts/s2-decision-dynamic-filter-v1/exposure-audit"
    write_json(prior / "exposure-pool.json", {"identities": ["old"], "scaffolds": ["s0", ""],
                                             "b09_document_ids": ["D0"]})
    dataset = study / "artifacts/other/data"
    dataset.mkdir(parents=True)
    pd.DataFrame({"identity": ["fit", "test"], "murcko_scaffold": ["s1", "s2"],
                  "document_id": ["D1", "D2"], "document_chembl_id": ["D3", ""],
                  "source_document_ids": ['["D4", "D5"]', "D6;D7"]}).to_parquet(
        dataset / "records.parquet", index=False)
    pd.DataFrame({"identity": ["external"], "source_document_ids": [["D8", "D9"]]}).to_parquet(
        dataset / "external.parquet", index=False)
    (dataset / "records.jsonl").write_text(json.dumps({"identity": "jsonl", "murcko_scaffold": "s3"}) + "\n")
    identities, scaffolds, documents, audit = tolerance_module.historical_exposure(study, output)
    assert identities == {"old", "fit", "test", "external", "jsonl"}
    assert scaffolds == {"s0", "s1", "s2", "s3"}
    assert documents == {f"D{i}" for i in range(10)}
    assert len(audit["source_files"]) == 4
    assert (output / "exposure/sets.json").exists()


def test_compatible_duplicate_pairs_preserve_median_and_document_sets():
    frame = pd.DataFrame({"identity": ["a", "a"], "task_id": ["t", "t"],
                          "pActivity": [6.8, 7.2], "primary_positive": [True, True],
                          "activity_status": ["positive", "positive"], "record_id": [1, 2],
                          "document_id": ["d1", "d2"], "assay_id": ["a", "a"],
                          "source_document_ids": ['["d1"]', '["d2"]'],
                          "source_assay_ids": ['["a"]', '["a"]']})
    clean, conflicts = collapse_positive_pairs(frame)
    assert clean.pActivity.iloc[0] == 7
    assert clean.document_id.iloc[0] == ""
    assert json.loads(clean.source_document_ids.iloc[0]) == ["d1", "d2"]
    assert json.loads(clean.source_rows.iloc[0]) == [1, 2]
    assert conflicts.empty
