"""Direct ChEMBL preparation contract tests; no network or model fitting."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def runner():
    spec = importlib.util.spec_from_file_location(
        "prepare_target_panel",
        Path(__file__).parents[2] / "benchmarks/prepare_target_panel.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TARGETS = {
    "ACHE_IC50": {
        "target_chembl_id": "CHEMBL220",
        "accession": "P22303",
        "endpoint": "IC50",
    }
}
ALLOW = {
    "CHEMBL1": {
        "task_id": "ACHE_IC50",
        "target_chembl_id": "CHEMBL220",
        "organism": "Homo sapiens",
        "target_type": "SINGLE PROTEIN",
        "confidence_score": 9,
        "relationship_type": "D",
        "mode": "biochemical_enzyme_inhibition",
        "decision": "include",
        "rationale": "Audited enzyme inhibition of human WT protein.",
        "document_chembl_id": "CHEMBL3",
        "variant": "no_annotated_or_explicit_variant",
    }
}


def record(**changes):
    return dict(
        {
            "activity_id": 1,
            "assay_chembl_id": "CHEMBL1",
            "molecule_chembl_id": "CHEMBL2",
            "target_chembl_id": "CHEMBL220",
            "canonical_smiles": "CCO",
            "document_chembl_id": "CHEMBL3",
            "document_year": 2020,
            "target_organism": "Homo sapiens",
            "pchembl_value": "7",
            "standard_type": "IC50",
            "standard_relation": "=",
            "standard_value": "100",
            "standard_units": "nM",
            "standard_flag": 1,
            "data_validity_comment": None,
            "potential_duplicate": 0,
        },
        **changes,
    )


def test_direct_join_preserves_evidence_and_duplicates():
    module = runner()
    source = record()
    accepted, excluded = module.curate_records([source, dict(source)], ALLOW, TARGETS)
    assert len(accepted) == 1 and len(excluded) == 1
    assert excluded.reason.tolist() == ["repeated_activity_id"]
    assert accepted.iloc[0].y_active == 1
    assert json.loads(accepted.iloc[0].raw_record_json) == source
    assert accepted.iloc[0].measurement_id == "1"
    with pytest.raises(ValueError, match="Conflicting activity_id"):
        module.curate_records([source, record(pchembl_value="5")], ALLOW, TARGETS)


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"assay_chembl_id": "OTHER"}, "assay_not_allowed"),
        ({"target_chembl_id": "OTHER"}, "target_mismatch"),
        ({"standard_type": "Ki"}, "endpoint_mismatch"),
        ({"standard_relation": "<"}, "nonexact_relation"),
        ({"standard_units": "uM"}, "nonstandard_units"),
        ({"standard_flag": 0}, "nonstandard_record"),
        ({"potential_duplicate": 1}, "potential_duplicate"),
        ({"data_validity_comment": "Outside typical range"}, "validity_comment"),
        ({"pchembl_value": None}, "invalid_pactivity"),
        ({"pchembl_value": "NaN"}, "invalid_pactivity"),
        ({"standard_value": "0"}, "invalid_standard_value"),
        ({"standard_value": "1000"}, "inconsistent_pactivity"),
        ({"document_year": None}, "unknown_year"),
        ({"canonical_smiles": ""}, "missing_provenance"),
    ],
)
def test_ineligible_is_unknown_not_negative(changes, reason):
    accepted, excluded = runner().curate_records([record(**changes)], ALLOW, TARGETS)
    assert accepted.empty and excluded.reason.tolist() == [reason]


def test_assay_allowlist_fails_closed():
    module = runner()
    for key, value in [
        ("variant", "mutant"),
        ("confidence_score", 8),
        ("mode", "cellular"),
        ("organism", "Mus musculus"),
        ("relationship_type", "H"),
        ("rationale", ""),
    ]:
        bad = {"CHEMBL1": dict(ALLOW["CHEMBL1"], **{key: value})}
        with pytest.raises(ValueError, match="allowlist"):
            module.curate_records([record()], bad, TARGETS)


def test_snapshot_complete_pages_and_hashes(tmp_path):
    module = runner()
    pages = []
    for i in range(2):
        path = tmp_path / f"{i}.json"
        path.write_text(
            json.dumps(
                {
                    "activities": [record(activity_id=i + 1)],
                    "page_meta": {
                        "offset": i,
                        "limit": 1,
                        "total_count": 2,
                        "next": f"https://example.test/{i+1}" if i == 0 else None,
                    },
                }
            )
        )
        pages.append(
            {
                "path": path.name,
                "sha256": module.file_hash(path),
                "url": f"https://example.test/{i}",
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"collection": "activities", "total_count": 2, "pages": pages})
    )
    assert len(module.load_snapshot(manifest)[0]) == 2
    pages[0]["sha256"] = "bad"
    manifest.write_text(
        json.dumps({"collection": "activities", "total_count": 2, "pages": pages})
    )
    with pytest.raises(ValueError, match="checksum"):
        module.load_snapshot(manifest)


def test_temporal_reservation_conflicts_and_missing_labels():
    module = runner()
    frame = pd.DataFrame(
        {
            "identity": ["a", "a", "b", "b", "c", "d", "d"],
            "task_id": ["x", "x", "x", "x", "y", "x", "x"],
            "year": [2020, 2024, 2020, 2020, 2025, 2024, 2024],
            "y_active": [1, 1, 0, 1, 1, 0, 1],
            "source_row": range(7),
            "measurement_id": list("1234567"),
        }
    )
    dev, external, excluded, conflicts = module.reserve_observations(frame)
    assert dev.identity.tolist() == ["a"]
    assert external.identity.tolist() == ["c"]
    assert excluded.identity.tolist() == ["a"]
    assert set(conflicts.identity) == {"b", "d"}
    labels = module.make_labels(["a", "c"], ["x", "y"], pd.concat([dev, external]))
    np.testing.assert_equal(labels, [[1, np.nan], [np.nan, 1]])


def test_threshold_uses_unrounded_measurement_and_variants_rejected():
    module = runner()
    accepted, excluded = module.curate_records(
        [
            record(activity_id=1, standard_value="1000.1", pchembl_value="6"),
            record(activity_id=2, standard_value="1000", pchembl_value="6"),
            record(activity_id=3, assay_variant_mutation="Y537S"),
        ],
        ALLOW,
        TARGETS,
    )
    assert accepted.y_active.tolist() == [0, 1]
    assert excluded.reason.tolist() == ["variant_annotation"]


def test_historical_exposure_and_future_conflict_do_not_change_training():
    frame = pd.DataFrame(
        {
            "identity": ["a", "a", "b"],
            "task_id": ["x"] * 3,
            "year": [2020, 2024, 2024],
            "y_active": [1, 0, 1],
            "source_row": [1, 2, 3],
            "measurement_id": ["1", "2", "3"],
        }
    )
    dev, external, overlap, conflicts = runner().reserve_observations(frame, {"b"})
    assert dev.y_active.tolist() == [1]
    assert external.empty and conflicts.empty
    assert set(overlap.identity) == {"a", "b"}


def test_prepare_dataset_writes_train_only_arrays(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from s2s_decision.schema import PROPERTY_NAMES

    module = runner()
    n = 84
    features = pd.DataFrame(
        {
            "identity": [f"m{i:03d}" for i in range(n)],
            "original_smiles": [f"source{i}" for i in range(n)],
            "murcko_scaffold": [f"scaffold{i//2}" for i in range(n)],
            "eligible": [True] * n,
            **{name: np.arange(n, dtype=float) for name in PROPERTY_NAMES},
        }
    )
    features.loc[81, "murcko_scaffold"] = "scaffold0"
    accepted = pd.DataFrame(
        {
            "original_smiles": features.original_smiles,
            "task_id": ["ACHE_IC50"] * n,
            "target_id": ["P22303_WT"] * n,
            "endpoint": ["IC50"] * n,
            "year": [2020] * 80 + [2024] * 4,
            "y_active": np.arange(n) % 2,
            "source_row": np.arange(n),
            "measurement_id": [str(i) for i in range(n)],
        }
    )
    monkeypatch.setattr(
        module, "featurize", lambda frame, bits: SimpleNamespace(records=features)
    )
    monkeypatch.setattr(
        module,
        "fingerprint_matrix",
        lambda frame, bits: np.zeros((len(frame), bits), dtype=np.float32),
    )
    monkeypatch.setattr(module, "chemistry_manifest", lambda bits: {"bits": bits})
    output = tmp_path / "data"
    report = module.prepare_dataset(
        accepted,
        output,
        expected_task_ids=["ACHE_IC50"],
        historical_identities={"m080"},
        train_min=1,
        holdout_min=1,
    )
    assert report["observed_labels"] == 80
    arrays = np.load(output / "arrays.npz")
    assert arrays["x"].shape == (80, 2088)
    ext_arrays = np.load(output / "external-arrays.npz")
    assert ext_arrays["identities"].tolist() == ["m081", "m082", "m083"]
    assert ext_arrays["scaffold_novel"].tolist() == [False, True, True]
    assert ext_arrays["x"].shape == (3, 2088)
    ext_molecules = pd.read_parquet(output / "external-molecules.parquet")
    assert ext_molecules.year.tolist() == [2024, 2024, 2024]
    molecules = pd.read_parquet(output / "molecules.parquet")
    assert molecules.groupby("murcko_scaffold").split.nunique().max() == 1
    preprocess = json.loads((output / "preprocess.json").read_text())
    assert (
        preprocess["median"][0]
        == molecules.loc[molecules.split.eq("train"), PROPERTY_NAMES[0]].median()
    )
    completion = json.loads((output / "completion.json").read_text())
    assert all(
        module.file_hash(output / name) == digest
        for name, digest in completion["files"].items()
    )
    with pytest.raises(ValueError, match="already exist"):
        module.prepare_dataset(accepted, output, expected_task_ids=["ACHE_IC50"])
    with pytest.raises(ValueError, match="support gates"):
        module.prepare_dataset(
            accepted,
            tmp_path / "rejected",
            expected_task_ids=["ACHE_IC50"],
            train_min=10000,
        )
    assert (tmp_path / "rejected" / "features.parquet").exists()
    assert (tmp_path / "rejected" / "failure.json").exists()
    assert not (tmp_path / "rejected" / "completion.json").exists()
    exposure = json.loads((output / "development-exposure.json").read_text())
    assert len(exposure["identities"]) == 80
    with pytest.raises(ValueError, match="No accepted"):
        module.prepare_dataset(
            pd.DataFrame(), tmp_path / "empty", expected_task_ids=["ACHE_IC50"]
        )


def test_exact_task_catalog_and_cross_source_provenance(tmp_path):
    module = runner()
    accepted, excluded = module.curate_records(
        [
            record(activity_id=1, document_chembl_id="OTHER"),
            record(activity_id=2, target_organism="Mus musculus"),
        ],
        ALLOW,
        TARGETS,
    )
    assert accepted.empty
    assert excluded.reason.tolist() == ["document_mismatch", "organism_mismatch"]
    accepted, _ = module.curate_records([record()], ALLOW, TARGETS)
    with pytest.raises(ValueError, match="prespecified"):
        module.prepare_dataset(
            accepted, tmp_path / "absent", expected_task_ids=["ACHE_IC50", "missing"]
        )
    assert not (tmp_path / "absent").exists()


def test_actual_chembl_activity_schema_has_target_not_assay_organism():
    # Relevant fields transcribed from ChEMBL37 activity 33969 source snapshot.
    activity = record(activity_id=33969, assay_chembl_id="CHEMBL643384",
                      canonical_smiles="CCOc1nn(-c2cccc(OCc3ccccc3)c2)c(=O)o1",
                      molecule_chembl_id="CHEMBL133897", document_chembl_id="CHEMBL1148382",
                      document_year=2004, standard_value="750.0", pchembl_value="6.12",
                      assay_variant_accession=None, assay_variant_mutation=None,
                      bao_format="BAO_0000357", target_tax_id="9606")
    assert "assay_organism" not in activity
    audited = {"CHEMBL643384": dict(ALLOW["CHEMBL1"], document_chembl_id="CHEMBL1148382")}
    accepted, rejected = runner().curate_records([activity], audited, TARGETS)
    assert len(accepted) == 1 and rejected.empty
    assert accepted.iloc[0].y_active == 1
