"""Contract checks using real SMILES2Select chemistry and exported run files."""

import json
import sqlite3

import pandas as pd
import pytest
from rdkit import Chem

from s2s_decision.features import featurize
from s2s_decision.smiles2select_io import read_smiles2select
from smiles2select.chemistry.fingerprints import FingerprintConfig, fingerprint
from smiles2select.chemistry.standardization import StandardizationConfig
from smiles2select.export.excel import export as export_excel
from smiles2select.export.excel import export_frame
from smiles2select.export.parquet import export as export_parquet
from smiles2select.io.importer import ColumnMapping, SourceFile
from smiles2select.pipeline.config import RunConfig
from smiles2select.pipeline.runner import run
from smiles2select.pipeline.workers import process_record
from smiles2select.profiles.loader import builtin_registry
from smiles2select.rules.evaluator import evaluate_profiles

DESCRIPTORS = (
    "mol_wt",
    "rdkit_wlogp",
    "mol_mr",
    "tpsa",
    "hbd_lipinski",
    "hba_lipinski",
    "rotatable_bonds",
    "heavy_atom_count",
    "heteroatom_count",
    "ring_count",
    "aromatic_ring_count",
    "formal_charge",
    "fraction_csp3",
    "qed",
    "sa_score",
    "np_score",
)


@pytest.fixture(scope="module")
def exported_run(tmp_path_factory):
    folder = tmp_path_factory.mktemp("real_smiles2select_run")
    source = folder / "entrada_µ.csv"
    pd.DataFrame(
        {
            "ID": ["001-aspirina", "002-cafeína", "003-duplicata", "004-inválida"],
            "SMILES": [
                "CC(=O)Oc1ccccc1C(=O)O",
                "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
                "CC(=O)Oc1ccccc1C(=O)O",
                "not a SMILES",
            ],
        }
    ).to_csv(source, index=False)
    result = run(
        RunConfig(
            sources=(SourceFile(source, ColumnMapping("SMILES", "ID")),),
            profile_ids=("lipinski", "veber"),
            n_jobs=1,
            compute_sa=True,
            compute_np=True,
            final_count=1,
            database_path=folder / "run.sqlite",
        )
    )
    export_excel(result, folder / "run.xlsx")
    export_parquet(result, folder / "run.parquet")
    export_frame(result).to_csv(folder / "run.csv", index=False)
    return folder, result


@pytest.mark.parametrize("bits", [1024, 2048])
def test_features_match_shared_chemistry_and_profile_definitions(bits):
    text = "CC(=O)Oc1ccccc1C(=O)O.[Na+]"
    frame = pd.DataFrame(
        {
            "record_id": [73],
            "molecule_id": ["original-α"],
            "original_smiles": [text],
            "eligible": [True],
        }
    )
    before = frame.copy(deep=True)
    actual = featurize(frame, fingerprint_bits=bits).records.iloc[0]
    direct = process_record(
        73, text, DESCRIPTORS, StandardizationConfig(remove_stereo=True), ("pains", "brenk")
    )
    assert direct.valid
    assert actual["record_id"] == 73
    assert actual["molecule_id"] == "original-α"
    assert actual["original_smiles"] == text
    assert actual["model_smiles"] == direct.standardized_smiles
    assert actual["valid"] and actual["eligible"]
    for name in DESCRIPTORS:
        assert actual[name] == pytest.approx(direct.descriptors[name]), name
    profiles = builtin_registry().select(("lipinski", "veber"))
    evaluation = evaluate_profiles(pd.DataFrame([direct.descriptors], index=[73]), profiles)
    for name in ("lipinski", "veber"):
        assert actual[f"{name}_violations"] == evaluation.violation_count(name).iloc[0]
    for catalog in ("pains", "brenk"):
        hits = {row["alert_name"] for row in direct.alert_rows if row["catalog_id"] == catalog}
        assert actual[f"{catalog}_count"] == len(hits)
    packed = bytes.fromhex(actual["fingerprint_hex"])
    assert len(packed) * 8 == bits
    expected = fingerprint(
        Chem.MolFromSmiles(direct.standardized_smiles), FingerprintConfig(size=bits)
    )
    assert sum(byte.bit_count() for byte in packed) == expected.GetNumOnBits()
    pd.testing.assert_frame_equal(frame, before)


def test_existing_source_features_remain_evidence_and_do_not_override_model_chemistry():
    source = pd.DataFrame(
        {
            "record_id": [9],
            "molecule_id": ["keep-me"],
            "original_smiles": ["CCO"],
            "mol_wt": [999.0],
            "eligible": [False],
            "selected": [False],
            "duplicate_of": [7],
            "source_file": ["usuário.csv"],
        }
    )
    row = featurize(source).records.iloc[0]
    assert row["mol_wt"] == pytest.approx(46.069)
    assert row["source__mol_wt"] == 999.0
    assert row["molecule_id"] == "keep-me"
    assert not row["eligible"]
    assert row["duplicate_of"] == 7
    assert row["source_file"] == "usuário.csv"
    assert "y_active" not in row.index


def test_invalid_smiles_never_becomes_eligible():
    result = featurize(pd.DataFrame({"original_smiles": ["not SMILES", "", None]}))
    assert not result.records["valid"].any()
    assert not result.records["eligible"].any()
    assert result.records["fingerprint_hex"].fillna("").eq("").all()
    assert result.records["invalid_reason"].fillna("").str.len().gt(0).all()


@pytest.mark.parametrize("bits", [0, 512, 2049])
def test_feature_schema_rejects_unsupported_fingerprint_size(bits):
    with pytest.raises(ValueError, match="1024|2048|fingerprint"):
        featurize(pd.DataFrame({"original_smiles": ["CCO"]}), fingerprint_bits=bits)


def test_sqlite_preserves_record_ids_and_does_not_invent_missing_descriptors(exported_run):
    folder, original = exported_run
    loaded = read_smiles2select(folder / "run.sqlite")
    frame = loaded.records.set_index("record_id")
    assert set(frame.index) == set(original.descriptors.index)
    assert frame["molecule_id"].to_dict() == original.descriptors["molecule_id"].to_dict()
    assert frame["source_row"].to_dict() == original.descriptors["source_row"].to_dict()
    assert frame["sa_score"].isna().all()
    assert frame["np_score"].isna().all()
    assert frame["aromatic_ring_count"].isna().all()
    valid_id = original.descriptors.index[original.descriptors["valid"]][0]
    assert frame.loc[valid_id, "hbd_lipinski"] == original.descriptors.loc[valid_id, "hbd_lipinski"]
    assert frame.loc[valid_id, "hba_lipinski"] == original.descriptors.loc[valid_id, "hba_lipinski"]
    assert frame["source__selected"].notna().any()
    assert "y_active" not in frame
    # final_count narrowed this run. Original chemical eligibility was not persisted.
    assert frame["eligible"].sum() == original.decision.decisions["selected"].sum() == 1
    assert loaded.manifest
    populated = featurize(loaded.records).records
    assert populated.loc[populated["valid"], "sa_score"].notna().all()
    assert populated.loc[populated["valid"], "np_score"].notna().all()
    assert populated["eligible"].sum() == 1


@pytest.mark.parametrize("suffix", [".xlsx", ".csv", ".parquet"])
def test_real_tabular_exports_keep_source_identity_and_selection_status(exported_run, suffix):
    folder, original = exported_run
    loaded = read_smiles2select(folder / f"run{suffix}")
    frame = loaded.records.set_index("molecule_id")
    expected = export_frame(original).set_index("ID")
    assert set(frame.index) == set(expected.index)
    assert frame["original_smiles"].to_dict() == expected["Original_SMILES"].to_dict()
    assert frame["source_row"].to_dict() == expected["Source_Row"].to_dict()
    assert frame["source__Final_Status"].to_dict() == expected["Final_Status"].to_dict()
    assert frame["eligible"].sum() == 1
    assert frame.loc["001-aspirina", "mol_wt"] == pytest.approx(expected.loc["001-aspirina", "MW"])
    assert frame.loc["001-aspirina", "source__MW"] == pytest.approx(
        expected.loc["001-aspirina", "MW"]
    )
    assert frame.loc["001-aspirina", "sa_score"] == pytest.approx(
        expected.loc["001-aspirina", "SA"]
    )
    assert "y_active" not in frame


def test_uncomputed_alert_catalog_is_missing_not_zero(tmp_path):
    source = tmp_path / "source.csv"
    pd.DataFrame({"SMILES": ["CCO"]}).to_csv(source, index=False)
    run(
        RunConfig(
            sources=(SourceFile(source, ColumnMapping("SMILES")),),
            profile_ids=("lipinski",),
            alert_catalogs=(),
            n_jobs=1,
            database_path=tmp_path / "run.sqlite",
        )
    )
    loaded = read_smiles2select(tmp_path / "run.sqlite")
    assert loaded.records["pains_count"].isna().all()
    assert loaded.records["brenk_count"].isna().all()
    computed = featurize(loaded.records).records
    assert computed["pains_count"].eq(0).all()
    assert computed["brenk_count"].eq(0).all()
    assert computed["source__pains_count"].isna().all()


def test_reader_rejects_unrelated_database_without_modifying_it(tmp_path):
    path = tmp_path / "unrelated.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE user_notes (body TEXT)")
        connection.execute("INSERT INTO user_notes VALUES ('preserve')")
    before = path.read_bytes()
    with pytest.raises(ValueError):
        read_smiles2select(path)
    assert path.read_bytes() == before


def test_reader_rejects_missing_smiles_column(tmp_path):
    path = tmp_path / "unrelated.csv"
    pd.DataFrame({"name": ["compound"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="SMILES|smiles"):
        read_smiles2select(path)


def test_blank_duplicate_marker_does_not_exclude_plain_csv_row(tmp_path):
    path = tmp_path / "duplicates.csv"
    pd.DataFrame({"SMILES": ["CCO", "CCO"], "duplicate_of": [None, 1]}).to_csv(path, index=False)
    result = read_smiles2select(path)
    assert result.records.eligible.tolist() == [True, False]


def test_csv_explicit_ineligible_flag_cannot_be_widened(tmp_path):
    path = tmp_path / "pool.csv"
    pd.DataFrame({"original_smiles": ["CCO", "CCN"], "eligible": [False, True]}).to_csv(
        path, index=False
    )
    result = read_smiles2select(path)
    assert result.records["eligible"].tolist() == [False, True]
    assert featurize(result.records).records["eligible"].tolist() == [False, True]


def test_direct_features_do_not_rehabilitate_upstream_invalid_record():
    frame = pd.DataFrame({"original_smiles": ["CCO"], "valid": [False]})
    actual = featurize(frame).records.iloc[0]
    assert not actual["eligible"]
    assert not actual["source__valid"]


def test_real_workbook_keeps_run_configuration_and_profile_rules(exported_run):
    folder, original = exported_run
    manifest = read_smiles2select(folder / "run.xlsx").manifest
    run_config = dict(original.config.summary_rows())
    assert manifest["source_config"]["app_version"] == run_config["app_version"]
    assert manifest["source_config"]["rdkit_version"] == run_config["rdkit_version"]
    assert manifest["source_config"]["profiles"] == run_config["profiles"]
    provenance = json.dumps(manifest, ensure_ascii=False)
    for profile in original.profiles:
        assert f"profile:{profile.id}" in provenance
        for rule in profile.rules:
            assert rule.id in provenance
            assert rule.describe() in provenance


@pytest.mark.parametrize("suffix", [".csv", ".xlsx"])
def test_literal_missing_value_tokens_are_valid_source_ids(tmp_path, suffix):
    path = tmp_path / f"ids{suffix}"
    frame = pd.DataFrame({"ID": ["NA", "NULL", "001"], "Original_SMILES": ["CCO", "CCN", "CCC"]})
    if suffix == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_excel(path, index=False)
    actual = read_smiles2select(path).records
    assert actual["molecule_id"].tolist() == ["NA", "NULL", "001"]
    assert actual["source__ID"].tolist() == ["NA", "NULL", "001"]


@pytest.mark.parametrize(
    "extra_column, extra_value, canonical, expected",
    [
        ("SMILES", "CCN", "original_smiles", "CCO"),
        ("access_code", "secondary", "molecule_id", "primary"),
    ],
)
def test_alias_collisions_are_explicit_or_preserve_preferred_source_column(
    tmp_path, extra_column, extra_value, canonical, expected
):
    path = tmp_path / "aliases.csv"
    pd.DataFrame(
        {"ID": ["primary"], "Original_SMILES": ["CCO"], extra_column: [extra_value]}
    ).to_csv(path, index=False)
    try:
        result = read_smiles2select(path)
    except ValueError as exc:
        # Rejecting ambiguous aliases is allowed, but the message must identify columns.
        assert extra_column in str(exc)
        assert "alias" in str(exc).lower() or "column" in str(exc).lower()
    else:
        assert result.records.columns.is_unique
        row = result.records.iloc[0]
        assert row[canonical] == expected
        assert row[f"source__{extra_column}"] == extra_value
