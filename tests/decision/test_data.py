import math

import pandas as pd
import pytest

from s2s_decision.data import audit_source, load_measurements


def source(tmp_path, rows):
    path = tmp_path / "measurements.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_custom_preserves_unknown_censor_and_provenance(tmp_path):
    rows = [
        {
            "target_id": "T",
            "endpoint": "IC50",
            "pactivity": p,
            "relation": r,
            "original_smiles": "CCO",
            "molecule_id": str(i),
            "source": "assay",
            "year": "2020",
        }
        for i, (p, r) in enumerate([(7, "="), ("", "="), (7, ">"), (5, ">")])
    ]
    path = source(tmp_path, rows)
    frame = load_measurements(path, "T", "IC50")
    assert frame.y_active.iloc[0] == 1
    assert math.isnan(frame.y_active.iloc[1])
    assert frame.y_active.iloc[2] == 1 and math.isnan(frame.pactivity.iloc[2])
    assert math.isnan(frame.y_active.iloc[3])
    assert frame.raw_record_json.str.contains("assay").all()
    assert len(frame.attrs["source_sha256"]) == 64
    audit = audit_source(path, chunk_size=2)
    assert audit["rows"] == 4 and audit["dated_records"] == 4
    assert audit["unknown_labels"] == 2


def test_papyrus_flags_and_inverse_censor(tmp_path):
    rows = [
        {
            "target_id": "T",
            "type_IC50": "1",
            "type_EC50": e,
            "type_KD": "0",
            "type_Ki": "0",
            "type_other": "0",
            "pchembl_value_Mean": "5",
            "relation": ">",
            "Activity_ID": str(i),
            "Quality": "high",
            "source": "db",
            "connectivity": "ABC",
        }
        for i, e in enumerate(["0", "1"])
    ]
    path = source(tmp_path, rows)
    frame = load_measurements(path, "T", "IC50")
    assert len(frame) == 1 and frame.y_active.iloc[0] == 0
    assert frame.original_smiles.iloc[0] == ""
    assert audit_source(path)["mixed_endpoint_rows"] == 1


def test_invalid_finite_and_bound(tmp_path):
    path = source(tmp_path, [{"target_id": "T", "endpoint": "Ki", "pactivity": "inf"}])
    with pytest.raises(ValueError, match="finite"):
        load_measurements(path, "T", "Ki")
    path = source(tmp_path, [{"target_id": "T", "endpoint": "Ki", "pactivity": 7}] * 2)
    with pytest.raises(ValueError, match="max_rows"):
        load_measurements(path, "T", "Ki", max_rows=1)


def test_censor_threshold_boundaries_and_gzip(tmp_path):
    path = tmp_path / "input.csv.gz"
    pd.DataFrame(
        [
            {"target_id": "T", "endpoint": "Ki", "pactivity": 6, "relation": r}
            for r in ["<", "<=", ">", ">=", "~"]
        ]
    ).to_csv(path, index=False)
    frame = load_measurements(path, "T", "Ki")
    assert frame.y_active.iloc[0] == 0
    assert math.isnan(frame.y_active.iloc[1])
    assert frame.y_active.iloc[2:4].eq(1).all()
    assert math.isnan(frame.y_active.iloc[4])
    assert frame.pactivity.isna().all()


def test_papyrus_quality_and_exact_date_preserved(tmp_path):
    common = {
        "target_id": "T",
        "type_IC50": "1",
        "type_EC50": "0",
        "type_KD": "0",
        "type_Ki": "0",
        "type_other": "0",
        "pchembl_value_Mean": "7",
        "relation": "=",
        "date": "2022-02-03",
        "source": "db",
        "connectivity": "ABC",
        "SMILES": "CCO",
    }
    path = source(tmp_path, [dict(common, Quality="medium"), dict(common, Quality="high")])
    frame = load_measurements(path, "T", "IC50")
    assert len(frame) == 1 and frame.pactivity.iloc[0] == 7
    assert frame.year.iloc[0] == 2022 and frame.original_smiles.iloc[0] == "CCO"


def test_empty_selection_is_explicit(tmp_path):
    path = source(tmp_path, [{"target_id": "T", "endpoint": "IC50", "pactivity": 7}])
    with pytest.raises(ValueError, match="No matching"):
        load_measurements(path, "X", "IC50")


def test_audit_exposes_target_endpoint_support_and_integer_record_id(tmp_path):
    path = source(
        tmp_path,
        [
            {"target_id": "T", "endpoint": "IC50", "pactivity": 7, "year": 2020},
            {"target_id": "T", "endpoint": "IC50", "pactivity": 5, "year": 2021},
            {"target_id": "T", "endpoint": "Ki", "pactivity": "NA"},
        ],
    )
    frame = load_measurements(path, "T", "IC50")
    assert frame.record_id.tolist() == [1, 2]
    assert frame.source_row.tolist() == [1, 2]
    counts = audit_source(path)["target_endpoint_counts"]
    ic50 = next(row for row in counts if row["endpoint"] == "IC50")
    assert ic50["active_records"] == ic50["inactive_records"] == 1
    assert ic50["exact_records"] == ic50["dated_records"] == 2


def test_official_papyrus_empty_other_flag_and_latest_aggregate_year(tmp_path):
    path = source(
        tmp_path,
        [
            {
                "Activity_ID": "AAAAEENPAALFRN_on_P49654_WT",
                "Quality": "High",
                "source": "ChEMBL",
                "SMILES": "CCO",
                "connectivity": "AAAAEENPAALFRN",
                "target_id": "P49654_WT",
                "Year": "2009.0",
                "all_years": "2009;2017;nan",
                "type_IC50": "1",
                "type_EC50": "0",
                "type_KD": "0",
                "type_Ki": "0",
                "type_other": "",
                "relation": "=",
                "pchembl_value_Mean": "7.8",
            }
        ],
    )
    frame = load_measurements(path, "P49654_WT", "IC50")
    assert frame.year.iloc[0] == 2017
    assert frame.Year.iloc[0] == "2009.0"
    assert frame.all_years.iloc[0] == "2009;2017;nan"
    assert frame.y_active.iloc[0] == 1
