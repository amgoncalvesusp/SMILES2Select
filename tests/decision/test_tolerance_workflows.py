import json

import pandas as pd
import pytest

from s2s_decision.cli import parser
from s2s_decision.tolerance_profile import read_tolerance_package
from s2s_decision.tolerance_workflows import (
    curate_custom,
    predict_tolerance_file,
    prepare_tolerance,
    train_tolerance,
)


def test_custom_concentrations_reverse_censoring_and_keep_unknown():
    frame = pd.DataFrame({"smiles": ["CCO", "CCN", "CCC", "CCCl"],
        "value": [100, 100, 400, None], "unit": ["nM"] * 4,
        "relation": ["=", "<", "=", "="], "endpoint": ["IC50"] * 4,
        "target_id": ["T1"] * 4, "species": ["human"] * 4})
    records, audit = curate_custom(frame)
    assert records.y_active.tolist()[:3] == [1.0, 1.0, 0.0]
    assert pd.isna(records.y_active.iloc[3])
    assert pd.isna(records.pactivity.iloc[1])
    assert audit["exact_positive_rows"] == 1
    assert audit["unknown_activity_rows"] == 1
    assert records.species.tolist() == ["human"] * 4


def test_custom_boundary_is_inclusive_and_defaults_are_explicit():
    records, audit = curate_custom(pd.DataFrame({"SMILES": ["CCO"], "pActivity": [6.5]}),
        target_id="T1", endpoint="Ki")
    assert records.pactivity.iloc[0] == 6.5
    assert records.y_active.iloc[0] == 1
    assert records.species.iloc[0] == "unknown"
    assert audit["defaults"]["species"] == "unknown"


def test_custom_rejects_bad_units_and_conflicting_potency_values():
    with pytest.raises(ValueError, match="unit"):
        curate_custom(pd.DataFrame({"smiles": ["CCO"], "value": [10], "unit": ["mg"]}))
    with pytest.raises(ValueError, match="conflict"):
        curate_custom(pd.DataFrame({"smiles": ["CCO"], "pactivity": [7.0],
                                   "value": [1000], "unit": ["nM"]}))


def test_prepare_does_not_split_same_identity_between_tasks(tmp_path):
    source = tmp_path / "source.csv"
    pd.DataFrame({"smiles": ["CCO", "CCO", "c1ccccc1"], "pactivity": [7, 7, 7],
        "target_id": ["T1", "T2", "T1"], "endpoint": ["Ki"] * 3}).to_csv(source, index=False)
    report = prepare_tolerance(source, tmp_path / "prepared")
    assert report["rows"] == 3
    data = pd.read_json(tmp_path / "prepared" / "records.jsonl", orient="records", lines=True)
    assert data.groupby("identity")["split"].nunique().max() == 1


def test_cli_exposes_separate_tolerance_commands():
    args = parser().parse_args(["tolerance-train", "--input", "source.csv", "--output", "profile",
                               "--mode", "hierarchical", "--recipe", "B", "--target", "T1",
                               "--species", "human", "--endpoint", "Ki"])
    assert args.mode == "hierarchical"
    assert args.recipe == "B"


def _prepared_fixture(tmp_path, name="source"):
    path = tmp_path / f"{name}.csv"
    pd.DataFrame({"smiles": ["CCO", "CCN", "CCCC", "CCCl", "CCCBr", "CCCO"],
                  "pactivity": [7.] * 6, "endpoint": ["Ki"] * 6,
                  "target_id": ["T1"] * 6, "species": ["human"] * 6}).to_csv(path, index=False)
    root = tmp_path / name
    prepare_tolerance(path, root)
    return root


def test_training_target_without_optional_endpoint_uses_input_context(tmp_path):
    from s2s_decision.tolerance_profile import read_tolerance_package

    source = _prepared_fixture(tmp_path)
    report = train_tolerance(source, tmp_path / "profile", target_id="T1")
    profile = read_tolerance_package(tmp_path / "profile")
    assert report["training_count"] > 0
    assert profile["context"] == {"target_id": "T1"}
    assert profile["provenance"]["qualification"] == "experimental"


@pytest.mark.parametrize("endpoint", ["unknown", "Ki;IC50", ""])
def test_unknown_endpoint_never_becomes_positive_training(tmp_path, endpoint):
    source = tmp_path / "unknown.csv"
    pd.DataFrame({"smiles": ["CCO", "CCN"], "pactivity": [7, 8],
                  "endpoint": [endpoint] * 2}).to_csv(source, index=False)
    with pytest.raises(ValueError, match="No exact positive training"):
        train_tolerance(source, tmp_path / "profile")
    assert not (tmp_path / "profile").exists()


def test_external_calibration_cannot_relabel_original_fit_groups(tmp_path):
    source = _prepared_fixture(tmp_path)
    data = pd.read_json(source / "records.jsonl", orient="records", lines=True)
    overlap = data.loc[data.split.ne("policy_cal")].iloc[0].original_smiles
    positive = tmp_path / "positive.csv"
    background = tmp_path / "background.csv"
    pd.DataFrame({"smiles": [overlap], "pactivity": [7], "endpoint": ["Ki"]}).to_csv(positive, index=False)
    pd.DataFrame({"smiles": ["CCF"], "endpoint": ["Ki"]}).to_csv(background, index=False)
    with pytest.raises(ValueError, match="overlap"):
        train_tolerance(source, tmp_path / "profile", positive_cal=positive, background_cal=background)
    assert not (tmp_path / "profile").exists()


@pytest.mark.parametrize("unit,value", [
    ("M", 1e-7), ("mM", 1e-4), ("uM", .1), ("μM", .1),
    ("µM", .1), ("nM", 100), ("pM", 100000),
])
def test_custom_supported_units_have_identical_experimental_potency(unit, value):
    source = pd.DataFrame({"smiles": ["CCO"], "value": [value],
                           "unit": [unit], "endpoint": ["Ki"]})
    before = source.copy(deep=True)

    records, audit = curate_custom(source)

    assert records.pactivity.iloc[0] == pytest.approx(7.)
    assert records.y_active.iloc[0] == 1
    assert records.activity_status.iloc[0] == "exact"
    assert audit["exact_positive_rows"] == 1
    pd.testing.assert_frame_equal(source, before)


@pytest.mark.parametrize("value", [0, -1, "inf", "-inf", "nan"])
def test_custom_rejects_nonpositive_or_nonfinite_concentrations(value):
    source = pd.DataFrame({"smiles": ["CCO"], "value": [value], "unit": ["nM"]})
    with pytest.raises(ValueError, match="finite and positive"):
        curate_custom(source)


@pytest.mark.parametrize("alias,canonical", [
    ("SMILES", "smiles"), ("pActivity", "pactivity"), ("target", "target_id"),
    ("units", "unit"), ("activity_value", "value"),
])
def test_custom_rejects_ambiguous_column_aliases(alias, canonical):
    source = pd.DataFrame({"smiles": ["CCO"], alias: ["A"], canonical: ["B"]})
    with pytest.raises(ValueError, match="Ambiguous custom column"):
        curate_custom(source)


@pytest.mark.parametrize("source,message", [
    (pd.DataFrame({"pactivity": [7]}), "SMILES column"),
    (pd.DataFrame(columns=["smiles"]), "no rows"),
])
def test_custom_rejects_missing_structures_or_empty_data(source, message):
    with pytest.raises(ValueError, match=message):
        curate_custom(source)


def _write_custom(path, smiles, **values):
    frame = pd.DataFrame({"smiles": smiles, "endpoint": ["Ki"] * len(smiles), **values})
    frame.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def custom_pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("custom_pipeline")
    prepared = _prepared_fixture(root)
    positive = _write_custom(root / "positive.csv", ["CCF", "CCCN", "CCCF", "bad-smiles"],
                             pactivity=[7., 8., None, 7.], relation=["=", ">", "=", "="])
    background = _write_custom(root / "background.csv", ["CCBr", "COC", "CNC", "bad-smiles"],
                               pactivity=[None, 5., 7., None])
    model = root / "calibrated_profile"
    report = train_tolerance(prepared, model, positive_cal=positive, background_cal=background)
    return {"root": root, "prepared": prepared, "model": model, "report": report,
            "positive": positive, "background": background}


def test_custom_external_calibration_uses_exact_positives_and_valid_unknown_background(custom_pipeline):
    profile = read_tolerance_package(custom_pipeline["model"])
    assert profile["policy"]["positive_count"] == 1
    assert profile["policy"]["background_count"] == 2
    assert profile["policy"]["status"] == "baseline_sufficient"
    assert profile["policy"]["rescue_enabled"] is False
    assert custom_pipeline["report"]["training_count"] == 5
    sources = profile["provenance"]["custom_calibration_sources"]
    assert len(sources["positive_source_sha256"]) == 64
    assert len(sources["background_source_sha256"]) == 64
    assert sources["positive_source_sha256"] != sources["background_source_sha256"]


@pytest.mark.parametrize("column", ["SMILES", "smiles", "model_smiles"])
def test_custom_predict_csv_aliases_write_decisions_and_invalid_structure_evidence(
        tmp_path, custom_pipeline, column):
    source = tmp_path / "candidates.csv"
    pd.DataFrame({column: ["CCO", "not-a-molecule"]}).to_csv(source, index=False)
    destination = tmp_path / "prediction"

    report = predict_tolerance_file(source, destination, custom_pipeline["model"])

    records = pd.read_json(destination / "records.jsonl", orient="records", lines=True)
    csv = pd.read_csv(destination / "decisions.csv")
    saved = json.loads((destination / "report.json").read_text(encoding="utf-8"))
    assert report["rows"] == saved["rows"] == len(records) == len(csv) == 2
    assert report["state_counts"] == records.tolerance_state.value_counts().to_dict()
    assert records.tolerance_state.tolist() == ["base_eligible", "excluded"]
    assert records.tolerance_eligible.tolist() == [True, False]
    assert "invalid_structure" in records.tolerance_evidence.iloc[1]["reasons"]
    profile = read_tolerance_package(custom_pipeline["model"])
    assert records.tolerance_profile_sha256.tolist() == [profile["profile_sha256"]] * 2


def test_custom_predict_prepared_input_preserves_record_identity(tmp_path, custom_pipeline):
    source = pd.read_json(custom_pipeline["prepared"] / "records.jsonl", orient="records", lines=True)
    destination = tmp_path / "prediction"

    report = predict_tolerance_file(custom_pipeline["prepared"], destination, custom_pipeline["model"])

    result = pd.read_json(destination / "records.jsonl", orient="records", lines=True)
    assert report["rows"] == len(source)
    assert result.record_id.tolist() == source.record_id.tolist()
    assert result.identity.tolist() == source.identity.tolist()


def test_custom_prediction_preserves_original_structure_context_and_source_metadata(tmp_path, custom_pipeline):
    source = tmp_path / "candidates.csv"
    pd.DataFrame({
        "SMILES": ["CCO", "not-a-molecule"], "record_id": [501, 902],
        "molecule_id": ["candidate-α", "candidate-β"], "target_id": ["T2", "T3"],
        "species": ["human", "mouse"], "endpoint": ["Ki", "IC50"],
        "document_id": ["document-1", "document-2"], "assay_id": ["assay-1", "assay-2"],
        "note": ["original á😀", "=HYPERLINK(\"example\")"],
    }).to_csv(source, index=False)
    destination = tmp_path / "prediction"

    predict_tolerance_file(source, destination, custom_pipeline["model"])

    result = pd.read_json(destination / "records.jsonl", orient="records", lines=True)
    assert result.record_id.tolist() == [501, 902]
    assert result.original_smiles.tolist() == ["CCO", "not-a-molecule"]
    assert result.molecule_id.tolist() == ["candidate-α", "candidate-β"]
    assert result.target_id.tolist() == ["T2", "T3"]
    assert result.species.tolist() == ["human", "mouse"]
    assert result.endpoint.tolist() == ["Ki", "IC50"]
    assert result.document_id.tolist() == ["document-1", "document-2"]
    assert result.assay_id.tolist() == ["assay-1", "assay-2"]
    assert result.note.tolist() == ["original á😀", "=HYPERLINK(\"example\")"]
    assert result.valid.tolist() == [True, False]
    assert result.tolerance_state.tolist() == ["base_eligible", "excluded"]
    spreadsheet = pd.read_csv(destination / "decisions.csv")
    assert spreadsheet.molecule_id.tolist() == ["candidate-α", "candidate-β"]
    assert spreadsheet.note.tolist() == ["original á😀", "'=HYPERLINK(\"example\")"]


def test_custom_predict_rejects_ambiguous_csv_structure_columns(tmp_path, custom_pipeline):
    source = tmp_path / "ambiguous.csv"
    pd.DataFrame({"SMILES": ["CCO"], "smiles": ["CCN"]}).to_csv(source, index=False)
    with pytest.raises(ValueError, match="ambiguous SMILES"):
        predict_tolerance_file(source, tmp_path / "prediction", custom_pipeline["model"])
    assert not (tmp_path / "prediction").exists()


def test_prepare_conflicting_activity_does_not_invent_training_positives(tmp_path):
    source = _write_custom(tmp_path / "measurements.tsv", ["CCO", "CCO", "CCN", "invalid"],
                           pactivity=[7., 5., 7., 7.], document_id=["á😀;DROP TABLE", "b", "c", "d"])
    pd.read_csv(source).to_csv(source, sep="\t", index=False)
    destination = tmp_path / "prepared"

    report = prepare_tolerance(source, destination)

    records = pd.read_json(destination / "records.jsonl", orient="records", lines=True)
    conflicting = records.loc[records.original_smiles.eq("CCO")]
    assert conflicting.y_active.isna().all()
    assert conflicting.pactivity.isna().all()
    assert conflicting.curation_status.eq("conflicting_activity").all()
    assert records.loc[records.original_smiles.eq("CCN"), "y_active"].tolist() == [1.]
    assert records.document_id.iloc[0] == "á😀;DROP TABLE"
    assert report["validation_report"]["conflicting_rows"] == 2
    assert report["validation_report"]["invalid_rows"] == 1


@pytest.mark.parametrize("tamper,message", [
    ("records", "checksum mismatch"), ("schema", "Not a prepared tolerance dataset"),
])
def test_prepared_data_integrity_failure_prevents_profile_publication(tmp_path, tamper, message):
    prepared = _prepared_fixture(tmp_path)
    if tamper == "records":
        path = prepared / "records.jsonl"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    else:
        path = prepared / "manifest.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps({**metadata, "schema": "unrelated/1"}), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        train_tolerance(prepared, tmp_path / "model")
    assert not (tmp_path / "model").exists()


@pytest.mark.parametrize("cohort", ["positive_cal", "background_cal"])
def test_custom_requires_paired_external_calibration_files(tmp_path, cohort):
    with pytest.raises(ValueError, match="both positive and background"):
        train_tolerance(tmp_path / "unused.csv", tmp_path / "profile", **{cohort: tmp_path / "cal.csv"})
    assert not (tmp_path / "profile").exists()


def test_custom_without_positive_calibration_is_descriptive_and_cannot_rescue(tmp_path):
    prepared = _prepared_fixture(tmp_path)
    destination = tmp_path / "model"
    report = train_tolerance(prepared, destination)
    profile = read_tolerance_package(destination)

    assert report["calibration_status"]["status"] == "uncalibrated"
    assert profile["policy"]["tau"] is None
    assert profile["policy"]["rescue_enabled"] is False
    assert profile["custom_validation"]["status"] == "insufficient_calibration"
    assert profile["custom_validation"]["positive_cal_count"] == 0
    assert profile["provenance"]["custom_calibration_sources"]["source"] == (
        "reserved custom scaffold policy_cal partition")


def test_custom_positive_only_reserved_calibration_cannot_enable_rescue(tmp_path):
    source = _write_custom(tmp_path / "source.csv", ["CCO", "CCF"], pactivity=[7., 7.])
    destination = tmp_path / "model"

    report = train_tolerance(source, destination)
    profile = read_tolerance_package(destination)

    assert report["training_count"] == 1
    assert profile["policy"]["positive_count"] == 1
    assert profile["policy"]["background_count"] == 0
    assert profile["policy"]["reason"] == "independent_background_missing"
    assert profile["policy"]["rescue_enabled"] is False


def test_training_context_uses_only_matching_existing_target_fit_and_calibration_rows(tmp_path):
    source = _write_custom(tmp_path / "targets.csv", ["CCO", "CCO", "CCF", "CCF"],
                           pactivity=[7.] * 4, target_id=["T1", "T2", "T1", "T2"],
                           species=["human"] * 4)
    prepared = tmp_path / "prepared"
    prepare_tolerance(source, prepared)
    destination = tmp_path / "target_profile"

    report = train_tolerance(prepared, destination, target_id="T1")

    profile = read_tolerance_package(destination)
    assert report["training_count"] == profile["training_counts"]["rows"] == 1
    assert profile["policy"]["positive_count"] == 1
    contexts = [context for reference in profile["references"] for context in reference["contexts"]]
    assert {context["target_id"] for context in contexts} == {"T1"}
    assert all(context["species"] == "human" and context["endpoint"] == "Ki" for context in contexts)


def test_custom_calibration_records_requested_criterion_scope(tmp_path, custom_pipeline):
    criterion = "rule:lipinski:lip_mw_max"
    destination = tmp_path / "selected_criterion"

    report = train_tolerance(custom_pipeline["prepared"], destination,
                             positive_cal=custom_pipeline["positive"],
                             background_cal=custom_pipeline["background"],
                             reconsidered_criteria=[criterion])

    profile = read_tolerance_package(destination)
    assert report["calibration_status"]["scope"]["criteria"] == [criterion]
    assert profile["policy"]["scope"]["criteria"] == [criterion]


@pytest.mark.parametrize("criteria", [["rule:lipinski:lip_mw_max"], []])
def test_cli_custom_calibration_can_lock_selected_or_empty_criterion_scope(criteria):
    args = parser().parse_args([
        "tolerance-train", "--input", "source.csv", "--output", "profile",
        "--reconsider-criteria", *criteria,
    ])
    assert args.reconsider_criteria == criteria


@pytest.mark.parametrize("cohort", ["positive", "background"])
def test_empty_external_calibration_reports_no_rows_without_publishing(tmp_path, custom_pipeline, cohort):
    empty = _write_custom(tmp_path / "empty.csv", [])
    positive = empty if cohort == "positive" else custom_pipeline["positive"]
    background = empty if cohort == "background" else custom_pipeline["background"]
    with pytest.raises(ValueError, match="Custom data contains no rows"):
        train_tolerance(custom_pipeline["prepared"], tmp_path / "model",
                         positive_cal=positive, background_cal=background)
    assert not (tmp_path / "model").exists()


@pytest.mark.parametrize("operation", ["prepare", "train", "predict"])
def test_custom_workflow_existing_output_is_preserved(tmp_path, custom_pipeline, operation):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "user-data.txt"
    marker.write_bytes(b"preserve this exact content\x00")
    with pytest.raises(ValueError, match="already exists"):
        if operation == "prepare":
            prepare_tolerance(custom_pipeline["prepared"], output)
        elif operation == "train":
            train_tolerance(custom_pipeline["prepared"], output)
        else:
            predict_tolerance_file(custom_pipeline["prepared"], output, custom_pipeline["model"])
    assert marker.read_bytes() == b"preserve this exact content\x00"
    assert list(output.iterdir()) == [marker]


@pytest.mark.parametrize("operation", ["prepare", "predict"])
def test_custom_workflow_failed_write_does_not_publish_partial_output(
        tmp_path, custom_pipeline, monkeypatch, operation):
    from s2s_decision import tolerance_workflows

    def failed_write(*args, **kwargs):
        raise OSError("storage unavailable")

    monkeypatch.setattr(tolerance_workflows, "write_json", failed_write)
    output = tmp_path / "result"
    with pytest.raises(OSError, match="storage unavailable"):
        if operation == "prepare":
            prepare_tolerance(custom_pipeline["prepared"], output)
        else:
            predict_tolerance_file(custom_pipeline["prepared"], output, custom_pipeline["model"])
    assert not output.exists()
    assert not list(tmp_path.glob(".s2s-tolerance-*"))


@pytest.mark.parametrize("role", ["fit", "tune", "test"])
@pytest.mark.parametrize("cohort", ["positive", "background"])
def test_external_calibration_rejects_all_original_protected_identity_groups(tmp_path, role, cohort):
    source = _write_custom(tmp_path / "source.csv", ["CCO", "CCCO", "C1CCCCC1", "C1CCCC1"],
                           pactivity=[7.] * 4)
    prepared = tmp_path / "prepared"
    prepare_tolerance(source, prepared)
    records = pd.read_json(prepared / "records.jsonl", orient="records", lines=True)
    overlap = records.loc[records.split.eq(role)].iloc[0].original_smiles
    positive = _write_custom(tmp_path / "positive.csv", [overlap if cohort == "positive" else "CCF"],
                             pactivity=[7.])
    background = _write_custom(tmp_path / "background.csv", [overlap if cohort == "background" else "CCBr"])

    with pytest.raises(ValueError, match="identity/scaffold overlap"):
        train_tolerance(prepared, tmp_path / "model", positive_cal=positive, background_cal=background)
    assert not (tmp_path / "model").exists()


def test_external_calibration_rejects_new_identity_on_protected_nonempty_scaffold(tmp_path):
    source = _write_custom(tmp_path / "source.csv", ["CCO", "C1CCCC1"], pactivity=[7., 7.])
    prepared = tmp_path / "prepared"
    prepare_tolerance(source, prepared)
    positive = _write_custom(tmp_path / "positive.csv", ["CC1CCCC1"], pactivity=[7.])
    background = _write_custom(tmp_path / "background.csv", ["CCBr"])

    with pytest.raises(ValueError, match="identity/scaffold overlap"):
        train_tolerance(prepared, tmp_path / "model", positive_cal=positive, background_cal=background)
    assert not (tmp_path / "model").exists()


def test_custom_training_can_adapt_shared_profile_without_changing_base_artifact(tmp_path, custom_pipeline):
    parent_path = custom_pipeline["model"]
    parent = read_tolerance_package(parent_path)
    source = _write_custom(tmp_path / "new-context.csv", ["CCCCN", "CCCNC"], pactivity=[7., 7.],
                           target_id=["new-target"] * 2, species=["human"] * 2)
    destination = tmp_path / "adapted"

    report = train_tolerance(source, destination, base_profile=parent_path, mode="hierarchical",
                             target_id="new-target", species="human", endpoint="Ki")

    adapted = read_tolerance_package(destination)
    assert report["training_mode"] == "adapt_shared"
    assert adapted["representation"] == parent["representation"]
    assert adapted["provenance"]["adaptation"]["base_profile_sha256"] == parent["profile_sha256"]
    assert {row["identity"] for row in parent["references"]}.issubset(
        row["identity"] for row in adapted["references"])
    assert adapted["context"] == {"target_id": "new-target", "species": "human", "endpoint": "Ki"}
    assert adapted["policy"]["rescue_enabled"] is False
    assert read_tolerance_package(parent_path) == parent
