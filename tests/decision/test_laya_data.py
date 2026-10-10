"""Synthetic contracts for matched Laya/numeric evidence, never activity evidence."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import DataStructs

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import laya_data as data

from s2s_decision.contextual_model import feature_matrix
from s2s_decision.schema import PROPERTY_NAMES


def panel():
    rows = []
    for index, split in enumerate(["train"] * 24 + sum(([s] * 4 for s in
            ("validation", "calibration", "test", "recent")), [])):
        fp = DataStructs.ExplicitBitVect(2048)
        fp.SetBit(index)
        fp.SetBit(100 + index % 2)
        rows.append({"identity": f"synthetic-{index}", "record_id": index + 1,
            "task_id": data.DEFAULT_TASK, "split": split, "y_active": index % 2,
            "murcko_scaffold": f"scaffold-{index // 2}", "model_smiles": "CC",
            "fingerprint_hex": DataStructs.BitVectToBinaryText(fp).hex(),
            **{name: index / 7 + position for position, name in enumerate(PROPERTY_NAMES)}})
    frame = pd.DataFrame(rows)
    frame.loc[0, "np_score"] = np.nan
    return frame


def test_fold_provenance_excludes_whole_scaffolds_and_own_identities():
    prepared, manifest, models = data.prepare_panel(panel(), folds=3)
    assert manifest["numeric_columns"] == list(data.NUMERIC_COLUMNS)
    assert len(manifest["numeric_columns"]) == 43
    assert prepared.row_id.is_unique
    assert prepared.set_index("identity").split.equals(panel().set_index("identity").split)
    train_ids = set(prepared.loc[prepared.split.eq("train"), "identity"])
    covered = set()
    for name, model in models.items():
        fitted, queried = set(model["fit_identities"]), set(model["query_identities"])
        assert fitted <= train_ids
        assert not fitted & queried
        if name.startswith("fold_"):
            covered |= queried
            assert not set(model["fit_groups"]) & set(model["query_groups"])
    assert covered == train_ids
    assert set(models["full_train"]["fit_identities"]) == train_ids


def test_heldout_labels_do_not_change_any_features_or_fitted_models():
    source = panel()
    changed = source.assign(y_active=np.where(source.split.eq("train"), source.y_active,
                                               1 - source.y_active))
    first, _, fitted = data.prepare_panel(source, folds=3)
    second, _, repeated = data.prepare_panel(changed, folds=3)
    np.testing.assert_array_equal(first[list(data.NUMERIC_COLUMNS)], second[list(data.NUMERIC_COLUMNS)])
    assert first.state_text.equals(second.state_text)
    assert fitted == repeated


def test_oof_own_group_labels_and_outlier_are_excluded_from_predictor_fit():
    source = panel()
    first, _, models = data.prepare_panel(source, folds=3)
    fold = models["fold_0"]
    queried = source.identity.isin(fold["query_identities"])
    changed = source.assign(y_active=np.where(queried, 1 - source.y_active, source.y_active))
    second, _, updated = data.prepare_panel(changed, folds=3)
    np.testing.assert_array_equal(first.loc[queried, "auxiliary_molecular_logit"],
                                  second.loc[queried, "auxiliary_molecular_logit"])
    assert fold == updated["fold_0"]
    exaggerated = source.assign(mol_wt=np.where(queried, 1e9, source.mol_wt))
    _, _, checked = data.prepare_panel(exaggerated, folds=3)
    assert fold["spec"] == checked["fold_0"]["spec"]


def test_state_numeric_parity_rounding_missingness_and_no_identifiers():
    source = panel()
    before = source.copy(deep=True)
    prepared, manifest, _ = data.prepare_panel(source, folds=3)
    pd.testing.assert_frame_equal(source, before)
    for row in prepared.itertuples():
        values = np.array([getattr(row, name) for name in data.NUMERIC_COLUMNS])
        recovered = data.parse_state(row.state_text, data.DEFAULT_TASK)
        np.testing.assert_array_equal(recovered, values)
        assert row.identity not in row.state_text
        assert "y_active" not in row.state_text and "split" not in row.state_text
        np.testing.assert_array_equal(values, np.round(values, 3))
    missing = prepared.loc[prepared.identity.eq("synthetic-0")].iloc[0]
    assert missing.np_score == 0 and missing.missing_np_score == 1
    assert manifest["missing_values"] == "fixed zero sentinel plus explicit indicator; no imputation"


def test_standalone_auxiliary_logits_reproduce_saved_train_only_estimator():
    source = panel()
    prepared, manifest, models = data.prepare_panel(source, folds=3)
    fitted = models["full_train"]
    held = source.loc[~source.split.eq("train")]
    matrix = feature_matrix(held, fitted["spec"])
    expected = matrix @ np.asarray(fitted["estimator"]["weights"]) + fitted["estimator"]["intercept"]
    np.testing.assert_allclose(prepared.loc[held.index, manifest["raw_auxiliary_logit_column"]], expected)


def test_empty_scaffolds_use_identity_groups_and_recent_overlap_is_declared():
    source = panel()
    source.loc[[0, 1, 24, 36], "murcko_scaffold"] = ""
    source.loc[37, "murcko_scaffold"] = source.loc[2, "murcko_scaffold"]
    _, manifest, models = data.prepare_panel(source, folds=3)
    assert manifest["empty_scaffold_grouping"] == "identity"
    all_groups = set(models["full_train"]["fit_groups"])
    assert "identity:synthetic-0" in all_groups and "identity:synthetic-1" in all_groups
    assert manifest["evaluation_status"] == "historically observed; exploratory reuse, not a fresh holdout"


@pytest.mark.parametrize("change,match", [
    (lambda f: pd.concat([f, f.iloc[[0]]], ignore_index=True), "identity"),
    (lambda f: f.assign(y_active=np.nan), "binary"),
    (lambda f: f.assign(split="oops"), "split"),
    (lambda f: f.assign(fingerprint_hex="00"), "fingerprint"),
    (lambda f: f.assign(identity=""), "identity"),
    (lambda f: f.assign(record_id=1), "record"),
    (lambda f: f.assign(record_id=f.record_id.astype(str)), "record"),
    (lambda f: f.assign(record_id=f.record_id.astype(float)), "record"),
    (lambda f: f.assign(record_id=[True, *f.record_id.tolist()[1:]]), "record"),
    (lambda f: f.assign(mol_wt=np.inf), "finite"),
    (lambda f: f.assign(mol_wt="bad"), "numeric"),
    (lambda f: f.drop(columns="np_score"), "columns"),
    (lambda f: f.assign(murcko_scaffold="same"), "Scaffold"),
])
def test_invalid_panels_fail_before_fitting(change, match):
    with pytest.raises(ValueError, match=match):
        data.prepare_panel(change(panel()), folds=3)


@pytest.mark.parametrize("kwargs", [{"folds": True}, {"folds": 1}, {"folds": 99},
                                    {"seed": -1}, {"task_id": "unknown"}])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        data.prepare_panel(panel(), **kwargs)


def test_group_support_fails_without_reseeding():
    source = panel()
    source.loc[source.split.eq("train"), "y_active"] = 0
    with pytest.raises(ValueError, match="classes"):
        data.prepare_panel(source, folds=3)


def test_state_parser_rejects_extra_evidence_wrong_context_and_nonfinite_values():
    values = np.zeros(43)
    text = data.render_state(values, data.DEFAULT_TASK)
    with pytest.raises(ValueError):
        data.parse_state(text + ";truth=1", data.DEFAULT_TASK)
    with pytest.raises(ValueError):
        data.parse_state(text, "P00918_HUMAN_Ki")
    with pytest.raises(ValueError):
        data.render_state(np.full(43, np.inf), data.DEFAULT_TASK)
    with pytest.raises(ValueError):
        data.render_state(np.zeros(42), data.DEFAULT_TASK)


def test_build_writes_matching_frozen_artifacts_and_refuses_overwrite(tmp_path):
    source = tmp_path / "source.parquet"
    panel().to_parquet(source, index=False)
    output = tmp_path / "prepared"
    manifest = data.build_pilot_data(source, output, folds=3)
    saved = json.loads((output / "manifest.json").read_text())
    assert saved == manifest
    assert saved["input_sha256"] == data.file_hash(source)
    frame = pd.read_parquet(output / "records.parquet")
    states = [json.loads(line) for line in (output / "states.jsonl").read_text().splitlines()]
    assert [row["state_text"] for row in states] == frame.state_text.tolist()
    assert [row["row_id"] for row in states] == frame.row_id.tolist()
    assert all(data.file_hash(output / name) == digest for name, digest in saved["output_sha256"].items())
    with pytest.raises(FileExistsError):
        data.build_pilot_data(source, output, folds=3)


def test_source_metadata_retained_outside_state_and_document_overlap_declared():
    source = panel().assign(molecule_id=lambda f: "source-" + f.identity,
                            valid=True, eligible=True,
                            source_document_ids='["shared-document"]',
                            source_assay_ids='["shared-assay"]', source_rows=1)
    prepared, manifest, _ = data.prepare_panel(source, folds=3)
    assert prepared.molecule_id.tolist() == source.molecule_id.tolist()
    assert prepared.valid.all() and prepared.eligible.all()
    assert prepared.source_document_ids.tolist() == source.source_document_ids.tolist()
    assert not prepared.state_text.str.contains("shared-document").any()
    assert manifest["source_grouping"] == "not enforced; source/document overlap may remain"


def test_build_detects_source_change_before_writing_output(tmp_path, monkeypatch):
    source, output = tmp_path / "source.parquet", tmp_path / "prepared"
    panel().to_parquet(source, index=False)
    original = data.prepare_panel

    def change_after_read(*args, **kwargs):
        result = original(*args, **kwargs)
        source.write_bytes(b"source changed")
        return result

    monkeypatch.setattr(data, "prepare_panel", change_after_read)
    with pytest.raises(ValueError, match="source changed"):
        data.build_pilot_data(source, output, folds=3)
    assert not output.exists()


def test_malformed_state_and_cli_dispatch(tmp_path, monkeypatch, capsys):
    for text in (None, "", "unstructured"):
        with pytest.raises(ValueError, match="Malformed"):
            data.parse_state(text)
    invalid = np.zeros(43)
    invalid[20] = .5
    with pytest.raises(ValueError, match="binary"):
        data.render_state(invalid)
    monkeypatch.setattr(data, "build_pilot_data", lambda *args: {"splits": {"train": {"n": 12}}})
    data.main(["--panel", str(tmp_path / "in"), "--output", str(tmp_path / "out"), "--folds", "3"])
    assert json.loads(capsys.readouterr().out)["splits"]["train"]["n"] == 12


def test_inference_reuses_frozen_predictor_without_query_labels_and_matches_heldout():
    source = panel()
    prepared, manifest, models = data.prepare_panel(source, folds=3)
    query = source.loc[source.split.ne("train")].drop(columns="y_active")
    refs = source.loc[source.split.eq("train")]
    result = data.inference_states(query, refs, models)
    expected = prepared.loc[prepared.split.ne("train")].reset_index(drop=True)
    np.testing.assert_array_equal(result[list(data.NUMERIC_COLUMNS)], expected[list(data.NUMERIC_COLUMNS)])
    assert result.state_text.equals(expected.state_text)
    assert "y_active" not in result
    assert data.validate_prepared(prepared, manifest)["rows"] == len(prepared)
    with pytest.raises(ValueError, match="reference"):
        data.inference_states(query, refs.iloc[1:], models)
    altered = refs.assign(y_active=1-refs.y_active)
    with pytest.raises(ValueError, match="reference"):
        data.inference_states(query, altered, models)


def test_prepared_validation_rejects_numeric_text_metadata_and_schema_mismatch():
    prepared, manifest, _ = data.prepare_panel(panel(), folds=3)
    changed = prepared.assign(mol_wt=prepared.mol_wt + 1)
    with pytest.raises(ValueError, match="mismatch"):
        data.validate_prepared(changed, manifest)
    with pytest.raises(ValueError, match="schema"):
        data.validate_prepared(prepared, {**manifest, "numeric_columns": ["y_active"]})
    with pytest.raises(ValueError, match="rows"):
        data.validate_prepared(pd.concat([prepared, prepared.iloc[[0]]]), manifest)
