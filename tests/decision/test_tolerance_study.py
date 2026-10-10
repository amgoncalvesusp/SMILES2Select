"""Chemical-tolerance evaluator contracts; biological labels never decide eligibility."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from benchmarks.tolerance_source_models import (
    common_card_matrix,
    molecular_cards,
    resume_qex_model,
    source_arrays,
    train_numeric,
)
from benchmarks.tolerance_study import (
    _load_data,
    _resource_annotate,
    _resource_records,
    _simple_choice,
    capacity_eligibility,
    compare_capacity,
    evaluate_arms,
    filter_metrics,
    final_metrics,
    freeze_threshold,
    paired_bootstrap,
    validate_partitions,
)


def panel():
    return pd.DataFrame({
        "record_id": range(6), "identity": list("abcdef"),
        "murcko_scaffold": ["s1", "s1", "s2", "s3", "s4", "s5"],
        "baseline_eligible": [True, False, False, False, False, False],
        "valid": [True] * 6, "y_active": [1, 1, np.nan, 1, np.nan, np.nan],
        "rank_score": [0., .2, .1, .3, .4, .5],
        "supported": [True, True, True, False, True, True],
    })


def test_capacity_uses_unlabeled_ranks_and_keeps_base():
    frame = panel()
    eligible, metadata = capacity_eligibility(frame, .4)
    assert eligible.tolist() == [True, True, True, False, False, False]
    assert metadata["rescue_budget"] == 2
    changed = frame.assign(y_active=1 - frame.y_active.fillna(0))
    assert np.array_equal(eligible, capacity_eligibility(changed, .4)[0])


def test_capacity_preserves_abstention_and_shows_deficit():
    frame = panel().assign(supported=[True, False, True, False, False, False])
    eligible, metadata = capacity_eligibility(frame, .8)
    assert eligible.sum() == 2
    assert metadata["rescue_deficit"] == 3
    assert metadata["rescue_fraction_achieved"] == .2


def test_equal_capacity_does_not_claim_gain_from_more_slots():
    frame = panel()
    limited = frame.assign(supported=[True, False, True, False, False, False])
    result = compare_capacity(frame, limited, .8)
    assert result["common_rescue_count"] == 1
    assert result["left"]["rescue_count"] == result["right"]["rescue_count"] == 1
    assert result["active_rescue_retention_difference"] == 0
    assert compare_capacity(frame, limited.assign(supported=False), .8)["status"] == "not_estimable"


def test_tune_comparator_choice_excludes_zero_capacity_when_an_available_simple_arm_exists():
    frame = pd.DataFrame({"record_id": range(21), "identity": [str(i) for i in range(21)],
        "murcko_scaffold": [str(i) for i in range(21)], "baseline_eligible": [True]+[False]*20,
        "supported": True, "rank_score": np.arange(21, dtype=float),
        "y_active": [1]+[np.nan]*19+[1]})
    assert _simple_choice({"simple_1": frame, "S0": frame.assign(supported=False)}) == "simple_1"


def test_metrics_publish_unknown_count_without_false_positive_claim():
    frame = panel()
    result = filter_metrics(frame, [True, True, True, False, False, False])
    assert result["active_count"] == 3
    assert result["active_R_count"] == 2
    assert result["active_rescue_count"] == 1
    assert result["active_rescue_retention"] == .5
    assert result["unknown_rescue_count"] == 1
    assert not ({"precision", "false_positives", "fpr", "specificity"} & result.keys())


def test_threshold_includes_unsupported_positive_cases_and_reports_infeasible():
    positives = panel().loc[lambda x: x.y_active.eq(1)]
    background = panel().loc[lambda x: x.y_active.isna()]
    frozen = freeze_threshold(positives, background, target_retention=1., max_expansion=1.)
    assert frozen["status"] == "infeasible"
    assert frozen["maximum_supported_retention"] == pytest.approx(2 / 3)


def test_threshold_uses_calibration_only_with_exact_boundaries():
    frame = panel().assign(supported=True)
    frozen = freeze_threshold(frame.loc[frame.y_active.eq(1)],
                              frame.loc[frame.y_active.isna()],
                              target_retention=2 / 3, max_expansion=1.)
    assert frozen["status"] == "feasible"
    assert frozen["tau"] == .2


def test_partition_validation_checks_identity_across_tasks_and_background():
    frame = pd.DataFrame({"identity": ["a", "a"], "murcko_scaffold": ["s", "s"],
                          "split": ["fit", "test"], "task_id": ["t1", "t2"]})
    with pytest.raises(ValueError, match="identity"):
        validate_partitions(frame)
    with pytest.raises(ValueError, match="scaffold"):
        validate_partitions(frame.assign(identity=["a", "b"]))
    validate_partitions(frame.assign(split="fit"))
    validate_partitions(frame.assign(identity=["a", "b"], murcko_scaffold=""))


@pytest.mark.parametrize("fraction", [-1, 1.1, np.nan, True])
def test_capacity_validation(fraction):
    with pytest.raises(ValueError):
        capacity_eligibility(panel(), fraction)


def test_source_training_requires_source_labels_and_disjoint_chemical_identities():
    with pytest.raises(ValueError, match="origin"):
        source_arrays(pd.DataFrame({"identity": ["a"], "split": ["fit"], "y_active": [1]}))
    frame = pd.DataFrame({"identity": ["a", "a"], "split": ["fit", "test"],
                          "sample_origin": ["observed_active", "reference_background"]})
    with pytest.raises(ValueError, match="identity"):
        source_arrays(frame)


def test_cards_expose_identical_finite_numeric_information_without_source_or_smiles():
    x = np.array([[1.12345678, 0., np.nan]])
    with pytest.raises(ValueError, match="finite"):
        molecular_cards(x)
    cards, rounded = molecular_cards(np.nan_to_num(x))
    assert cards == ['{"features":{"0":1.123457},"implicit_zero":true}']
    assert rounded[0, 0] == 1.123457


def test_compact_cards_losslessly_encode_the_common_six_decimal_matrix():
    cards, rounded = molecular_cards(np.array([[1., 0., 1.12345678, -2.]]), compact=True)
    card = json.loads(cards[0])
    restored = np.zeros(4)
    restored[card["one"]] = 1.
    for index, value in card["value"].items():
        restored[int(index)] = float(value)
    assert np.array_equal(restored, rounded[0])
    assert card["zero_default"] is True
    assert "1.123457" in cards[0]


def test_common_card_whitelist_retains_descriptors_bools_alerts_and_their_missing_masks():
    names = ["mol_wt", "missing__mol_wt", "tolerance_rule__veber__x__violation",
             "missing__tolerance_rule__veber__x__violation", "tolerance_alert__alert:v2:pains:0:x",
             "missing__tolerance_alert__alert:v2:pains:0:x", "tolerance_rule__veber__x__magnitude",
             "pActivity", "source_document_ids"]
    selected, spec = common_card_matrix(np.ones((2, len(names))), {"feature_names": names})
    assert spec["feature_names"] == names[:6]
    assert selected.shape == (2, 6)


def test_numeric_source_models_fit_both_origins_and_return_log_density_not_bioactivity():
    pytest.importorskip("sklearn")
    x = np.array([[0., 0.], [.1, 0.], [2., 1.], [3., 1.], [0., 0.], [3., 1.]])
    y = np.array([0, 0, 1, 1, 0, 1])
    split = np.array(["fit"] * 4 + ["tune"] * 2)
    models = train_numeric(x, y, split, families=("logistic", "boosting"))
    assert set(models) == {"logistic", "boosting"}
    for result in models.values():
        assert result["scores"].shape == (6,)
        assert np.isfinite(result["scores"]).all()
        assert result["event"] == "active_sample_origin_vs_reference_origin"
        assert result["sampling_prior"] == .5


def test_new_source_tiny_uses_one_bce_head_and_exports_actual_onnx(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    x = np.array([[0., 0.], [.1, 0.], [2., 1.], [3., 1.], [0., 0.], [3., 1.]])
    y = np.array([0, 0, 1, 1, 0, 1])
    roles = np.array(["fit"] * 4 + ["tune"] * 2)
    result = train_numeric(x, y, roles, families=("tiny",), output=tmp_path)["tiny"]
    assert result["model"]["architecture"][-1] == 1
    assert result["model"]["loss"] == "BCEWithLogitsLoss"
    assert all(row["onnx_validated"] for row in result["grid"])
    assert (tmp_path/"tiny-epochs10.onnx").is_file()


def test_paired_bootstrap_preserves_shared_groups_and_is_explicitly_exploratory():
    rows = pd.DataFrame({"task": ["a", "a", "b", "b"], "group": ["s1", "s2", "s1", "s3"],
                         "left_rescued": [True, False, True, True], "right_rescued": [False] * 4})
    result = paired_bootstrap(rows, repeats=50)
    assert result["status"] == "exploratory"
    assert result["global_scaffold_groups"] == 3
    assert result["macro_difference"] == .75
    assert paired_bootstrap(rows.loc[rows.task.eq("a")])["status"] == "not_estimable"


def test_actual_annotated_molecules_use_canonical_qed_in_native_balanced_selector():
    from s2s_decision.selection import select_candidates
    from s2s_decision.tolerance_features import annotate_tolerance

    frame, _ = annotate_tolerance(pd.DataFrame({"record_id": [0, 1], "identity": ["a", "b"],
        "model_smiles": ["CC(=O)Oc1ccccc1C(=O)O", "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"],
        "y_active": [1, np.nan]}))
    result = final_metrics(frame, np.array([True, True]), 5, 3)
    assert result["selected_n"] == 2
    assert result["shortfall"] == 3
    assert result["active_selected"] == 1
    full = select_candidates(frame.assign(eligible=True, valid=True, priority_score=frame.qed),
                             5, max_per_scaffold=3)
    assert result["selected_ids"] == full.manifest["final_ids"]


def test_raw_annotated_universe_can_be_evaluated_using_each_explicit_arm_baseline():
    from s2s_decision.tolerance_features import annotate_tolerance

    frame, _ = annotate_tolerance(pd.DataFrame({"record_id": [0, 1], "identity": ["a", "b"],
        "model_smiles": ["CC(=O)Oc1ccccc1C(=O)O", "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"],
        "y_active": [1, np.nan]}))
    ranked = frame.assign(baseline_eligible=[True, False], supported=True, rank_score=[0., 1.])
    rows, final = evaluate_arms(frame, {"arm": ranked},
                              {"arm": {"status": "feasible", "tau": 1.}}, budgets=(2,))
    assert rows[0]["B_count"] == 1
    assert final[0]["selected_n"] == 2


def test_data_loader_rejects_mutated_consumed_file_before_reading(tmp_path):
    data, protocol = tmp_path/"data", tmp_path/"protocol"
    data.mkdir()
    protocol.mkdir()
    names = ["data/positive-records.parquet", "data/reference-records.parquet",
             "data/tasks.json", "protocol/protocol.json"]
    for name in names:
        (tmp_path/name).write_text("frozen", encoding="utf-8")
    hashes = {name: hashlib.sha256((tmp_path/name).read_bytes()).hexdigest() for name in names}
    (data/"manifest.json").write_text(json.dumps({"schema": "s2s-tolerance-data/1", "files": hashes}))
    (data/"tasks.json").write_text("mutated")
    with pytest.raises(ValueError, match="checksum"):
        _load_data(data)


def test_resource_workload_requires_actual_distinct_identity_and_smiles(tmp_path):
    path = tmp_path/"workload.parquet"
    pd.DataFrame({"record_id": [0, 1], "identity": ["a", "b"],
                  "model_smiles": ["CC", "CC"]}).to_parquet(path)
    with pytest.raises(ValueError, match="distinct"):
        _resource_records(path)
    pd.DataFrame({"record_id": [0, 1], "identity": ["a", "b"],
                  "model_smiles": ["CC", "CCC"]}).to_parquet(path)
    assert len(_resource_records(path)) == 2


def test_real_resource_annotation_can_be_saved_without_changing_sparse_alert_values(tmp_path):
    from s2s_decision.tolerance_features import annotate_tolerance, representation_matrix

    records = pd.DataFrame({"record_id": [0, 1], "identity": ["a", "b"],
        "model_smiles": ["CC(=O)Oc1ccccc1C(=O)O", "Oc1ccccc1O"]})
    original, schema = annotate_tolerance(records)
    expected, _ = representation_matrix(original, schema)
    path = tmp_path/"annotations.parquet"
    result = _resource_annotate((records, path))
    actual, _ = representation_matrix(pd.read_parquet(path), schema)
    assert result["rows"] == 2
    assert np.array_equal(expected, actual)


def test_qex_resume_rejects_different_training_chemistry(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from rdkit import rdBase

    from s2s_decision.qex import QEXModel

    values = np.arange(64., dtype=float).reshape(8, 8)
    model = SimpleNamespace(training_count=8,
        training_sha256=hashlib.sha256(values.astype('<f8').tobytes()).hexdigest(),
        rdkit_version=rdBase.rdkitVersion)
    monkeypatch.setattr(QEXModel, 'from_dict', lambda value: model)
    path = tmp_path/'qex.json'
    path.write_text(json.dumps({'status': 'fit', 'model': {}}))
    resumed, receipt = resume_qex_model(path, values)
    assert resumed is model and receipt['status'] == 'fit'
    with pytest.raises(ValueError, match='training'):
        resume_qex_model(path, values + 1.)
