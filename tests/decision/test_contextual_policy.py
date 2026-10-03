"""Behavioral contracts for contextual decisions, independent of model training."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, asdict, replace

import numpy as np
import pandas as pd
import pytest

from s2s_decision.contextual_policy import (
    FeatureAction,
    PolicyContext,
    PolicySettings,
    ProfileAction,
    ScoreEvidence,
    apply_contextual_policy,
    select_information_queue,
)
from s2s_decision.schema import FeatureSet


def context(**changes):
    return replace(PolicyContext("P22303", "Homo sapiens", "IC50", "hit_finding",
                                 "biochemical", "ChEMBL37", "model/1", "chem/1", 1000, "nM"), **changes)


def evidence(**changes):
    return replace(ScoreEvidence(context(), True, "platt+mondrian", 40, 20, 20,
                                 "heldout-calibration", "activity"), **changes)


def candidates():
    return pd.DataFrame({
        "record_id": [1, 2, 3, 4], "identity": ["a", "b", "c", "d"],
        "valid": [True] * 4, "eligible": [True] * 4,
        "murcko_scaffold": ["A", "A", "B", ""], "cluster_id": [1, 1, 2, 3],
        "model_smiles": ["CCO", "CCN", "c1ccccc1", "CC(=O)O"],
        "activity_score": [0.9, 0.8, 0.7, 0.6], "risk_score": [None] * 4,
        "in_domain": [True] * 4, "activity_prediction_set": [(1,)] * 4,
        "alert_ids": [(), ("pains:test",), (), ()],
        "mol_wt": [600., 400., 300., 200.], "rdkit_wlogp": [4., 4., 2., 1.],
        "hbd_lipinski": [1, 1, 1, 1], "hba_lipinski": [2, 2, 2, 2],
        "tpsa": [40., 40., 40., 40.], "rotatable_bonds": [3, 3, 3, 3],
    })


def settings(**changes):
    return replace(PolicySettings(2, activity_evidence=evidence()), **changes)


def test_preserves_inputs_and_separates_unknown_risk_from_activity():
    frame = candidates()
    original = frame.copy(deep=True)
    result = apply_contextual_policy(FeatureSet(frame, {"source": "fixture"}), context(), settings())
    pd.testing.assert_frame_equal(frame, original)
    assert result.manifest["final_ids"] == [1, 2]
    assert result.records.risk_score.isna().all()
    assert result.records.activity_score.tolist() == original.activity_score.tolist()
    assert result.records.review_required.all()  # missing risk evidence stays visible
    assert result.manifest["contextual_policy"]["context"] == asdict(context())
    json.dumps(result.manifest, allow_nan=False)


def test_rule_magnitude_and_default_no_filter():
    result = apply_contextual_policy(candidates(), context(), settings())
    rules = result.records.iloc[0].rule_evidence
    mw = next(item for item in rules if item["rule_id"] == "lip_mw_max")
    assert mw["violated"] is True
    assert mw["normalized_excess"] == pytest.approx(0.2)
    assert result.records.eligible.all()


@pytest.mark.parametrize("action,eligible,priority,review", [
    ("inform", True, .9, True), ("warn", True, .9, True),
    ("penalize", True, .8, True), ("exclude", False, .9, True),
])
def test_four_explicit_rule_actions(action, eligible, priority, review):
    policy = FeatureAction("lip_mw_max", action)
    row = apply_contextual_policy(candidates(), context(), settings(rule_actions=(policy,))).records.iloc[0]
    assert bool(row.eligible) == eligible
    assert row.priority_score == pytest.approx(priority)
    assert bool(row.review_required) == review


def test_explicit_alert_exclusion_is_not_cancelled_by_high_activity():
    frame = candidates().assign(activity_score=[.1, 1., .2, .3])
    result = apply_contextual_policy(frame, context(), settings(
        alert_actions=(FeatureAction("pains:test", "exclude"),)))
    assert not result.records.iloc[1].eligible
    assert 2 not in result.manifest["final_ids"]
    assert "pains:test" in result.records.iloc[1].policy_explanation


@pytest.mark.parametrize("action", ["penalize", "exclude", "inform"])
def test_unsupported_learned_alert_becomes_warning(action):
    result = apply_contextual_policy(candidates(), context(), settings(alert_actions=(
        FeatureAction("pains:test", action, "learned"),)))
    row = result.records.iloc[1]
    assert row.eligible and row.priority_score == .8
    assert row.alert_evidence[0]["effective_action"] == "warn"
    assert row.alert_evidence[0]["supported"] is False


def test_learned_activity_cannot_establish_risk_exclusion():
    action = FeatureAction("pains:test", "exclude", "learned", .1, 30, 15, 15, 6, 4,
                           "activity", "training-ledger")
    result = apply_contextual_policy(candidates(), context(), settings(alert_actions=(action,)))
    assert result.records.iloc[1].alert_evidence[0]["effective_action"] == "warn"


def test_supported_learned_penalty_requires_matching_calibrated_context():
    action = FeatureAction("pains:test", "penalize", "learned", .1, 30, 15, 15, 6, 4,
                           "activity", "training-ledger")
    result = apply_contextual_policy(candidates(), context(), settings(alert_actions=(action,)))
    assert result.records.iloc[1].priority_score == pytest.approx(.7)
    mismatch = apply_contextual_policy(candidates(), context(stage="fragment"), settings(alert_actions=(action,)))
    assert mismatch.records.iloc[1].priority_score == .8
    assert "context_mismatch" in mismatch.records.iloc[1].policy_warnings


def test_profiles_tolerances_and_stages_are_hand_defined_not_probabilities():
    frame = candidates().assign(rdkit_wlogp=[6., 4., 2., 1.])
    strict = apply_contextual_policy(frame, context(), settings(
        profiles=(ProfileAction("lipinski", "exclude", 0),)))
    tolerant = apply_contextual_policy(frame, context(), settings(
        profiles=(ProfileAction("lipinski", "exclude", 2),)))
    assert strict.records.eligible.tolist() == [False, True, True, True]
    assert tolerant.records.eligible.all()
    fragment = apply_contextual_policy(frame, context(stage="fragment"), settings())
    assert {item["profile_id"] for item in fragment.records.iloc[0].rule_evidence} == {"ro3_core"}
    assert fragment.records.activity_score.tolist() == frame.activity_score.tolist()
    lead = apply_contextual_policy(frame, context(stage="lead"), settings())
    assert {item["profile_id"] for item in lead.records.iloc[0].rule_evidence} == {"lead_like"}


def test_hard_eligibility_pins_shortfall_and_quotas_preserved():
    frame = candidates().assign(eligible=[False, True, True, True])
    result = apply_contextual_policy(frame, context(), settings(n=5, max_per_scaffold=1, pinned_ids=(4,)))
    assert result.manifest["final_ids"] == [4, 2, 3]
    assert result.manifest["shortfall"] == 2
    assert not result.records.iloc[0].eligible
    with pytest.raises(ValueError, match="not eligible"):
        apply_contextual_policy(frame, context(), settings(pinned_ids=(1,)))


def test_optional_smarts_require_exclude_and_prefer():
    result = apply_contextual_policy(candidates(), context(), settings(
        required_smarts=("[#6]",), excluded_smarts=("[NX3]",), preferred_smarts=("c1ccccc1",),
        smarts_bonus=.25))
    assert not result.records.iloc[1].eligible
    assert result.records.iloc[2].priority_score == pytest.approx(.95)
    assert result.records.iloc[2].smarts_evidence["preferred"] == ["c1ccccc1"]
    oxygen = apply_contextual_policy(candidates(), context(), settings(required_smarts=("[O]",)))
    assert oxygen.records.eligible.tolist() == [True, False, False, True]


def test_unknown_activity_not_negative_and_information_queue_is_descriptive():
    frame = candidates().assign(activity_score=[None, .8, .7, .6], in_domain=[None, False, True, True],
                                activity_prediction_set=[None, (0, 1), (), (1,)])
    result = apply_contextual_policy(frame, context(), settings())
    assert pd.isna(result.records.iloc[0].activity_score)
    assert result.records.iloc[0].priority_score == 0
    assert "activity_unknown" in result.records.iloc[0].policy_warnings
    assert result.records.iloc[1].information_priority > result.records.iloc[3].information_priority
    assert result.records.information_rank.notna().all()
    assert "descriptive" in result.manifest["contextual_policy"]["information_queue"]


def test_risk_threshold_never_treats_missing_as_safe():
    frame = candidates().assign(risk_score=[.9, None, .1, .1])
    risk = evidence(endpoint="cytotoxicity")
    result = apply_contextual_policy(frame, context(), settings(risk_evidence=risk, risk_exclude_at=.8))
    assert result.records.eligible.tolist() == [False, True, True, True]
    assert "risk_unknown" in result.records.iloc[1].policy_warnings
    assert result.manifest["contextual_policy"]["risk_evidence"]["endpoint"] == "cytotoxicity"


def test_uncalibrated_and_ambiguous_predictions_force_review():
    frame = candidates().assign(risk_score=.1, risk_in_domain=True, risk_prediction_set=[(0,)] * 4)
    clean = apply_contextual_policy(frame, context(), settings(risk_evidence=evidence(endpoint="risk")))
    assert not clean.records.iloc[2].review_required
    result = apply_contextual_policy(frame, context(), settings(activity_evidence=evidence(calibrated=False)))
    assert result.records.review_required.all()
    assert "activity_uncalibrated" in result.records.iloc[2].policy_warnings


@pytest.mark.parametrize("changes,match", [
    ({"stage": "drug"}, "stage"), ({"target": ""}, "target"), ({"species": None}, "species"),
])
def test_invalid_context(changes, match):
    with pytest.raises(ValueError, match=match):
        context(**changes)


@pytest.mark.parametrize("column,value,match", [
    ("activity_score", np.inf, "activity_score"), ("activity_score", 1.1, "activity_score"),
    ("risk_score", -.1, "risk_score"), ("activity_score", "0.5", "activity_score"),
    ("in_domain", "yes", "in_domain"), ("valid", 1, "valid"),
    ("mol_wt", np.inf, "mol_wt"), ("mol_wt", "600", "mol_wt"),
])
def test_invalid_frame_values(column, value, match):
    with pytest.raises(ValueError, match=match):
        apply_contextual_policy(candidates().assign(**{column: value}), context(), settings())


@pytest.mark.parametrize("value", [[2], [True], [1, 1], "1", {1}])
def test_invalid_prediction_sets(value):
    frame = candidates()
    frame["activity_prediction_set"] = [value] * 4
    with pytest.raises(ValueError, match="prediction_set"):
        apply_contextual_policy(frame, context(), settings())


def test_missing_descriptors_review_without_fabricated_violations():
    result = apply_contextual_policy(candidates().drop(columns="mol_wt"), context(), settings())
    evidence_row = next(item for item in result.records.iloc[0].rule_evidence if item["rule_id"] == "lip_mw_max")
    assert evidence_row["violated"] is None
    assert evidence_row["normalized_excess"] is None
    assert "descriptor_missing:mol_wt" in result.records.iloc[0].policy_warnings


def test_prediction_context_tampering_rejected():
    frame = candidates()
    frame.attrs["score_context"] = asdict(context(target="different"))
    with pytest.raises(ValueError, match="score_context"):
        apply_contextual_policy(frame, context(), settings())


@pytest.mark.parametrize("changes,match", [
    ({"n": 0}, "n"), ({"n": True}, "n"), ({"risk_weight": -1}, "risk_weight"),
    ({"default_alert_action": "erase"}, "action"), ({"required_smarts": ("[invalid",)}, "SMARTS"),
    ({"profiles": (ProfileAction("../secret"),)}, "profile"),
    ({"rule_actions": (FeatureAction("no_rule"),)}, "rule"),
])
def test_bad_settings(changes, match):
    with pytest.raises(ValueError, match=match):
        apply_contextual_policy(candidates(), context(), settings(**changes))


def test_dataclasses_are_frozen_and_nested_mutable_settings_rejected():
    with pytest.raises(FrozenInstanceError):
        context().stage = "fragment"
    with pytest.raises(ValueError, match="tuple"):
        settings(alert_actions=[])


def test_empty_shape_rejected_but_zero_rows_supported():
    with pytest.raises(ValueError, match="DataFrame"):
        apply_contextual_policy(None, context(), settings())
    with pytest.raises(ValueError, match="Missing"):
        apply_contextual_policy(pd.DataFrame(), context(), settings())
    result = apply_contextual_policy(candidates().iloc[:0], context(), settings())
    assert result.manifest["shortfall"] == 2


def test_duplicate_columns_ids_and_alias_actions_rejected():
    with pytest.raises(ValueError, match="unique"):
        apply_contextual_policy(pd.concat([candidates(), candidates()[["valid"]]], axis=1), context(), settings())
    with pytest.raises(ValueError, match="unique"):
        apply_contextual_policy(candidates().assign(record_id=1), context(), settings())
    with pytest.raises(ValueError, match="duplicate"):
        apply_contextual_policy(candidates(), context(), settings(alert_actions=(FeatureAction("x"), FeatureAction("x"))))


def test_unicode_evidence_provenance_and_parallel_replay():
    frame = candidates()
    frame["alert_ids"] = [("custom:α'🧪",)] * 4
    config = settings(alert_actions=(FeatureAction("custom:α'🧪", "inform"),))
    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: apply_contextual_policy(frame, context(), config), range(3)))
    assert [item.manifest["final_ids"] for item in outcomes] == [[1, 2]] * 3
    assert "α" in json.dumps(outcomes[0].manifest, ensure_ascii=False)


def test_large_input_10000_rows_uses_same_selector():
    frame = pd.concat([candidates().iloc[[2]]] * 10_000, ignore_index=True).assign(
        record_id=np.arange(10_000), identity=[f"id{i}" for i in range(10_000)])
    result = apply_contextual_policy(frame, context(), settings(n=123))
    assert len(result.records) == 10_000
    assert result.manifest["final_count"] == 123
    assert result.manifest["final_ids"] == list(range(123))


@pytest.mark.parametrize("changes,match", [
    ({"activity_unit": "", "activity_threshold": 1000}, "activity_unit"),
    ({"activity_unit": "nM", "activity_threshold": None}, "activity_unit"),
    ({"activity_relation": "=="}, "activity_relation"),
    ({"activity_threshold": -1}, "activity_threshold"),
])
def test_context_threshold_validation(changes, match):
    with pytest.raises(ValueError, match=match):
        context(**changes)


@pytest.mark.parametrize("changes,match", [
    ({"context": None}, "PolicyContext"), ({"calibrated": 1}, "boolean"),
    ({"calibration_n": 1}, "class counts"), ({"negative_n": 0}, "both classes"),
    ({"source": ""}, "source"), ({"calibration_method": ""}, "calibration_method"),
    ({"source": []}, "source"),
])
def test_score_evidence_validation(changes, match):
    with pytest.raises(ValueError, match=match):
        evidence(**changes)


@pytest.mark.parametrize("changes,match", [
    ({"origin": "hidden"}, "origin"), ({"support_n": 0, "positive_n": 1}, "support"),
    ({"source": None}, "source"), ({"action": []}, "action"),
])
def test_feature_action_validation(changes, match):
    with pytest.raises(ValueError, match=match):
        FeatureAction("x", **changes)


@pytest.mark.parametrize("changes,match", [
    ({"activity_evidence": 42}, "ScoreEvidence"),
    ({"risk_weight": .1}, "risk evidence"), ({"risk_exclude_at": .7}, "risk evidence"),
    ({"activity_evidence": evidence(endpoint="toxicity")}, "activity endpoint"),
    ({"risk_evidence": evidence()}, "risk endpoint"),
])
def test_prediction_axes_cannot_be_swapped_or_actions_without_risk_evidence(changes, match):
    with pytest.raises(ValueError, match=match):
        settings(**changes)


def test_missing_alerts_boolean_shapes_and_no_context_silent_changes():
    unknown = apply_contextual_policy(candidates().drop(columns="alert_ids"), context(), settings())
    assert "alerts_unknown" in unknown.records.iloc[0].policy_warnings
    with pytest.raises(ValueError, match="alert_ids"):
        apply_contextual_policy(candidates().assign(alert_ids="pains:x"), context(), settings())
    with pytest.raises(ValueError, match="invalid structures"):
        apply_contextual_policy(candidates().assign(valid=False), context(), settings())
    with pytest.raises(ValueError, match="PolicyContext"):
        apply_contextual_policy(candidates(), None, settings())
    frame = candidates()
    frame.attrs["risk_context"] = asdict(context(target="x"))
    with pytest.raises(ValueError, match="risk_context"):
        apply_contextual_policy(frame, context(), settings(risk_evidence=evidence(endpoint="toxicity")))


def test_invalid_structure_smarts_kept_and_valid_structure_smarts_failure_rejected():
    frame = candidates().assign(valid=[False, True, True, True], eligible=[False, True, True, True],
                                model_smiles=[None, "CCN", "c1ccccc1", "CCO"])
    result = apply_contextual_policy(frame, context(), settings(required_smarts=("C",)))
    assert result.records.iloc[0].smarts_evidence == {"status": "invalid_structure"}
    with pytest.raises(ValueError, match="model_smiles"):
        apply_contextual_policy(candidates().assign(model_smiles=None), context(), settings(required_smarts=("C",)))


def test_numpy_metadata_serializes_and_manifest_is_detached():
    calibrated = evidence(calibration_n=np.int64(40))
    frame = candidates()
    frame.attrs["version"] = np.int64(2)
    result = apply_contextual_policy(frame, context(), settings(activity_evidence=calibrated))
    json.dumps(result.manifest, allow_nan=False)
    result.manifest["contextual_policy"]["source"]["version"] = 7
    assert frame.attrs["version"] == 2


@pytest.mark.parametrize("column", ["alert_ids", "activity_prediction_set"])
def test_zero_dimensional_arrays_rejected_as_sequences(column):
    frame = candidates()
    frame[column] = [np.array(1)] * 4
    with pytest.raises(ValueError, match=column):
        apply_contextual_policy(frame, context(), settings())


def test_supported_risk_exclusion_separate_endpoint_and_calibration():
    action = FeatureAction("pains:test", "exclude", "learned", .1, 30, 15, 15, 6, 4,
                           "cytotoxicity", "risk-training", assays=3)
    result = apply_contextual_policy(candidates().assign(risk_in_domain=True), context(), settings(
        alert_actions=(action,), risk_evidence=evidence(endpoint="cytotoxicity")))
    assert not result.records.iloc[1].eligible
    assert result.records.iloc[1].alert_evidence[0]["evidence"]["assays"] == 3


@pytest.mark.parametrize("changes", [
    {"target": "different"}, {"species": "mouse"}, {"endpoint": "Ki"},
    {"activity_threshold": 10}, {"activity_unit": "uM"}, {"activity_relation": ">="},
    {"model_version": "other/2"}, {"chemistry_version": "chem/2"}, {"source_version": "ChEMBL38"},
])
def test_incompatible_activity_task_fails_closed(changes):
    with pytest.raises(ValueError, match="Incompatible activity context"):
        apply_contextual_policy(candidates(), context(**changes), settings())


def test_risk_uses_own_biological_context_and_domain():
    risk_context = context(target="HepG2", endpoint="cell_viability", assay_context="ATP signal",
                           source_version="PubChem", model_version="risk/1", activity_threshold=None,
                           activity_unit="")
    risk = evidence(context=risk_context, endpoint="cytotoxicity")
    frame = candidates().assign(risk_score=.1, risk_in_domain=True, risk_prediction_set=[(0,)] * 4)
    result = apply_contextual_policy(frame, context(), settings(risk_evidence=risk))
    assert not result.records.iloc[2].review_required
    unknown = apply_contextual_policy(frame.drop(columns="risk_in_domain"), context(), settings(risk_evidence=risk))
    assert "risk_domain_unknown" in unknown.records.iloc[2].policy_warnings
    assert unknown.records.iloc[2].review_required
    with pytest.raises(ValueError, match="Incompatible risk chemistry"):
        apply_contextual_policy(frame, context(), settings(risk_evidence=replace(risk, context=replace(risk_context, chemistry_version="x"))))


def test_missing_hard_profile_descriptor_fails_closed():
    result = apply_contextual_policy(candidates().drop(columns="mol_wt"), context(), settings(
        profiles=(ProfileAction("lipinski", "exclude", 0),)))
    assert not result.records.eligible.any()
    assert result.manifest["shortfall"] == 2
    assert "lipinski: unknown mandatory profile" in result.records.iloc[0].policy_explanation


def test_native_profile_operator_semantics_and_magnitude_are_distinct():
    result = apply_contextual_policy(candidates().assign(mol_wt=250.), context(stage="lead"), settings())
    lower = next(item for item in result.records.iloc[0].rule_evidence if item["rule_id"] == "ll_mw_min")
    assert lower["violated"] is False and lower["normalized_excess"] == 0


def test_unknown_explicit_exclusion_inputs_fail_closed():
    missing_rule = apply_contextual_policy(candidates().drop(columns="mol_wt"), context(), settings(
        rule_actions=(FeatureAction("lip_mw_max", "exclude"),)))
    assert not missing_rule.records.eligible.any()
    missing_alert = apply_contextual_policy(candidates().drop(columns="alert_ids"), context(), settings(
        alert_actions=(FeatureAction("pains:test", "exclude"),)))
    assert not missing_alert.records.eligible.any()
    learned = apply_contextual_policy(candidates().drop(columns="alert_ids"), context(), settings(
        alert_actions=(FeatureAction("pains:test", "exclude", "learned"),)))
    assert learned.records.eligible.all()


@pytest.mark.parametrize("changes", [{"stage": []}, {"activity_threshold": None, "activity_unit": None}])
def test_context_nested_mutable_invalid_types(changes):
    with pytest.raises(ValueError):
        context(**changes)


def test_non_json_provenance_rejected_without_changing_input():
    frame = candidates()
    frame.attrs["bad"] = object()
    with pytest.raises(ValueError, match="JSON-compatible"):
        apply_contextual_policy(frame, context(), settings())
    assert "bad" in frame.attrs


def test_separate_budgeted_information_queue_never_changes_main_basket_or_truth():
    frame = candidates().assign(in_domain=[False, False, True, True], activity_prediction_set=[(0, 1)] * 4)
    main = apply_contextual_policy(frame, context(), settings(n=1))
    original = main.records.copy(deep=True)
    queue = select_information_queue(main, 3, max_per_scaffold=1)
    assert queue.manifest["requested_count"] == 3
    assert queue.manifest["final_ids"] == [1, 3, 4]
    assert queue.records.in_information_queue.tolist() == [True, False, True, True]
    assert queue.records.main_basket_selected.tolist() == [True, False, False, False]
    assert not {"activity_label", "risk_label", "truth"} & set(queue.records)
    assert queue.records.loc[0, "information_reason"]
    pd.testing.assert_frame_equal(main.records, original)
    json.dumps(queue.manifest, allow_nan=False)


def test_information_queue_optional_rejects_require_explicit_opt_in_hard_exclusions_preserved():
    frame = candidates().assign(in_domain=False, eligible=[True, False, True, True])
    main = apply_contextual_policy(frame, context(), settings(
        excluded_ids=(4,), rule_actions=(FeatureAction("lip_mw_max", "exclude"),)))
    default = select_information_queue(main, 10)
    optin = select_information_queue(main, 10, include_policy_exclusions=True)
    assert default.manifest["final_ids"] == [3]
    assert optin.manifest["final_ids"] == [1, 3]
    assert optin.manifest["shortfall"] == 8
    assert optin.records.loc[0, "information_source"] == "optional_policy_exclusion"
    assert not optin.records.loc[1, "in_information_queue"]
    assert not optin.records.loc[3, "in_information_queue"]


def test_information_queue_validates_annotated_source_and_explicit_optin():
    with pytest.raises(ValueError, match="annotated"):
        select_information_queue(candidates(), 2)
    main = apply_contextual_policy(candidates(), context(), settings())
    with pytest.raises(ValueError, match="boolean"):
        select_information_queue(main, 2, include_policy_exclusions="yes")
    with pytest.raises(ValueError, match="information"):
        select_information_queue(FeatureSet(main.records.drop(columns="information_priority"), main.manifest), 2)


def test_information_queue_zero_information_reports_shortfall_and_does_not_copy_pins():
    frame = candidates().iloc[[2, 3]].assign(in_domain=True, pinned=True,
                                           activity_prediction_set=[(1,), (1,)])
    main = apply_contextual_policy(frame, context(), settings())
    queue = select_information_queue(main, 1)
    assert queue.manifest["final_count"] == 0
    assert queue.manifest["shortfall"] == 1
    assert queue.manifest["pinned_ids"] == []


def test_unknown_prediction_uncertainty_is_eligible_for_information_queue():
    frame = candidates().iloc[[2]].assign(activity_prediction_set=None)
    main = apply_contextual_policy(frame, context(), settings())
    assert main.records.iloc[0].information_priority > 0
    queue = select_information_queue(main, 1)
    assert queue.manifest["final_ids"] == [3]


def test_queue_annotations_reject_nonfinite_and_nonboolean_fields():
    main = apply_contextual_policy(candidates(), context(), settings())
    for column, value in (("information_priority", np.inf), ("policy_hard_eligible", 1)):
        with pytest.raises(ValueError, match=column):
            select_information_queue(FeatureSet(main.records.assign(**{column: value}), main.manifest), 1)


@pytest.mark.parametrize("stage", ["lead", "fragment"])
def test_real_learned_rule_support_remains_auditable_when_stage_profiles_change(stage):
    action = FeatureAction("lip_mw_max", "penalize", "learned", .1, 30, 15, 15, 6, 4,
                           "activity", "training-ledger")
    result = apply_contextual_policy(candidates(), context(stage=stage), settings(rule_actions=(action,)))
    assert result.records.priority_score.tolist() == candidates().activity_score.tolist()
    inactive = result.manifest["contextual_policy"]["inactive_rule_actions"]
    assert inactive == [asdict(action)]
    assert "inactive_rule_action:lip_mw_max" in result.records.iloc[0].policy_warnings
    assert "lip_mw_max" not in {item["rule_id"] for item in result.records.iloc[0].rule_evidence}
    with pytest.raises(ValueError, match="selected profiles"):
        apply_contextual_policy(candidates(), context(stage=stage), settings(
            rule_actions=(FeatureAction("lip_mw_max", "exclude"),)))
    with pytest.raises(ValueError, match="Unknown rule"):
        apply_contextual_policy(candidates(), context(stage=stage), settings(
            rule_actions=(FeatureAction("arbitrary_unknown", "warn", "learned"),)))
