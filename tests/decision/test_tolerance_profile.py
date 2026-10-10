"""Frozen numerical fixtures for active-only tolerance and abstention."""

import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from s2s_decision import tolerance_profile as model

SCHEMA = {
    "schema": "s2s-tolerance-annotations/1", "chemistry_hash": "a" * 64,
    "descriptor_columns": ["descriptor"], "numeric_columns": ["magnitude"],
    "binary_columns": ["failed"],
    "criterion_columns": {"rule:lipinski:mw": {
        "violation_column": "failed", "magnitude_column": "magnitude",
        "upper_normalized_column": "magnitude", "lower_normalized_column": "lower",
    }},
}


def records(n=24, *, split="fit"):
    return pd.DataFrame({
        "record_id": list(range(n)), "identity": [f"id-{i}" for i in range(n)],
        "model_smiles": ["c1ccccc1"] * n, "murcko_scaffold": [f"s-{i % 6}" for i in range(n)],
        "document_id": [f"d-{i % 3}" for i in range(n)],
        "assay_id": [f"a-{i % 4}" for i in range(n)], "task_id": ["T"] * n,
        "target_id": ["target"] * n, "species": ["Human"] * n,
        "endpoint": ["IC50"] * n, "pactivity": [7.] * n, "relation": ["="] * n,
        "split": [split] * n, "descriptor": np.arange(n, dtype=float),
        "magnitude": np.linspace(.01, .24, n), "lower": [0.] * n, "failed": [True] * n,
    })


@pytest.fixture(autouse=True)
def numerical_annotations(monkeypatch):
    def annotate(frame, **kwargs):
        result = frame.copy()
        result["tolerance_failed_criteria"] = [
            [name for name, meta in SCHEMA["criterion_columns"].items()
             if pd.notna(row.get(meta["violation_column"])) and bool(row.get(meta["violation_column"]))]
            for _, row in result.iterrows()]
        result["tolerance_unknown_criteria"] = [
            [name for name, meta in SCHEMA["criterion_columns"].items()
             if pd.isna(row.get(meta["violation_column"]))]
            for _, row in result.iterrows()]
        return result, copy.deepcopy(SCHEMA)

    def matrix(frame, schema, recipe="A"):
        return frame[["descriptor"]].to_numpy(dtype=np.float32), {"recipe": recipe}

    monkeypatch.setattr(model, "annotate_tolerance", annotate)
    monkeypatch.setattr(model, "representation_matrix", matrix)


def candidate(magnitude=.10, **changes):
    row = records(1).assign(identity="new-id", magnitude=magnitude, descriptor=10.)
    return row.assign(**changes)


def calibrated(profile, tau=1.):
    from s2s_decision.tolerance_schema import seal_profile
    contexts = [{key: item[key] for key in ("task_id", "target_id", "species", "endpoint") if item.get(key)}
                for row in profile["references"] for item in row["contexts"]]
    return seal_profile({**profile, "policy": {
        "status": "feasible", "tau": tau, "rescue_enabled": True,
        "target_retention": .95, "max_expansion": .1,
        "scope": {"criteria": sorted(model._active_criteria(profile["annotation_schema"])),
                  "context_mode": "row_context" if profile["mode"] in {"hierarchical", "contextual"} else "global",
                  "requested_context": {}, "contexts": contexts if profile["mode"] in {"hierarchical", "contextual"} else [{}]},
    }})


def test_fit_accepts_exact_boundary_and_never_uses_other_splits():
    frame = records().assign(pactivity=6.5)
    withheld = records(3, split="test").assign(identity=["test-a", "test-b", "test-c"], magnitude=99.)
    profile = model.fit_tolerance(pd.concat([frame, withheld], ignore_index=True))
    assert profile["training_counts"]["identities"] == 24
    assert all(row["magnitudes"]["rule:lipinski:mw"] < 1 for row in profile["references"])
    assert profile["activity_threshold"] == {"pactivity": 6.5, "relation": ">=", "fit_relation": "="}
    with pytest.raises(ValueError, match="exact"):
        model.fit_tolerance(frame.assign(relation="<"))
    with pytest.raises(ValueError, match="exact"):
        model.fit_tolerance(frame.assign(pactivity=np.nan))


def test_joint_support_and_unsupported_magnitudes_abstain():
    profile = calibrated(model.fit_tolerance(records()))
    accepted = model.predict_tolerance(profile, candidate())
    assert accepted.state.tolist() == ["supported_relaxation"]
    assert accepted.nonconformity.iloc[0] == pytest.approx(10 / 24)
    assert accepted.tolerance_support.iloc[0]["joint"]["identities"] == 24
    outside = model.predict_tolerance(profile, candidate(.30))
    assert outside.state.iloc[0] == "review"
    assert "magnitude_outside_support" in outside.tolerance_evidence.iloc[0]["reasons"]
    insufficient = calibrated(model.fit_tolerance(records(8)))
    assert model.predict_tolerance(insufficient, candidate()).state.iloc[0] == "review"


def test_self_identity_never_counts_as_support():
    profile = calibrated(model.fit_tolerance(records(20)))
    result = model.predict_tolerance(profile, candidate(identity="id-0"))
    assert result.tolerance_support.iloc[0]["joint"]["identities"] == 19
    assert result.state.iloc[0] == "review"


def test_baseline_eligibility_hard_restrictions_and_unreconsidered_criteria():
    profile = calibrated(model.fit_tolerance(records()))
    baseline = model.predict_tolerance(profile, candidate(10), baseline_eligible=[True])
    assert baseline.state.iloc[0] == "base_eligible"
    assert baseline.eligible.iloc[0]
    hard = model.predict_tolerance(profile, candidate(hard_excluded=True), baseline_eligible=[True])
    assert hard.state.iloc[0] == "excluded"
    excluded = model.predict_tolerance(profile, candidate(), reconsidered_criteria=[], baseline_eligible=[False])
    assert excluded.state.iloc[0] == "excluded"
    unknown = model.predict_tolerance(profile, candidate(failed=np.nan))
    assert unknown.state.iloc[0] == "review"


def test_uncalibrated_scores_are_available_but_not_automatically_rescued():
    profile = model.fit_tolerance(records())
    prediction = model.predict_tolerance(profile, candidate())
    assert prediction.tolerance_supported.iloc[0]
    assert prediction.state.iloc[0] == "review"
    assert "policy_uncalibrated" in prediction.tolerance_evidence.iloc[0]["reasons"]


def test_hierarchy_reports_parent_fallback_and_context_incompatibility():
    frame = pd.concat([
        records(24).assign(task_id="parent", target_id="parent-target"),
        records(3).assign(identity=["small-a", "small-b", "small-c"], task_id="small", target_id="small-target"),
    ], ignore_index=True)
    profile = calibrated(model.fit_tolerance(frame, mode="hierarchical"))
    context = {"task_id": "small", "species": "Human", "endpoint": "IC50"}
    result = model.predict_tolerance(profile, candidate(), context=context)
    assert result.tolerance_fallback.iloc[0] == "general_parent"
    assert result.tolerance_support.iloc[0]["task"]["identities"] == 3
    assert result.tolerance_support.iloc[0]["parent"]["identities"] == 27
    wrong = model.predict_tolerance(profile, candidate(), context={**context, "species": "Mouse"})
    assert wrong.state.iloc[0] == "review"
    assert "incompatible_context" in wrong.tolerance_evidence.iloc[0]["reasons"]


def test_contextual_fallback_is_not_reported_as_local_support():
    profile = calibrated(model.fit_tolerance(records(), mode="contextual", support_settings={"k": 4, "domain_threshold": 100.}))
    result = model.predict_tolerance(profile, candidate(), context={"task_id": "T", "species": "Human", "endpoint": "IC50"})
    support = result.tolerance_support.iloc[0]
    assert support["local"]["identities"] == 4
    assert result.tolerance_fallback.iloc[0] == "task_parent"
    assert result.state.iloc[0] == "supported_relaxation"


def test_calibration_counts_unsupported_positives_and_baseline_minimal_expansion():
    profile = model.fit_tolerance(records())
    cal = pd.concat([candidate(.10), candidate(.30)], ignore_index=True).assign(
        identity=["cal-a", "cal-b"], split="policy_cal", baseline_eligible=False)
    background = candidate(.10, identity="bg-a", split="policy_cal", baseline_eligible=False)
    impossible = model.calibrate_tolerance(profile, cal, background, target_retention=1., max_expansion=1.)
    assert impossible["policy"]["status"] == "infeasible"
    assert impossible["policy"]["positive_count"] == 2
    assert impossible["policy"]["supported_positive_count"] == 1
    minimum = model.calibrate_tolerance(profile, cal.assign(baseline_eligible=True), background)
    assert minimum["policy"]["status"] == "baseline_sufficient"
    assert minimum["policy"]["tau"] is None
    assert not minimum["policy"]["rescue_enabled"]
    with pytest.raises(ValueError, match="policy_cal"):
        model.calibrate_tolerance(profile, cal.assign(split="test"), background)
    with pytest.raises(ValueError, match="overlap"):
        model.calibrate_tolerance(profile, records(1, split="policy_cal"), background)


def test_fit_is_deterministic_and_does_not_mutate_inputs():
    frame = records()
    before = frame.copy(deep=True)
    one = model.fit_tolerance(frame)
    two = model.fit_tolerance(frame.sample(frac=1, random_state=14))
    assert one["profile_sha256"] == two["profile_sha256"]
    pd.testing.assert_frame_equal(frame, before)


def test_unknown_background_remains_descriptive():
    profile = model.fit_tolerance(records())
    cal = candidate(identity="cal", split="policy_cal")
    result = model.calibrate_tolerance(profile, cal, None)
    assert result["policy"]["status"] == "uncalibrated"
    assert result["policy"]["expansion"] is None
    assert not result["policy"]["rescue_enabled"]


def test_successful_calibration_expansion_cap_and_frozen_quantile():
    profile = model.fit_tolerance(records())
    cal = candidate(.10, identity="cal", split="policy_cal", baseline_eligible=False)
    bg = candidate(.20, identity="background", split="policy_cal", baseline_eligible=False)
    result = model.calibrate_tolerance(profile, cal, bg, target_retention=1., max_expansion=0.)
    assert result["policy"]["status"] == "feasible"
    assert result["policy"]["tau"] == pytest.approx(10 / 24)
    assert model.predict_tolerance(result, cal).eligible.iloc[0]
    blocked = model.calibrate_tolerance(profile, cal, bg.assign(magnitude=.02), target_retention=1., max_expansion=0.)
    assert blocked["policy"]["status"] == "infeasible"
    fixed = calibrated(model.fit_tolerance(records(), mode="fixed", support_settings={"quantile": .5}))
    assert model.predict_tolerance(fixed, candidate(.20)).state.iloc[0] == "review"


def test_positive_conflicts_are_quarantined_before_fit_and_metadata_is_consolidated():
    frame = records()
    conflict = frame.iloc[[0]].assign(pactivity=5.)
    profile = model.fit_tolerance(pd.concat([frame, conflict], ignore_index=True))
    assert profile["training_counts"]["identities"] == 23
    duplicate = frame.iloc[[0]].assign(task_id="another", document_id="new-document")
    result = model.fit_tolerance(pd.concat([frame, duplicate], ignore_index=True))
    reference = next(row for row in result["references"] if row["identity"] == "id-0")
    assert set(reference["document_ids"]) == {"d-0", "new-document"}
    assert len(reference["contexts"]) == 2


def test_marginal_evidence_cannot_replace_joint_coexistence(monkeypatch):
    second = "rule:lipinski:logp"
    monkeypatch.setitem(SCHEMA["criterion_columns"], second, {
        "violation_column": "second_failed", "magnitude_column": "second_magnitude",
        "upper_normalized_column": "second_magnitude", "lower_normalized_column": "lower",
    })
    frame = records().assign(failed=[True] * 12 + [False] * 12,
        second_failed=[False] * 12 + [True] * 12, second_magnitude=.2)
    profile = calibrated(model.fit_tolerance(frame, support_settings={"min_identities": 10}))
    output = model.predict_tolerance(profile, candidate(second_failed=True, second_magnitude=.1))
    support = output.tolerance_support.iloc[0]
    assert support["marginal"]["rule:lipinski:mw"]["identities"] == 12
    assert support["marginal"][second]["identities"] == 12
    assert support["joint"]["identities"] == 0
    assert output.state.iloc[0] == "review"


def test_missing_document_strings_do_not_create_support():
    profile = calibrated(model.fit_tolerance(records().assign(document_id="NA")))
    output = model.predict_tolerance(profile, candidate())
    assert output.tolerance_support.iloc[0]["joint"]["documents"] == 0
    assert output.state.iloc[0] == "review"


def test_zero_magnitude_strict_boundary_keeps_violation_side(monkeypatch):
    meta = {**SCHEMA["criterion_columns"]["rule:lipinski:mw"],
            "descriptor": "descriptor", "lower_limit": 0., "upper_limit": 2.}
    monkeypatch.setitem(SCHEMA["criterion_columns"], "rule:lipinski:mw", meta)
    frame = records().assign(descriptor=0., magnitude=0., lower=0.)
    profile = calibrated(model.fit_tolerance(frame))
    assert profile["references"][0]["sides"]["rule:lipinski:mw"] == ["lower"]
    unsupported_side = model.predict_tolerance(profile, candidate(0., descriptor=2.))
    assert unsupported_side.tolerance_support.iloc[0]["joint"]["identities"] == 0
    assert unsupported_side.state.iloc[0] == "review"


def test_actual_chemical_recipe_roundtrip_and_invalid_structure(monkeypatch, tmp_path):
    from s2s_decision import tolerance_features
    monkeypatch.setattr(model, "annotate_tolerance", tolerance_features.annotate_tolerance)
    monkeypatch.setattr(model, "representation_matrix", tolerance_features.representation_matrix)
    train = pd.DataFrame({"record_id": range(4), "model_smiles": ["C" * n for n in range(50, 54)],
                          "pactivity": 7., "task_id": "T", "species": "Human", "endpoint": "IC50"})
    for recipe in "ABCD":
        profile = calibrated(model.fit_tolerance(train, recipe=recipe, mode="contextual",
            support_settings={"min_identities": 1, "min_scaffolds": 0, "min_documents": 0, "domain_threshold": 100.}))
        path = tmp_path / f"{recipe}.json"
        model.write_tolerance_package(profile, path)
        loaded = model.read_tolerance_package(path)
        candidates = pd.DataFrame({"record_id": [50, 51], "model_smiles": ["C" * 49, "invalid?"],
                                   "task_id": "T", "species": "Human", "endpoint": "IC50"})
        output = model.predict_tolerance(loaded, candidates)
        assert output.state.tolist() == ["supported_relaxation", "excluded"]
        assert output.tolerance_support.iloc[0]["joint"]["identities"] == 4


def test_calibration_binds_criteria_and_context_but_preserves_baseline(monkeypatch):
    second = "rule:veber:rotatable"
    monkeypatch.setitem(SCHEMA["criterion_columns"], second, {
        "violation_column": "second_failed", "magnitude_column": "second_magnitude",
        "upper_normalized_column": "second_magnitude", "lower_normalized_column": "lower",
    })
    profile = model.fit_tolerance(records().assign(second_failed=True, second_magnitude=.3))
    cal = candidate(.10, identity="cal", split="policy_cal", second_failed=True, second_magnitude=.2)
    bg = candidate(.20, identity="bg", split="policy_cal", second_failed=True, second_magnitude=.3)
    first = ["rule:lipinski:mw"]
    result = model.calibrate_tolerance(profile, cal, bg, target_retention=1., max_expansion=1., reconsidered_criteria=first)
    assert result["policy"]["scope"]["criteria"] == first
    assert model.predict_tolerance(result, cal, reconsidered_criteria=first).state.iloc[0] == "supported_relaxation"
    changed = model.predict_tolerance(result, cal, reconsidered_criteria=[second])
    assert changed.state.iloc[0] == "review"
    assert "calibration_scope_mismatch" in changed.tolerance_evidence.iloc[0]["reasons"]
    target_changed = model.predict_tolerance(result, cal, reconsidered_criteria=first, context={"target_id": "another"})
    assert target_changed.state.iloc[0] == "review"
    assert model.predict_tolerance(result, cal, reconsidered_criteria=[second], baseline_eligible=[True]).state.iloc[0] == "base_eligible"


def test_shared_adaptation_keeps_frozen_transform_and_real_fit_references():
    base = calibrated(model.fit_tolerance(records()))
    before = copy.deepcopy(base)
    custom = records(6).assign(identity=[f"custom-{i}" for i in range(6)], task_id="custom", target_id="custom-target", descriptor=999.)
    heldout = records(2, split="test").assign(identity=["held-a", "held-b"], magnitude=999.)
    adapted = model.adapt_tolerance(base, pd.concat([custom, heldout], ignore_index=True),
                                    context={"task_id": "custom", "species": "Human", "endpoint": "IC50"})
    assert base == before
    assert adapted["representation"] == base["representation"]
    assert adapted["annotation_schema"] == base["annotation_schema"]
    assert adapted["support_settings"]["domain_threshold"] == base["support_settings"]["domain_threshold"]
    assert adapted["policy"]["status"] == "uncalibrated"
    assert adapted["training_counts"]["identities"] == 30
    assert adapted["provenance"]["adaptation"]["base_profile_sha256"] == base["profile_sha256"]
    assert adapted["provenance"]["adaptation"]["custom_fit_rows"] == 6
    assert not {"held-a", "held-b"} & {row["identity"] for row in adapted["references"]}


def test_adaptation_unions_existing_fit_identity_without_double_counting():
    base = model.fit_tolerance(records())
    custom = records(1).assign(document_id="new-document", task_id="custom", target_id="custom-target")
    adapted = model.adapt_tolerance(base, custom)
    assert adapted["training_counts"]["identities"] == 24
    row = next(row for row in adapted["references"] if row["identity"] == "id-0")
    assert set(row["document_ids"]) == {"d-0", "new-document"}
    assert len(row["contexts"]) == 2
    with pytest.raises(ValueError, match="inconsistent"):
        model.adapt_tolerance(base, custom.assign(magnitude=10.))


def test_feature_morgan_distance_retains_numeric_and_binary_blocks():
    raw = {"schema": "s2s-tolerance-representation/1", "numeric_columns": ["x"],
           "binary_columns": ["alert"], "representation_offset": 4, "representation_size": 2048}
    width = 2053
    spec = {**raw, "width": width, "fingerprint_width": 2048, "center": [0.] * width,
            "scale": [1.] * width, **model._distance_recipe(raw, width, "B")}
    vector = np.zeros(width)
    vector[4] = 1.
    refs = [{"vector": vector.tolist()}]
    same = model._distances(vector, refs, spec)[0]
    numeric = vector.copy()
    numeric[0] = 3.
    binary = vector.copy()
    binary[1] = 1.
    assert same == pytest.approx(0.)
    assert model._distances(numeric, refs, spec)[0] == pytest.approx(1.)
    assert model._distances(binary, refs, spec)[0] == pytest.approx(1 / 12)


def test_erg_distance_uses_continuous_values_with_equal_block_weight():
    raw = {"schema": "s2s-tolerance-representation/1", "numeric_columns": ["x"],
           "binary_columns": ["alert"], "representation_offset": 4, "representation_size": 315}
    width = 320
    spec = {**raw, "width": width, "fingerprint_width": 0, "center": [0.] * width,
            "scale": [1.] * width, **model._distance_recipe(raw, width, "C")}
    vector = np.zeros(width)
    vector[4] = 1.
    candidate_vector = vector.copy()
    candidate_vector[4] = 2.
    assert model._distances(candidate_vector, [{"vector": vector.tolist()}], spec)[0] == pytest.approx(1 / (3 * np.sqrt(315)))


@pytest.mark.parametrize("mode", ["hierarchical", "contextual"])
def test_multitask_calibration_retains_distinct_identity_context_pairs(mode):
    train = pd.concat([records(), records().assign(identity=lambda x: "other-" + x.identity,
                      task_id="T2", target_id="target2")], ignore_index=True)
    profile = model.fit_tolerance(train, mode=mode)
    original = copy.deepcopy(profile)
    positive = pd.concat([candidate(split="policy_cal", baseline_eligible=True),
                          candidate(record_id=1, split="policy_cal", task_id="T2",
                                    target_id="target2", baseline_eligible=False)], ignore_index=True)
    background = pd.concat([candidate(identity="background", split="policy_cal", magnitude=.24),
                            candidate(identity="background", record_id=1, split="policy_cal", magnitude=.24,
                                      task_id="T2", target_id="target2")], ignore_index=True)
    result = model.calibrate_tolerance(profile, positive, background,
                                      target_retention=1., max_expansion=1.)
    policy = result["policy"]
    assert policy["status"] == "feasible"
    assert policy["calibration_count_basis"] == "identity_context"
    assert policy["positive_count"] == 2
    assert policy["positive_unique_identities"] == 1
    assert policy["baseline_positive_count"] == 1
    assert policy["positive_input_rows"] == 2
    assert {item["task_id"] for item in policy["scope"]["contexts"]} == {"T", "T2"}
    assert "nonindependent" in policy["retention_interpretation"]
    assert result["references"] == original["references"]
    assert profile == original
    with pytest.raises(ValueError, match="identity-context pairs"):
        model.calibrate_tolerance(profile, pd.concat([positive, positive.iloc[[0]]], ignore_index=True), background)
    with pytest.raises(ValueError, match="positive/background"):
        model.calibrate_tolerance(profile, positive, background.assign(identity="new-id", task_id="unseen"))


@pytest.mark.parametrize("mode", ["general", "fixed"])
def test_global_calibration_deduplicates_chemical_identity_and_counts_actual_rows(mode):
    profile = model.fit_tolerance(records(), mode=mode)
    positive = pd.concat([candidate(split="policy_cal", baseline_eligible=True),
                          candidate(record_id=1, split="policy_cal", task_id="T2",
                                    target_id="target2", baseline_eligible=True)], ignore_index=True)
    result = model.calibrate_tolerance(profile, positive, None)
    assert result["policy"]["calibration_count_basis"] == "identity"
    assert result["policy"]["positive_count"] == 1
    assert result["policy"]["positive_input_rows"] == 2
    assert result["policy"]["positive_unique_identities"] == 1
    assert result["policy"]["background_unique_identities"] == 0
    with pytest.raises(ValueError, match="inconsistent global decisions"):
        model.calibrate_tolerance(profile, positive.assign(baseline_eligible=[True, False]), None)


@pytest.mark.parametrize("mode", ["hierarchical", "contextual"])
def test_contextual_calibration_needs_background_evaluated_for_each_positive_query(mode):
    train = pd.concat([records(), records().assign(identity=lambda x: "other-" + x.identity,
                      task_id="T2", target_id="target2")], ignore_index=True)
    profile = model.fit_tolerance(train, mode=mode)
    positive = pd.concat([candidate(split="policy_cal"),
                          candidate(identity="positive2", record_id=1, split="policy_cal",
                                    task_id="T2", target_id="target2")], ignore_index=True)
    bank = candidate(identity="background", split="policy_cal", magnitude=.24,
                     task_id="", target_id="", species="", endpoint="")
    unknown = model.calibrate_tolerance(profile, positive, bank, target_retention=1., max_expansion=1.)
    assert unknown["policy"]["status"] == "uncalibrated"
    assert unknown["policy"]["reason"] == "missing_context_matched_background"
    assert unknown["policy"]["expansion"] is None
    assert unknown["policy"]["tau"] is None
    assert not unknown["policy"]["rescue_enabled"]
    assert {item["task_id"] for item in unknown["policy"]["missing_background_contexts"]} == {"T", "T2"}
    matched = pd.concat([bank.assign(task_id="T", target_id="target", species="Human", endpoint="IC50"),
                         bank.assign(record_id=1, task_id="T2", target_id="target2",
                                     species="Human", endpoint="IC50")], ignore_index=True)
    qualified = model.calibrate_tolerance(profile, positive, matched, target_retention=1., max_expansion=0.)
    assert qualified["policy"]["status"] == "feasible"
    assert qualified["policy"]["background_count"] == 2
    assert qualified["policy"]["background_unique_identities"] == 1
    assert qualified["policy"]["background_contexts"] == qualified["policy"]["scope"]["contexts"]
    partial = model.calibrate_tolerance(profile, positive, matched.iloc[[0]],
                                       target_retention=1., max_expansion=1.)
    assert partial["policy"]["reason"] == "missing_context_matched_background"
    assert [item["task_id"] for item in partial["policy"]["missing_background_contexts"]] == ["T2"]
    sufficient = model.calibrate_tolerance(profile, positive.assign(baseline_eligible=True), bank)
    assert sufficient["policy"]["status"] == "baseline_sufficient"


@pytest.mark.parametrize("mode", ["hierarchical", "contextual"])
@pytest.mark.parametrize("context_source", ["profile", "calibration"])
def test_explicit_query_context_qualifies_unlabelled_reference_bank(mode, context_source):
    query = {"task_id": "T"}
    profile = model.fit_tolerance(records(), mode=mode, context=query if context_source == "profile" else None)
    bank = candidate(identity="background", split="policy_cal", magnitude=.24,
                     task_id="", target_id="", species="", endpoint="")
    result = model.calibrate_tolerance(profile, candidate(split="policy_cal"), bank,
                                      target_retention=1., max_expansion=0.,
                                      context=query if context_source == "calibration" else None)
    assert result["policy"]["status"] == "feasible"
    assert result["policy"]["scope"]["context_mode"] == "fixed"


@pytest.mark.parametrize("recipe", list("ABCD"))
def test_annotation_metadata_is_not_propagated_to_scoring_rows(monkeypatch, recipe):
    from s2s_decision import tolerance_features
    monkeypatch.setattr(model, "annotate_tolerance", tolerance_features.annotate_tolerance)
    monkeypatch.setattr(model, "representation_matrix", tolerance_features.representation_matrix)
    train = pd.DataFrame({"record_id": range(4), "model_smiles": ["C" * n for n in range(50, 54)],
                          "pactivity": 7., "task_id": "T", "species": "Human", "endpoint": "IC50"})
    settings = {"min_identities": 1, "min_scaffolds": 0, "min_documents": 0, "domain_threshold": 100.}
    profile = model.fit_tolerance(train, recipe=recipe, mode="contextual", support_settings=settings)
    query = train.iloc[[0]].assign(record_id=99, model_smiles="C" * 49)
    expected = model.predict_tolerance(profile, query)
    annotated, _ = tolerance_features.annotate_tolerance(query)
    annotated_train, _ = tolerance_features.annotate_tolerance(train)
    original_attrs = copy.deepcopy(annotated.attrs)
    training_attrs = copy.deepcopy(annotated_train.attrs)
    original_pattern = model._row_pattern

    def attrs_free_pattern(row, schema):
        assert row.attrs == {}, "Scoring must not copy the annotation schema into each Series"
        return original_pattern(row, schema)

    monkeypatch.setattr(model, "_row_pattern", attrs_free_pattern)
    pd.testing.assert_frame_equal(model.predict_tolerance(profile, annotated), expected)
    assert model.fit_tolerance(annotated_train, recipe=recipe, mode="contextual", support_settings=settings) == profile
    assert annotated.attrs == original_attrs
    assert annotated_train.attrs == training_attrs


@pytest.mark.parametrize("frame, options, message", [
    (None, {}, "nonempty"),
    (records(0), {}, "nonempty"),
    (records().drop(columns="pactivity"), {}, "measured pactivity"),
    (records().assign(pactivity=np.inf), {}, "finite"),
    (records(), {"mode": "classifier"}, "mode or recipe"),
    (records(), {"recipe": "unknown"}, "mode or recipe"),
    (records(), {"support_settings": {"invented": 20}}, "support setting"),
    (records(), {"context": ["target"]}, "context must be an object"),
])
def test_training_rejects_invalid_input_instead_of_creating_a_fallback_model(frame, options, message):
    with pytest.raises(ValueError, match=message):
        model.fit_tolerance(frame, **options)


def test_training_accepts_activity_alias_and_regenerates_repeated_record_ids():
    frame = records().rename(columns={"pactivity": "pActivity"}).assign(record_id=1)
    frame = frame.drop(columns=["relation", "task_id", "target_id", "species", "endpoint"])
    profile = model.fit_tolerance(frame)
    assert profile["training_counts"]["rows"] == 24
    assert profile["training_counts"]["tasks"] == 1
    assert {item["task_id"] for row in profile["references"] for item in row["contexts"]} == {"general"}
    assert frame.record_id.tolist() == [1] * 24


@pytest.mark.parametrize("mask", [[1], ["True"], [True, False], [[True]]])
def test_prediction_rejects_ambiguous_or_misaligned_baseline_masks(mask):
    profile = model.fit_tolerance(records())
    with pytest.raises(ValueError, match="one boolean per candidate"):
        model.predict_tolerance(profile, candidate(), baseline_eligible=mask)


def test_prediction_rejects_unknown_policy_criteria_and_preserves_empty_input_shape():
    profile = model.fit_tolerance(records())
    with pytest.raises(ValueError, match="Unknown tolerance reconsidered criterion"):
        model.predict_tolerance(profile, candidate(), reconsidered_criteria=["rule:invented"])
    empty = model.predict_tolerance(profile, records(0))
    assert empty.empty
    assert {"record_id", "eligible", "tolerance_supported", "tolerance_evidence"} <= set(empty)


@pytest.mark.parametrize("mode", ["hierarchical", "contextual"])
def test_disabled_fallback_abstains_despite_sufficient_parent_support(mode):
    profile = calibrated(model.fit_tolerance(records(), mode=mode,
                         support_settings={"allow_fallback": False, "k": 4}))
    output = model.predict_tolerance(profile, candidate(task_id="unseen", target_id="unseen"))
    assert output.tolerance_fallback.iloc[0] == "disabled"
    assert output.state.iloc[0] == "review"
    assert not output.tolerance_supported.iloc[0]
    assert output.tolerance_support.iloc[0]["parent"]["identities"] == 24
    assert output.tolerance_support.iloc[0]["joint"]["identities"] == 0


@pytest.mark.parametrize("mode", ["hierarchical", "contextual"])
def test_unknown_species_or_incomplete_query_context_never_enables_rescue(mode):
    profile = calibrated(model.fit_tolerance(records(), mode=mode))
    incompatible = model.predict_tolerance(profile, candidate(species="Mouse"))
    assert incompatible.state.iloc[0] == "review"
    assert "incompatible_context" in incompatible.tolerance_evidence.iloc[0]["reasons"]
    incomplete = model.predict_tolerance(profile, candidate(task_id="", target_id="", species="", endpoint=""))
    assert incomplete.state.iloc[0] == "review"
    assert "incomplete_context" in incomplete.tolerance_evidence.iloc[0]["reasons"]


def test_baseline_only_training_does_not_supply_evidence_for_unobserved_violations():
    profile = calibrated(model.fit_tolerance(records().assign(failed=False)))
    assert profile["curves"] == {}
    baseline = model.predict_tolerance(profile, candidate(failed=False))
    assert baseline.state.iloc[0] == "base_eligible"
    unsupported = model.predict_tolerance(profile, candidate())
    assert unsupported.state.iloc[0] == "review"
    assert "insufficient_joint_support" in unsupported.tolerance_evidence.iloc[0]["reasons"]


@pytest.mark.parametrize("options, message", [
    ({"mode": "classifier"}, "adaptation mode"),
    ({"support_settings": {"domain_threshold": 999.}}, "frozen base domain"),
    ({"support_settings": {"invented": 1}}, "recognized support settings"),
])
def test_shared_profile_adaptation_rejects_algorithm_and_domain_drift(options, message):
    base = model.fit_tolerance(records())
    before = copy.deepcopy(base)
    with pytest.raises(ValueError, match=message):
        model.adapt_tolerance(base, candidate(split="fit"), **options)
    assert base == before


def test_shared_profile_adaptation_rejects_changed_chemical_representation(monkeypatch):
    base = model.fit_tolerance(records())
    monkeypatch.setattr(model, "representation_matrix", lambda frame, schema, recipe="A":
                        (np.ones((len(frame), 2), dtype=np.float32), {"recipe": recipe}))
    with pytest.raises(ValueError, match="frozen base transform"):
        model.adapt_tolerance(base, candidate(split="fit"))


def multicontext_calibration_fixture(mode):
    train = pd.concat([records(), records().assign(identity=lambda x: "other-" + x.identity,
                      task_id="T2", target_id="target2", magnitude=lambda x: x.magnitude + .08,
                      descriptor=lambda x: x.descriptor + .5)], ignore_index=True)
    profile = model.fit_tolerance(train, mode=mode)
    positive = pd.concat([
        candidate(identity="shared-positive", split="policy_cal"),
        candidate(identity="shared-positive", record_id=1, split="policy_cal", task_id="T2", target_id="target2"),
        candidate(identity="other-positive", record_id=2, split="policy_cal", magnitude=.12),
        candidate(identity="baseline-positive", record_id=3, split="policy_cal", task_id="T2",
                  target_id="target2", baseline_eligible=True),
    ], ignore_index=True).fillna({"baseline_eligible": False})
    background = pd.concat([
        candidate(identity="shared-bank", split="policy_cal", magnitude=.30),
        candidate(identity="shared-bank", record_id=1, split="policy_cal", magnitude=.30,
                  task_id="T2", target_id="target2"),
    ], ignore_index=True)
    probe = pd.concat([positive, candidate(identity="above-threshold", record_id=4,
                                           split="test", magnitude=.23)], ignore_index=True)
    return profile, positive, background, probe.fillna({"baseline_eligible": False})


@pytest.mark.parametrize("mode", ["hierarchical", "contextual"])
def test_calibration_resolves_each_exact_context_once_per_call_and_returns_independent_copies(monkeypatch, mode):
    profile, _, _, _ = multicontext_calibration_fixture(mode)
    frame = records(4, split="policy_cal").assign(identity=[f"cal-{index}" for index in range(4)])
    scored = model.predict_tolerance(profile, frame)
    contexts = [{"task_id": "T", "audit_tag": "one"},
                {"audit_tag": "one", "task_id": "T"},
                {"task_id": "T", "audit_tag": "two"},
                {"task_id": "T2", "audit_tag": "one"}]
    scored = scored.assign(tolerance_applied_context=copy.deepcopy(contexts))
    monkeypatch.setattr(model, "predict_tolerance", lambda *args, **kwargs: scored.copy())
    original = model._resolve_context
    calls = []

    def counted(refs, context):
        calls.append(copy.deepcopy(context))
        return original(refs, context)

    monkeypatch.setattr(model, "_resolve_context", counted)
    first = model._calibration_cohort(profile, frame, positive=True)
    assert len(calls) == 3
    assert {item["audit_tag"] for item in calls} == {"one", "two"}
    first.tolerance_applied_context.iloc[0]["species"] = "Modified"
    assert first.tolerance_applied_context.iloc[1]["species"] == "Human"
    assert scored.tolerance_applied_context.tolist() == contexts
    second = model._calibration_cohort(profile, frame, positive=True)
    assert len(calls) == 6
    assert second.tolerance_applied_context.iloc[0]["species"] == "Human"


@pytest.mark.parametrize("mode, expected_profile_hash, expected_prediction_hash", [
    ("hierarchical", "bd4b1691748090ae35d0c1b7537042043df1001b5cf0bf83ab3ec7bcca1e15bb",
     "27e44440b5b2544df5c96d6576f68213ccdee82bc6d2d92fb36addd41b658f07"),
    ("contextual", "6056951d5082132893748139ef21fe12ccb00ce34668857c93a7afcc7af45b08",
     "80b8eb9f32d75a8b8d69e9d32be9e3672665e8956972eabc965860234afacdba"),
])
def test_calibration_context_memo_preserves_frozen_policy_and_every_prediction_field(
        mode, expected_profile_hash, expected_prediction_hash):
    profile, positive, background, probe = multicontext_calibration_fixture(mode)
    frozen = model.calibrate_tolerance(profile, positive, background, target_retention=1., max_expansion=0.)
    assert frozen["profile_sha256"] == expected_profile_hash
    predicted = model.predict_tolerance(frozen, probe)
    serialized = json.dumps(predicted.to_dict("records"), sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert hashlib.sha256(serialized.encode()).hexdigest() == expected_prediction_hash
    assert predicted.state.tolist() == ["supported_relaxation", "supported_relaxation",
                                        "supported_relaxation", "base_eligible", "excluded"]
    with pytest.raises(ValueError, match="identity-context pairs"):
        model.calibrate_tolerance(profile, pd.concat([positive, positive.iloc[[0]]], ignore_index=True), background)
