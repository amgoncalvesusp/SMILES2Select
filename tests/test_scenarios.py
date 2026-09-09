"""Cached scenario replay must preserve scientific eligibility and input identity."""

from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from smiles2select.decision.policies import DecisionPolicy
from smiles2select.pipeline.config import RunConfig
from smiles2select.profiles.registry import Profile
from smiles2select.rules.engine import Rule
from smiles2select.selection_intelligence.constrained_selection import SelectionConstraints
from smiles2select.selection_intelligence.objectives import Objective
from smiles2select.selection_intelligence.scenarios import (
    ScenarioSpec,
    compare_scenarios,
    evaluate_scenario,
    evaluate_study,
    stability_table,
)


@pytest.fixture
def cached():
    descriptors = pd.DataFrame(
        {
            "mw": [100.0, 200.0, 600.0, 900.0],
            "qed": [0.4, 0.8, 0.9, 0.95],
            "evaluable": [True] * 4,
            "valid": [True] * 4,
            "murcko_scaffold": ["A", "B", "C", "D"],
        },
        index=pd.Index([1, 2, 3, 4], name="record_id"),
    )
    profile = Profile(
        "p", "P", "custom", "1", (Rule("mw_limit", "p", "mw", "<=", 500),), {"type": "all_rules"}
    )
    result = SimpleNamespace(
        descriptors=descriptors,
        profiles=(profile,),
        alerts=pd.DataFrame(),
        scores=pd.DataFrame({"qed": descriptors.qed, "alert_count": 0}),
        config=RunConfig((), ("p",), policy=DecisionPolicy(roles={"p": "mandatory"})),
        reference_duplicates=None,
    )
    return result, descriptors.copy()


def test_threshold_replay_changes_eligibility_without_mutation(cached):
    result, candidates = cached
    a = ScenarioSpec("A", constraints=SelectionConstraints(target_count=1))
    b = replace(a, name="B", thresholds={"mw_limit": 700})
    first, second = evaluate_study(result, candidates, (a, b))
    assert tuple(first.eligible_ids) == (1, 2)
    assert first.outcome.selected_ids == (2,)
    assert second.outcome.selected_ids == (3,)
    assert result.profiles[0].rules[0].threshold == 500
    comparison = compare_scenarios(first, second)
    assert comparison.entered_ids == (3,)
    assert comparison.left_ids == (2,)
    assert comparison.gained_scaffolds == ("C",)
    assert comparison.jaccard == 0
    table = stability_table((first, second))
    assert table.loc[2, "eligibility_frequency"] == 1
    assert table.loc[2, "selection_frequency"] == 0.5


def test_allowed_violations_and_informative_profiles_are_preserved(cached):
    result, candidates = cached
    result.profiles = (
        replace(result.profiles[0], pass_policy={"type": "max_violations", "value": 1}),
    )
    assert len(evaluate_scenario(result, candidates, ScenarioSpec("A")).eligible_ids) == 4
    result.profiles = (replace(result.profiles[0], pass_policy={"type": "all_rules"}),)
    result.config = replace(result.config, policy=DecisionPolicy(roles={"p": "informative"}))
    assert len(evaluate_scenario(result, candidates, ScenarioSpec("A")).eligible_ids) == 4


def test_manual_flags_separate_from_chemical_frequency(cached):
    result, candidates = cached
    snapshot = evaluate_scenario(
        result, candidates, ScenarioSpec("A"), pinned_ids=(4,), rescued_ids=(3,), excluded_ids=(2,)
    )
    assert set(snapshot.outcome.selected_ids) == {1, 3}
    table = stability_table((snapshot,))
    assert table.loc[4, "pinned"] and table.loc[4, "eligibility_frequency"] == 0
    assert table.loc[3, "rescued"] and table.loc[2, "excluded"]


def test_changed_universe_or_data_rejected(cached):
    result, candidates = cached
    first = evaluate_scenario(result, candidates, ScenarioSpec("A"))
    candidates.loc[1, "qed"] = 0.99
    second = evaluate_scenario(result, candidates, ScenarioSpec("B"))
    with pytest.raises(ValueError, match="same input"):
        compare_scenarios(first, second)
    with pytest.raises(ValueError, match="complete"):
        evaluate_scenario(result, candidates.iloc[:2], ScenarioSpec("A"))


def test_invalid_specs_and_unsupported_zones_fail_explicitly(cached):
    result, candidates = cached
    with pytest.raises(ValueError):
        ScenarioSpec("", thresholds={"mw_limit": float("nan")})
    with pytest.raises(ValueError, match="unknown"):
        evaluate_scenario(result, candidates, ScenarioSpec("A", thresholds={"typo": 1}))
    result.config = replace(result.config, zones=({"zone_id": "a"},))
    with pytest.raises(ValueError, match="zones"):
        evaluate_scenario(result, candidates, ScenarioSpec("A"))


def test_large_objective_ranking_never_calls_quadratic_pareto(cached, monkeypatch):
    import smiles2select.selection_intelligence.scenarios as module

    result, candidates = cached
    monkeypatch.setattr(module, "EXACT_PARETO_LIMIT", 2)
    monkeypatch.setattr(module, "rank_candidates", lambda *a: pytest.fail("quadratic ranking"))
    scenario = ScenarioSpec(
        "A",
        thresholds={"mw_limit": 1000},
        objectives=(Objective("qed"),),
        constraints=SelectionConstraints(target_count=1),
    )
    snapshot = evaluate_scenario(result, candidates, scenario)
    assert snapshot.outcome.selected_ids == (4,)
    assert snapshot.provenance["ranking_method"] == "weighted_percentile"
    assert any("Pareto" in warning for warning in snapshot.warnings)


def test_json_replay_and_reject_tampered_recipe(cached, tmp_path):
    import json

    from smiles2select.selection_intelligence.scenario_io import load_study, save_study

    result, candidates = cached
    snapshots = evaluate_study(result, candidates, (ScenarioSpec("A"), ScenarioSpec("B")))
    path = tmp_path / "study.json"
    save_study(path, snapshots)
    loaded = load_study(path, result, candidates)
    assert loaded[0].outcome.selected_ids == snapshots[0].outcome.selected_ids
    payload = json.loads(path.read_text())
    payload["scenarios"][0]["spec"]["constraints"]["target_count"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_study(path, result, candidates)
    save_study(path, snapshots)
    candidates.loc[1, "mw"] = 42
    with pytest.raises(ValueError, match="same input"):
        load_study(path, result, candidates)


def test_reference_restrictions_not_bypassed_by_rescue(cached):
    result, candidates = cached
    result.descriptors["is_reference_duplicate"] = [False, True, False, False]
    result.config = replace(result.config, exclude_reference_duplicates=True)
    snapshot = evaluate_scenario(result, candidates, ScenarioSpec("A"), rescued_ids=(2,))
    assert snapshot.outcome.selected_ids == (1,)


def test_objective_ranking_exact_and_threshold_immutable(cached):
    result, candidates = cached
    thresholds = {"mw_limit": 700}
    spec = ScenarioSpec("A", thresholds, (Objective("qed"),), SelectionConstraints(target_count=1))
    thresholds["mw_limit"] = 10
    assert evaluate_scenario(result, candidates, spec).outcome.selected_ids == (3,)
    with pytest.raises(TypeError):
        spec.thresholds["mw_limit"] = 10


def test_alert_policy_and_chunk_boundaries(cached, monkeypatch):
    import smiles2select.selection_intelligence.scenarios as module
    from smiles2select.alerts.policies import AlertPolicy

    result, candidates = cached
    result.alerts = pd.DataFrame({"record_id": [1, 1, 3], "catalog_id": ["pains"] * 3})
    result.config = replace(
        result.config,
        policy=replace(
            result.config.policy, alert_policy=AlertPolicy(actions={"pains": "exclude"})
        ),
    )
    monkeypatch.setattr(module, "EVALUATION_CHUNK_SIZE", 1)
    assert evaluate_scenario(result, candidates, ScenarioSpec("A")).outcome.selected_ids == (2,)


def test_unknown_scaffolds_cannot_bypass_quotas(cached):
    result, candidates = cached
    candidates["murcko_scaffold"] = pd.NA
    with pytest.raises(ValueError, match="complete cached"):
        evaluate_scenario(
            result,
            candidates,
            ScenarioSpec("A", constraints=SelectionConstraints(max_per_scaffold=1)),
        )


def test_disabled_pin_preservation_does_not_rescue(cached):
    result, candidates = cached
    snapshot = evaluate_scenario(
        result,
        candidates,
        ScenarioSpec("A", constraints=SelectionConstraints(preserve_pinned=False)),
        pinned_ids=(4,),
    )
    assert 4 not in snapshot.outcome.selected_ids


def test_pin_only_preserves_a_failed_record_after_explicit_rescue(cached):
    result, candidates = cached
    spec = ScenarioSpec("A", constraints=SelectionConstraints(target_count=1))
    assert evaluate_scenario(result, candidates, spec, pinned_ids=(4,)).outcome.selected_ids == (2,)
    assert evaluate_scenario(
        result, candidates, spec, pinned_ids=(4,), rescued_ids=(4,)
    ).outcome.selected_ids == (4,)


def test_ranking_uses_recomputed_consensus_after_threshold_override(cached):
    result, candidates = cached
    result.config = replace(result.config, policy=DecisionPolicy(roles={"p": "informative"}))
    candidates["consensus_score"] = [1.0, 0.9, 0.8, 0.7]
    spec = ScenarioSpec(
        "A",
        thresholds={"mw_limit": 1000},
        objectives=(Objective("consensus_score"),),
        constraints=SelectionConstraints(target_count=1),
    )
    assert evaluate_scenario(result, candidates, spec).outcome.selected_ids == (4,)


def test_qed_percentile_is_global_and_independent_of_chunk_size(cached, monkeypatch):
    import smiles2select.selection_intelligence.scenarios as module
    from smiles2select.scores.qed import QedSelection

    result, candidates = cached
    result.config = replace(
        result.config,
        policy=DecisionPolicy(
            roles={"p": "informative"}, qed=QedSelection(mode="top_percentile", percentile=50)
        ),
    )
    for chunk_size in (1, 2, 100):
        monkeypatch.setattr(module, "EVALUATION_CHUNK_SIZE", chunk_size)
        snapshot = evaluate_scenario(result, candidates, ScenarioSpec("A"))
        assert tuple(snapshot.eligible_ids) == (3, 4)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, [], None])
def test_json_rejects_invalid_objective_weight(bad):
    from smiles2select.selection_intelligence.scenario_io import spec_from_dict, spec_to_dict

    payload = spec_to_dict(ScenarioSpec("A", objectives=(Objective("qed"),)))
    payload["objectives"][0]["weight"] = bad
    with pytest.raises(ValueError):
        spec_from_dict(payload)
