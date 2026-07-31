"""Selection policies: mandatory profiles, consensus, QED and expressions."""

from __future__ import annotations

import pandas as pd
import pytest

from smiles2select.decision.engine import DecisionEngine
from smiles2select.decision.explanations import policy_sentence, restrictiveness_warning
from smiles2select.decision.expression_parser import ExpressionError, evaluate, variable_names
from smiles2select.decision.policies import (
    DecisionPolicy,
    all_profiles_policy,
    consensus_policy,
    recommended_policy,
    single_profile_policy,
)
from smiles2select.scores.qed import QedSelection

pytestmark = pytest.mark.unit


def status_frame(**profiles: list[bool]) -> pd.DataFrame:
    """Build an evaluation status table from per-profile verdicts."""
    size = len(next(iter(profiles.values())))
    frame = pd.DataFrame(index=pd.RangeIndex(1, size + 1, name="record_id"))
    for profile_id, verdicts in profiles.items():
        frame[f"{profile_id}__passed"] = verdicts
        frame[f"{profile_id}__violations"] = [0 if verdict else 1 for verdict in verdicts]
    return frame


def score_frame(index: pd.Index, **columns) -> pd.DataFrame:
    frame = pd.DataFrame(index=index)
    frame["hard_violation_count"] = columns.get("hard", [0] * len(index))
    frame["alert_count"] = columns.get("alerts", [0] * len(index))
    frame["consensus_score"] = columns.get("consensus", [0.5] * len(index))
    if "qed" in columns:
        frame["qed"] = columns["qed"]
    return frame


def test_single_mandatory_profile():
    status = status_frame(lipinski=[True, False])
    result = DecisionEngine(single_profile_policy("lipinski")).decide(
        status, score_frame(status.index)
    )
    assert result.decisions["selected"].tolist() == [True, False]
    assert "lipinski" in result.decisions["exclusion_reasons"].iloc[1]


def test_all_profiles_must_pass():
    status = status_frame(lipinski=[True, True], veber=[True, False])
    policy = all_profiles_policy(["lipinski", "veber"])
    result = DecisionEngine(policy).decide(status, score_frame(status.index))
    assert result.decisions["selected"].tolist() == [True, False]


def test_consensus_two_of_three():
    status = status_frame(
        lipinski=[True, True, False],
        veber=[True, False, False],
        ghose=[False, False, True],
    )
    policy = consensus_policy(["lipinski", "veber", "ghose"], 2)
    result = DecisionEngine(policy).decide(status, score_frame(status.index))
    assert result.decisions["selected"].tolist() == [True, False, False]


def test_consensus_three_of_five():
    status = status_frame(
        lipinski=[True, True],
        veber=[True, True],
        ghose=[True, False],
        egan=[False, False],
        muegge=[False, False],
    )
    policy = consensus_policy(["lipinski", "veber", "ghose", "egan", "muegge"], 3)
    result = DecisionEngine(policy).decide(status, score_frame(status.index))
    assert result.decisions["selected"].tolist() == [True, False]


def test_consensus_of_one_means_any_profile():
    status = status_frame(lipinski=[False, False], veber=[True, False])
    policy = consensus_policy(["lipinski", "veber"], 1)
    result = DecisionEngine(policy).decide(status, score_frame(status.index))
    assert result.decisions["selected"].tolist() == [True, False]


def test_consensus_beyond_the_number_of_profiles_is_rejected():
    with pytest.raises(ValueError, match="exceeds"):
        consensus_policy(["lipinski", "veber"], 3)


def test_informative_profiles_never_exclude():
    """The whole point of the recommended default."""
    status = status_frame(
        lipinski=[True], veber=[True], ghose=[False], egan=[False], muegge=[False]
    )
    result = DecisionEngine(recommended_policy()).decide(status, score_frame(status.index))
    assert result.decisions["selected"].tolist() == [True]


def test_qed_ranking_mode_excludes_nothing():
    status = status_frame(lipinski=[True, True])
    policy = DecisionPolicy(id="t", roles={"lipinski": "mandatory"}, qed=QedSelection(mode="rank"))
    result = DecisionEngine(policy).decide(status, score_frame(status.index, qed=[0.1, 0.9]))
    assert result.decisions["selected"].tolist() == [True, True]


def test_qed_threshold_mode_excludes_below_the_cutoff():
    status = status_frame(lipinski=[True, True])
    policy = DecisionPolicy(
        id="t",
        roles={"lipinski": "mandatory"},
        qed=QedSelection(mode="threshold", threshold=0.5),
    )
    result = DecisionEngine(policy).decide(status, score_frame(status.index, qed=[0.1, 0.9]))
    assert result.decisions["selected"].tolist() == [False, True]


def test_qed_threshold_mode_requires_a_threshold():
    with pytest.raises(ValueError):
        QedSelection(mode="threshold")


def test_expression_with_and_or_not():
    status = status_frame(lipinski=[True, True, False], veber=[True, False, True])
    policy = DecisionPolicy(
        id="t",
        roles={"lipinski": "informative", "veber": "informative"},
        expression="(lipinski AND veber) OR NOT lipinski",
    )
    result = DecisionEngine(policy).decide(status, score_frame(status.index))
    assert result.decisions["selected"].tolist() == [True, False, True]


def test_expression_can_combine_profiles_and_qed():
    status = status_frame(lipinski=[True, True])
    policy = DecisionPolicy(
        id="t",
        roles={"lipinski": "informative"},
        expression="lipinski AND qed >= 0.50",
    )
    result = DecisionEngine(policy).decide(status, score_frame(status.index, qed=[0.4, 0.6]))
    assert result.decisions["selected"].tolist() == [False, True]


def test_expression_rejects_function_calls():
    with pytest.raises(ExpressionError, match="forbidden syntax"):
        variable_names("__import__('os').system('echo')")


def test_expression_rejects_attribute_access():
    with pytest.raises(ExpressionError, match="forbidden syntax"):
        variable_names("qed.__class__")


def test_expression_reports_unknown_names():
    frame = pd.DataFrame({"lipinski": [True]}, index=pd.Index([1], name="record_id"))
    with pytest.raises(ExpressionError, match="unknown name"):
        evaluate("lipinski AND unknown_profile", frame)


def test_policy_sentence_states_that_alerts_do_not_exclude():
    sentence = policy_sentence(recommended_policy())
    assert "não" in sentence
    assert "PAINS" in sentence


def test_stacking_many_mandatory_profiles_raises_a_warning():
    policy = all_profiles_policy(["lipinski", "veber", "ghose", "egan", "muegge"])
    warning = restrictiveness_warning(policy)
    assert warning is not None
    assert "restritiva" in warning


def test_two_mandatory_profiles_do_not_trigger_the_warning():
    assert restrictiveness_warning(recommended_policy()) is None
