"""Operators, rule evaluation and pass policies.

Boundary behaviour is tested on synthetic descriptor tables: a molecule
"exactly at the limit" is expressed directly as the limit value, which is the
only way to test the boundary without depending on a structure that happens to
land there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smiles2select.profiles.registry import Profile
from smiles2select.rules import operators
from smiles2select.rules.engine import Rule
from smiles2select.rules.evaluator import evaluate_profiles
from smiles2select.rules.explanations import explain_failure
from tests.conftest import descriptor_frame

pytestmark = pytest.mark.unit


def make_profile(rules, policy=None) -> Profile:
    return Profile(
        id="test",
        name="Test",
        category="custom",
        version="1.0.0",
        rules=tuple(rules),
        pass_policy=policy or {"type": "all_rules"},
    )


def test_less_or_equal_accepts_the_exact_limit():
    values = pd.Series([499.0, 500.0, 500.001])
    assert operators.apply("<=", values, 500).tolist() == [True, True, False]


def test_greater_or_equal_accepts_the_exact_limit():
    values = pd.Series([159.9, 160.0, 160.1])
    assert operators.apply(">=", values, 160).tolist() == [False, True, True]


def test_strict_operators_reject_the_limit():
    values = pd.Series([4, 5])
    assert operators.apply(">", values, 4).tolist() == [False, True]


def test_between_inclusive_includes_both_ends():
    values = pd.Series([19, 20, 45, 70, 71])
    assert operators.apply("between_inclusive", values, [20, 70]).tolist() == [
        False,
        True,
        True,
        True,
        False,
    ]


def test_missing_values_fail_closed():
    """An uncomputable descriptor must never pass a rule by accident."""
    values = pd.Series([100.0, np.nan])
    assert operators.apply("<=", values, 500).tolist() == [True, False]


def test_range_operator_rejects_a_single_bound():
    with pytest.raises(ValueError):
        operators.bounds("between_inclusive", 5)


def test_unknown_operator_is_rejected_at_rule_construction():
    with pytest.raises(operators.UnknownOperatorError):
        Rule(id="r", profile_id="p", descriptor="mol_wt", operator="≈", threshold=1)


def test_only_broken_rules_are_materialised():
    frame = descriptor_frame({"mol_wt": [100.0, 600.0], "tpsa": [50.0, 50.0]})
    profile = make_profile(
        [
            Rule("mw", "test", "mol_wt", "<=", 500, failure_code="MW_HIGH"),
            Rule("tpsa", "test", "tpsa", "<=", 140, failure_code="TPSA_HIGH"),
        ]
    )
    evaluation = evaluate_profiles(frame, [profile])

    assert evaluation.passed("test").tolist() == [True, False]
    assert len(evaluation.failures) == 1
    failure = evaluation.failures.iloc[0]
    assert failure["record_id"] == 2
    assert failure["failure_code"] == "MW_HIGH"
    assert failure["observed_value"] == 600.0
    assert failure["upper_limit"] == 500.0
    assert pd.isna(failure["lower_limit"])


def test_all_rules_policy_needs_zero_violations():
    frame = descriptor_frame({"mol_wt": [600.0], "tpsa": [50.0]})
    profile = make_profile(
        [
            Rule("mw", "test", "mol_wt", "<=", 500, failure_code="MW_HIGH"),
            Rule("tpsa", "test", "tpsa", "<=", 140, failure_code="TPSA_HIGH"),
        ]
    )
    assert evaluate_profiles(frame, [profile]).passed("test").tolist() == [False]


def test_max_violations_policy_tolerates_one():
    """The classical Lipinski reading: one violation still passes, two do not."""
    frame = descriptor_frame({"mol_wt": [600.0, 600.0], "tpsa": [50.0, 300.0]})
    profile = make_profile(
        [
            Rule("mw", "test", "mol_wt", "<=", 500, failure_code="MW_HIGH"),
            Rule("tpsa", "test", "tpsa", "<=", 140, failure_code="TPSA_HIGH"),
        ],
        policy={"type": "max_violations", "value": 1},
    )
    evaluation = evaluate_profiles(frame, [profile])
    assert evaluation.passed("test").tolist() == [True, False]
    assert evaluation.violation_count("test").tolist() == [1, 2]


def test_soft_rules_do_not_block_approval():
    frame = descriptor_frame({"mol_wt": [600.0]})
    profile = make_profile(
        [Rule("mw", "test", "mol_wt", "<=", 500, severity="soft", failure_code="MW_HIGH")]
    )
    evaluation = evaluate_profiles(frame, [profile])
    assert evaluation.passed("test").tolist() == [True]
    assert len(evaluation.failures) == 1


def test_missing_column_is_an_explicit_error():
    frame = descriptor_frame({"mol_wt": [100.0]})
    profile = make_profile([Rule("tpsa", "test", "tpsa", "<=", 140, failure_code="TPSA_HIGH")])
    with pytest.raises(KeyError, match="tpsa"):
        evaluate_profiles(frame, [profile])


def test_failure_explanation_names_value_and_limit():
    rule = Rule("mw", "test", "mol_wt", "<=", 500, failure_code="MW_HIGH", label="Molecular weight")
    assert explain_failure(rule, 612.4) == "Molecular weight = 612.4 above limit 500"


def test_unevaluable_descriptor_is_explained_as_such():
    rule = Rule("mw", "test", "mol_wt", "<=", 500, failure_code="MW_HIGH", label="Molecular weight")
    assert "not evaluable" in explain_failure(rule, float("nan"))
