"""Vectorized evaluation of profiles over a descriptor table.

One pass per rule over the whole library, not one Python call per
molecule x rule. For a 10^6-row library and 30 rules this is 30 vector
operations instead of 3 x 10^7 function calls.

Only broken rules are materialised (:attr:`ProfileEvaluation.failures`); storing
one row per satisfied rule would dwarf the useful output.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from smiles2select.rules import operators
from smiles2select.rules.engine import Rule

#: Rules per profile that still fit a signed 64-bit failure mask.
MAX_MASK_BITS = 63

FAILURE_COLUMNS = (
    "record_id",
    "profile_id",
    "rule_id",
    "failure_code",
    "observed_value",
    "lower_limit",
    "upper_limit",
)


@dataclass(frozen=True)
class ProfileEvaluation:
    """Per-profile verdicts plus the sparse table of broken rules."""

    status: pd.DataFrame
    failures: pd.DataFrame

    def passed(self, profile_id: str) -> pd.Series:
        return self.status[f"{profile_id}__passed"]

    def violation_count(self, profile_id: str) -> pd.Series:
        return self.status[f"{profile_id}__violations"]

    def failure_mask(self, profile_id: str) -> pd.Series:
        """Bitmask of broken rules, in the profile's own rule order."""
        return self.status[f"{profile_id}__mask"]

    def profile_ids(self) -> list[str]:
        suffix = "__passed"
        return [column[: -len(suffix)] for column in self.status.columns if column.endswith(suffix)]


def evaluate_profiles(
    descriptors: pd.DataFrame,
    profiles: Sequence[object],
) -> ProfileEvaluation:
    """Evaluate every profile against a descriptor table.

    ``descriptors`` is indexed by ``record_id``; its columns are descriptor ids
    plus, when substructure rules are active, boolean ``smarts__<rule_id>``
    columns produced by the chemistry workers.
    """
    status = pd.DataFrame(index=descriptors.index)
    failure_frames: list[pd.DataFrame] = []

    for profile in profiles:
        rules: tuple[Rule, ...] = tuple(getattr(profile, "rules", ()))
        hard_violations = pd.Series(0, index=descriptors.index, dtype="int64")
        soft_violations = pd.Series(0, index=descriptors.index, dtype="int64")
        # Bit i of the mask marks rule i of this profile as broken. SQLite
        # stores 64-bit integers, so the mask is only filled for profiles with
        # at most 63 rules; beyond that it stays 0 and rule_failures remains
        # the authoritative record.
        failure_mask = pd.Series(0, index=descriptors.index, dtype="int64")
        maskable = len(rules) <= MAX_MASK_BITS

        for position, rule in enumerate(rules):
            column = _column_for(rule)
            if column not in descriptors.columns:
                raise KeyError(
                    f"profile {profile.id}: rule {rule.id} needs column '{column}', "
                    f"which is not in the descriptor table"
                )
            values = descriptors[column]
            satisfied = operators.apply(rule.operator, values, rule.threshold)
            broken = ~satisfied
            if rule.severity == "hard":
                hard_violations = hard_violations + broken.astype("int64")
            else:
                soft_violations = soft_violations + broken.astype("int64")

            if maskable:
                failure_mask = failure_mask + broken.astype("int64") * (1 << position)

            if broken.any():
                failure_frames.append(_failure_frame(rule, profile.id, values, broken))

        status[f"{profile.id}__passed"] = _apply_pass_policy(profile, hard_violations)
        status[f"{profile.id}__violations"] = hard_violations
        status[f"{profile.id}__soft_violations"] = soft_violations
        status[f"{profile.id}__mask"] = failure_mask

    failures = (
        pd.concat(failure_frames, ignore_index=True)
        if failure_frames
        else pd.DataFrame(columns=list(FAILURE_COLUMNS))
    )
    return ProfileEvaluation(status=status, failures=failures)


def _column_for(rule: Rule) -> str:
    return f"smarts__{rule.id}" if rule.is_substructure else rule.descriptor


def _failure_frame(
    rule: Rule, profile_id: str, values: pd.Series, broken: pd.Series
) -> pd.DataFrame:
    lower, upper = rule.bounds
    observed = values[broken]
    return pd.DataFrame(
        {
            "record_id": observed.index.to_numpy(),
            "profile_id": profile_id,
            "rule_id": rule.id,
            "failure_code": rule.failure_code or rule.id.upper(),
            "observed_value": pd.to_numeric(observed.to_numpy(), errors="coerce"),
            "lower_limit": lower if lower is not None else np.nan,
            "upper_limit": upper if upper is not None else np.nan,
        }
    )


def _apply_pass_policy(profile: object, hard_violations: pd.Series) -> pd.Series:
    """Translate a profile's pass policy into a boolean verdict.

    ``all_rules`` demands zero hard violations; ``max_violations`` implements
    the classical Lipinski reading that tolerates a single violation.
    """
    policy = getattr(profile, "pass_policy", None) or {"type": "all_rules"}
    policy_type = policy.get("type", "all_rules")
    if policy_type == "all_rules":
        return hard_violations == 0
    if policy_type == "max_violations":
        allowed = int(policy.get("value", 0))
        return hard_violations <= allowed
    raise ValueError(f"profile {getattr(profile, 'id', '?')}: unknown pass policy '{policy_type}'")
