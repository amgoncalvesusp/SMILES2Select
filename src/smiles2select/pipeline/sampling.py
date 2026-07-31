"""Impact analysis on a sample.

Runs the selected profiles over a subset so the user can see, before committing
to the full library, how restrictive each filter is on their own data. The
analysis reports; it never decides which rules to use.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from smiles2select.rules.evaluator import ProfileEvaluation

_SOLE_FAILURE_COLUMNS = ["profile_id", "rule_id", "failure_code", "sole_failures"]


@dataclass(frozen=True)
class ImpactReport:
    """Per-profile and cumulative impact over a sample."""

    sample_size: int
    per_profile: pd.DataFrame
    cumulative: pd.DataFrame
    only_one_rule: pd.DataFrame
    recovered_with_one_violation: pd.DataFrame

    def most_restrictive(self, top: int = 3) -> list[str]:
        ordered = self.per_profile.sort_values("pass_rate")
        return ordered["profile_id"].head(top).tolist()

    def estimated_output(self, library_size: int) -> int:
        """Extrapolate the surviving count to the full library."""
        if self.cumulative.empty or self.sample_size == 0:
            return library_size
        final_rate = float(self.cumulative["pass_rate"].iloc[-1])
        return int(round(final_rate * library_size))


def analyse(evaluation: ProfileEvaluation, profile_ids: list[str]) -> ImpactReport:
    """Build the report shown on the profiles screen."""
    status = evaluation.status
    sample_size = len(status)

    per_profile = pd.DataFrame(
        [
            {
                "profile_id": profile_id,
                "passed": int(status[f"{profile_id}__passed"].sum()),
                "pass_rate": float(status[f"{profile_id}__passed"].mean()) if sample_size else 0.0,
            }
            for profile_id in profile_ids
        ]
    )

    cumulative_rows = []
    surviving = pd.Series(True, index=status.index)
    for profile_id in profile_ids:
        surviving = surviving & status[f"{profile_id}__passed"].astype(bool)
        cumulative_rows.append(
            {
                "profile_id": profile_id,
                "surviving": int(surviving.sum()),
                "pass_rate": float(surviving.mean()) if sample_size else 0.0,
            }
        )

    return ImpactReport(
        sample_size=sample_size,
        per_profile=per_profile,
        cumulative=pd.DataFrame(cumulative_rows),
        only_one_rule=_failed_by_single_rule(evaluation, profile_ids),
        recovered_with_one_violation=_recovered_with_one_violation(status, profile_ids),
    )


def _failed_by_single_rule(evaluation: ProfileEvaluation, profile_ids: list[str]) -> pd.DataFrame:
    """How many molecules each rule rejects on its own.

    A high count for one rule means that rule alone drives the filter, which is
    worth knowing before running a million structures.
    """
    failures = evaluation.failures
    if failures.empty:
        return pd.DataFrame(columns=_SOLE_FAILURE_COLUMNS)

    relevant = failures[failures["profile_id"].isin(profile_ids)]
    per_record = relevant.groupby("record_id").size()
    sole = relevant[relevant["record_id"].isin(per_record[per_record == 1].index)]
    if sole.empty:
        return pd.DataFrame(columns=_SOLE_FAILURE_COLUMNS)
    return (
        sole.groupby(["profile_id", "rule_id", "failure_code"])
        .size()
        .reset_index(name="sole_failures")
        .sort_values("sole_failures", ascending=False)
    )


def _recovered_with_one_violation(status: pd.DataFrame, profile_ids: list[str]) -> pd.DataFrame:
    """Molecules that would return if a profile tolerated a single violation."""
    return pd.DataFrame(
        [
            {
                "profile_id": profile_id,
                "recovered": int((status[f"{profile_id}__violations"] == 1).sum()),
            }
            for profile_id in profile_ids
        ]
    )
