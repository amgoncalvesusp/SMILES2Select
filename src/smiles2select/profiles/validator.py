"""Validation of profiles against the descriptor registry.

Catches the mistakes that would otherwise surface only after an hour of
computation: a descriptor typo, an operator that needs two bounds but got one,
a duplicated rule id, or a threshold that cannot be compared numerically.
"""

from __future__ import annotations

from collections.abc import Iterable

from smiles2select.chemistry.descriptor_registry import DescriptorRegistry
from smiles2select.profiles.registry import CATEGORIES, Profile
from smiles2select.rules import operators

_NON_NUMERIC_OPERATORS = frozenset({"in", "not_in", "==", "!="})


class ProfileValidationError(ValueError):
    """Raised when a profile cannot be used for a run."""


def validate_profile(profile: Profile, descriptors: DescriptorRegistry) -> list[str]:
    """Return a list of problems; an empty list means the profile is usable."""
    issues: list[str] = []

    if profile.category not in CATEGORIES:
        issues.append(f"unknown category '{profile.category}' (expected one of {list(CATEGORIES)})")

    seen_rule_ids: set[str] = set()
    for rule in profile.rules:
        if rule.id in seen_rule_ids:
            issues.append(f"duplicated rule id '{rule.id}'")
        seen_rule_ids.add(rule.id)

        if rule.profile_id != profile.id:
            issues.append(f"rule '{rule.id}' claims profile '{rule.profile_id}'")

        if not rule.is_substructure and rule.descriptor not in descriptors:
            issues.append(f"rule '{rule.id}' uses unregistered descriptor '{rule.descriptor}'")

        if rule.operator in operators.RANGE_OPERATORS:
            try:
                operators.bounds(rule.operator, rule.threshold)
            except ValueError as exc:
                issues.append(f"rule '{rule.id}': {exc}")
        elif (
            rule.operator not in operators.SUBSTRUCTURE_OPERATORS
            and rule.operator not in _NON_NUMERIC_OPERATORS
            and not isinstance(rule.threshold, (int, float))
        ):
            issues.append(
                f"rule '{rule.id}': operator '{rule.operator}' needs a numeric threshold, "
                f"got {rule.threshold!r}"
            )

        if not rule.failure_code:
            issues.append(f"rule '{rule.id}': failure_code is empty")

    if profile.pass_policy.get("type") == "max_violations":
        allowed = profile.pass_policy.get("value")
        if not isinstance(allowed, int) or isinstance(allowed, bool) or allowed < 0:
            issues.append(
                f"pass policy max_violations needs a non-negative integer, got {allowed!r}"
            )
        elif allowed >= len(profile.rules):
            issues.append(
                f"pass policy allows {allowed} violations for {len(profile.rules)} rules, "
                "which makes the profile impossible to fail"
            )

    return issues


def validate_all(
    profiles: Iterable[Profile], descriptors: DescriptorRegistry, *, raise_on_error: bool = True
) -> dict[str, list[str]]:
    """Validate several profiles, optionally raising on the first broken one."""
    report = {profile.id: validate_profile(profile, descriptors) for profile in profiles}
    broken = {profile_id: issues for profile_id, issues in report.items() if issues}
    if broken and raise_on_error:
        lines = [f"{profile_id}: {'; '.join(issues)}" for profile_id, issues in broken.items()]
        raise ProfileValidationError("invalid profile(s):\n" + "\n".join(lines))
    return report
