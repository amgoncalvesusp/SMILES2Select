"""Rule definitions.

A rule is exactly one condition on one descriptor. Profiles are groups of
rules; nothing else in the system is allowed to hide a second condition inside
a single rule, because the report has to be able to name precisely which
threshold a molecule crossed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smiles2select.rules import operators


@dataclass(frozen=True)
class Rule:
    """One condition, with the metadata needed to explain a failure."""

    id: str
    profile_id: str
    descriptor: str
    operator: str
    threshold: Any = None
    severity: str = "hard"
    failure_code: str = ""
    label: str = ""
    reference: str = ""
    smarts: str | None = None

    def __post_init__(self) -> None:
        operators.get(self.operator)  # raises UnknownOperatorError on a typo
        if self.severity not in {"hard", "soft"}:
            raise ValueError(f"rule {self.id}: severity must be 'hard' or 'soft'")
        if not self.descriptor:
            raise ValueError(f"rule {self.id}: descriptor is required")
        if self.operator in operators.SUBSTRUCTURE_OPERATORS and not self.smarts:
            raise ValueError(f"rule {self.id}: substructure operator requires a SMARTS pattern")

    @property
    def is_substructure(self) -> bool:
        return self.operator in operators.SUBSTRUCTURE_OPERATORS

    @property
    def bounds(self) -> tuple[float | None, float | None]:
        return operators.bounds(self.operator, self.threshold)

    def describe(self) -> str:
        label = self.label or self.descriptor
        return f"{label}: {operators.describe(self.operator, self.threshold)}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "profile_id": self.profile_id,
            "descriptor": self.descriptor,
            "operator": self.operator,
            "threshold": self.threshold,
            "severity": self.severity,
            "failure_code": self.failure_code or self.id.upper(),
            "label": self.label,
            "reference": self.reference,
            "smarts": self.smarts,
        }


@dataclass(frozen=True)
class RuleResult:
    """Outcome of one rule for one molecule."""

    rule_id: str
    passed: bool
    observed_value: float | int | None
    threshold: Any
    failure_code: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "passed": self.passed,
            "observed_value": self.observed_value,
            "threshold": self.threshold,
            "failure_code": self.failure_code,
        }


def rule_from_dict(payload: dict[str, Any], profile_id: str) -> Rule:
    """Build a :class:`Rule` from its JSON representation."""
    try:
        rule_id = payload["id"]
    except KeyError as exc:
        raise ValueError(f"profile {profile_id}: a rule is missing its 'id'") from exc
    return Rule(
        id=rule_id,
        profile_id=payload.get("profile_id", profile_id),
        descriptor=payload.get("descriptor", ""),
        operator=payload.get("operator", "<="),
        threshold=payload.get("threshold"),
        severity=payload.get("severity", "hard"),
        failure_code=payload.get("failure_code", rule_id.upper()),
        label=payload.get("label", ""),
        reference=payload.get("reference", ""),
        smarts=payload.get("smarts"),
    )
