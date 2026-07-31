"""Counterfactual explanations.

Answers one question: what is the smallest change to the *criteria* that would
flip this molecule's classification?

Deliberately limited to the criteria. This version never suggests structural
modifications - it explains the filter, not the chemistry. And it never applies
the change: it states what would happen, leaving the decision with the user.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from smiles2select.rules.engine import Rule

#: The spec caps a counterfactual at three simultaneous changes; beyond that the
#: explanation stops being an explanation.
MAX_SIMULTANEOUS_CHANGES = 3

NO_SIMPLE_COUNTERFACTUAL = (
    "Esta classificação depende de múltiplas regras. Nenhuma alteração única "
    "dentro dos limites configurados muda o resultado."
)

TABLE_COLUMNS = [
    "record_id",
    "current_status",
    "affected_rule",
    "current_threshold",
    "counterfactual_threshold",
    "absolute_delta",
    "relative_delta",
    "policy_change",
    "new_status",
]


@dataclass(frozen=True)
class Counterfactual:
    """One minimal change and what it would do."""

    record_id: int
    affected_rule: str
    current_status: str
    classification_after_change: str
    current_threshold: float | None = None
    counterfactual_threshold: float | None = None
    absolute_delta: float | None = None
    relative_delta: float | None = None
    policy_change: str | None = None

    def describe(self) -> str:
        if self.policy_change:
            return f"A molécula seria {self.classification_after_change} se {self.policy_change}."
        return (
            f"A molécula seria {self.classification_after_change} se o limite de "
            f"{self.affected_rule} fosse alterado de {self.current_threshold:g} "
            f"para {self.counterfactual_threshold:g}."
        )

    def as_row(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "current_status": self.current_status,
            "affected_rule": self.affected_rule,
            "current_threshold": self.current_threshold,
            "counterfactual_threshold": self.counterfactual_threshold,
            "absolute_delta": self.absolute_delta,
            "relative_delta": self.relative_delta,
            "policy_change": self.policy_change,
            "new_status": self.classification_after_change,
        }


def _threshold_change(rule: Rule, observed: float) -> tuple[float, float] | None:
    """Threshold that would just admit ``observed``, and the value to move to."""
    lower, upper = rule.bounds
    if upper is not None and observed > upper:
        return float(upper), float(observed)
    if lower is not None and observed < lower:
        return float(lower), float(observed)
    return None


def for_failed_rules(
    record_id: int,
    failures: pd.DataFrame,
    rules_by_id: dict[str, Rule],
    max_changes: int = MAX_SIMULTANEOUS_CHANGES,
) -> list[Counterfactual]:
    """Threshold moves that would make a rejected molecule pass.

    One entry per broken rule. When a molecule broke more rules than
    ``max_changes``, no counterfactual is offered: the classification is not
    the consequence of a single adjustable limit.
    """
    broken = failures[failures["record_id"] == record_id]
    if broken.empty or len(broken) > max_changes:
        return []

    results: list[Counterfactual] = []
    for row in broken.itertuples():
        rule = rules_by_id.get(row.rule_id)
        if rule is None or rule.is_substructure or pd.isna(row.observed_value):
            continue
        change = _threshold_change(rule, float(row.observed_value))
        if change is None:
            continue
        current, proposed = change
        results.append(
            Counterfactual(
                record_id=record_id,
                affected_rule=rule.id,
                current_status="AUTO_FAIL",
                classification_after_change="aprovada",
                current_threshold=current,
                counterfactual_threshold=proposed,
                absolute_delta=round(proposed - current, 6),
                relative_delta=round((proposed - current) / current, 6) if current else None,
            )
        )
    return sorted(results, key=lambda item: abs(item.absolute_delta or 0.0))


def for_passing_molecule(
    record_id: int,
    margins: pd.DataFrame,
    rules_by_id: dict[str, Rule],
) -> Counterfactual | None:
    """The tightest threshold that would start rejecting a passing molecule.

    The mirror image of the rescue case, and the one that shows how fragile an
    approval is: it names the limit that would have to move the least.
    """
    rows = margins[(margins["record_id"] == record_id) & (margins["normalized_margin"] >= 0)]
    if rows.empty:
        return None
    nearest = rows.sort_values("normalized_margin").iloc[0]
    rule = rules_by_id.get(nearest["rule_id"])
    if rule is None or rule.is_substructure:
        return None

    lower, upper = rule.bounds
    current = upper if upper is not None else lower
    if current is None:
        return None

    observed = float(nearest["observed_value"])
    return Counterfactual(
        record_id=record_id,
        affected_rule=rule.id,
        current_status="AUTO_PASS",
        classification_after_change="reprovada",
        current_threshold=float(current),
        counterfactual_threshold=observed,
        absolute_delta=round(observed - float(current), 6),
        relative_delta=round((observed - float(current)) / float(current), 6) if current else None,
    )


def violation_policy_change(
    record_id: int, failures: pd.DataFrame, profile_id: str, allowed_violations: int
) -> Counterfactual | None:
    """Whether tolerating one more violation would approve the molecule."""
    broken = failures[(failures["record_id"] == record_id) & (failures["profile_id"] == profile_id)]
    needed = len(broken)
    if needed == 0 or needed > allowed_violations + 1:
        return None
    return Counterfactual(
        record_id=record_id,
        affected_rule=profile_id,
        current_status="AUTO_FAIL",
        classification_after_change="aprovada",
        policy_change=(
            f"fosse permitida uma {needed}ª violação em {profile_id} "
            f"(hoje são permitidas {allowed_violations})"
        ),
    )


def explain(
    record_id: int,
    failures: pd.DataFrame,
    margins: pd.DataFrame,
    rules_by_id: dict[str, Rule],
) -> list[Counterfactual] | str:
    """Every simple counterfactual for one molecule, or why there is none."""
    rescue = for_failed_rules(record_id, failures, rules_by_id)
    if rescue:
        return rescue
    if not failures.empty and (failures["record_id"] == record_id).any():
        return NO_SIMPLE_COUNTERFACTUAL
    fragile = for_passing_molecule(record_id, margins, rules_by_id)
    return [fragile] if fragile else NO_SIMPLE_COUNTERFACTUAL


def table(counterfactuals: Sequence[Counterfactual]) -> pd.DataFrame:
    """Rows for the COUNTERFACTUALS sheet."""
    if not counterfactuals:
        return pd.DataFrame(columns=TABLE_COLUMNS)
    return pd.DataFrame([item.as_row() for item in counterfactuals])
