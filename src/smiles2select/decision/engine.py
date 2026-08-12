"""Final selection.

Every exclusion is recorded with its reason, so a molecule that did not make
the cut can always be traced back to the specific mandatory profile, alert
catalogue or score threshold that removed it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from smiles2select.alerts import engine as alert_engine
from smiles2select.decision import expression_parser
from smiles2select.decision.policies import DecisionPolicy
from smiles2select.scores import qed as qed_scores

DECISION_COLUMNS = (
    "record_id",
    "selected",
    "decision_policy_id",
    "hard_failure_count",
    "alert_count",
    "consensus_score",
    "exclusion_reasons",
)


@dataclass(frozen=True)
class DecisionResult:
    """Per-record verdict plus the reasons behind it."""

    decisions: pd.DataFrame

    @property
    def selected_count(self) -> int:
        return int(self.decisions["selected"].sum())

    @property
    def excluded_count(self) -> int:
        return int((~self.decisions["selected"].astype(bool)).sum())

    def selected_ids(self) -> pd.Index:
        return self.decisions.index[self.decisions["selected"].astype(bool)]


class DecisionEngine:
    """Applies a :class:`DecisionPolicy` to the results of one run."""

    def __init__(self, policy: DecisionPolicy) -> None:
        self.policy = policy

    def decide(
        self,
        status: pd.DataFrame,
        scores: pd.DataFrame,
        alerts: pd.DataFrame | None = None,
    ) -> DecisionResult:
        index = status.index
        selected = pd.Series(True, index=index)
        reasons = pd.Series([[] for _ in range(len(index))], index=index, dtype="object")

        selected, reasons = self._apply_mandatory(status, selected, reasons)
        selected, reasons = self._apply_consensus(status, selected, reasons)
        selected, reasons = self._apply_qed(scores, selected, reasons)
        selected, reasons = self._apply_alerts(alerts, index, selected, reasons)
        selected, reasons = self._apply_expression(status, scores, alerts, index, selected, reasons)

        decisions = pd.DataFrame(
            {
                "selected": selected.astype(bool),
                "decision_policy_id": self.policy.id,
                "hard_failure_count": scores.get(
                    "hard_violation_count", pd.Series(0, index=index)
                ).astype("int64"),
                "alert_count": scores.get("alert_count", pd.Series(0, index=index)).astype("int64"),
                "consensus_score": scores.get(
                    "consensus_score", pd.Series(float("nan"), index=index)
                ),
                "exclusion_reasons": reasons.apply(lambda items: "; ".join(items)),
            },
            index=index,
        )
        return DecisionResult(decisions=decisions)

    def _apply_mandatory(
        self, status: pd.DataFrame, selected: pd.Series, reasons: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        for profile_id in self.policy.mandatory_profiles():
            column = f"{profile_id}__passed"
            if column not in status.columns:
                raise KeyError(f"mandatory profile '{profile_id}' was not evaluated")
            failed = ~status[column].astype(bool)
            selected = selected & ~failed
            reasons = _add_reason(reasons, failed, f"failed mandatory profile {profile_id}")
        return selected, reasons

    def _apply_consensus(
        self, status: pd.DataFrame, selected: pd.Series, reasons: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        consensus_profiles = self.policy.consensus_profiles()
        if not consensus_profiles or self.policy.consensus_min_pass is None:
            return selected, reasons
        columns = [f"{profile_id}__passed" for profile_id in consensus_profiles]
        missing = [column for column in columns if column not in status.columns]
        if missing:
            raise KeyError(f"consensus profiles were not evaluated: {missing}")
        passes = status[columns].astype(bool).sum(axis=1)
        failed = passes < int(self.policy.consensus_min_pass)
        selected = selected & ~failed
        reasons = _add_reason(
            reasons,
            failed,
            f"insufficient consensus (< {self.policy.consensus_min_pass} of {len(columns)})",
        )
        return selected, reasons

    def _apply_qed(
        self, scores: pd.DataFrame, selected: pd.Series, reasons: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        if not self.policy.qed.excludes:
            return selected, reasons
        if "qed" not in scores.columns:
            raise KeyError("the QED policy excludes molecules but QED was not computed")
        passes = qed_scores.passes(scores["qed"], self.policy.qed)
        selected = selected & passes
        reasons = _add_reason(reasons, ~passes, self.policy.qed.describe())
        return selected, reasons

    def _apply_alerts(
        self,
        alerts: pd.DataFrame | None,
        index: pd.Index,
        selected: pd.Series,
        reasons: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        if alerts is None or alerts.empty:
            return selected, reasons
        excluded = alert_engine.exclusion_mask(alerts, index, self.policy.alert_policy)
        if not excluded.any():
            return selected, reasons
        catalogs = ", ".join(self.policy.alert_policy.excluding_catalogs())
        selected = selected & ~excluded
        reasons = _add_reason(reasons, excluded, f"excluding structural alert ({catalogs})")
        return selected, reasons

    def _apply_expression(
        self,
        status: pd.DataFrame,
        scores: pd.DataFrame,
        alerts: pd.DataFrame | None,
        index: pd.Index,
        selected: pd.Series,
        reasons: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        if not self.policy.expression:
            return selected, reasons
        variables = expression_variables(status, scores, alerts, index)
        satisfied = expression_parser.evaluate(self.policy.expression, variables)
        selected = selected & satisfied
        reasons = _add_reason(reasons, ~satisfied, "custom expression not satisfied")
        return selected, reasons


def expression_variables(
    status: pd.DataFrame,
    scores: pd.DataFrame,
    alerts: pd.DataFrame | None,
    index: pd.Index,
) -> pd.DataFrame:
    """Names an expression may reference: profile ids, scores and alert flags."""
    variables = pd.DataFrame(index=index)
    suffix = "__passed"
    for column in status.columns:
        if column.endswith(suffix):
            variables[column[: -len(suffix)]] = status[column].astype(bool)
    for column in ("qed", "consensus_score", "alert_count", "hard_violation_count"):
        if column in scores.columns:
            variables[column] = scores[column]
    if alerts is not None and not alerts.empty:
        for catalog_id in alerts["catalog_id"].unique():
            hits = alerts.loc[alerts["catalog_id"] == catalog_id, "record_id"].unique()
            variables[str(catalog_id)] = variables.index.isin(hits)
    return variables


def _add_reason(reasons: pd.Series, mask: pd.Series, text: str) -> pd.Series:
    """Append ``text`` to the reason list of every record the mask selects."""
    if not mask.any():
        return reasons
    updated = reasons.copy()
    for record_id in reasons.index[mask.astype(bool)]:
        updated.at[record_id] = [*updated.at[record_id], text]
    return updated
