"""Why each molecule is, or is not, in the final set.

Every selected molecule carries the reason it was picked, and every rejected
one the reason it was not. Without that, a final list is an assertion; with it,
it is an argument someone can check.
"""

from __future__ import annotations

import pandas as pd

from smiles2select.selection_intelligence.basket import SelectionBasket
from smiles2select.selection_intelligence.constrained_selection import (
    SelectionConstraints,
    SelectionOutcome,
)
from smiles2select.selection_intelligence.states import SelectionOrigin

MANUAL_COLUMNS = ["ID", "Previous_Status", "New_Status", "Action", "Reason", "Timestamp"]

#: A single scaffold holding this share of the final set is worth flagging.
SCAFFOLD_DOMINANCE = 0.30


def selection_reason(outcome: SelectionOutcome, record_id: int) -> str:
    return "; ".join(outcome.reasons.get(record_id, [])) or "-"


def rejection_reason(outcome: SelectionOutcome, record_id: int) -> str:
    return "; ".join(outcome.rejections.get(record_id, [])) or "-"


def explanation_table(outcome: SelectionOutcome, candidates: pd.Index) -> pd.DataFrame:
    """One row per candidate with its verdict and the reason behind it."""
    selected = set(outcome.selected_ids)
    return pd.DataFrame(
        {
            "record_id": list(candidates),
            "selected": [int(record_id) in selected for record_id in candidates],
            "reason": [
                selection_reason(outcome, int(record_id))
                if int(record_id) in selected
                else rejection_reason(outcome, int(record_id))
                for record_id in candidates
            ],
        }
    ).set_index("record_id")


def manual_decisions(basket: SelectionBasket) -> pd.DataFrame:
    """Rows for the MANUAL_DECISIONS sheet, from the action history.

    Read from the log rather than from the current state: a decision that was
    later undone still belongs in the audit trail.
    """
    rows = []
    for action in basket.log.applied:
        for record_id, after in action.new_state.items():
            before = action.previous_state.get(record_id, {})
            rows.append(
                {
                    "ID": record_id,
                    "Previous_Status": before.get("selection_status", "-"),
                    "New_Status": after.get("selection_status", "-"),
                    "Action": action.action_type.value,
                    "Reason": action.reason or "-",
                    "Timestamp": action.timestamp,
                }
            )
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=MANUAL_COLUMNS)


def final_alerts(
    outcome: SelectionOutcome,
    constraints: SelectionConstraints,
    basket: SelectionBasket | None = None,
    unrepresented_clusters: list[int] | None = None,
    borderline_ids: list[int] | None = None,
) -> list[str]:
    """The warnings shown before exporting.

    Each names a specific way the final set may not be what the user thinks it
    is: too few molecules, one scaffold dominating, fragile approvals, or
    compounds kept against the chemistry.
    """
    messages = list(outcome.warnings(constraints))

    if unrepresented_clusters:
        messages.append(f"{len(unrepresented_clusters)} clusters have no representatives.")

    if outcome.scaffold_usage and outcome.count:
        share = max(outcome.scaffold_usage.values()) / outcome.count
        if share >= SCAFFOLD_DOMINANCE:
            messages.append(f"{share * 100:.0f}% of the selection belongs to one scaffold.")

    if borderline_ids:
        selected = set(outcome.selected_ids)
        overlap = [record_id for record_id in borderline_ids if record_id in selected]
        if overlap:
            messages.append(
                f"{len(overlap)} selected molecules are close to at least one threshold."
            )

    if basket is not None:
        selected = set(outcome.selected_ids)
        overrides = [
            state
            for state in basket.overrides()
            if state.record_id in selected
            or state.origin in {SelectionOrigin.MANUAL, SelectionOrigin.RESCUED}
        ]
        if overrides:
            messages.append(
                f"{len(overrides)} molecules were selected manually despite a chemical failure."
            )

    return messages


def summary_rows(
    outcome: SelectionOutcome, constraints: SelectionConstraints
) -> list[tuple[str, object]]:
    """Counts for the SELECTION_SUMMARY sheet."""
    return [
        ("strategy", outcome.strategy.value),
        ("final selected", outcome.count),
        ("requested", constraints.target_count or "unlimited"),
        ("shortfall", outcome.shortfall(constraints)),
        ("scaffolds covered", outcome.scaffolds_covered),
        ("clusters covered", outcome.clusters_covered),
        ("rejected by quota or limit", len(outcome.rejections)),
    ]
