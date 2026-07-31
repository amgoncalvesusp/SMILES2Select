"""The borderline panel.

Molecules sitting just inside or just outside a limit are where a threshold
decision actually costs something. This module assembles the table that lets a
human look at them one by one instead of trusting the cut-off blindly.

It reports; it never reclassifies. A molecule listed here keeps whatever the
rules decided about it.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

import pandas as pd

from smiles2select.selection_intelligence.margins import MarginStatus
from smiles2select.selection_intelligence.robustness import robustness_table

PANEL_COLUMNS = (
    "record_id",
    "molecule_id",
    "smiles",
    "closest_rule",
    "observed_value",
    "threshold",
    "margin",
    "margin_status",
    "borderline_rule_count",
    "pareto_rank",
    "qed",
    "scaffold",
)

_CLOSEST_COLUMNS = [
    "record_id",
    "closest_rule",
    "observed_value",
    "threshold",
    "margin",
    "margin_status",
]

_BORDERLINE_STATUSES = frozenset(
    {MarginStatus.NEAR_PASS_LIMIT.value, MarginStatus.NEAR_FAIL_LIMIT.value}
)


class BorderlineSort(str, Enum):
    """Orderings the panel offers."""

    TIGHTEST_MARGIN = "menor margem"
    MOST_BORDERLINE_RULES = "mais critérios limítrofes"
    BEST_PARETO = "melhor Pareto rank"
    HIGHEST_QED = "maior QED"
    LOWEST_SA = "menor SA Score"


_SORT_KEYS: dict[BorderlineSort, tuple[str, bool]] = {
    BorderlineSort.TIGHTEST_MARGIN: ("margin", True),
    BorderlineSort.MOST_BORDERLINE_RULES: ("borderline_rule_count", False),
    BorderlineSort.BEST_PARETO: ("pareto_rank", True),
    BorderlineSort.HIGHEST_QED: ("qed", False),
    BorderlineSort.LOWEST_SA: ("sa_score", True),
}


def closest_rule(margins: pd.DataFrame) -> pd.DataFrame:
    """For each molecule, the rule it is nearest to breaking (or has broken)."""
    if margins.empty:
        return pd.DataFrame(columns=_CLOSEST_COLUMNS)
    ordered = margins.sort_values("normalized_margin", kind="stable")
    nearest = ordered.groupby("record_id", as_index=False).first()
    return pd.DataFrame(
        {
            "record_id": nearest["record_id"],
            "closest_rule": nearest["rule_id"],
            "observed_value": nearest["observed_value"],
            "threshold": nearest["threshold_high"].fillna(nearest["threshold_low"]),
            "margin": nearest["normalized_margin"],
            "margin_status": nearest["margin_status"],
        }
    )


def borderline_ids(margins: pd.DataFrame) -> pd.Index:
    """Molecules with at least one rule inside the borderline band."""
    if margins.empty:
        return pd.Index([], name="record_id")
    hits = margins[margins["margin_status"].isin(_BORDERLINE_STATUSES)]
    return pd.Index(sorted(hits["record_id"].unique()), name="record_id")


def build_panel(
    margins: pd.DataFrame,
    context: pd.DataFrame | None = None,
    only_borderline: bool = True,
) -> pd.DataFrame:
    """The borderline table, joined with whatever context is available.

    ``context`` is any per-record table - identifiers, QED, Pareto rank,
    scaffold. Missing columns are simply absent from the result rather than
    invented.
    """
    if margins.empty:
        return pd.DataFrame(columns=list(PANEL_COLUMNS))

    panel = closest_rule(margins).set_index("record_id")
    summary = robustness_table(margins)
    panel = panel.join(summary[["borderline_rule_count", "failed_rule_count"]], how="left")

    if context is not None and not context.empty:
        panel = panel.join(context, how="left")

    if only_borderline:
        panel = panel.loc[panel.index.intersection(borderline_ids(margins))]

    return panel.reset_index()


def sort_panel(panel: pd.DataFrame, order: BorderlineSort) -> pd.DataFrame:
    """Apply one of the panel's orderings, ignoring one it cannot compute."""
    column, ascending = _SORT_KEYS[order]
    if column not in panel.columns:
        return panel
    return panel.sort_values(column, ascending=ascending, kind="stable")


def rules_at_risk(margins: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Rules with the most molecules sitting in their borderline band.

    A rule appearing here is one whose exact value is deciding the outcome for
    many molecules - the first place to look when calibrating a threshold.
    """
    empty = pd.DataFrame(columns=["rule_id", "borderline_molecules"])
    if margins.empty:
        return empty
    hits = margins[margins["margin_status"].isin(_BORDERLINE_STATUSES)]
    if hits.empty:
        return empty
    return (
        hits.groupby("rule_id")
        .size()
        .sort_values(ascending=False)
        .head(top)
        .reset_index(name="borderline_molecules")
    )


def near_misses(margins: pd.DataFrame, record_ids: Sequence[int] | None = None) -> pd.DataFrame:
    """Molecules that failed, but only just.

    These are the candidates a small, deliberate threshold change would
    recover - the input to the counterfactual view.
    """
    if margins.empty:
        return pd.DataFrame(columns=_CLOSEST_COLUMNS)
    failing = margins[margins["margin_status"] == MarginStatus.NEAR_FAIL_LIMIT.value]
    if record_ids is not None:
        failing = failing[failing["record_id"].isin(record_ids)]
    if failing.empty:
        return pd.DataFrame(columns=_CLOSEST_COLUMNS)
    return closest_rule(failing).sort_values("margin", ascending=False)
