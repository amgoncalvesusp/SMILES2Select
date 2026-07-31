"""Per-molecule robustness.

Summarises the margins of one molecule across every rule it was tested against.
The governing number is the *minimum* margin: a molecule is exactly as robust
as its most fragile criterion, and averaging would hide precisely the rule that
is about to break.

The robustness score is an internal metric of this software, useful for ranking
within one run. It is not a measure of molecular quality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smiles2select.selection_intelligence.margins import MarginStatus

ROBUSTNESS_COLUMNS = (
    "minimum_rule_margin",
    "mean_rule_margin",
    "borderline_rule_count",
    "failed_rule_count",
    "robustness_score",
)


def robustness_score(table: pd.DataFrame) -> pd.Series:
    """Squash the minimum margin into [0, 1], zero for anything that fails.

    A molecule that breaks a rule scores zero regardless of how comfortable its
    other margins are - robustness describes a passing molecule, and a failing
    one is not made safer by the rules it did satisfy.
    """
    minimum = table["minimum_rule_margin"].astype(float)
    scaled = np.tanh(minimum.clip(lower=0) * 3.0)
    return pd.Series(np.where(minimum < 0, 0.0, scaled), index=table.index).round(4)


def robustness_table(margins: pd.DataFrame) -> pd.DataFrame:
    """One row per molecule summarising its margins."""
    if margins.empty:
        return pd.DataFrame(columns=list(ROBUSTNESS_COLUMNS))

    grouped = margins.groupby("record_id")
    table = pd.DataFrame(
        {
            "minimum_rule_margin": grouped["normalized_margin"].min(),
            "mean_rule_margin": grouped["normalized_margin"].mean(),
            "borderline_rule_count": grouped["margin_status"].apply(
                lambda values: int(
                    sum(MarginStatus(value).is_borderline for value in values.dropna())
                )
            ),
            "failed_rule_count": grouped["normalized_margin"].apply(
                lambda values: int((values < 0).sum())
            ),
        }
    )
    table["robustness_score"] = robustness_score(table)
    return table


def attach(target: pd.DataFrame, margins: pd.DataFrame) -> pd.DataFrame:
    """Add the robustness columns to a table indexed by ``record_id``."""
    summary = robustness_table(margins)
    if summary.empty:
        return target
    return target.join(summary, how="left")


def most_fragile(margins: pd.DataFrame, top: int = 20) -> pd.DataFrame:
    """Passing molecules closest to breaking, tightest margin first."""
    if margins.empty:
        return pd.DataFrame(columns=list(ROBUSTNESS_COLUMNS))
    table = robustness_table(margins)
    passing = table[table["minimum_rule_margin"] >= 0]
    return passing.sort_values("minimum_rule_margin").head(top)


def limiting_rule(margins: pd.DataFrame) -> pd.Series:
    """The rule that defines each molecule's robustness."""
    if margins.empty:
        return pd.Series(dtype="object")
    ordered = margins.sort_values("normalized_margin")
    return ordered.groupby("record_id")["rule_id"].first()
