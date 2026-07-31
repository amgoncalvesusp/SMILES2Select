"""Human-readable renderings of rule failures.

The wording used in the Excel report, the GUI table and the CLI is defined once
here so a molecule's exclusion always reads the same way wherever it appears.
"""

from __future__ import annotations

import pandas as pd

from smiles2select.rules.engine import Rule


def explain_failure(rule: Rule, observed_value: float | int | None) -> str:
    """One sentence naming the descriptor, the observation and the limit."""
    label = rule.label or rule.descriptor
    if observed_value is None or pd.isna(observed_value):
        return f"{label}: not evaluable (descriptor could not be computed)"
    lower, upper = rule.bounds
    if lower is not None and upper is not None:
        return f"{label} = {observed_value:g} outside [{lower:g}, {upper:g}]"
    if upper is not None:
        return f"{label} = {observed_value:g} above limit {upper:g}"
    if lower is not None:
        return f"{label} = {observed_value:g} below limit {lower:g}"
    return f"{label} = {observed_value:g} violates {rule.describe()}"


def summarise_failures(
    failures: pd.DataFrame, rules_by_id: dict[str, Rule], separator: str = "; "
) -> pd.Series:
    """Collapse the sparse failure table into one sentence per record."""
    if failures.empty:
        return pd.Series(dtype="object")

    def _row(row: pd.Series) -> str:
        rule = rules_by_id.get(row["rule_id"])
        if rule is None:
            return f"{row['failure_code']} = {row['observed_value']}"
        return explain_failure(rule, row["observed_value"])

    rendered = failures.assign(text=failures.apply(_row, axis=1))
    return rendered.groupby("record_id")["text"].apply(lambda texts: separator.join(texts))


def failed_profiles(failures: pd.DataFrame, separator: str = ", ") -> pd.Series:
    """Comma-separated list of the profiles a record broke at least one rule in."""
    if failures.empty:
        return pd.Series(dtype="object")
    return failures.groupby("record_id")["profile_id"].apply(
        lambda ids: separator.join(sorted(set(ids)))
    )


def failed_rule_codes(failures: pd.DataFrame, separator: str = ", ") -> pd.Series:
    if failures.empty:
        return pd.Series(dtype="object")
    return failures.groupby("record_id")["failure_code"].apply(
        lambda codes: separator.join(sorted(set(codes)))
    )
