"""Ranking helpers shared by the CLI, the Excel export and the GUI table."""

from __future__ import annotations

import pandas as pd


def rank_by(table: pd.DataFrame, column: str, *, ascending: bool = False) -> pd.Series:
    """1-based ranking over one score column; ties share the best rank."""
    if column not in table.columns:
        raise KeyError(f"cannot rank by '{column}': column not present")
    return table[column].rank(ascending=ascending, method="min").astype("Int64")


def top_n(table: pd.DataFrame, column: str, count: int, *, ascending: bool = False) -> pd.DataFrame:
    if count <= 0:
        raise ValueError("count must be positive")
    return table.sort_values(column, ascending=ascending).head(count)


def top_percentile(
    table: pd.DataFrame, column: str, percentile: float, *, ascending: bool = False
) -> pd.DataFrame:
    """Rows in the best ``percentile`` percent of one score column."""
    if not 0 < percentile <= 100:
        raise ValueError("percentile must lie in (0, 100]")
    quantile = percentile / 100.0 if ascending else 1.0 - percentile / 100.0
    cutoff = table[column].quantile(quantile)
    mask = table[column] <= cutoff if ascending else table[column] >= cutoff
    return table[mask.fillna(False)]
