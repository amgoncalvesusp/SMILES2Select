"""QED handling.

QED is a continuous desirability score in [0, 1], not a rule. The application
imposes no universal cut-off: the user may compute it, rank by it, take a top
percentile, or apply a threshold they choose themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

QedMode = Literal["compute", "rank", "top_percentile", "threshold"]


@dataclass(frozen=True)
class QedSelection:
    """How QED participates in a run."""

    mode: QedMode = "compute"
    threshold: float | None = None
    percentile: float | None = None

    def __post_init__(self) -> None:
        if self.mode == "threshold" and self.threshold is None:
            raise ValueError("QED threshold mode requires an explicit threshold")
        if self.mode == "top_percentile":
            if self.percentile is None:
                raise ValueError("QED top_percentile mode requires a percentile")
            if not 0 < self.percentile <= 100:
                raise ValueError("QED percentile must lie in (0, 100]")

    @property
    def excludes(self) -> bool:
        """True when this mode can remove molecules from the selection."""
        return self.mode in {"threshold", "top_percentile"}

    def describe(self) -> str:
        if self.mode == "compute":
            return "QED calculado e reportado (sem limite)"
        if self.mode == "rank":
            return "QED usado para ranqueamento"
        if self.mode == "top_percentile":
            return f"QED: percentil superior {self.percentile:g}%"
        return f"QED >= {self.threshold:g}"


def rank(qed_values: pd.Series, *, ascending: bool = False) -> pd.Series:
    """Dense ranking, best QED first by default."""
    return qed_values.rank(ascending=ascending, method="min").astype("Int64")


def percentile(qed_values: pd.Series) -> pd.Series:
    """Percentile of each value within the library (0-100)."""
    return qed_values.rank(pct=True) * 100.0


def passes(qed_values: pd.Series, selection: QedSelection) -> pd.Series:
    """Boolean mask implementing the selected QED mode.

    ``compute`` and ``rank`` never exclude anything - they return all True, so
    a user who only wanted a ranking cannot lose molecules by accident.
    """
    if not selection.excludes:
        return pd.Series(True, index=qed_values.index)
    if selection.mode == "threshold":
        return (qed_values >= float(selection.threshold)).fillna(False)
    cutoff = qed_values.quantile(1.0 - float(selection.percentile) / 100.0)
    return (qed_values >= cutoff).fillna(False)


def distribution(qed_values: pd.Series, bins: int = 10) -> pd.DataFrame:
    """Histogram of QED over [0, 1], for the summary sheet and the GUI charts."""
    clean = pd.to_numeric(qed_values, errors="coerce").dropna()
    if clean.empty:
        return pd.DataFrame(columns=["bin_lower", "bin_upper", "count"])
    edges = [index / bins for index in range(bins + 1)]
    grouped = pd.cut(clean, bins=edges, include_lowest=True).value_counts().sort_index()
    # The nominal edges are reported rather than the interval bounds: pandas
    # shifts the first left edge slightly below zero to make it inclusive.
    return pd.DataFrame(
        {
            "bin_lower": edges[:-1],
            "bin_upper": edges[1:],
            "count": grouped.to_numpy(),
        }
    )
