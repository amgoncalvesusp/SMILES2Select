"""Comparison operators available to rules.

Every operator is vectorized over a pandas Series so a whole library is tested
against a rule in one pass instead of one Python call per molecule x rule.

Missing values fail closed: a descriptor that could not be computed produces
``False`` (rule not satisfied) and is reported as an unevaluable observation,
never silently treated as a pass.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

Comparator = Callable[[pd.Series, Any], pd.Series]


def _as_pair(threshold: Any) -> tuple[float, float]:
    if not isinstance(threshold, Sequence) or isinstance(threshold, (str, bytes)):
        raise ValueError(f"range operator needs [lower, upper], got {threshold!r}")
    if len(threshold) != 2:
        raise ValueError(f"range operator needs exactly two bounds, got {threshold!r}")
    lower, upper = float(threshold[0]), float(threshold[1])
    if lower > upper:
        raise ValueError(f"lower bound {lower} exceeds upper bound {upper}")
    return lower, upper


def _lt(values: pd.Series, threshold: Any) -> pd.Series:
    return values < float(threshold)


def _le(values: pd.Series, threshold: Any) -> pd.Series:
    return values <= float(threshold)


def _gt(values: pd.Series, threshold: Any) -> pd.Series:
    return values > float(threshold)


def _ge(values: pd.Series, threshold: Any) -> pd.Series:
    return values >= float(threshold)


def _eq(values: pd.Series, threshold: Any) -> pd.Series:
    return values == threshold


def _ne(values: pd.Series, threshold: Any) -> pd.Series:
    return (values != threshold) & values.notna()


def _between_inclusive(values: pd.Series, threshold: Any) -> pd.Series:
    lower, upper = _as_pair(threshold)
    return (values >= lower) & (values <= upper)


def _between_exclusive(values: pd.Series, threshold: Any) -> pd.Series:
    lower, upper = _as_pair(threshold)
    return (values > lower) & (values < upper)


def _in(values: pd.Series, threshold: Any) -> pd.Series:
    if not isinstance(threshold, Sequence) or isinstance(threshold, (str, bytes)):
        raise ValueError(f"'in' needs a list of allowed values, got {threshold!r}")
    return values.isin(list(threshold))


def _not_in(values: pd.Series, threshold: Any) -> pd.Series:
    return ~_in(values, threshold) & values.notna()


def _contains_substructure(values: pd.Series, threshold: Any) -> pd.Series:
    """True where the precomputed substructure flag is set.

    The SMARTS match itself runs once per molecule in the chemistry worker; the
    rule layer only reads the resulting boolean column.
    """
    del threshold
    return values.fillna(False).astype(bool)


def _does_not_contain_substructure(values: pd.Series, threshold: Any) -> pd.Series:
    del threshold
    return ~values.fillna(False).astype(bool)


OPERATORS: dict[str, Comparator] = {
    "<": _lt,
    "<=": _le,
    ">": _gt,
    ">=": _ge,
    "==": _eq,
    "!=": _ne,
    "between_inclusive": _between_inclusive,
    "between_exclusive": _between_exclusive,
    "in": _in,
    "not_in": _not_in,
    "contains_substructure": _contains_substructure,
    "does_not_contain_substructure": _does_not_contain_substructure,
}

RANGE_OPERATORS = frozenset({"between_inclusive", "between_exclusive"})
SUBSTRUCTURE_OPERATORS = frozenset({"contains_substructure", "does_not_contain_substructure"})


class UnknownOperatorError(KeyError):
    """Raised when a profile file names an operator that does not exist."""


def get(operator: str) -> Comparator:
    try:
        return OPERATORS[operator]
    except KeyError as exc:
        raise UnknownOperatorError(
            f"unknown operator '{operator}'; supported: {sorted(OPERATORS)}"
        ) from exc


def apply(operator: str, values: pd.Series, threshold: Any) -> pd.Series:
    """Evaluate one condition over a column, with missing values failing closed."""
    comparator = get(operator)
    result = comparator(values, threshold)
    result = result.fillna(False).astype(bool)
    if operator not in SUBSTRUCTURE_OPERATORS:
        # NaN comparisons already yield False; make the intent explicit and
        # protect against operators that propagate NA instead.
        result = result & values.notna()
    return result


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def bounds(operator: str, threshold: Any) -> tuple[float | None, float | None]:
    """Lower/upper limits reported in the failure table for one rule."""
    if operator in RANGE_OPERATORS:
        lower, upper = _as_pair(threshold)
        return lower, upper
    if operator in {"<", "<="}:
        return None, _to_float(threshold)
    if operator in {">", ">="}:
        return _to_float(threshold), None
    return None, None


def describe(operator: str, threshold: Any) -> str:
    """Human-readable form of the condition, used in reports and tooltips."""
    if operator in RANGE_OPERATORS:
        lower, upper = _as_pair(threshold)
        symbol = "<=" if operator == "between_inclusive" else "<"
        return f"{lower:g} {symbol} value {symbol} {upper:g}"
    if operator in SUBSTRUCTURE_OPERATORS:
        prefix = "" if operator == "contains_substructure" else "does not "
        return f"{prefix}contain {threshold}"
    if operator in {"in", "not_in"}:
        return f"{operator.replace('_', ' ')} {list(threshold)}"
    return f"value {operator} {threshold}"


def is_numeric_series(values: pd.Series) -> bool:
    return bool(np.issubdtype(values.dtype, np.number))
