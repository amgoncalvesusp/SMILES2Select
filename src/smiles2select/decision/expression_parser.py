"""Custom selection expressions.

Supports the documented form::

    (Lipinski AND Veber) AND QED >= 0.50 AND NOT reactive_group

``AND``/``OR``/``NOT`` are rewritten to Python operators and the result is
parsed with :mod:`ast`. Only boolean operations, comparisons and plain names or
numbers survive validation - calls, attributes, subscripts, imports and
comprehensions are rejected, so an expression from a saved configuration file
cannot execute arbitrary code.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from typing import Any

import pandas as pd

_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.BoolOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)

_KEYWORD_PATTERN = re.compile(r"\b(AND|OR|NOT)\b")
_KEYWORD_REPLACEMENTS = {"AND": "and", "OR": "or", "NOT": "not"}


class ExpressionError(ValueError):
    """Raised when an expression is malformed or uses forbidden syntax."""


def normalise(expression: str) -> str:
    """Rewrite the documented AND/OR/NOT keywords into Python operators."""
    return _KEYWORD_PATTERN.sub(lambda match: _KEYWORD_REPLACEMENTS[match.group(0)], expression)


def compile_expression(expression: str) -> ast.Expression:
    """Parse and validate an expression, returning its AST."""
    text = normalise(expression).strip()
    if not text:
        raise ExpressionError("expression is empty")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"invalid expression: {exc.msg}") from exc

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ExpressionError(f"expression uses forbidden syntax: {type(node).__name__}")
    return tree


def variable_names(expression: str) -> tuple[str, ...]:
    """Identifiers referenced by the expression, in sorted order."""
    tree = compile_expression(expression)
    return tuple(sorted({node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}))


def validate(expression: str, known_names: Mapping[str, Any]) -> list[str]:
    """Names used by the expression that the run cannot provide."""
    lowered = {name.lower() for name in known_names}
    return [name for name in variable_names(expression) if name.lower() not in lowered]


def evaluate(expression: str, variables: pd.DataFrame) -> pd.Series:
    """Evaluate the expression per row.

    Note: row-wise evaluation, O(n) Python calls. Vectorising would need
    ``&``/``|``, whose precedence silently breaks mixed comparisons such as
    ``qed >= 0.5 AND flag``. Custom expressions are the rare path; switch to a
    compiled vector form only if profiling shows it matters.
    """
    tree = compile_expression(expression)
    code = compile(tree, filename="<selection-expression>", mode="eval")
    columns = {name.lower(): name for name in variables.columns}
    names = variable_names(expression)

    missing = [name for name in names if name.lower() not in columns]
    if missing:
        raise ExpressionError(
            f"expression references unknown name(s): {missing}; "
            f"available: {sorted(variables.columns)}"
        )

    results: list[bool] = []
    for _, row in variables.iterrows():
        namespace = {name: row[columns[name.lower()]] for name in names}
        try:
            results.append(bool(eval(code, {"__builtins__": {}}, namespace)))  # noqa: S307
        except Exception as exc:  # a NaN comparison or a type mismatch
            raise ExpressionError(f"expression failed on a record: {exc}") from exc
    return pd.Series(results, index=variables.index, dtype=bool)
