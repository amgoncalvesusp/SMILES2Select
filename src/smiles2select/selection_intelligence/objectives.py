"""Pareto objectives.

An objective says which column matters and in which direction. Everything
downstream works on a single convention - **higher is better** - so the
dominance code never has to know whether a column was to be maximised,
minimised or kept inside a range.

Two deliberate constraints:

* the original value is never modified; desirability is computed alongside it,
  so a report can always show the molecule's real TPSA next to its score;
* ``weight`` never changes dominance. Pareto dominance is a formal relation;
  weighting it would quietly turn it into a single arbitrary score, which is
  exactly what Pareto analysis exists to avoid. Weights only break ties.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class Direction(str, Enum):
    """What "better" means for one column."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    TARGET_RANGE = "target_range"
    TARGET_VALUE = "target_value"


class ObjectiveError(ValueError):
    """Raised when an objective is inconsistent or names a missing column."""


@dataclass(frozen=True)
class Objective:
    """One optimisation direction on one numeric column."""

    field: str
    direction: Direction = Direction.MAXIMIZE
    weight: float = 1.0
    enabled: bool = True
    transform: str = "none"
    target_low: float | None = None
    target_high: float | None = None
    target_value: float | None = None

    def __post_init__(self) -> None:
        if not self.field:
            raise ObjectiveError("objective needs a field")
        if self.weight <= 0:
            raise ObjectiveError(f"{self.field}: weight must be positive")
        if self.direction is Direction.TARGET_RANGE:
            if self.target_low is None or self.target_high is None:
                raise ObjectiveError(f"{self.field}: target_range needs target_low and target_high")
            if self.target_low > self.target_high:
                raise ObjectiveError(f"{self.field}: target_low is above target_high")
        if self.direction is Direction.TARGET_VALUE and self.target_value is None:
            raise ObjectiveError(f"{self.field}: target_value is required")

    @property
    def label(self) -> str:
        if self.direction is Direction.TARGET_RANGE:
            return f"{self.field} em [{self.target_low:g}, {self.target_high:g}]"
        if self.direction is Direction.TARGET_VALUE:
            return f"{self.field} ≈ {self.target_value:g}"
        verb = "maximizar" if self.direction is Direction.MAXIMIZE else "minimizar"
        return f"{verb} {self.field}"

    def desirability(self, values: pd.Series) -> pd.Series:
        """Map the column onto "higher is better", without touching the original.

        Missing values become the worst possible score rather than being
        dropped: a molecule whose objective could not be computed must never
        dominate one whose value is known.
        """
        numeric = pd.to_numeric(values, errors="coerce")
        if self.direction is Direction.MAXIMIZE:
            scored = numeric
        elif self.direction is Direction.MINIMIZE:
            scored = -numeric
        elif self.direction is Direction.TARGET_RANGE:
            below = (self.target_low - numeric).clip(lower=0)
            above = (numeric - self.target_high).clip(lower=0)
            scored = -(below + above)
        else:
            scored = -(numeric - float(self.target_value)).abs()
        worst = scored.min() - 1.0 if scored.notna().any() else 0.0
        return scored.fillna(worst)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "field": self.field,
            "direction": self.direction.value,
            "weight": self.weight,
            "enabled": self.enabled,
            "transform": self.transform,
        }
        for key in ("target_low", "target_high", "target_value"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload


def objective_from_dict(payload: dict[str, Any]) -> Objective:
    try:
        return Objective(
            field=payload["field"],
            direction=Direction(payload.get("direction", "maximize")),
            weight=float(payload.get("weight", 1.0)),
            enabled=bool(payload.get("enabled", True)),
            transform=payload.get("transform", "none"),
            target_low=payload.get("target_low"),
            target_high=payload.get("target_high"),
            target_value=payload.get("target_value"),
        )
    except KeyError as exc:
        raise ObjectiveError(f"objective is missing {exc}") from exc
    except ValueError as exc:
        raise ObjectiveError(str(exc)) from exc


#: Beyond this, most molecules end up non-dominated and the ranking stops
#: discriminating; the interface warns instead of silently degrading.
MAX_USEFUL_OBJECTIVES = 8


class ObjectiveSet:
    """The objectives in force, and the matrix they produce."""

    def __init__(self, objectives: Iterable[Objective] = ()) -> None:
        self._objectives = tuple(objectives)
        fields = [objective.field for objective in self._objectives]
        duplicated = {field for field in fields if fields.count(field) > 1}
        if duplicated:
            raise ObjectiveError(f"repeated objective field(s): {sorted(duplicated)}")

    def __len__(self) -> int:
        return len(self.active)

    def __iter__(self):
        return iter(self._objectives)

    @property
    def all(self) -> tuple[Objective, ...]:
        return self._objectives

    @property
    def active(self) -> tuple[Objective, ...]:
        return tuple(objective for objective in self._objectives if objective.enabled)

    def fields(self) -> tuple[str, ...]:
        return tuple(objective.field for objective in self.active)

    def validate(self, columns: Sequence[str]) -> list[str]:
        problems = [f"coluna ausente: {field}" for field in self.fields() if field not in columns]
        if not self.active:
            problems.append("nenhum objetivo ativo")
        if len(self.active) > MAX_USEFUL_OBJECTIVES:
            problems.append(
                f"{len(self.active)} objetivos ativos; o máximo suportado é {MAX_USEFUL_OBJECTIVES}"
            )
        return problems

    def matrix(self, frame: pd.DataFrame) -> np.ndarray:
        """Desirability matrix, one column per active objective, higher is better."""
        problems = self.validate(list(frame.columns))
        if problems:
            raise ObjectiveError("; ".join(problems))
        columns = [
            objective.desirability(frame[objective.field]).to_numpy(dtype=float)
            for objective in self.active
        ]
        return np.column_stack(columns)

    def weights(self) -> np.ndarray:
        return np.array([objective.weight for objective in self.active], dtype=float)

    def as_dicts(self) -> tuple[dict[str, Any], ...]:
        return tuple(objective.as_dict() for objective in self._objectives)

    def diagnostics(self, frame: pd.DataFrame) -> list[str]:
        """Warnings about objectives that cannot discriminate.

        A near-constant objective adds no information but does add a dimension,
        which inflates the first front; two highly correlated objectives count
        the same preference twice.
        """
        warnings: list[str] = []
        active = self.active
        for objective in active:
            values = pd.to_numeric(frame[objective.field], errors="coerce").dropna()
            if values.empty:
                warnings.append(f"objetivo '{objective.field}' não tem valores calculados")
            elif float(values.std(ddof=0)) < 1e-9:
                warnings.append(
                    f"objetivo '{objective.field}' é praticamente constante "
                    "e não diferencia moléculas"
                )

        for first in range(len(active)):
            for second in range(first + 1, len(active)):
                pair = frame[[active[first].field, active[second].field]].apply(
                    pd.to_numeric, errors="coerce"
                )
                if pair.dropna().shape[0] < 3:
                    continue
                correlation = pair.corr().iloc[0, 1]
                if pd.notna(correlation) and abs(correlation) > 0.95:
                    warnings.append(
                        f"objetivos '{active[first].field}' e '{active[second].field}' são "
                        f"redundantes (correlação {correlation:.2f})"
                    )
        return warnings


def objectives_from_dicts(payload: Iterable[dict[str, Any]]) -> ObjectiveSet:
    return ObjectiveSet(objective_from_dict(item) for item in payload)
