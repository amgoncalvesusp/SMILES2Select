"""How far each molecule sits from each threshold.

A pass at 499 Da and a pass at 180 Da are both passes, and the rule engine is
right to treat them the same. For a human deciding what to buy, they are not
the same at all: one survives a re-measurement, the other does not.

The margin is normalised so limits on different scales can be compared. The
denominator is recorded in the recipe, because a margin means nothing without
knowing what it was divided by.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from smiles2select.rules.engine import Rule

MARGIN_COLUMNS = (
    "record_id",
    "rule_id",
    "profile_id",
    "descriptor",
    "observed_value",
    "threshold_low",
    "threshold_high",
    "normalized_margin",
    "margin_status",
)

#: Default width of the borderline band, as a share of the scale.
DEFAULT_BORDERLINE_FRACTION = 0.10


class MarginScale(str, Enum):
    """What the raw distance is divided by."""

    IQR = "iqr_normalized"
    RANGE = "range_normalized"
    THRESHOLD = "threshold_normalized"


class MarginStatus(str, Enum):
    """How safely a molecule sits on its side of the limit."""

    ROBUST_PASS = "ROBUST_PASS"
    NEAR_PASS_LIMIT = "NEAR_PASS_LIMIT"
    NEAR_FAIL_LIMIT = "NEAR_FAIL_LIMIT"
    ROBUST_FAIL = "ROBUST_FAIL"

    @property
    def is_borderline(self) -> bool:
        return self in {MarginStatus.NEAR_PASS_LIMIT, MarginStatus.NEAR_FAIL_LIMIT}

    @property
    def passed(self) -> bool:
        return self in {MarginStatus.ROBUST_PASS, MarginStatus.NEAR_PASS_LIMIT}


@dataclass(frozen=True)
class MarginConfig:
    """Scale and band width, both recorded with the results."""

    scale: MarginScale = MarginScale.IQR
    borderline_fraction: float = DEFAULT_BORDERLINE_FRACTION

    def __post_init__(self) -> None:
        if not 0 < self.borderline_fraction < 1:
            raise ValueError("borderline_fraction must lie in (0, 1)")


def scale_for(values: pd.Series, threshold: float, scale: MarginScale) -> float:
    """Denominator used to normalise the distance to a threshold.

    Falls back to the threshold, then to 1.0, when the chosen statistic
    degenerates - a constant descriptor has no spread to divide by, and a zero
    denominator would turn every margin into infinity.
    """
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    candidate = 0.0
    if not numeric.empty:
        if scale is MarginScale.IQR:
            candidate = float(numeric.quantile(0.75) - numeric.quantile(0.25))
        elif scale is MarginScale.RANGE:
            candidate = float(numeric.max() - numeric.min())
    if scale is MarginScale.THRESHOLD or candidate <= 0:
        candidate = abs(float(threshold))
    return candidate if candidate > 0 else 1.0


def classify(margin: float, borderline_fraction: float) -> MarginStatus:
    """Positive margin means the rule is satisfied; near zero means fragile."""
    if margin >= 0:
        return (
            MarginStatus.NEAR_PASS_LIMIT
            if margin <= borderline_fraction
            else MarginStatus.ROBUST_PASS
        )
    return (
        MarginStatus.NEAR_FAIL_LIMIT if margin >= -borderline_fraction else MarginStatus.ROBUST_FAIL
    )


def rule_margins(
    rule: Rule,
    descriptors: pd.DataFrame,
    config: MarginConfig = MarginConfig(),
    threshold: float | None = None,
) -> pd.DataFrame:
    """Normalised margin of every molecule against one rule.

    For a range rule the margin is the *smaller* of the two distances: a
    molecule is only as safe as its nearest limit.
    """
    if rule.descriptor not in descriptors.columns:
        return pd.DataFrame(columns=list(MARGIN_COLUMNS))

    values = pd.to_numeric(descriptors[rule.descriptor], errors="coerce")
    lower, upper = rule.bounds
    if threshold is not None:
        lower = lower if lower is None else threshold
        upper = upper if upper is None else threshold

    reference = upper if upper is not None else lower
    if reference is None:
        return pd.DataFrame(columns=list(MARGIN_COLUMNS))
    denominator = scale_for(values, float(reference), config.scale)

    if lower is not None and upper is not None:
        raw = np.minimum(values - lower, upper - values)
    elif upper is not None:
        raw = upper - values
    else:
        raw = values - lower

    frame = pd.DataFrame(
        {
            "record_id": descriptors.index,
            "rule_id": rule.id,
            "profile_id": rule.profile_id,
            "descriptor": rule.descriptor,
            "observed_value": values.to_numpy(),
            "threshold_low": lower if lower is not None else np.nan,
            "threshold_high": upper if upper is not None else np.nan,
            "normalized_margin": (raw / denominator).to_numpy(),
        }
    )
    frame["margin_status"] = [
        MarginStatus.ROBUST_FAIL.value
        if pd.isna(value)
        else classify(float(value), config.borderline_fraction).value
        for value in frame["normalized_margin"]
    ]
    return frame


def margin_table(
    rules: Sequence[Rule],
    descriptors: pd.DataFrame,
    config: MarginConfig = MarginConfig(),
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Long table of margins: one row per molecule x rule."""
    overrides = thresholds or {}
    frames = [
        rule_margins(rule, descriptors, config, overrides.get(rule.id))
        for rule in rules
        if not rule.is_substructure
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=list(MARGIN_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def margin_matrix(margins: pd.DataFrame, record_ids: Sequence[int] | None = None) -> pd.DataFrame:
    """Molecules x rules matrix of normalised margins.

    Restricted to the requested records on purpose: a matrix with hundreds of
    thousands of rows is not a visualisation, it is a memory problem.
    """
    if margins.empty:
        return pd.DataFrame()
    subset = margins if record_ids is None else margins[margins["record_id"].isin(record_ids)]
    return subset.pivot_table(
        index="record_id", columns="rule_id", values="normalized_margin", aggfunc="first"
    )


def rows_for_storage(margins: pd.DataFrame) -> list[dict[str, object]]:
    """Rows for the ``rule_margins`` table."""
    if margins.empty:
        return []
    columns = [
        "record_id",
        "rule_id",
        "observed_value",
        "threshold_low",
        "threshold_high",
        "normalized_margin",
        "margin_status",
    ]
    return margins[columns].to_dict(orient="records")
