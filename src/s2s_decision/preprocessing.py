"""Train-only scalar transforms; missing indicators follow values in each branch."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import CONTEXT_NAMES, LOG_NAMES, PROPERTY_NAMES


def _values(frame):
    names = (*PROPERTY_NAMES, *CONTEXT_NAMES)
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    values = (
        frame.loc[:, list(names)]
        .apply(pd.to_numeric, errors="raise")
        .to_numpy(dtype=float, copy=True)
    )
    if np.isinf(values).any():
        raise ValueError("Infinite feature values are invalid")
    for index, name in enumerate(names):
        if name in LOG_NAMES:
            if (values[:, index] < 0).any():
                raise ValueError(f"Negative count in {name}")
            values[:, index] = np.log1p(values[:, index])
    return values


@dataclass(frozen=True)
class Preprocessor:
    median: tuple
    mean: tuple
    scale: tuple
    context_all_missing: tuple

    @classmethod
    def fit(cls, frame):
        if frame.empty or ("split" in frame and not frame["split"].eq("train").all()):
            raise ValueError("Preprocessor requires nonempty train rows only")
        values = _values(frame)
        absent = np.isnan(values).all(axis=0)
        if absent[:20].any():
            names = [PROPERTY_NAMES[i] for i in np.flatnonzero(absent[:20])]
            raise ValueError(f"All-missing required molecular features: {names}")
        median = np.array([np.nanmedian(values[:, i]) if not absent[i] else 0.0 for i in range(27)])
        filled = np.where(np.isnan(values), median, values)
        mean, scale = filled.mean(axis=0), filled.std(axis=0)
        scale = np.where(scale > 0, scale, 1.0)
        # Unit-interval scores keep their documented scale.
        for index, name in enumerate((*PROPERTY_NAMES, *CONTEXT_NAMES)):
            if name in {"fraction_csp3", "qed"} or name in CONTEXT_NAMES:
                mean[index], scale[index] = 0, 1
        return cls(
            tuple(median),
            tuple(mean),
            tuple(scale),
            tuple(CONTEXT_NAMES[i] for i in np.flatnonzero(absent[20:])),
        )

    def transform(self, frame):
        values = _values(frame)
        missing = np.isnan(values)
        scaled = (np.where(missing, self.median, values) - self.mean) / self.scale
        properties = np.concatenate((scaled[:, :20], missing[:, :20]), axis=1).astype(np.float32)
        context = np.concatenate((scaled[:, 20:], missing[:, 20:]), axis=1).astype(np.float32)
        if not np.isfinite(properties).all() or not np.isfinite(context).all():
            raise ValueError("Feature transform produced nonfinite values")
        return properties, context

    def to_dict(self):
        return {
            "median": list(self.median),
            "mean": list(self.mean),
            "scale": list(self.scale),
            "context_all_missing": list(self.context_all_missing),
            "property_names": list(PROPERTY_NAMES),
            "context_names": list(CONTEXT_NAMES),
            "log_names": sorted(LOG_NAMES),
        }

    @classmethod
    def from_dict(cls, data):
        if (
            data.get("property_names") != list(PROPERTY_NAMES)
            or data.get("context_names") != list(CONTEXT_NAMES)
            or data.get("log_names") != sorted(LOG_NAMES)
        ):
            raise ValueError("Incompatible preprocessing schema")
        for key in ("median", "mean", "scale"):
            if len(data[key]) != 27 or not np.isfinite(data[key]).all():
                raise ValueError(f"Invalid preprocessing {key}")
        if (np.asarray(data["scale"]) <= 0).any():
            raise ValueError("Invalid preprocessing scale")
        return cls(
            *(tuple(data[key]) for key in ("median", "mean", "scale", "context_all_missing"))
        )
