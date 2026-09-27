"""Deterministic identity/scaffold partitions and conservative temporal availability."""

from __future__ import annotations

import numpy as np
import pandas as pd

SPLITS = ("train", "validation", "calibration", "test")


def available_dates(frame):
    """A group's evidence becomes available only when its latest record is available."""
    dates = pd.to_datetime(
        frame.get("date", pd.Series("", index=frame.index)), errors="coerce", utc=True
    )
    if "year" in frame:
        years = pd.to_numeric(frame["year"], errors="coerce")
        # Year-only records cannot be assumed available on January 1.
        year_dates = pd.to_datetime(
            years.map(lambda y: f"{int(y)}-12-31" if pd.notna(y) else None),
            errors="coerce",
            utc=True,
        )
        dates = dates.fillna(year_dates)
    if dates.isna().any():
        raise ValueError("Temporal protocol requires usable date or year for every record")
    return dates.groupby(frame["identity"]).transform("max")


def _groups(frame, scaffold):
    identities = frame["identity"].astype(str)
    parents = {identity: identity for identity in identities.unique()}

    def root(identity):
        while parents[identity] != identity:
            identity = parents[identity]
        return identity

    if scaffold:
        if "murcko_scaffold" not in frame or frame["murcko_scaffold"].isna().any():
            raise ValueError(
                "Scaffold partition requires calculated murcko_scaffold; empty acyclic scaffold is valid"
            )
        for _, group in frame.groupby("murcko_scaffold", sort=False):
            group_ids = group["identity"].astype(str).unique()
            anchor = root(group_ids[0])
            for identity in group_ids[1:]:
                parents[root(identity)] = anchor
    return identities.map(root)


def assign_splits(frame, method="scaffold", seed=42):
    """Split whole groups approximately 70/10/10/10; never silently split large groups."""
    if method not in {"scaffold", "random", "temporal"}:
        raise ValueError("method must be scaffold, random or temporal")
    if (
        "identity" not in frame
        or frame.identity.isna().any()
        or frame.identity.astype(str).str.strip().eq("").any()
    ):
        raise ValueError("Nonempty standardized identity required")
    groups = _groups(frame, method == "scaffold")
    unique = sorted(groups.unique())
    if len(unique) < 4:
        raise ValueError(
            "At least four independent groups required for train/validation/calibration/test"
        )
    result = frame.copy()
    if method == "temporal":
        dates = available_dates(frame)
        result["availability_date"] = dates
        group_dates = dates.groupby(groups).max()
        unique = sorted(unique, key=lambda key: (group_dates[key], key))
        # Equal dates form one indivisible time group, avoiding same-year information leakage.
        groups = groups.map(group_dates)
        unique = sorted(groups.unique())
        if len(unique) < 4:
            raise ValueError(
                "At least four distinct date/year groups required for temporal splitting"
            )
    else:
        np.random.default_rng(seed).shuffle(unique)
    n = len(unique)
    n_train = max(1, min(n - 3, int(n * 0.7)))
    n_val = max(1, min(n - n_train - 2, int(n * 0.1)))
    n_cal = max(1, min(n - n_train - n_val - 1, int(n * 0.1)))
    ends = (n_train, n_train + n_val, n_train + n_val + n_cal, n)
    mapping = {
        key: SPLITS[next(i for i, end in enumerate(ends) if index < end)]
        for index, key in enumerate(unique)
    }
    result["split"] = groups.map(mapping)
    result.attrs.update(
        frame.attrs,
        split_method=method,
        split_seed=seed,
        temporal_interpretation="group latest evidence availability; retrospective, not prospective",
    )
    return result
