"""Exact reference-context features; own identity never serves as evidence."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import DataStructs

from .schema import CONTEXT_NAMES
from .splits import available_dates


def _fingerprint(value):
    try:
        binary = bytes.fromhex(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("fingerprint_hex must be valid hexadecimal RDKit binary bytes") from exc
    if not binary:
        raise ValueError("fingerprint_hex cannot be empty")
    return DataStructs.CreateFromBinaryText(binary)


def _validate(frame):
    required = {"identity", "murcko_scaffold", "fingerprint_hex"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing context columns: {sorted(required - set(frame.columns))}")
    if (
        frame[list(required)].isna().any().any()
        or frame.identity.astype(str).str.strip().eq("").any()
    ):
        raise ValueError("Context requires valid identity, scaffold and fingerprint")


def _references(references, threshold):
    _validate(references)
    refs = references.copy()
    if "y_active" not in refs:
        if "pactivity" not in refs:
            raise ValueError("References require y_active or exact pactivity")
        values = pd.to_numeric(refs.pactivity, errors="raise")
        if np.isinf(values).any():
            raise ValueError("Reference pactivity must be finite")
        if "relation" in refs:
            values = values.where(refs.relation.eq("="))
        refs["y_active"] = values.ge(threshold).astype(float).where(values.notna())
    labels = pd.to_numeric(refs.y_active, errors="raise")
    if not labels.dropna().isin([0, 1]).all():
        raise ValueError("Reference y_active must be 0, 1 or missing")
    refs = refs.assign(y_active=labels)
    # Conflicting replicates are not evidence for either class.
    consistent = refs.groupby("identity").y_active.nunique().le(1)
    refs = refs.loc[refs.identity.map(consistent) & refs.y_active.notna()]
    if refs.groupby("identity").fingerprint_hex.nunique().gt(1).any():
        raise ValueError("Same identity has inconsistent fingerprints")
    return refs.drop_duplicates("identity")


def build_context(queries, references, threshold=6.0):
    """Seven exact Tanimoto features, independently filtered per query identity."""
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    _validate(queries)
    for field in ("target_id", "endpoint"):
        parts = [frame[field] for frame in (queries, references) if field in frame]
        if parts and pd.concat(parts).dropna().nunique() > 1:
            raise ValueError(f"Context requires one consistent {field}")
    refs = _references(references, threshold)
    reference_fps = [_fingerprint(value) for value in refs.fingerprint_hex]
    reference_dimensions = {fp.GetNumBits() for fp in reference_fps}
    reference_identities = refs.identity.to_numpy()
    reference_labels = refs.y_active.to_numpy()
    reference_scaffolds = refs.murcko_scaffold.to_numpy()
    result = []
    # ponytail: exact O(queries * references); approximate indexing needs measured recall.
    for row in queries.itertuples():
        query_fp = _fingerprint(row.fingerprint_hex)
        if reference_dimensions and reference_dimensions != {query_fp.GetNumBits()}:
            raise ValueError("Query/reference fingerprint dimensions differ")
        allowed = reference_identities != row.identity
        if not allowed.any():
            result.append([np.nan] * 7)
            continue
        similarities = np.array(DataStructs.BulkTanimotoSimilarity(query_fp, reference_fps))[
            allowed
        ]
        labels = reference_labels[allowed]
        values = []
        for label in (1, 0):
            subset = similarities[labels == label]
            values.extend(
                [float(subset.max()), float(np.sort(subset)[-5:].mean())]
                if len(subset)
                else [np.nan, np.nan]
            )
        values.extend(
            [
                1 - float(similarities.max()),
                float((reference_scaffolds[allowed] == row.murcko_scaffold).mean()),
                float(np.sort(similarities)[-5:].mean()),
            ]
        )
        result.append(values)
    return pd.DataFrame(result, index=queries.index, columns=CONTEXT_NAMES)


def training_context(frame, seed=42, folds=5, temporal=False):
    """Train: scaffold OOF; held-out: train only. Temporal: strictly earlier evidence."""
    _validate(frame)
    if (
        "split" not in frame
        or not frame.split.isin(["train", "validation", "calibration", "test"]).all()
    ):
        raise ValueError("Valid split required before context generation")
    if not isinstance(folds, int) or folds < 2:
        raise ValueError("folds must be at least two")
    if not frame.index.is_unique:
        raise ValueError("Context frame index must be unique")
    if frame.groupby("identity").split.nunique().gt(1).any():
        raise ValueError("Identity leakage across splits")
    train = frame.loc[frame.split.eq("train")]
    result = pd.DataFrame(np.nan, index=frame.index, columns=CONTEXT_NAMES)
    threshold = frame.attrs.get("threshold", 6.0)
    if temporal:
        dates = available_dates(frame)
        train_dates = dates.loc[train.index]
        for date, indices in dates.groupby(dates, sort=False).groups.items():
            references = train.loc[train_dates.lt(date)]
            result.loc[indices] = build_context(frame.loc[indices], references, threshold)
    else:
        scaffolds = sorted(train.murcko_scaffold.unique())
        np.random.default_rng(seed).shuffle(scaffolds)
        fold_for = {scaffold: i % folds for i, scaffold in enumerate(scaffolds)}
        for fold in range(min(folds, len(scaffolds))):
            query = train.loc[train.murcko_scaffold.map(fold_for).eq(fold)]
            refs = train.loc[train.murcko_scaffold.map(fold_for).ne(fold)]
            # Excluding scaffold also excludes own identity in normal feature contracts.
            result.loc[query.index] = build_context(query, refs, threshold)
        held = frame.loc[~frame.split.eq("train")]
        result.loc[held.index] = build_context(held, train, threshold)
    return result
