"""Score adapter for the existing SMILES2Select constrained selector."""

from copy import deepcopy
from dataclasses import asdict
from numbers import Integral, Real

import numpy as np

from smiles2select.selection_intelligence.constrained_selection import (
    CORE_VERSION,
    SelectionConstraints,
    Strategy,
    select,
)

from .schema import FeatureSet


def _ids(values, name, available):
    values = tuple(values)
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) for value in values
    ):
        raise ValueError(f"{name} IDs must be integers")
    unknown = set(values) - available
    if unknown:
        raise ValueError(f"Unknown {name} IDs: {sorted(unknown)}")
    return tuple(sorted(set(map(int, values))))


def _validate(frame):
    required = {"record_id", "eligible", "valid", "priority_score", "identity", "murcko_scaffold"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing selection fields: {sorted(missing)}")
    if not frame.columns.is_unique:
        raise ValueError("Selection columns must be unique")
    ids = frame.record_id
    if (
        ids.isna().any()
        or ids.duplicated().any()
        or any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) for value in ids
        )
    ):
        raise ValueError("record_id values must be unique integers")
    boolean_columns = (
        ("eligible", "valid", "pinned") if "pinned" in frame else ("eligible", "valid")
    )
    for name in boolean_columns:
        if any(not isinstance(value, (bool, np.bool_)) for value in frame[name]):
            raise ValueError(f"{name} must contain boolean values")
    if (frame.eligible.astype(bool) & ~frame.valid.astype(bool)).any():
        raise ValueError("Eligible candidates cannot have invalid structures")


def _validate_pool(pool, max_per_cluster):
    if any(
        not isinstance(value, Real)
        or isinstance(value, (bool, np.bool_))
        or not np.isfinite(value)
        or not 0 <= value <= 1
        for value in pool.priority_score
    ):
        raise ValueError("Eligible priority_score must be finite numbers in [0, 1]")
    if any(not isinstance(value, str) or not value for value in pool.identity):
        raise ValueError("Eligible candidates require a nonempty chemical identity")
    if pool.identity.duplicated().any():
        raise ValueError(
            "Duplicate chemical identities among eligible candidates; "
            "resolve duplicates before selection"
        )
    if any(not isinstance(value, str) for value in pool.murcko_scaffold):
        raise ValueError("Eligible candidates require known murcko_scaffold; empty means acyclic")
    if max_per_cluster is not None:
        if "cluster_id" not in pool or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or int(value) != value
            for value in pool.cluster_id
        ):
            raise ValueError("Cluster quotas require known integer cluster_id for every candidate")


def select_candidates(
    frame,
    n: int,
    max_per_scaffold=None,
    min_scaffolds=None,
    max_per_cluster=None,
    pinned_ids=(),
    excluded_ids=(),
    selection_scaffold_col="murcko_scaffold",
) -> FeatureSet:
    """Preserve source rows and select only eligible candidates, with explicit conflicts.

    Scores rank descending; record_id breaks ties. Inherited and explicit pins
    are combined. Pins override count and quotas,
    with violations recorded. No model probability is reinterpreted as P(advance).
    """
    provenance = frame.manifest if isinstance(frame, FeatureSet) else frame.attrs
    records = frame.records if isinstance(frame, FeatureSet) else frame
    _validate(records)
    if selection_scaffold_col not in records:
        raise ValueError(f"Missing selection scaffold: {selection_scaffold_col}")
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer")
    constraints = SelectionConstraints(n, max_per_scaffold, min_scaffolds, max_per_cluster)
    available = set(records.record_id)
    pins = _ids(pinned_ids, "pinned", available)
    if "pinned" in records:
        inherited = records.loc[records.pinned.astype(bool), "record_id"]
        pins = tuple(sorted(set(pins) | set(map(int, inherited))))
    excluded = _ids(excluded_ids, "excluded", available)
    if set(pins) & set(excluded):
        raise ValueError("Candidates cannot be both pinned and excluded")
    pool = records.loc[records.eligible.astype(bool) & ~records.record_id.isin(excluded)]
    ineligible_pins = set(pins) - set(pool.record_id)
    if ineligible_pins:
        raise ValueError(
            f"Pinned candidates are not eligible: {sorted(ineligible_pins)}; "
            "resolve eligibility or unpin these records before selection"
        )
    native_pool = pool.assign(murcko_scaffold=pool[selection_scaffold_col])
    _validate_pool(native_pool, max_per_cluster)
    # Ignore inherited ranking fields: reasons must describe this score-based run.
    columns = ["record_id", "murcko_scaffold"]
    if "cluster_id" in pool and max_per_cluster is not None:
        columns.append("cluster_id")
    candidates = native_pool[columns].assign(
        selection_priority=-pool.priority_score
    ).set_index("record_id")
    outcome = select(candidates, constraints, strategy=Strategy.BALANCED, pinned_ids=pins)
    final_ids = list(outcome.selected_ids)
    warnings = outcome.warnings(constraints)
    explanations = {
        record_id: "; ".join(reasons)
        for record_id, reasons in {**outcome.rejections, **outcome.reasons}.items()
    }
    reasons = [
        "excluded by user"
        if row.record_id in excluded
        else "not eligible"
        if not row.eligible
        else explanations.get(row.record_id, "not selected")
        for row in records.itertuples(index=False)
    ]
    selected = records.assign(is_final=records.record_id.isin(final_ids), selection_reason=reasons)
    manifest = {
        "requested_count": n,
        "final_count": len(final_ids),
        "shortfall": max(0, n - len(final_ids)),
        "excess": max(0, len(final_ids) - n),
        "status": "constraint_conflict" if warnings else "complete",
        "warnings": warnings,
        "final_ids": final_ids,
        "selector_version": CORE_VERSION,
        "selector": "smiles2select.selection_intelligence.constrained_selection",
        "strategy": Strategy.BALANCED.value,
        "constraints": asdict(constraints),
        "pinned_ids": list(pins),
        "excluded_ids": list(excluded),
        "scaffold_counts": dict(outcome.scaffold_usage),
        "scaffolds_covered": outcome.scaffolds_covered,
        "cluster_counts": dict(outcome.cluster_usage),
        "input_provenance": deepcopy(provenance),
        "duplicate_policy": "reject eligible duplicate chemical identities",
        "ranking": "priority_score descending, record_id ascending; pins and coverage precede filling",
        "selection_scaffold_column": selection_scaffold_col,
    }
    return FeatureSet(selected, manifest)
