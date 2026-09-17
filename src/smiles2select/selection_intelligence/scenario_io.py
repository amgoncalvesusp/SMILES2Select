"""Portable scenario recipes bound to exact cached input; no molecule tables in JSON."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

from smiles2select.selection_intelligence.constrained_selection import (
    CORE_VERSION,
    SelectionConstraints,
)
from smiles2select.selection_intelligence.objectives import objective_from_dict
from smiles2select.selection_intelligence.scenarios import (
    ScenarioSnapshot,
    ScenarioSpec,
    data_fingerprint,
    evaluate_scenario,
    selected_ids_fingerprint,
)

SCHEMA_VERSION = 2
MAX_RECIPE_BYTES = 8_000_000


def spec_to_dict(spec: ScenarioSpec) -> dict:
    return {
        "name": spec.name,
        "thresholds": dict(spec.thresholds),
        "objectives": [obj.as_dict() for obj in spec.objectives],
        "constraints": asdict(spec.constraints),
        "strategy": spec.strategy.value,
    }


def spec_from_dict(payload: dict) -> ScenarioSpec:
    _keys(payload, {"name", "thresholds", "objectives", "constraints", "strategy"})
    _keys(payload["constraints"], set(SelectionConstraints.__dataclass_fields__))
    if not isinstance(payload["thresholds"], dict) or not isinstance(payload["objectives"], list):
        raise ValueError("invalid thresholds or objectives")
    objective_keys = {
        "field",
        "direction",
        "weight",
        "enabled",
        "transform",
        "target_low",
        "target_high",
        "target_value",
    }
    for obj in payload["objectives"]:
        if not isinstance(obj, dict) or set(obj) - objective_keys or "field" not in obj:
            raise ValueError("invalid objective fields")
        if not isinstance(obj["field"], str) or type(obj.get("enabled", True)) is not bool:
            raise ValueError("invalid objective field or enabled flag")
        for name in ("weight", "target_low", "target_high", "target_value"):
            if name in obj and (
                type(obj[name]) not in (float, int) or not math.isfinite(obj[name])
            ):
                raise ValueError(f"objective {name} must be finite numeric")
    return ScenarioSpec(
        payload["name"],
        payload["thresholds"],
        tuple(objective_from_dict(obj) for obj in payload["objectives"]),
        SelectionConstraints(**payload["constraints"]),
        payload["strategy"],
    )


def _keys(payload: dict, expected: set[str]) -> None:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(f"expected fields: {sorted(expected)}")


def save_study(path: str | Path, snapshots: tuple[ScenarioSnapshot, ...]) -> None:
    if not snapshots or len({s.data_fingerprint for s in snapshots}) != 1:
        raise ValueError("study requires scenarios from the same input")
    if len({s.spec.name for s in snapshots}) != len(snapshots):
        raise ValueError("scenario names must be distinct")
    if any(s.provenance.get("selection_algorithm_version") != CORE_VERSION for s in snapshots):
        raise ValueError("selection algorithm version changed; create a new scenario study")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "data_fingerprint": snapshots[0].data_fingerprint,
        "scenarios": [
            {
                "spec": spec_to_dict(s.spec),
                "pinned_ids": s.pinned_ids,
                "rescued_ids": s.rescued_ids,
                "excluded_ids": s.excluded_ids,
                "ranking_method": s.provenance["ranking_method"],
                "exact_pareto_limit": s.provenance["exact_pareto_limit"],
                "selection_algorithm_version": s.provenance["selection_algorithm_version"],
                "selected_ids_sha256": selected_ids_fingerprint(s.outcome.selected_ids),
                "scaffold_inputs_sha256": s.provenance["scaffold_inputs_sha256"],
            }
            for s in snapshots
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_study(path: str | Path, result, candidates) -> tuple[ScenarioSnapshot, ...]:
    """Validate recipe and replay it; incompatible data or ranking version fails closed."""
    recipe = Path(path)
    if recipe.stat().st_size > MAX_RECIPE_BYTES:
        raise ValueError("scenario recipe exceeds 8 MB")
    payload = json.loads(recipe.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)
    _keys(payload, {"schema_version", "data_fingerprint", "scenarios"})
    if type(payload["schema_version"]) is int and payload["schema_version"] == 1:
        raise ValueError(
            "This legacy scenario recipe cannot safely replay: schema 1 did not record "
            "the selection algorithm version or selected IDs hash. Create a new scenario study."
        )
    if type(payload["schema_version"]) is not int or payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported scenario schema version")
    fingerprint = data_fingerprint(result, candidates)
    if payload["data_fingerprint"] != fingerprint:
        raise ValueError("scenario recipe requires the same input data and policy")
    rows = payload["scenarios"]
    if not isinstance(rows, list) or not rows or len(rows) > 100:
        raise ValueError("recipe requires 1 to 100 scenarios")
    snapshots = []
    for row in rows:
        _keys(
            row,
            {
                "spec",
                "pinned_ids",
                "rescued_ids",
                "excluded_ids",
                "ranking_method",
                "exact_pareto_limit",
                "selection_algorithm_version",
                "selected_ids_sha256",
                "scaffold_inputs_sha256",
            },
        )
        if row["selection_algorithm_version"] != CORE_VERSION:
            raise ValueError("selection algorithm version changed; create a new scenario study")
        flags = {name: row[name] for name in ("pinned_ids", "rescued_ids", "excluded_ids")}
        if any(
            not isinstance(ids, list) or any(type(i) is not int for i in ids)
            for ids in flags.values()
        ):
            raise ValueError("manual flags must be integer ID lists")
        snapshot = evaluate_scenario(
            result, candidates, spec_from_dict(row["spec"]), _fingerprint=fingerprint, **flags
        )
        for name in ("ranking_method", "exact_pareto_limit"):
            if snapshot.provenance[name] != row[name]:
                raise ValueError("ranking implementation changed; create a new scenario study")
        if selected_ids_fingerprint(snapshot.outcome.selected_ids) != row["selected_ids_sha256"]:
            raise ValueError("Replayed selected IDs differ from the saved study; adoption is blocked.")
        if snapshot.provenance["scaffold_inputs_sha256"] != row["scaffold_inputs_sha256"]:
            raise ValueError("Replayed scaffold inputs differ from the saved study; adoption is blocked.")
        snapshots.append(snapshot)
    if len({s.spec.name for s in snapshots}) != len(snapshots):
        raise ValueError("scenario names must be distinct")
    return tuple(snapshots)
