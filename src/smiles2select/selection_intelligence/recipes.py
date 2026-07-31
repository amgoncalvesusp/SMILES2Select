"""Selection recipes.

A recipe is what makes a selection reproducible: the objectives, the thresholds
actually applied, the robustness settings, the final-selection strategy and the
manual overrides. Saved next to the Excel report, it lets the same decision be
replayed - or applied to a different library - months later.

It records ``input_hash`` and ``descriptor_version`` so a replay that silently
uses different data or a different toolkit can be detected instead of assumed
equivalent.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from smiles2select.app_metadata import APP_VERSION, rdkit_version
from smiles2select.selection_intelligence.action_log import utc_timestamp

SCHEMA_VERSION = "2.0"
RECIPE_SUFFIX = ".selection.json"


class RecipeError(ValueError):
    """Raised when a recipe file cannot be read or is from another schema."""


@dataclass(frozen=True)
class ManualOverride:
    """One decision a human took against, or beyond, the automatic result."""

    record_id: int
    previous_status: str
    new_status: str
    action: str
    reason: str
    timestamp: str = field(default_factory=utc_timestamp)


@dataclass(frozen=True)
class SelectionRecipe:
    """Everything needed to reproduce a selection."""

    name: str = "selection"
    input_hash: str = ""
    descriptor_version: str = field(default_factory=rdkit_version)
    required_filters: tuple[str, ...] = ()
    objectives: tuple[dict[str, Any], ...] = ()
    max_front: int | None = None
    original_thresholds: dict[str, float] = field(default_factory=dict)
    applied_thresholds: dict[str, float] = field(default_factory=dict)
    robustness_method: str = "iqr_normalized"
    borderline_fraction: float = 0.10
    target_count: int | None = None
    strategy: str = "balanced"
    max_per_scaffold: int | None = None
    max_per_cluster: int | None = None
    manual_overrides: tuple[ManualOverride, ...] = ()
    pinned_ids: tuple[int, ...] = ()
    excluded_ids: tuple[int, ...] = ()
    created_at: str = field(default_factory=utc_timestamp)
    updated_at: str = field(default_factory=utc_timestamp)

    @property
    def changed_thresholds(self) -> dict[str, tuple[float, float]]:
        """Thresholds that differ from the ones the profiles ship with."""
        return {
            rule_id: (self.original_thresholds[rule_id], value)
            for rule_id, value in self.applied_thresholds.items()
            if rule_id in self.original_thresholds and self.original_thresholds[rule_id] != value
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "app_version": APP_VERSION,
            "name": self.name,
            "input_hash": self.input_hash,
            "descriptor_version": self.descriptor_version,
            "required_filters": list(self.required_filters),
            "pareto": {
                "objectives": [dict(objective) for objective in self.objectives],
                "max_front": self.max_front,
            },
            "sensitivity": {
                "original_thresholds": dict(self.original_thresholds),
                "applied_thresholds": dict(self.applied_thresholds),
            },
            "robustness": {
                "method": self.robustness_method,
                "borderline_fraction": self.borderline_fraction,
            },
            "final_selection": {
                "target_count": self.target_count,
                "strategy": self.strategy,
                "max_per_scaffold": self.max_per_scaffold,
                "max_per_cluster": self.max_per_cluster,
            },
            "manual_overrides": [asdict(override) for override in self.manual_overrides],
            "pinned_ids": list(self.pinned_ids),
            "excluded_ids": list(self.excluded_ids),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def summary_rows(self) -> list[tuple[str, object]]:
        """Tabular form for the SELECTION_RECIPE sheet."""
        return [
            ("schema_version", SCHEMA_VERSION),
            ("name", self.name),
            ("input_hash", self.input_hash or "-"),
            ("descriptor_version", self.descriptor_version),
            ("objectives", len(self.objectives)),
            ("max_front", self.max_front or "-"),
            ("robustness_method", self.robustness_method),
            ("borderline_fraction", self.borderline_fraction),
            ("target_count", self.target_count or "-"),
            ("strategy", self.strategy),
            ("max_per_scaffold", self.max_per_scaffold or "-"),
            ("max_per_cluster", self.max_per_cluster or "-"),
            ("thresholds alterados", len(self.changed_thresholds)),
            ("manual_overrides", len(self.manual_overrides)),
            ("pinned", len(self.pinned_ids)),
            ("excluded", len(self.excluded_ids)),
            ("created_at", self.created_at),
            ("updated_at", self.updated_at),
        ]


def from_dict(payload: dict[str, Any], *, source: str = "<dict>") -> SelectionRecipe:
    """Rebuild a recipe, refusing a schema this version does not understand."""
    schema = payload.get("schema_version", SCHEMA_VERSION)
    if schema != SCHEMA_VERSION:
        raise RecipeError(
            f"{source}: schema '{schema}' is not supported (this version reads '{SCHEMA_VERSION}')"
        )

    pareto = payload.get("pareto") or {}
    sensitivity = payload.get("sensitivity") or {}
    robustness = payload.get("robustness") or {}
    final = payload.get("final_selection") or {}

    return SelectionRecipe(
        name=payload.get("name", "selection"),
        input_hash=payload.get("input_hash", ""),
        descriptor_version=payload.get("descriptor_version", ""),
        required_filters=tuple(payload.get("required_filters", ())),
        objectives=tuple(pareto.get("objectives", ())),
        max_front=pareto.get("max_front"),
        original_thresholds=dict(sensitivity.get("original_thresholds", {})),
        applied_thresholds=dict(sensitivity.get("applied_thresholds", {})),
        robustness_method=robustness.get("method", "iqr_normalized"),
        borderline_fraction=robustness.get("borderline_fraction", 0.10),
        target_count=final.get("target_count"),
        strategy=final.get("strategy", "balanced"),
        max_per_scaffold=final.get("max_per_scaffold"),
        max_per_cluster=final.get("max_per_cluster"),
        manual_overrides=tuple(
            ManualOverride(**override) for override in payload.get("manual_overrides", ())
        ),
        pinned_ids=tuple(payload.get("pinned_ids", ())),
        excluded_ids=tuple(payload.get("excluded_ids", ())),
        created_at=payload.get("created_at", utc_timestamp()),
        updated_at=payload.get("updated_at", utc_timestamp()),
    )


def save(recipe: SelectionRecipe, path: str | Path) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(recipe.as_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return file_path


def load(path: str | Path) -> SelectionRecipe:
    file_path = Path(path)
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RecipeError(f"{file_path.name}: invalid JSON ({exc})") from exc
    except OSError as exc:
        raise RecipeError(f"{file_path}: cannot be read ({exc})") from exc
    return from_dict(payload, source=file_path.name)


def default_path(project_name: str, directory: str | Path = ".") -> Path:
    return Path(directory) / f"{project_name}{RECIPE_SUFFIX}"


def compare(first: SelectionRecipe, second: SelectionRecipe) -> list[str]:
    """Human-readable differences between two recipes."""
    differences: list[str] = []
    if first.input_hash != second.input_hash:
        differences.append("bibliotecas de entrada diferentes (input_hash)")
    if first.descriptor_version != second.descriptor_version:
        differences.append(
            f"versão de descritores: {first.descriptor_version} -> {second.descriptor_version}"
        )
    if first.objectives != second.objectives:
        differences.append(
            f"objetivos de Pareto: {len(first.objectives)} -> {len(second.objectives)}"
        )
    for rule_id in sorted(set(first.applied_thresholds) | set(second.applied_thresholds)):
        before = first.applied_thresholds.get(rule_id)
        after = second.applied_thresholds.get(rule_id)
        if before != after:
            differences.append(f"limite {rule_id}: {before} -> {after}")
    if first.target_count != second.target_count:
        differences.append(f"quantidade final: {first.target_count} -> {second.target_count}")
    if first.strategy != second.strategy:
        differences.append(f"estratégia: {first.strategy} -> {second.strategy}")
    return differences
