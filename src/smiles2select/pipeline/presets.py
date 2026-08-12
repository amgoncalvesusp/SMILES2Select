"""Saved run presets.

A preset stores the *decisions* of a run - which profiles, in which roles, with
which alert actions, QED mode and standardization - but never the input files
or the output paths. The same preset therefore applies to any library, which is
the point: a group agrees on one selection policy and reuses it.

Profile definitions themselves are not copied into the preset. It references
them by id, so a preset stays valid when a profile is corrected, and fails
loudly when a profile it names no longer exists.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from smiles2select.alerts.custom_smarts import SmartsAlert
from smiles2select.alerts.policies import DEFAULT_ACTIONS, AlertAction, AlertPolicy
from smiles2select.app_metadata import APP_VERSION
from smiles2select.chemistry.standardization import StandardizationConfig
from smiles2select.decision.policies import DecisionPolicy, ProfileRole
from smiles2select.scores.qed import QedSelection

PRESET_FORMAT = "1.0"


class PresetError(ValueError):
    """Raised when a preset file is malformed or references unknown profiles."""


@dataclass(frozen=True)
class RunPreset:
    """Everything a user configures, minus the data and the destinations."""

    name: str = "preset"
    roles: dict[str, ProfileRole] = field(default_factory=dict)
    consensus_min_pass: int | None = None
    expression: str | None = None
    qed: QedSelection = field(default_factory=QedSelection)
    active_catalogs: tuple[str, ...] = ("pains", "brenk")
    alert_actions: dict[str, AlertAction] = field(default_factory=lambda: dict(DEFAULT_ACTIONS))
    custom_alerts: tuple[SmartsAlert, ...] = ()
    standardization: StandardizationConfig = field(default_factory=StandardizationConfig)
    compute_qed: bool = True
    drop_duplicates: bool = True
    detailed_export: bool = True

    def profile_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.roles))

    def build_policy(self) -> DecisionPolicy:
        return DecisionPolicy(
            id=f"preset:{self.name}",
            roles=dict(self.roles),
            consensus_min_pass=self.consensus_min_pass,
            expression=self.expression,
            qed=self.qed,
            alert_policy=AlertPolicy(actions=dict(self.alert_actions)),
        )

    def validate(self, available_profiles: tuple[str, ...]) -> list[str]:
        """Problems that prevent this preset from being applied."""
        problems: list[str] = []
        if not self.roles:
            problems.append("the preset does not select any profile")
        unknown = [pid for pid in self.roles if pid not in available_profiles]
        if unknown:
            problems.append(f"unknown profiles: {', '.join(sorted(unknown))}")
        if self.qed.excludes and not self.compute_qed:
            problems.append("the QED policy excludes molecules, but QED is disabled")
        try:
            self.build_policy()
        except ValueError as exc:
            problems.append(str(exc))
        return problems

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": PRESET_FORMAT,
            "app_version": APP_VERSION,
            "name": self.name,
            "roles": dict(self.roles),
            "consensus_min_pass": self.consensus_min_pass,
            "expression": self.expression,
            "qed": {
                "mode": self.qed.mode,
                "threshold": self.qed.threshold,
                "percentile": self.qed.percentile,
            },
            "active_catalogs": list(self.active_catalogs),
            "alert_actions": dict(self.alert_actions),
            "custom_alerts": [asdict(alert) for alert in self.custom_alerts],
            "standardization": asdict(self.standardization),
            "compute_qed": self.compute_qed,
            "drop_duplicates": self.drop_duplicates,
            "detailed_export": self.detailed_export,
        }


def natural_product_exploration_preset() -> RunPreset:
    """Built-in policy for natural-product exploration.

    Dockability is the only hard eligibility layer. Classical profiles remain
    classifications/informative outputs unless a caller deliberately changes
    their roles.
    """

    profile_ids = (
        "dockability_envelope",
        "lipinski",
        "veber",
        "ghose",
        "egan",
        "muegge",
        "beyond_ro5",
    )
    roles: dict[str, ProfileRole] = dict.fromkeys(profile_ids, "informative")
    roles["dockability_envelope"] = "mandatory"
    return RunPreset(
        name="Natural Product Exploration",
        roles=roles,
        active_catalogs=("pains", "brenk"),
        alert_actions={"pains": "warn", "brenk": "warn"},
        compute_qed=True,
        drop_duplicates=True,
    )


def from_dict(payload: dict[str, Any], *, source: str = "<dict>") -> RunPreset:
    """Rebuild a preset, rejecting anything the current version cannot honour."""
    file_format = payload.get("format", PRESET_FORMAT)
    if file_format != PRESET_FORMAT:
        raise PresetError(
            f"{source}: preset format '{file_format}' is not supported "
            f"(this version reads '{PRESET_FORMAT}')"
        )

    qed_payload = payload.get("qed") or {}
    try:
        qed = QedSelection(
            mode=qed_payload.get("mode", "compute"),
            threshold=qed_payload.get("threshold"),
            percentile=qed_payload.get("percentile"),
        )
    except ValueError as exc:
        raise PresetError(f"{source}: {exc}") from exc

    try:
        alerts = tuple(
            SmartsAlert(
                id=item["id"],
                name=item.get("name", item["id"]),
                smarts=item["smarts"],
                description=item.get("description", ""),
            )
            for item in payload.get("custom_alerts", [])
        )
        for alert in alerts:
            alert.compile_pattern()  # a broken SMARTS must fail on load
    except (KeyError, ValueError) as exc:
        raise PresetError(f"{source}: invalid custom alert ({exc})") from exc

    standardization_payload = payload.get("standardization") or {}
    known = set(StandardizationConfig.__dataclass_fields__)
    unexpected = set(standardization_payload) - known
    if unexpected:
        raise PresetError(f"{source}: unknown standardization option(s): {sorted(unexpected)}")

    return RunPreset(
        name=payload.get("name", "preset"),
        roles=dict(payload.get("roles", {})),
        consensus_min_pass=payload.get("consensus_min_pass"),
        expression=payload.get("expression"),
        qed=qed,
        active_catalogs=tuple(payload.get("active_catalogs", ())),
        alert_actions=dict(payload.get("alert_actions", DEFAULT_ACTIONS)),
        custom_alerts=alerts,
        standardization=StandardizationConfig(**standardization_payload),
        compute_qed=payload.get("compute_qed", True),
        drop_duplicates=payload.get("drop_duplicates", True),
        detailed_export=payload.get("detailed_export", True),
    )


def save(preset: RunPreset, path: str | Path) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(preset.as_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return file_path


def load(path: str | Path) -> RunPreset:
    file_path = Path(path)
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PresetError(f"{file_path.name}: invalid JSON ({exc})") from exc
    except OSError as exc:
        raise PresetError(f"{file_path}: cannot be read ({exc})") from exc
    return from_dict(payload, source=file_path.name)
