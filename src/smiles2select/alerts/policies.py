"""What a catalogue hit does to a molecule.

Flagging and excluding are different decisions and the software keeps them
apart. PAINS and Brenk warn by default; nothing is excluded unless the user
explicitly asks for it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

AlertAction = Literal["inform", "warn", "penalize", "exclude"]

ACTION_LABELS: dict[str, str] = {
    "inform": "Informar — apenas registra o alerta",
    "warn": "Advertir — destaca a molécula na interface",
    "penalize": "Penalizar — reduz o escore combinado",
    "exclude": "Excluir — remove da seleção final",
}

DEFAULT_ACTIONS: dict[str, AlertAction] = {
    "pains": "warn",
    "pains_a": "warn",
    "pains_b": "warn",
    "pains_c": "warn",
    "brenk": "warn",
    "nih": "inform",
    "zinc": "inform",
    "custom_smarts": "inform",
}


@dataclass(frozen=True)
class AlertPolicy:
    """Action per catalogue, with a conservative default."""

    actions: Mapping[str, AlertAction] = field(default_factory=lambda: dict(DEFAULT_ACTIONS))
    default_action: AlertAction = "warn"

    def action_for(self, catalog_id: str) -> AlertAction:
        return self.actions.get(catalog_id, self.default_action)

    def excludes(self, catalog_id: str) -> bool:
        return self.action_for(catalog_id) == "exclude"

    def penalizes(self, catalog_id: str) -> bool:
        return self.action_for(catalog_id) in {"penalize", "exclude"}

    def excluding_catalogs(self) -> tuple[str, ...]:
        return tuple(sorted(cid for cid in self.actions if self.excludes(cid)))

    def with_action(self, catalog_id: str, action: AlertAction) -> "AlertPolicy":
        if action not in ACTION_LABELS:
            raise ValueError(f"unknown alert action '{action}'")
        updated = dict(self.actions)
        updated[catalog_id] = action
        return AlertPolicy(actions=updated, default_action=self.default_action)

    def describe(self) -> list[str]:
        return [
            f"{catalog_id}: {ACTION_LABELS[self.action_for(catalog_id)]}"
            for catalog_id in sorted(self.actions)
        ]
