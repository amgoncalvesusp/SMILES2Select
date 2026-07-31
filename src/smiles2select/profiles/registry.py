"""Profiles: versioned, named groups of rules.

A profile owns its rules and its pass policy, nothing else. It does not know
whether the user treats it as mandatory, informative or as part of a consensus
- that is the decision layer's business, and keeping it out of here is what
lets the same Lipinski definition be mandatory in one run and informative in
the next.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from smiles2select.rules.engine import Rule

CATEGORIES = (
    "oral_druglikeness",
    "chemical_space",
    "fragment_space",
    "lead_space",
    "cns_space",
    "beyond_ro5",
    "custom",
)


@dataclass(frozen=True)
class Profile:
    """A versioned group of rules with a pass policy."""

    id: str
    name: str
    category: str
    version: str
    rules: tuple[Rule, ...]
    pass_policy: Mapping[str, Any]
    #: Compact label for report columns ("Egan"), as opposed to the full
    #: qualified name ("Egan — implementação compatível com SwissADME").
    short_name: str = ""
    logp_method: str = ""
    atom_count_definition: str = ""
    reference: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.rules:
            raise ValueError(f"profile {self.id}: at least one rule is required")
        policy_type = self.pass_policy.get("type")
        if policy_type not in {"all_rules", "max_violations"}:
            raise ValueError(f"profile {self.id}: unknown pass policy '{policy_type}'")

    @property
    def label(self) -> str:
        """Short label used for report column names."""
        return self.short_name or self.name

    def descriptor_ids(self) -> tuple[str, ...]:
        """Descriptors this profile needs, without duplicates."""
        seen = {rule.descriptor for rule in self.rules if not rule.is_substructure}
        return tuple(sorted(seen))

    def rules_by_id(self) -> dict[str, Rule]:
        return {rule.id: rule for rule in self.rules}

    def with_pass_policy(self, policy: Mapping[str, Any]) -> Profile:
        """Return a copy under a different policy (e.g. strict vs classical Lipinski)."""
        return replace(self, pass_policy=dict(policy))

    def policy_label(self) -> str:
        if self.pass_policy.get("type") == "max_violations":
            allowed = int(self.pass_policy.get("value", 0))
            return f"até {allowed} violação(ões)" if allowed else "nenhuma violação"
        return "todos os critérios"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "short_name": self.short_name,
            "category": self.category,
            "version": self.version,
            "logp_method": self.logp_method,
            "atom_count_definition": self.atom_count_definition,
            "reference": self.reference,
            "notes": self.notes,
            "pass_policy": dict(self.pass_policy),
            "rules": [rule.as_dict() for rule in self.rules],
        }


class ProfileRegistry:
    """Lookup table of available profiles."""

    def __init__(self, profiles: Iterable[Profile] = ()) -> None:
        self._profiles: dict[str, Profile] = {}
        for profile in profiles:
            self.add(profile)

    def add(self, profile: Profile, *, overwrite: bool = False) -> None:
        if profile.id in self._profiles and not overwrite:
            raise ValueError(f"profile already registered: {profile.id}")
        self._profiles[profile.id] = profile

    def __contains__(self, profile_id: object) -> bool:
        return profile_id in self._profiles

    def __len__(self) -> int:
        return len(self._profiles)

    def get(self, profile_id: str) -> Profile:
        try:
            return self._profiles[profile_id]
        except KeyError as exc:
            raise KeyError(
                f"unknown profile '{profile_id}'; available: {sorted(self._profiles)}"
            ) from exc

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._profiles))

    def all(self) -> tuple[Profile, ...]:
        return tuple(self._profiles[profile_id] for profile_id in self.ids())

    def select(self, profile_ids: Iterable[str]) -> tuple[Profile, ...]:
        return tuple(self.get(profile_id) for profile_id in profile_ids)

    def by_category(self, category: str) -> tuple[Profile, ...]:
        return tuple(profile for profile in self.all() if profile.category == category)

    def descriptor_ids(self, profile_ids: Iterable[str] | None = None) -> tuple[str, ...]:
        chosen = self.select(profile_ids) if profile_ids is not None else self.all()
        needed: set[str] = set()
        for profile in chosen:
            needed.update(profile.descriptor_ids())
        return tuple(sorted(needed))

    def rules_by_id(self) -> dict[str, Rule]:
        collected: dict[str, Rule] = {}
        for profile in self.all():
            collected.update(profile.rules_by_id())
        return collected
