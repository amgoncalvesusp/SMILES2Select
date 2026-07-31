"""Selection policies.

A profile's *role* lives here, not in the profile itself, so the same Lipinski
definition can be mandatory in one run and informative in the next.

The recommended default deliberately does not require every profile to pass at
once: the profiles were built with different objectives, data sets and
descriptors, and intersecting all of them produces a needlessly small selection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from smiles2select.alerts.policies import AlertPolicy
from smiles2select.scores.qed import QedSelection

ProfileRole = Literal[
    "mandatory",
    "consensus",
    "informative",
    "ranking",
    "warning",
    "exclusion",
]

ROLE_LABELS: dict[str, str] = {
    "mandatory": "Obrigatório",
    "consensus": "Consenso",
    "informative": "Informativo",
    "ranking": "Ranqueamento",
    "warning": "Advertência",
    "exclusion": "Exclusão",
}

#: Roles that can remove a molecule from the final selection.
EXCLUDING_ROLES = frozenset({"mandatory", "exclusion"})


@dataclass(frozen=True)
class DecisionPolicy:
    """How profile verdicts, scores and alerts combine into a selection."""

    id: str = "recommended"
    roles: Mapping[str, ProfileRole] = field(default_factory=dict)
    consensus_min_pass: int | None = None
    expression: str | None = None
    qed: QedSelection = field(default_factory=QedSelection)
    alert_policy: AlertPolicy = field(default_factory=AlertPolicy)

    def __post_init__(self) -> None:
        for profile_id, role in self.roles.items():
            if role not in ROLE_LABELS:
                raise ValueError(f"profile '{profile_id}': unknown role '{role}'")
        consensus = self.consensus_profiles()
        if consensus and self.consensus_min_pass is None:
            raise ValueError("consensus profiles were selected but consensus_min_pass is not set")
        if self.consensus_min_pass is not None:
            if self.consensus_min_pass < 1:
                raise ValueError("consensus_min_pass must be at least 1")
            if consensus and self.consensus_min_pass > len(consensus):
                raise ValueError(
                    f"consensus_min_pass={self.consensus_min_pass} exceeds the "
                    f"{len(consensus)} consensus profile(s) selected"
                )

    def profiles_with_role(self, role: ProfileRole) -> tuple[str, ...]:
        return tuple(sorted(pid for pid, value in self.roles.items() if value == role))

    def mandatory_profiles(self) -> tuple[str, ...]:
        return self.profiles_with_role("mandatory")

    def consensus_profiles(self) -> tuple[str, ...]:
        return self.profiles_with_role("consensus")

    def informative_profiles(self) -> tuple[str, ...]:
        return self.profiles_with_role("informative")

    def evaluated_profiles(self) -> tuple[str, ...]:
        """Every profile that must be computed, whatever its role."""
        return tuple(sorted(self.roles))

    def excludes_on_alert(self) -> bool:
        return bool(self.alert_policy.excluding_catalogs())

    def with_role(self, profile_id: str, role: ProfileRole) -> DecisionPolicy:
        updated = dict(self.roles)
        updated[profile_id] = role
        return DecisionPolicy(
            id=self.id,
            roles=updated,
            consensus_min_pass=self.consensus_min_pass,
            expression=self.expression,
            qed=self.qed,
            alert_policy=self.alert_policy,
        )

    def describe(self) -> list[str]:
        """Plain-language summary shown before a run starts."""
        lines = [
            f"{profile_id}: {ROLE_LABELS[self.roles[profile_id]]}"
            for profile_id in sorted(self.roles)
        ]
        if self.consensus_min_pass is not None and self.consensus_profiles():
            lines.append(
                f"Consenso: aprovar em pelo menos {self.consensus_min_pass} de "
                f"{len(self.consensus_profiles())} perfis"
            )
        lines.append(self.qed.describe())
        lines.extend(self.alert_policy.describe())
        if self.expression:
            lines.append(f"Expressão personalizada: {self.expression}")
        return lines


#: Roles the specification recommends as the starting configuration.
RECOMMENDED_ROLES: dict[str, ProfileRole] = {
    "lipinski": "mandatory",
    "veber": "mandatory",
    "ghose": "informative",
    "egan": "informative",
    "muegge": "informative",
}


def recommended_policy(profile_ids: Sequence[str] = ()) -> DecisionPolicy:
    """The default offered to the user.

    Lipinski (up to one violation) and Veber are mandatory; Ghose, Egan and
    Muegge are informative; QED is computed and used for ranking; PAINS and
    Brenk warn without excluding.

    Passing ``profile_ids`` restricts the policy to those profiles, so a run
    that evaluates three profiles never carries roles for two it did not
    select.
    """
    if not profile_ids:
        roles = dict(RECOMMENDED_ROLES)
    else:
        roles = {
            profile_id: RECOMMENDED_ROLES.get(profile_id, "informative")
            for profile_id in profile_ids
        }
    return DecisionPolicy(
        id="recommended",
        roles=roles,
        qed=QedSelection(mode="rank"),
        alert_policy=AlertPolicy(),
    )


def single_profile_policy(profile_id: str) -> DecisionPolicy:
    """Approve on one profile alone."""
    return DecisionPolicy(id=f"single:{profile_id}", roles={profile_id: "mandatory"})


def all_profiles_policy(profile_ids: Sequence[str]) -> DecisionPolicy:
    """Approve only when every listed profile passes (an intentionally strict choice)."""
    return DecisionPolicy(
        id="all_profiles",
        roles={profile_id: "mandatory" for profile_id in profile_ids},
    )


def consensus_policy(profile_ids: Sequence[str], min_pass: int) -> DecisionPolicy:
    """Approve on N of M profiles."""
    return DecisionPolicy(
        id=f"consensus:{min_pass}_of_{len(profile_ids)}",
        roles={profile_id: "consensus" for profile_id in profile_ids},
        consensus_min_pass=min_pass,
    )
