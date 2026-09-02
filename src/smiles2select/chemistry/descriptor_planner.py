"""Execution planning for descriptor calculation.

The planner answers one question before any molecule is read: which descriptors
does this particular selection of profiles, scores and alerts actually need?
Anything not required is never computed - if only Lipinski is active, molar
refractivity, ring counts, QED and the PAINS catalogue stay untouched.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from smiles2select.chemistry.descriptor_registry import DescriptorRegistry

# Scores that are themselves descriptors depend on that descriptor; consensus
# needs no extra property beyond the profile results it summarises.
SCORE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "qed": ("qed",),
    "sa_score": ("sa_score",),
    "np_score": ("np_score",),
    "preparability": (
        "undefined_stereocenters",
        "defined_stereocenters",
        "fragment_count",
        "largest_ring_size",
        "amide_bond_count",
    ),
    "preparability_tautomers": ("tautomer_count",),
    "consensus": (),
    "profile_pass_fraction": (),
}


@dataclass(frozen=True)
class DescriptorPlan:
    """Descriptors to compute, with the reason each one is needed."""

    descriptor_ids: tuple[str, ...]
    reasons: dict[str, tuple[str, ...]]

    def __len__(self) -> int:
        return len(self.descriptor_ids)

    def explain(self) -> list[str]:
        return [
            f"{descriptor_id}: required by {', '.join(self.reasons[descriptor_id])}"
            for descriptor_id in self.descriptor_ids
        ]


class DescriptorPlanner:
    """Resolves the minimal descriptor set for a run."""

    def __init__(self, registry: DescriptorRegistry) -> None:
        self._registry = registry

    def resolve(
        self,
        profiles: Iterable[object] = (),
        scores: Iterable[str] = (),
        alerts: Iterable[str] = (),
    ) -> DescriptorPlan:
        """Return the plan for the selected profiles, scores and alert catalogs.

        ``profiles`` are :class:`~smiles2select.profiles.registry.Profile`
        objects (duck-typed here to keep the layers decoupled). ``scores`` are
        score ids such as ``qed``. ``alerts`` are catalog ids; structural alerts
        need no descriptors but are accepted so callers can pass the whole
        selection in one call.
        """
        reasons: dict[str, list[str]] = {}

        for profile in profiles:
            for rule in getattr(profile, "rules", ()):
                descriptor_id = getattr(rule, "descriptor", None)
                if not descriptor_id:
                    continue  # substructure rules carry a SMARTS, not a descriptor
                self._registry.get(descriptor_id)  # fail fast on typos in a profile file
                reasons.setdefault(descriptor_id, []).append(f"profile:{profile.id}")

        for score_id in scores:
            for descriptor_id in SCORE_DEPENDENCIES.get(score_id, ()):
                self._registry.get(descriptor_id)
                reasons.setdefault(descriptor_id, []).append(f"score:{score_id}")

        for _catalog_id in alerts:
            # Structural alerts run on the molecule graph, not on descriptors.
            pass

        ordered = tuple(sorted(reasons))
        return DescriptorPlan(
            descriptor_ids=ordered,
            reasons={key: tuple(dict.fromkeys(reasons[key])) for key in ordered},
        )
