"""Wording of the selection policy, shown before and after a run."""

from __future__ import annotations

from smiles2select.app_metadata import SELECTION_SENTENCE
from smiles2select.decision.policies import ROLE_LABELS, DecisionPolicy


def policy_sentence(policy: DecisionPolicy) -> str:
    """The sentence the interface must display before executing.

    It exists so a user who ticked eight filters realises they have built an
    intersection, and so nobody mistakes a warning for an exclusion.
    """
    mandatory = policy.mandatory_profiles()
    informative = policy.informative_profiles()
    parts = [SELECTION_SENTENCE]
    if mandatory:
        parts.append(f"Perfis obrigatórios: {', '.join(mandatory)}.")
    if informative:
        parts.append(f"Perfis informativos: {', '.join(informative)}.")
    if policy.excludes_on_alert():
        catalogs = ", ".join(policy.alert_policy.excluding_catalogs())
        parts.append(f"ATENÇÃO: os catálogos {catalogs} estão configurados para EXCLUIR.")
    return " ".join(parts)


def policy_rows(policy: DecisionPolicy) -> list[dict[str, str]]:
    """Table rows for the CONFIG sheet and the GUI profile table."""
    return [
        {
            "profile_id": profile_id,
            "role": ROLE_LABELS[policy.roles[profile_id]],
            "excludes": "sim" if policy.roles[profile_id] in {"mandatory", "exclusion"} else "não",
        }
        for profile_id in sorted(policy.roles)
    ]


def restrictiveness_warning(policy: DecisionPolicy) -> str | None:
    """Warn when many mandatory profiles are stacked into one intersection."""
    mandatory = policy.mandatory_profiles()
    if len(mandatory) < 4:
        return None
    return (
        f"{len(mandatory)} perfis estão marcados como obrigatórios "
        f"({', '.join(mandatory)}). Eles foram desenvolvidos com objetivos, conjuntos de "
        "dados e descritores diferentes; exigir aprovação simultânea em todos tende a "
        "produzir uma interseção excessivamente restritiva."
    )
