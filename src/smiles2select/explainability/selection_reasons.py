"""Deterministic explanations for inclusion and exclusion decisions."""

from __future__ import annotations

from collections.abc import Mapping


def _value(row: Mapping[str, object], key: str, default: object = None) -> object:
    try:
        return row.get(key, default)
    except AttributeError:
        return default


def explain_selected(row: Mapping[str, object]) -> list[str]:
    """Generate stable reasons from recorded metadata, never from an LLM."""

    reasons: list[str] = []
    if _value(row, "pinned", False):
        reasons.append("Pinned manually by the user")
    zone = _value(row, "selection_zone") or _value(row, "zone_id")
    if zone:
        reasons.append(f"Selected from zone {zone}")
    novelty = _value(row, "reference_novelty")
    if novelty is not None:
        try:
            reasons.append(f"Reference novelty = {float(novelty):.3f}")
        except (TypeError, ValueError):
            pass
    scaffold = _value(row, "murcko_scaffold")
    if _value(row, "first_scaffold_representative", False) and scaffold:
        reasons.append(f"First representative of scaffold {scaffold}")
    strategy = _value(row, "selection_strategy") or _value(row, "strategy")
    if strategy:
        reasons.append(f"Selected by {str(strategy).replace('_', ' ').title()}")
    explicit = _value(row, "selection_reason")
    if explicit and str(explicit) not in reasons:
        reasons.append(str(explicit))
    return reasons or ["Selected by the configured eligibility and selection policy"]


def explain_not_selected(row: Mapping[str, object]) -> list[str]:
    """Generate stable exclusion reasons from the recorded decision metadata."""

    reasons: list[str] = []
    for key, label in (
        ("is_reference_duplicate", "Exact reference duplicate"),
        ("dockability_failed", "Dockability Envelope failure"),
        ("scaffold_quota_reached", "Scaffold quota reached"),
        ("zone_quota_reached", "Zone quota reached"),
        ("manually_excluded", "Manually excluded"),
    ):
        if _value(row, key, False):
            reasons.append(label)
    explicit = _value(row, "exclusion_reason") or _value(row, "exclusion_reasons")
    if explicit:
        reasons.extend(
            item.strip() for item in str(explicit).split(";") if item.strip() and item.strip() not in reasons
        )
    return reasons or ["Not selected after applying the configured selection plan"]
