"""Consequence-aware messages for controls that change many molecules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Consequence:
    control: str
    before: object
    after: object
    affected_count: int | None
    message: str


def explain_change(
    control: str,
    before: object,
    after: object,
    *,
    affected_count: int | None = None,
) -> Consequence:
    """Return a plain-English explanation suitable for a persistent GUI panel."""

    key = control.strip().lower().replace(" ", "_")
    if key in {"lipinski", "mandatory_lipinski", "lipinski_action"} and str(after).lower() in {
        "mandatory",
        "exclude",
        "hard",
        "true",
    }:
        message = (
            "Making Lipinski mandatory preferentially removes compounds outside "
            "conventional oral drug-like space, including many natural products "
            "and beyond-Rule-of-Five molecules."
        )
    elif key in {"pains", "brenk"} and str(after).lower() in {"warn", "warning", "inform"}:
        message = (
            f"{control.upper()} matches remain eligible and will be flagged in the "
            "report and Molecule Inspector."
        )
    elif key in {"pains", "brenk"} and str(after).lower() in {"exclude", "hard"}:
        message = (
            f"{control.upper()} exclusion removes alert-matching molecules from the "
            "eligible pool; the alert is not universal proof of unsuitability."
        )
    elif key in {"max_per_scaffold", "scaffold_quota"}:
        message = (
            "A scaffold quota reduces domination by repeated chemotypes and increases "
            "framework coverage; dense analogue series lose representatives."
        )
    elif key in {"reference_similarity", "similarity_threshold"}:
        message = (
            "Changing the similarity threshold changes which reference neighborhoods "
            "are considered covered; it does not change the fingerprint definition."
        )
    else:
        message = (
            "This change affects the eligible or ranked pool. Review the before/after "
            "counts and the saved recipe before committing it."
        )
    if affected_count is not None:
        message += f" Current estimated affected molecules: {affected_count}."
    return Consequence(control, before, after, affected_count, message)
