"""English method cards shown by the Chemical Space Hub."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodCard:
    """A complete explanation for a selectable method or operational control."""

    method_id: str
    title: str
    what_it_does: str
    how_it_works: str
    what_it_favors: str
    may_underrepresent: str
    practical_consequence: str
    when_to_use: str
    when_not_to_use: str
    reproducibility_notes: str
    methodological_reference: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "method_id": self.method_id,
            "title": self.title,
            "what_it_does": self.what_it_does,
            "how_it_works": self.how_it_works,
            "what_it_favors": self.what_it_favors,
            "may_underrepresent": self.may_underrepresent,
            "practical_consequence": self.practical_consequence,
            "when_to_use": self.when_to_use,
            "when_not_to_use": self.when_not_to_use,
            "reproducibility_notes": self.reproducibility_notes,
            "methodological_reference": self.methodological_reference,
        }


_CARDS = (
    MethodCard(
        "balanced",
        "Balanced",
        "Combines several configured objectives without letting one criterion dominate.",
        "Ranks candidates using the active score and eligibility information.",
        "Compromise libraries with broad practical quality.",
        "Extremes that would be favored by a single exploratory objective.",
        "The result is a compromise and should be compared with focused strategies.",
        "A general-purpose starting point.",
        "A clear exploration or exploitation hypothesis is required.",
        "Store every objective, threshold, tie-break rule and seed in the recipe.",
    ),
    MethodCard(
        "diversity_first",
        "Diversity First",
        "Prioritizes molecules that expand structural coverage of the selected set.",
        "Uses fingerprint distance to update each candidate's distance from selected molecules.",
        "Rare chemotypes and broad structural coverage.",
        "Dense analogue series and locally optimized chemistry.",
        "The final library spans more distinct regions but contains fewer close analogues.",
        "Exploratory virtual screening.",
        "Focused screening around one known chemotype.",
        "Record Morgan radius, bit count, chirality and seed; 2D map distances are not used.",
    ),
    MethodCard(
        "reference_novelty",
        "Reference Novelty",
        "Prioritizes candidates poorly represented by reference libraries.",
        "Ranks by 1 minus the maximum exact Tanimoto similarity to the references.",
        "Candidate-only scaffolds and remote fingerprint neighborhoods.",
        "Close analogues of known bioactives.",
        "The result explores chemistry that may be less connected to known activity space.",
        "When the scientific question emphasizes unexplored chemical space.",
        "When known-bioactive neighborhood coverage is the primary objective.",
        "Record reference library IDs, search mode, fingerprint settings and the threshold used.",
    ),
    MethodCard(
        "reference_neighborhood",
        "Reference Neighborhood",
        "Selects candidates within a user-defined similarity window around references.",
        "Filters by exact maximum reference Tanimoto and keeps the configured interval visible.",
        "Chemical neighborhoods already represented by known compounds.",
        "Remote or candidate-only chemistry.",
        "Higher similarity favors exploitation and reduces exploration of distant regions.",
        "Analogue expansion and focused follow-up libraries.",
        "A novelty-first screen with no intended reference complementarity.",
        "Store both bounds; similarity thresholds are not universal scientific cutoffs.",
    ),
    MethodCard(
        "reference_aware_diversity",
        "Reference-Aware Diversity",
        "Builds a structurally diverse library complementary to the references.",
        "Initial distance comes from reference novelty, then MaxMin updates distance from selected candidates.",
        "Chemistry far from both covered reference space and already selected molecules.",
        "Repeated analogues and chemistry close to the reference library.",
        "The result balances exploration against internal coverage rather than optimizing either alone.",
        "Complementary screening around a known bioactive library.",
        "A purely local analogue campaign.",
        "Record exact/approximate discovery mode, verified similarities, fingerprint settings and seed; never use 2D map distance as novelty.",
    ),
    MethodCard(
        "scaffold_coverage",
        "Scaffold Coverage",
        "Prioritizes distinct Murcko frameworks.",
        "Ranks candidates to represent under-covered scaffolds before applying quotas.",
        "Framework breadth and rare scaffold representation.",
        "Multiple valuable analogues from a dense scaffold family.",
        "A rare scaffold may receive a representative even when its individual score is modest.",
        "Early chemical-space exploration.",
        "A campaign requiring many analogues from a validated series.",
        "Record scaffold definition, frequencies, quota and deterministic tie-breaking.",
    ),
    MethodCard(
        "dockability_envelope",
        "Dockability Envelope",
        "Removes operational extremes that are unattractive for the intended docking workflow.",
        "Applies editable MW, TPSA, HBD, HBA, rotatable-bond and WLogP boundaries.",
        "Structures likely to be manageable in downstream preparation.",
        "Large, highly polar, highly flexible or otherwise operationally difficult molecules.",
        "These are operational defaults, not universal biological or drug-likeness thresholds.",
        "Preparing a library for a specified docking toolchain.",
        "When unusual natural products are scientifically central and can be prepared downstream.",
        "Store every bound, enabled flag and the software/workflow context.",
    ),
    MethodCard(
        "property_pca",
        "Property PCA",
        "Visualizes broad physicochemical differences.",
        "Standardizes configured descriptors and projects them with deterministic PCA.",
        "Property-profile similarity.",
        "Scaffold similarity that is not reflected by the chosen descriptors.",
        "Nearby points have similar descriptors, not necessarily similar scaffolds.",
        "Comparing broad physicochemical envelopes.",
        "Inferring structural novelty from map distance.",
        "Store features, scaling, input hash, software versions and projection settings.",
    ),
    MethodCard(
        "structural_umap",
        "Structural UMAP",
        "Visualizes local structural neighborhoods and chemical families.",
        "Embeds Morgan fingerprints with configurable UMAP parameters and seed.",
        "Local fingerprint neighborhoods.",
        "Literal global distance interpretation and reproducibility across changed parameters.",
        "The map is for exploration; fingerprint similarity remains the novelty metric.",
        "Interactive investigation of large structural libraries.",
        "Using a 2D distance as a selection or novelty threshold.",
        "Store n_neighbors, min_dist, metric, seed, fingerprint and input hash.",
    ),
)

_BY_ID = {card.method_id: card for card in _CARDS}


def method_cards() -> tuple[MethodCard, ...]:
    return _CARDS


def get_method_card(method_id: str) -> MethodCard:
    try:
        return _BY_ID[method_id]
    except KeyError as exc:
        raise KeyError(f"unknown method card '{method_id}'; available: {sorted(_BY_ID)}") from exc
