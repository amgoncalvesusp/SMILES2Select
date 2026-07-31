"""RDKit FilterCatalog wrappers.

Catalogues are built once per process and reused; construction is the expensive
part and a worker that scans a million molecules must not rebuild PAINS a
million times.
"""

from __future__ import annotations

from functools import lru_cache

from rdkit import Chem
from rdkit.Chem import FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams

# Catalogue id -> RDKit enum member name.
CATALOG_SPECS: dict[str, str] = {
    "pains": "PAINS",
    "pains_a": "PAINS_A",
    "pains_b": "PAINS_B",
    "pains_c": "PAINS_C",
    "brenk": "BRENK",
    "nih": "NIH",
    "zinc": "ZINC",
    "chembl_dundee": "CHEMBL_Dundee",
    "chembl_glaxo": "CHEMBL_Glaxo",
}

CATALOG_LABELS: dict[str, str] = {
    "pains": "PAINS (A+B+C)",
    "pains_a": "PAINS A",
    "pains_b": "PAINS B",
    "pains_c": "PAINS C",
    "brenk": "Brenk",
    "nih": "NIH",
    "zinc": "ZINC",
    "chembl_dundee": "CHEMBL Dundee",
    "chembl_glaxo": "CHEMBL Glaxo",
}

CATALOG_NOTES: dict[str, str] = {
    "pains": (
        "Substructures associated with recurrent assay interference. "
        "A hit is a flag, not a verdict on the molecule."
    ),
    "brenk": "Substructures considered problematic when assembling screening libraries.",
    "nih": "NIH filter set for undesirable functionality.",
    "zinc": "ZINC drug-likeness / unwanted functionality filters.",
}


class UnknownCatalogError(KeyError):
    """Raised when a catalogue id is not available in this RDKit build."""


def available_catalogs() -> tuple[str, ...]:
    """Catalogue ids supported by the installed RDKit."""
    return tuple(
        catalog_id
        for catalog_id, enum_name in CATALOG_SPECS.items()
        if hasattr(FilterCatalogParams.FilterCatalogs, enum_name)
    )


@lru_cache(maxsize=None)
def get_catalog(catalog_id: str) -> FilterCatalog.FilterCatalog:
    """Build (once) and return the RDKit catalogue for ``catalog_id``."""
    try:
        enum_name = CATALOG_SPECS[catalog_id]
    except KeyError as exc:
        raise UnknownCatalogError(
            f"unknown catalog '{catalog_id}'; available: {sorted(CATALOG_SPECS)}"
        ) from exc
    enum_member = getattr(FilterCatalogParams.FilterCatalogs, enum_name, None)
    if enum_member is None:
        raise UnknownCatalogError(f"catalog '{catalog_id}' is not available in this RDKit build")
    params = FilterCatalogParams()
    params.AddCatalog(enum_member)
    return FilterCatalog.FilterCatalog(params)


def match_catalog(mol: Chem.Mol, catalog_id: str) -> list[tuple[str, str]]:
    """Every entry of ``catalog_id`` that matches ``mol``.

    Returns ``(alert_name, description)`` pairs; an empty list means no hit.
    """
    catalog = get_catalog(catalog_id)
    hits: list[tuple[str, str]] = []
    for entry in catalog.GetMatches(mol):
        name = entry.GetDescription()
        # FilterCatalogEntry exposes properties only through GetPropList();
        # there is no HasProp on this class.
        scope = entry.GetProp("Scope") if "Scope" in entry.GetPropList() else ""
        hits.append((name, scope))
    return hits
