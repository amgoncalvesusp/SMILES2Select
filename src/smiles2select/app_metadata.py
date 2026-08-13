"""Application identity and provenance helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any

APP_NAME = "SMILES2Select"
APP_VERSION = "3.0.1"

AUTHOR = "Adriano Marques Gonçalves"
AFFILIATION = "Universidade de Araraquara (UNIARA)"
AUTHORSHIP = f"{AUTHOR} — {AFFILIATION}"

DISCLAIMER = (
    "Drug-likeness profiles are heuristics. A molecule that passes every "
    "profile is not thereby predicted to be orally bioavailable, effective or "
    "safe; a molecule that fails is not thereby disqualified."
)

SELECTION_SENTENCE = (
    "A molecule is selected when it passes the mandatory profiles. Informative "
    "profiles are calculated and included in the report but do not exclude "
    "molecules. PAINS and Brenk alerts do not exclude molecules by default."
)


def rdkit_version() -> str:
    """Return the installed RDKit version, or ``unknown`` if RDKit is absent."""
    try:
        import rdkit

        return str(rdkit.__version__)
    except Exception:  # pragma: no cover - only when RDKit is missing
        return "unknown"


def config_hash(config: Any) -> str:
    """Stable SHA-256 of a JSON-serialisable configuration object.

    Used to invalidate cached descriptor values whenever the standardization
    profile, the descriptor set or the toolkit version changes.
    """
    payload = json.dumps(config, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
