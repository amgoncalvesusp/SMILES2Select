"""SMILES parsing and validation.

Parsing failures are first-class results, never exceptions that abort a run:
one bad row in a million-molecule library must not lose the other 999 999.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem, RDLogger

# RDKit writes parse errors to stderr; we capture failures in the result object.
RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")


@dataclass(frozen=True)
class ParseResult:
    """Outcome of parsing a single SMILES string."""

    smiles: str
    mol: Chem.Mol | None
    valid: bool
    error: str | None = None

    @property
    def is_empty(self) -> bool:
        return self.mol is not None and self.mol.GetNumAtoms() == 0


def parse_smiles(smiles: str | None, *, sanitize: bool = True) -> ParseResult:
    """Parse a SMILES string into an RDKit molecule.

    Returns a :class:`ParseResult` whose ``valid`` flag is False for empty,
    non-string or chemically unparseable input.
    """
    if smiles is None:
        return ParseResult("", None, False, "empty input")
    text = str(smiles).strip()
    if not text:
        return ParseResult("", None, False, "empty input")

    try:
        mol = Chem.MolFromSmiles(text, sanitize=sanitize)
    except Exception as exc:  # RDKit can raise on pathological input
        return ParseResult(text, None, False, f"parser error: {exc}")

    if mol is None:
        return ParseResult(text, None, False, "invalid SMILES")
    if mol.GetNumAtoms() == 0:
        return ParseResult(text, None, False, "no atoms")
    return ParseResult(text, mol, True, None)


def canonical_smiles(mol: Chem.Mol) -> str:
    """Canonical isomeric SMILES used as the cache key for a structure."""
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)


def has_carbon(mol: Chem.Mol) -> bool:
    return any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())


def is_mixture(smiles: str) -> bool:
    """True when the SMILES encodes more than one disconnected component."""
    return "." in smiles
