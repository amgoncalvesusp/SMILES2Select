"""Descriptors that predict the cost of taking a molecule to 3D.

None of these judge a molecule. They state how many distinct structures a
docking run would have to prepare, and where the input left a decision open
that some downstream tool will otherwise make silently.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize


def undefined_stereocenters(mol: Chem.Mol) -> int:
    """Stereo elements RDKit perceives but the input never specified.

    Covers tetrahedral atoms and double bonds in a single pass. Each one
    doubles the number of structures a docking run must prepare, which is why
    it is counted rather than resolved here.
    """
    elements = Chem.FindPotentialStereo(mol)
    return sum(1 for item in elements if item.specified == Chem.StereoSpecified.Unspecified)


def defined_stereocenters(mol: Chem.Mol) -> int:
    """Stereo elements the input did specify."""
    elements = Chem.FindPotentialStereo(mol)
    return sum(1 for item in elements if item.specified == Chem.StereoSpecified.Specified)


def fragment_count(mol: Chem.Mol) -> int:
    """Disconnected fragments left after standardization.

    More than one means the record is a salt or mixture that survived the
    standardization profile in force.
    """
    return len(Chem.GetMolFrags(mol))


def largest_ring_size(mol: Chem.Mol) -> int:
    """Atoms in the largest ring; 0 for acyclic molecules.

    Twelve or more marks a macrocycle, whose conformational sampling behaves
    differently from an ordinary drug-like ring system.
    """
    rings = mol.GetRingInfo().AtomRings()
    return max((len(ring) for ring in rings), default=0)


def amide_bond_count(mol: Chem.Mol) -> int:
    """Amide bonds, a proxy for peptide-like character."""
    return rdMolDescriptors.CalcNumAmideBonds(mol)


def bridgehead_atom_count(mol: Chem.Mol) -> int:
    return rdMolDescriptors.CalcNumBridgeheadAtoms(mol)


def spiro_atom_count(mol: Chem.Mol) -> int:
    return rdMolDescriptors.CalcNumSpiroAtoms(mol)


#: Enumerating beyond this is pointless: a molecule with more than sixteen
#: plausible tautomers is ambiguous by any standard, and the exact count adds
#: nothing to that verdict while costing a great deal of time.
MAX_TAUTOMERS = 16


def tautomer_count(mol: Chem.Mol) -> int:
    """Plausible tautomers, capped at ``MAX_TAUTOMERS``.

    Roughly twenty-four times more expensive than every other descriptor here,
    which is why it is never computed unless explicitly requested.
    """
    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetMaxTautomers(MAX_TAUTOMERS)
    return len(enumerator.Enumerate(mol))


#: Above this, the count stops being a plan and becomes a warning.
MAX_REPORTED_STRUCTURES = 1024


def estimated_3d_structures(undefined_stereo: int, tautomers: int = 1) -> int:
    """How many distinct structures a preparation step would have to build.

    Each unspecified stereo element doubles the count; each additional
    plausible tautomer multiplies it. The number is capped because past a
    thousand structures the exact value changes no decision - the molecule is
    simply not worth preparing at that price.
    """
    if undefined_stereo < 0 or tautomers < 1:
        raise ValueError("undefined_stereo must be >= 0 and tautomers >= 1")
    total = (2**undefined_stereo) * tautomers
    return min(total, MAX_REPORTED_STRUCTURES)


def uncommon_elements(mol: Chem.Mol, common: frozenset[str]) -> tuple[str, ...]:
    """Element symbols outside the given set, in order of first appearance."""
    seen: list[str] = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in common and symbol not in seen:
            seen.append(symbol)
    return tuple(seen)
