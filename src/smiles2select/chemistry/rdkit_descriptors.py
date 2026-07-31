"""Concrete RDKit descriptor implementations.

Naming is deliberately explicit about the *method*, not only the property.
There is no generic ``logp`` field: ``rdkit_wlogp`` is the Wildman-Crippen
estimate returned by ``Crippen.MolLogP``, and a future XLOGP3 or MLOGP backend
gets its own column so values computed by different methods are never compared
as if they were interchangeable.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, QED, RDConfig, rdMolDescriptors


def mol_wt(mol: Chem.Mol) -> float:
    return Descriptors.MolWt(mol)


def rdkit_wlogp(mol: Chem.Mol) -> float:
    """Wildman-Crippen LogP (``Crippen.MolLogP``), the WLOGP of SwissADME."""
    return Crippen.MolLogP(mol)


def mol_mr(mol: Chem.Mol) -> float:
    """Wildman-Crippen molar refractivity."""
    return Crippen.MolMR(mol)


def hbd_lipinski(mol: Chem.Mol) -> int:
    """Hydrogen-bond donors, Lipinski definition (N-H and O-H count)."""
    return rdMolDescriptors.CalcNumLipinskiHBD(mol)


def hba_lipinski(mol: Chem.Mol) -> int:
    """Hydrogen-bond acceptors, Lipinski definition (N and O count)."""
    return rdMolDescriptors.CalcNumLipinskiHBA(mol)


def tpsa(mol: Chem.Mol) -> float:
    return rdMolDescriptors.CalcTPSA(mol)


def rotatable_bonds(mol: Chem.Mol) -> int:
    return rdMolDescriptors.CalcNumRotatableBonds(mol)


def heavy_atom_count(mol: Chem.Mol) -> int:
    """Non-hydrogen atoms."""
    return mol.GetNumHeavyAtoms()


def explicit_atom_count(mol: Chem.Mol) -> int:
    """Atoms present in the graph as drawn (implicit H excluded)."""
    return mol.GetNumAtoms()


def total_atom_count(mol: Chem.Mol) -> int:
    """All atoms including hydrogens.

    This is the count used by the Ghose profile: the original filter bounds the
    total number of atoms (20-70) with hydrogens included, not the heavy-atom
    count. The distinction changes the verdict for small molecules, so the
    descriptor id states which convention is in force.
    """
    return Chem.AddHs(mol).GetNumAtoms()


def carbon_count(mol: Chem.Mol) -> int:
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)


def heteroatom_count(mol: Chem.Mol) -> int:
    return rdMolDescriptors.CalcNumHeteroatoms(mol)


def ring_count(mol: Chem.Mol) -> int:
    return rdMolDescriptors.CalcNumRings(mol)


def aromatic_ring_count(mol: Chem.Mol) -> int:
    return rdMolDescriptors.CalcNumAromaticRings(mol)


def formal_charge(mol: Chem.Mol) -> int:
    return Chem.GetFormalCharge(mol)


def fraction_csp3(mol: Chem.Mol) -> float:
    return rdMolDescriptors.CalcFractionCSP3(mol)


def qed(mol: Chem.Mol) -> float:
    """Quantitative Estimate of Drug-likeness, continuous in [0, 1]."""
    return QED.qed(mol)


def nitrogen_oxygen_count(mol: Chem.Mol) -> int:
    """Nitrogen + oxygen atoms.

    Kept separate from ``hba_lipinski`` even though RDKit currently computes the
    Lipinski acceptor count the same way: profiles that bound "N + O" mean the
    atom count, and conflating the two would hide it if either definition
    changes.
    """
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in (7, 8))


@lru_cache(maxsize=1)
def _sascorer():
    """Import the SA_Score contrib module, which is not on the default path."""
    contrib = Path(RDConfig.RDContribDir) / "SA_Score"
    if str(contrib) not in sys.path:
        sys.path.append(str(contrib))
    import sascorer

    return sascorer


@lru_cache(maxsize=1)
def _npscorer():
    """Import the NP_Score contrib module and read its fragment model once."""
    contrib = Path(RDConfig.RDContribDir) / "NP_Score"
    if str(contrib) not in sys.path:
        sys.path.append(str(contrib))
    import npscorer

    return npscorer, npscorer.readNPModel()


def np_score(mol: Chem.Mol) -> float:
    """Natural-product likeness, roughly -5 (synthetic) to +5 (natural-like).

    Ertl, Roggo & Schuffenhauer's fragment score. Like QED and SA, it is a
    continuous score for ranking: a high value means the fragments resemble
    those of natural products, not that the molecule *is* a natural product or
    that it is preferable.
    """
    scorer, model = _npscorer()
    return float(scorer.scoreMol(mol, model))


def sa_score(mol: Chem.Mol) -> float:
    """Synthetic accessibility, 1 (easy to make) to 10 (hard).

    Ertl & Schuffenhauer's heuristic, from RDKit's contrib tree. It is a
    continuous score for ranking, never a pass/fail rule: a high value means the
    fragment contributions look unusual against a reference set, not that the
    molecule cannot be synthesised.
    """
    return float(_sascorer().calculateScore(mol))
