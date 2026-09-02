"""Unit tests for docking preparability descriptors."""

from __future__ import annotations

import pytest
from rdkit import Chem

from smiles2select.chemistry import preparability

pytestmark = pytest.mark.unit


def _mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"invalid test SMILES: {smiles}"
    return mol


def test_undefined_stereocenters():
    assert preparability.undefined_stereocenters(_mol("CC(O)C(N)C=CC1CCCCC1")) == 3
    assert preparability.undefined_stereocenters(_mol("C[C@H](O)C(=O)O")) == 0
    assert preparability.undefined_stereocenters(_mol("CCO")) == 0
    assert preparability.undefined_stereocenters(_mol("O=C(N)/C=C/C")) == 0


def test_defined_stereocenters():
    assert preparability.defined_stereocenters(_mol("CC(O)C(N)C=CC1CCCCC1")) == 0
    assert preparability.defined_stereocenters(_mol("C[C@H](O)C(=O)O")) == 1
    assert preparability.defined_stereocenters(_mol("CCO")) == 0
    assert preparability.defined_stereocenters(_mol("O=C(N)/C=C/C")) == 1


def test_fragment_count():
    assert preparability.fragment_count(_mol("CC(=O)O.[Na+]")) == 2
    assert preparability.fragment_count(_mol("CCO")) == 1


def test_largest_ring_size():
    assert preparability.largest_ring_size(_mol("C1CCCCCCCCCCC1")) == 12
    assert preparability.largest_ring_size(_mol("CCO")) == 0


def test_amide_bond_count():
    assert preparability.amide_bond_count(_mol("CC(=O)NCC(=O)NC")) == 2
    assert preparability.amide_bond_count(_mol("CCO")) == 0


def test_bridgehead_atom_count():
    assert preparability.bridgehead_atom_count(_mol("C1CC2CCC1CC2")) == 2
    assert preparability.bridgehead_atom_count(_mol("CCO")) == 0


def test_spiro_atom_count():
    assert preparability.spiro_atom_count(_mol("C1CCC2(CC1)CCCC2")) == 1
    assert preparability.spiro_atom_count(_mol("CCO")) == 0


def test_tautomer_count():
    assert preparability.tautomer_count(_mol("CCO")) == 1
    assert preparability.tautomer_count(_mol("O=C1CCCCC1")) == 2
    assert preparability.tautomer_count(_mol("c1ccccc1O")) == 2
    assert preparability.tautomer_count(_mol("CC(=O)CC(=O)C")) == 5


def test_estimated_structures_doubles_per_stereocenter():
    assert preparability.estimated_3d_structures(0, 1) == 1
    assert preparability.estimated_3d_structures(1, 1) == 2
    assert preparability.estimated_3d_structures(3, 1) == 8


def test_estimated_structures_multiplies_by_tautomers():
    assert preparability.estimated_3d_structures(2, 3) == 12


def test_estimated_structures_is_capped():
    assert preparability.estimated_3d_structures(20, 1) == 1024


def test_estimated_structures_rejects_invalid_input():
    with pytest.raises(ValueError):
        preparability.estimated_3d_structures(-1, 1)
    with pytest.raises(ValueError):
        preparability.estimated_3d_structures(1, 0)
