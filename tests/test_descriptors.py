"""Descriptor registry and planner."""

from __future__ import annotations

import pytest
from rdkit import Chem

from smiles2select.chemistry.descriptor_planner import DescriptorPlanner
from smiles2select.chemistry.descriptor_registry import (
    DescriptorDefinition,
    DescriptorRegistry,
    UnknownDescriptorError,
)
from tests.conftest import REFERENCE_SMILES

pytestmark = pytest.mark.unit


def mol(name: str) -> Chem.Mol:
    return Chem.MolFromSmiles(REFERENCE_SMILES[name])


def test_aspirin_descriptors_match_published_values(descriptors):
    values = descriptors.compute(mol("aspirin"), descriptors.ids())
    assert values["mol_wt"] == pytest.approx(180.16, abs=0.01)
    assert values["hbd_lipinski"] == 1
    assert values["hba_lipinski"] == 4
    assert values["tpsa"] == pytest.approx(63.60, abs=0.01)
    assert values["rotatable_bonds"] == 2
    assert values["ring_count"] == 1
    assert 0.0 <= values["qed"] <= 1.0


def test_atom_count_conventions_are_distinct(descriptors):
    """Ghose bounds atoms *with* hydrogens; the heavy-atom count is different."""
    values = descriptors.compute(
        mol("benzene"), ("heavy_atom_count", "explicit_atom_count", "total_atom_count")
    )
    assert values["heavy_atom_count"] == 6
    assert values["explicit_atom_count"] == 6
    assert values["total_atom_count"] == 12


def test_there_is_no_generic_logp_field(descriptors):
    """LogP columns must name their method so different estimates never mix."""
    assert "logp" not in descriptors.ids()
    assert "rdkit_wlogp" in descriptors.ids()
    assert "Crippen" in descriptors.get("rdkit_wlogp").function


def test_failing_descriptor_yields_none_not_exception():
    """One broken property must not discard the whole record."""

    def explode(_mol):
        raise RuntimeError("boom")

    registry = DescriptorRegistry(
        [
            DescriptorDefinition(
                id="broken",
                label="Broken",
                provider="test",
                function="explode",
                compute=explode,
            )
        ]
    )
    assert registry.compute(mol("aspirin"), ("broken",)) == {"broken": None}


def test_unknown_descriptor_is_reported(descriptors):
    with pytest.raises(UnknownDescriptorError):
        descriptors.get("does_not_exist")


def test_planner_computes_only_what_is_needed(descriptors, profiles):
    """Lipinski alone must not drag in MR, rings or QED."""
    plan = DescriptorPlanner(descriptors).resolve(profiles=[profiles.get("lipinski")])
    assert set(plan.descriptor_ids) == {"mol_wt", "rdkit_wlogp", "hbd_lipinski", "hba_lipinski"}
    assert "mol_mr" not in plan.descriptor_ids
    assert "ring_count" not in plan.descriptor_ids
    assert "qed" not in plan.descriptor_ids


def test_planner_adds_qed_only_when_the_score_is_selected(descriptors, profiles):
    plan = DescriptorPlanner(descriptors).resolve(
        profiles=[profiles.get("lipinski")], scores=["qed"]
    )
    assert "qed" in plan.descriptor_ids
    assert any("score:qed" in reason for reason in plan.reasons["qed"])


def test_planner_deduplicates_shared_descriptors(descriptors, profiles):
    """A descriptor used by two profiles appears once, with both reasons."""
    plan = DescriptorPlanner(descriptors).resolve(
        profiles=[profiles.get("lipinski"), profiles.get("ghose")]
    )
    assert len(plan.descriptor_ids) == len(set(plan.descriptor_ids))
    assert set(plan.reasons["mol_wt"]) == {"profile:lipinski", "profile:ghose"}


def test_alert_catalogs_need_no_descriptors(descriptors, profiles):
    plan = DescriptorPlanner(descriptors).resolve(
        profiles=[profiles.get("lipinski")], alerts=["pains", "brenk"]
    )
    assert set(plan.descriptor_ids) == {"mol_wt", "rdkit_wlogp", "hbd_lipinski", "hba_lipinski"}
