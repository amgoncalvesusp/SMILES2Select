"""Built-in profiles: definitions, boundaries and cross-profile disagreement."""

from __future__ import annotations

import pytest
from rdkit import Chem

from smiles2select.profiles.loader import BUILTIN_DIR, ProfileFileError, profile_from_dict
from smiles2select.profiles.validator import validate_all, validate_profile
from smiles2select.rules.evaluator import evaluate_profiles
from tests.conftest import REFERENCE_SMILES, descriptor_frame

pytestmark = pytest.mark.unit

EXPECTED_PROFILES = {
    "lipinski",
    "veber",
    "ghose",
    "egan",
    "muegge",
    "ro3_core",
    "ro3_extended",
    "lead_like",
    "cns_like",
    "beyond_ro5",
}


def test_all_expected_profiles_ship(profiles):
    assert EXPECTED_PROFILES.issubset(set(profiles.ids()))


def test_every_builtin_profile_is_valid(profiles, descriptors):
    assert validate_all(profiles.all(), descriptors) == {
        profile.id: [] for profile in profiles.all()
    }


def test_every_profile_file_is_loadable():
    assert len(sorted(BUILTIN_DIR.glob("*.json"))) == len(EXPECTED_PROFILES)


def test_lipinski_ships_with_the_classical_policy(profiles):
    """Both readings must be available; the tolerant one is the default."""
    lipinski = profiles.get("lipinski")
    assert lipinski.pass_policy == {"type": "max_violations", "value": 1}
    strict = lipinski.with_pass_policy({"type": "all_rules"})
    assert strict.pass_policy["type"] == "all_rules"
    assert lipinski.pass_policy["type"] == "max_violations"  # original untouched


def test_lipinski_thresholds(profiles):
    thresholds = {rule.descriptor: rule.threshold for rule in profiles.get("lipinski").rules}
    assert thresholds == {
        "mol_wt": 500,
        "rdkit_wlogp": 5,
        "hbd_lipinski": 5,
        "hba_lipinski": 10,
    }


def test_veber_thresholds(profiles):
    thresholds = {rule.descriptor: rule.threshold for rule in profiles.get("veber").rules}
    assert thresholds == {"rotatable_bonds": 10, "tpsa": 140}


def test_ghose_uses_total_atom_count_with_hydrogens(profiles):
    ghose = profiles.get("ghose")
    atom_rules = [rule for rule in ghose.rules if "atoms" in rule.id]
    assert {rule.descriptor for rule in atom_rules} == {"total_atom_count"}
    assert "hydrogens" in ghose.atom_count_definition


def test_egan_is_declared_as_swissadme_compatible_not_original(profiles):
    egan = profiles.get("egan")
    assert "SwissADME" in egan.name
    assert "NOT a reproduction" in egan.notes
    assert egan.logp_method.startswith("rdkit_wlogp")


def test_muegge_is_declared_as_an_rdkit_adaptation(profiles):
    muegge = profiles.get("muegge")
    assert "adaptação RDKit" in muegge.name
    assert "XLOGP3" in muegge.notes


def test_rule_of_three_is_not_presented_as_drug_likeness(profiles):
    for profile_id in ("ro3_core", "ro3_extended"):
        profile = profiles.get(profile_id)
        assert profile.category == "fragment_space"
        assert "fragment" in profile.notes.lower()


def test_lead_like_is_its_own_chemical_space(profiles):
    lead_like = profiles.get("lead_like")
    assert lead_like.category == "lead_space"
    assert "optimisation" in lead_like.notes


@pytest.mark.parametrize(("mol_wt", "expected"), [(499.9, True), (500.0, True), (500.1, False)])
def test_lipinski_molecular_weight_boundary(profiles, mol_wt, expected):
    """At the limit, just below and just above."""
    strict = profiles.get("lipinski").with_pass_policy({"type": "all_rules"})
    frame = descriptor_frame(
        {
            "mol_wt": [mol_wt],
            "rdkit_wlogp": [2.0],
            "hbd_lipinski": [1],
            "hba_lipinski": [3],
        }
    )
    assert evaluate_profiles(frame, [strict]).passed("lipinski").tolist() == [expected]


@pytest.mark.parametrize(("tpsa", "expected"), [(139.9, True), (140.0, True), (140.1, False)])
def test_veber_tpsa_boundary(profiles, tpsa, expected):
    frame = descriptor_frame({"tpsa": [tpsa], "rotatable_bonds": [5]})
    assert evaluate_profiles(frame, [profiles.get("veber")]).passed("veber").tolist() == [expected]


def test_ghose_reports_which_side_of_the_range_was_crossed(profiles):
    frame = descriptor_frame(
        {
            "mol_wt": [150.0, 500.0],
            "rdkit_wlogp": [1.0, 1.0],
            "mol_mr": [50.0, 50.0],
            "total_atom_count": [30, 30],
        }
    )
    evaluation = evaluate_profiles(frame, [profiles.get("ghose")])
    assert set(evaluation.failures["failure_code"]) == {"GHO_MW_LOW", "GHO_MW_HIGH"}


def test_a_molecule_can_pass_one_profile_and_fail_another(profiles, descriptors):
    """Aspirin passes Lipinski but is below the Muegge weight floor."""
    mol = Chem.MolFromSmiles(REFERENCE_SMILES["aspirin"])
    needed = set(profiles.get("lipinski").descriptor_ids()) | set(
        profiles.get("muegge").descriptor_ids()
    )
    values = descriptors.compute(mol, sorted(needed))
    frame = descriptor_frame({key: [value] for key, value in values.items()})

    evaluation = evaluate_profiles(frame, [profiles.get("lipinski"), profiles.get("muegge")])
    assert evaluation.passed("lipinski").tolist() == [True]
    assert evaluation.passed("muegge").tolist() == [False]
    assert "MUE_MW_LOW" in set(evaluation.failures["failure_code"])


def test_profile_with_unknown_descriptor_is_rejected(descriptors):
    profile = profile_from_dict(
        {
            "id": "broken",
            "name": "Broken",
            "rules": [
                {
                    "id": "broken_rule",
                    "descriptor": "does_not_exist",
                    "operator": "<=",
                    "threshold": 1,
                    "failure_code": "X",
                }
            ],
        }
    )
    issues = validate_profile(profile, descriptors)
    assert any("unregistered descriptor" in issue for issue in issues)


def test_profile_without_rules_is_rejected():
    with pytest.raises(ProfileFileError):
        profile_from_dict({"id": "empty", "name": "Empty", "rules": []})


def test_policy_that_cannot_fail_is_rejected(descriptors):
    profile = profile_from_dict(
        {
            "id": "loose",
            "name": "Loose",
            "pass_policy": {"type": "max_violations", "value": 2},
            "rules": [
                {
                    "id": "mw",
                    "descriptor": "mol_wt",
                    "operator": "<=",
                    "threshold": 500,
                    "failure_code": "MW",
                }
            ],
        }
    )
    issues = validate_profile(profile, descriptors)
    assert any("impossible to fail" in issue for issue in issues)
