"""v2.0 modules: SA score, CNS/bRo5 profiles, scaffolds, diversity and docking hand-off."""

from __future__ import annotations

import pandas as pd
import pytest
from rdkit import Chem

from smiles2select.chemistry.descriptor_planner import DescriptorPlanner
from smiles2select.chemistry.fingerprints import (
    FingerprintConfig,
    fingerprints_from_smiles,
    mean_pairwise_similarity,
    tanimoto,
)
from smiles2select.chemistry.scaffolds import (
    murcko_scaffold,
    scaffold_diversity,
    scaffold_ids,
    scaffold_table,
    scaffolds_from_smiles,
)
from smiles2select.export import docking
from smiles2select.io.importer import ColumnMapping, SourceFile
from smiles2select.pipeline.config import RunConfig
from smiles2select.pipeline.runner import run
from smiles2select.rules.evaluator import evaluate_profiles
from smiles2select.selection.diversity import max_min_pick, pick_by_scaffold, pick_diverse
from tests.conftest import REFERENCE_SMILES, descriptor_frame

pytestmark = pytest.mark.unit

# Three chemotypes, several analogues each - the shape diversity picking is for.
FAMILY = [
    "c1ccccc1C",
    "c1ccccc1CC",
    "c1ccccc1CCC",
    "c1ccncc1C",
    "c1ccncc1CC",
    "C1CCCCC1C",
    "C1CCCCC1CC",
]

COMPLEX_SUGAR = (
    "C[C@H]1O[C@@H](O[C@@H]2[C@H](O)[C@@H](CO)O[C@H](O)[C@H]2O)[C@H](O)[C@@H](O)[C@@H]1O"
)


# --- synthetic accessibility -------------------------------------------------


def test_sa_score_is_registered_as_a_score_not_a_rule(descriptors):
    definition = descriptors.get("sa_score")
    assert "not a rule" in definition.compatibility
    assert "sascorer" in definition.function


def test_sa_score_ranks_a_simple_molecule_below_a_complex_one(descriptors):
    simple = descriptors.compute(Chem.MolFromSmiles(REFERENCE_SMILES["aspirin"]), ("sa_score",))
    complicated = descriptors.compute(Chem.MolFromSmiles(COMPLEX_SUGAR), ("sa_score",))
    assert 1.0 <= simple["sa_score"] <= 10.0
    assert simple["sa_score"] < complicated["sa_score"]


def test_sa_score_is_only_computed_when_requested(descriptors, profiles):
    planner = DescriptorPlanner(descriptors)
    without = planner.resolve(profiles=[profiles.get("lipinski")], scores=["qed"])
    with_sa = planner.resolve(profiles=[profiles.get("lipinski")], scores=["qed", "sa_score"])
    assert "sa_score" not in without.descriptor_ids
    assert "sa_score" in with_sa.descriptor_ids


# --- CNS-like and beyond Rule of Five ---------------------------------------


def test_cns_profile_is_declared_as_a_heuristic_not_cns_mpo(profiles):
    cns = profiles.get("cns_like")
    assert cns.category == "cns_space"
    assert "NOT the CNS MPO" in cns.notes
    assert "pKa" in cns.notes


def test_beyond_ro5_is_declared_as_descriptive_not_predictive(profiles):
    bro5 = profiles.get("beyond_ro5")
    assert bro5.category == "beyond_ro5"
    assert "does not predict" in bro5.notes


def test_cns_profile_rejects_a_large_polar_molecule(profiles):
    frame = descriptor_frame(
        {
            "mol_wt": [350.0, 520.0],
            "tpsa": [60.0, 160.0],
            "hbd_lipinski": [2, 6],
            "rdkit_wlogp": [2.5, 0.5],
            "rotatable_bonds": [4, 12],
            "nitrogen_oxygen_count": [4, 11],
        }
    )
    evaluation = evaluate_profiles(frame, [profiles.get("cns_like")])
    assert evaluation.passed("cns_like").tolist() == [True, False]


def test_beyond_ro5_admits_a_molecule_lipinski_rejects(profiles):
    """The point of the profile: keep large modalities in the report."""
    frame = descriptor_frame(
        {
            "mol_wt": [820.0],
            "rdkit_wlogp": [4.0],
            "hbd_lipinski": [6],
            "hba_lipinski": [14],
            "rotatable_bonds": [18],
            "tpsa": [230.0],
        }
    )
    evaluation = evaluate_profiles(frame, [profiles.get("beyond_ro5"), profiles.get("lipinski")])
    assert evaluation.passed("beyond_ro5").tolist() == [True]
    assert evaluation.passed("lipinski").tolist() == [False]


# --- scaffolds ---------------------------------------------------------------


def test_murcko_scaffold_strips_side_chains():
    assert murcko_scaffold(Chem.MolFromSmiles(REFERENCE_SMILES["aspirin"])) == "c1ccccc1"


def test_acyclic_molecule_has_no_scaffold():
    assert murcko_scaffold(Chem.MolFromSmiles("CCCCO")) == ""


def test_generic_scaffold_collapses_heteroatoms():
    pyridine = murcko_scaffold(Chem.MolFromSmiles("c1ccncc1C"), generic=True)
    benzene = murcko_scaffold(Chem.MolFromSmiles("c1ccccc1C"), generic=True)
    assert pyridine == benzene


def test_scaffold_table_counts_every_molecule():
    table = scaffold_table(pd.Series(scaffolds_from_smiles(FAMILY)))
    assert table["molecules"].sum() == len(FAMILY)
    assert table["molecules"].is_monotonic_decreasing


def test_scaffold_ids_are_assigned_by_population():
    scaffolds = pd.Series(scaffolds_from_smiles(FAMILY))
    ids = scaffold_ids(scaffolds)
    assert ids.nunique() == scaffolds.nunique()
    assert ids.min() == 1


def test_scaffold_diversity_exposes_a_dominated_library():
    dominated = ["c1ccccc1C"] * 9 + ["C1CCCCC1C"]
    stats = scaffold_diversity(pd.Series(scaffolds_from_smiles(dominated)))
    assert stats["scaffold_count"] == 2
    assert stats["largest_scaffold_share"] == 0.9
    assert stats["singleton_scaffolds"] == 1


# --- fingerprints and diversity ---------------------------------------------


def test_fingerprint_config_states_its_parameters():
    assert FingerprintConfig().label() == "Morgan r=2, 2048 bits"
    assert "chiral" in FingerprintConfig(use_chirality=True).label()


def test_fingerprints_report_the_positions_they_came_from():
    vectors, positions = fingerprints_from_smiles(["CCO", "not_a_smiles", "CCC"])
    assert len(vectors) == 2
    assert positions == [0, 2]


def test_identical_molecules_have_similarity_one():
    vectors, _ = fingerprints_from_smiles(["CCO", "CCO"])
    assert tanimoto(vectors[0], vectors[1]) == pytest.approx(1.0)


def test_mean_similarity_of_a_single_molecule_is_zero():
    vectors, _ = fingerprints_from_smiles(["CCO"])
    assert mean_pairwise_similarity(vectors) == 0.0


def test_max_min_returns_everything_when_asked_for_more_than_exists():
    assert sorted(max_min_pick(FAMILY, 99)) == list(range(len(FAMILY)))


def test_max_min_is_reproducible():
    assert max_min_pick(FAMILY, 3, seed=7) == max_min_pick(FAMILY, 3, seed=7)


def test_max_min_rejects_a_non_positive_count():
    with pytest.raises(ValueError):
        max_min_pick(FAMILY, 0)


def test_diverse_pick_lowers_the_mean_similarity():
    report = pick_diverse(FAMILY, 3)
    assert len(report.picked_index) == 3
    assert report.mean_similarity_after <= report.mean_similarity_before
    assert report.available == len(FAMILY)


def test_pick_by_scaffold_keeps_one_of_each_chemotype():
    picked = pick_by_scaffold(FAMILY, per_scaffold=1)
    scaffolds = scaffolds_from_smiles([FAMILY[index] for index in picked])
    assert len(scaffolds) == len(set(scaffolds))
    assert len(picked) == len(set(scaffolds_from_smiles(FAMILY)))


def test_pick_by_scaffold_rejects_a_zero_limit():
    with pytest.raises(ValueError):
        pick_by_scaffold(FAMILY, per_scaffold=0)


# --- pipeline integration and docking hand-off -------------------------------


@pytest.fixture(scope="module")
def diverse_result(tmp_path_factory):
    directory = tmp_path_factory.mktemp("v2")
    csv = directory / "family.csv"
    pd.DataFrame(
        [(f"MOL{index:03d}", smiles) for index, smiles in enumerate(FAMILY, start=1)],
        columns=["ID", "SMILES"],
    ).to_csv(csv, index=False)

    return run(
        RunConfig(
            sources=(
                SourceFile(path=csv, mapping=ColumnMapping(smiles="SMILES", molecule_id="ID")),
            ),
            profile_ids=("lipinski",),
            alert_catalogs=(),
            compute_sa=True,
            per_scaffold_limit=1,
            n_jobs=1,
            chunk_size=4,
        )
    )


@pytest.mark.integration
def test_scaffold_limit_narrows_the_selection(diverse_result):
    scaffolds = diverse_result.descriptors["murcko_scaffold"]
    assert diverse_result.decision.selected_count == scaffolds.nunique()


@pytest.mark.integration
def test_diversity_drop_is_recorded_as_an_exclusion_reason(diverse_result):
    decisions = diverse_result.decision.decisions
    dropped = decisions[~decisions["selected"].astype(bool)]
    assert not dropped.empty
    assert dropped["exclusion_reasons"].str.contains("scaffold").all()


@pytest.mark.integration
def test_sa_score_reaches_the_descriptor_table(diverse_result):
    assert diverse_result.descriptors["sa_score"].notna().all()


@pytest.mark.integration
def test_docking_export_uses_the_columns_smiles2docking_reads(diverse_result, tmp_path):
    path = docking.export(diverse_result, tmp_path / "docking.xlsx")
    frame = pd.read_excel(path)
    assert list(frame.columns) == ["access_code", "smiles"]
    assert len(frame) == diverse_result.decision.selected_count
    assert frame["smiles"].str.strip().ne("").all()


@pytest.mark.integration
def test_docking_export_defaults_to_the_selection_only(diverse_result):
    selected = docking.build_frame(diverse_result)
    everything = docking.build_frame(
        diverse_result, docking.DockingExportOptions(selected_only=False)
    )
    assert len(selected) < len(everything)


@pytest.mark.integration
def test_docking_export_can_add_context_columns(diverse_result):
    frame = docking.build_frame(
        diverse_result, docking.DockingExportOptions(use_canonical=False, include_context=True)
    )
    assert "MW" in frame.columns
    assert frame["smiles"].notna().all()


@pytest.mark.integration
def test_docking_export_writes_csv_when_asked(diverse_result, tmp_path):
    path = docking.export(diverse_result, tmp_path / "docking.csv")
    frame = pd.read_csv(path)
    assert list(frame.columns) == ["access_code", "smiles"]
