"""Contract tests for the first Chemical Space Hub foundations."""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from smiles2select.chemical_space.clustering import ClusteringTooLargeError, cluster
from smiles2select.chemical_space.coverage import coverage
from smiles2select.chemical_space.projection_manager import ProjectionConfig, project
from smiles2select.chemical_space.zones import membership_table, zone_from_expression
from smiles2select.chemistry.fingerprints import FingerprintConfig
from smiles2select.explainability.consequences import explain_change
from smiles2select.explainability.method_cards import get_method_card, method_cards
from smiles2select.explainability.selection_reasons import explain_not_selected, explain_selected
from smiles2select.io.importer import ColumnMapping, SourceFile
from smiles2select.pipeline.config import RunConfig
from smiles2select.pipeline.runner import run
from smiles2select.reference.duplicates import compare_exact_duplicates
from smiles2select.reference.libraries import LibraryRole, prepare_library_frame
from smiles2select.reference.similarity import (
    compute_reference_similarity,
)
from smiles2select.selection.reference_diversity import (
    reference_neighborhood,
    reference_seeded_maxmin,
)
from smiles2select.selection.zone_allocator import allocate_zones
from smiles2select.selection_intelligence import recipes


def _library(rows: list[tuple[str, str]], name: str):
    return prepare_library_frame(
        pd.DataFrame(rows, columns=["ID", "SMILES"]),
        name,
        smiles_column="SMILES",
        id_column="ID",
        role=LibraryRole.REFERENCE,
    )


def test_exact_reference_overlap_preserves_library_and_source_ids():
    candidates = prepare_library_frame(
        pd.DataFrame(
            [(1, "C1=CC=CC=C1"), (2, "CCO"), (3, "CCC")],
            columns=["ID", "SMILES"],
        ),
        "candidates",
        smiles_column="SMILES",
        id_column="ID",
        role=LibraryRole.CANDIDATE,
    )
    references = {
        "bioactives": _library([("ECBD-1", "c1ccccc1"), ("ECBD-2", "CCN")], "bioactives")
    }
    report = compare_exact_duplicates(candidates.molecules, references)
    assert report.duplicate_count == 1
    assert report.annotations.loc[0, "is_reference_duplicate"]
    assert report.annotations.loc[0, "reference_duplicate_library"] == "bioactives"
    assert report.annotations.loc[0, "reference_duplicate_id"] == "ECBD-1"
    assert report.overlaps.iloc[0]["match_type"] in {"canonical_smiles", "InChIKey"}


def test_reference_similarity_reports_exact_nearest_reference_and_novelty():
    candidates = pd.DataFrame(
        {"molecule_id": ["A", "B"], "canonical_smiles": ["c1ccccc1C", "CCCCCCCC"]},
        index=pd.Index([10, 11], name="record_id"),
    )
    result = compute_reference_similarity(
        candidates,
        {"known": _library([("R1", "c1ccccc1"), ("R2", "CCO")], "known")},
    )
    assert result.search_mode == "Exact search"
    assert result.fingerprint == FingerprintConfig()
    assert result.annotations.loc[10, "nearest_reference_id"] == "R1"
    assert result.annotations.loc[10, "max_reference_similarity"] > 0
    assert result.annotations.loc[10, "reference_novelty"] == pytest.approx(
        1 - result.annotations.loc[10, "max_reference_similarity"]
    )


def test_fast_reference_search_is_explicitly_optional():
    class FakeIndex:
        def __init__(self, *, space: str, dim: int):
            self.matrix = None

        def init_index(self, *, max_elements: int, ef_construction: int, M: int):
            return None

        def add_items(self, matrix, labels):
            self.matrix = np.asarray(matrix)

        def set_ef(self, value):
            return None

        def knn_query(self, query, k: int):
            matrix = self.matrix
            scores = matrix @ query[0] / (
                np.linalg.norm(matrix, axis=1) * max(np.linalg.norm(query[0]), 1e-9)
            )
            labels = np.argsort(-scores)[:k]
            return np.asarray([labels]), np.asarray([1.0 - scores[labels]])

    fake_hnsw = types.SimpleNamespace(Index=FakeIndex)
    candidates = pd.DataFrame({"canonical_smiles": ["CCO"]})
    reference = _library([("R1", "CCO"), ("R2", "CCC")], "known")
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(sys.modules, {"hnswlib": fake_hnsw}):
        result = compute_reference_similarity(candidates, {"known": reference}, search="fast")
    assert "HNSW discovery" in result.search_mode
    assert result.annotations.iloc[0]["max_reference_similarity"] == pytest.approx(1.0)


def test_reference_seeded_maxmin_is_reproducible_and_returns_original_positions():
    result = reference_seeded_maxmin(
        ["c1ccccc1C", "c1ccccc1CC", "CCCCCCCC", "CCO"],
        ["c1ccccc1"],
        2,
        seed=17,
    )
    again = reference_seeded_maxmin(
        ["c1ccccc1C", "c1ccccc1CC", "CCCCCCCC", "CCO"],
        ["c1ccccc1"],
        2,
        seed=17,
    )
    assert result.picked_indices == again.picked_indices
    assert len(result.picked_indices) == 2
    assert set(result.picked_indices).issubset({0, 1, 2, 3})


def test_zone_expression_and_allocator_keep_overlaps_without_double_counting():
    frame = pd.DataFrame(
        {"qed": [0.9, 0.8, 0.4, 0.2], "reference_novelty": [0.1, 0.6, 0.8, 0.9]},
        index=pd.Index([1, 2, 3, 4], name="record_id"),
    )
    core = zone_from_expression(
        frame, "qed >= 0.8", zone_id="core", name="Conventional", quota=1, priority=1
    )
    frontier = zone_from_expression(
        frame,
        "reference_novelty >= 0.6",
        zone_id="frontier",
        name="Novel Frontier",
        quota=2,
        priority=2,
    )
    membership = membership_table([core, frontier], frame.index)
    assert membership[membership["record_id"] == 2]["zone_id"].tolist() == ["core", "frontier"]
    outcome = allocate_zones(
        frame,
        [core, frontier],
        final_count=3,
        reserve_count=1,
        ranking={1: 0.1, 2: 0.2, 3: 0.9, 4: 0.8},
    )
    assert len(outcome.final_ids) == 3
    assert len(outcome.reserve_ids) == 1
    assert set(outcome.final_ids).isdisjoint(outcome.reserve_ids)
    assert len(set(outcome.final_ids)) == 3
    assert set(outcome.assignments).issuperset(outcome.final_ids)


def test_reference_neighborhood_requires_explicit_user_window():
    frame = pd.DataFrame(
        {"max_reference_similarity": [0.2, 0.55, 0.85]},
        index=pd.Index([1, 2, 3], name="record_id"),
    )
    assert list(reference_neighborhood(frame, minimum_similarity=0.5, maximum_similarity=0.8)) == [2]
    with pytest.raises(ValueError):
        reference_neighborhood(frame, minimum_similarity=0.8, maximum_similarity=0.5)


def test_coverage_reports_multiple_metrics_and_threshold_used():
    frame = pd.DataFrame(
        {
            "canonical_smiles": ["c1ccccc1C", "c1ccccc1CC", "CCCCCCCC"],
            "murcko_scaffold": ["benzene", "benzene", ""],
            "reference_novelty": [0.2, 0.5, 0.9],
        },
        index=pd.Index([1, 2, 3], name="record_id"),
    )
    report = coverage(frame, [2, 3], novelty_threshold=0.55)
    assert report.selected_count == 2
    assert report.unique_scaffolds == 1
    assert report.selected_unique_scaffolds == 1
    assert report.reference_similarity_threshold == 0.55
    assert report.novel_candidate_count == 1
    assert report.selected_novel_candidate_count == 1


def test_projection_manager_records_input_hash_and_parameters():
    frame = pd.DataFrame(
        {"mol_wt": [100.0, 200.0, 300.0], "tpsa": [20.0, 40.0, 60.0]},
        index=pd.Index([1, 2, 3], name="record_id"),
    )
    result = project(frame, ProjectionConfig(method="property_pca", features=("mol_wt", "tpsa")))
    assert result.input_hash
    assert result.recipe_block()["method"] == "property_pca"
    assert result.recipe_block()["software_versions"]["rdkit"]
    assert result.projection.coordinates.index.tolist() == [1, 2, 3]


def test_large_exact_clustering_is_refused_before_quadratic_allocation():
    with pytest.raises(ClusteringTooLargeError, match="quadratic"):
        cluster(["CC"] * 4, pd.RangeIndex(4), max_exact_size=3)


def test_method_cards_and_reasons_are_english_and_deterministic():
    assert {"balanced", "reference_aware_diversity", "dockability_envelope"}.issubset(
        {card.method_id for card in method_cards()}
    )
    assert "2D map" in get_method_card("reference_aware_diversity").reproducibility_notes
    assert explain_selected(
        {"reference_novelty": 0.57, "selection_strategy": "reference_novelty"}
    )[:2] == ["Reference novelty = 0.570", "Selected by Reference Novelty"]
    assert "Exact reference duplicate" in explain_not_selected({"is_reference_duplicate": True})
    assert "Lipinski" in explain_change("lipinski", "informative", "mandatory").message


def test_hub_recipe_round_trips_reference_and_projection_provenance(tmp_path):
    recipe = recipes.SelectionRecipe(
        candidate_library="NPASS",
        reference_libraries=({"library_id": "ECBD", "role": "reference"},),
        fingerprint={"type": "Morgan", "radius": 2, "bits": 2048, "use_chirality": False},
        zones=({"zone_id": "frontier", "quota": 300},),
        projection={"method": "structural_umap", "seed": 42},
        target_count=3000,
        reserve_count=500,
        seed=42,
    )
    path = recipes.save(recipe, tmp_path / "hub.selection.json")
    restored = recipes.load(path)
    assert restored.candidate_library == "NPASS"
    assert restored.reference_libraries[0]["library_id"] == "ECBD"
    assert restored.reserve_count == 500
    assert restored.projection["seed"] == 42
    assert recipes.from_dict({"schema_version": "2.0"}).candidate_library == ""


def test_pipeline_attaches_reference_columns_and_can_exclude_exact_overlap(tmp_path):
    candidates_path = tmp_path / "candidates.csv"
    references_path = tmp_path / "references.csv"
    pd.DataFrame(
        [("C1", "c1ccccc1"), ("C2", "CCO"), ("C3", "CCCCCCCC")],
        columns=["ID", "SMILES"],
    ).to_csv(candidates_path, index=False)
    pd.DataFrame(
        [("R1", "c1ccccc1")], columns=["ID", "SMILES"]
    ).to_csv(references_path, index=False)
    result = run(
        RunConfig(
            sources=(
                SourceFile(
                    candidates_path, ColumnMapping(smiles="SMILES", molecule_id="ID")
                ),
            ),
            profile_ids=("lipinski",),
            alert_catalogs=(),
            n_jobs=1,
            reference_sources=(
                SourceFile(
                    references_path, ColumnMapping(smiles="SMILES", molecule_id="ID")
                ),
            ),
            exclude_reference_duplicates=True,
            database_path=tmp_path / "run.sqlite",
        )
    )
    assert result.reference_similarity is not None
    assert result.reference_duplicates is not None
    assert result.descriptors.loc[1, "is_reference_duplicate"]
    assert result.descriptors.loc[1, "reference_novelty"] == pytest.approx(0.0)
    assert not bool(result.decision.decisions.loc[1, "selected"])
    assert "reference duplicate" in result.decision.decisions.loc[1, "exclusion_reasons"]
    assert "reference_libraries" in __import__("sqlite3").connect(result.database_path).execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='reference_libraries'"
    ).fetchone()


def test_pipeline_reference_aware_strategy_produces_final_and_reserve(tmp_path):
    candidates_path = tmp_path / "candidates.csv"
    references_path = tmp_path / "references.csv"
    pd.DataFrame(
        [("C1", "c1ccccc1"), ("C2", "CCO"), ("C3", "CCCCCCCC")],
        columns=["ID", "SMILES"],
    ).to_csv(candidates_path, index=False)
    pd.DataFrame(
        [("R1", "c1ccccc1")], columns=["ID", "SMILES"]
    ).to_csv(references_path, index=False)
    source = SourceFile(candidates_path, ColumnMapping(smiles="SMILES", molecule_id="ID"))
    reference = SourceFile(references_path, ColumnMapping(smiles="SMILES", molecule_id="ID"))
    result = run(
        RunConfig(
            sources=(source,),
            profile_ids=("lipinski",),
            alert_catalogs=(),
            n_jobs=1,
            reference_sources=(reference,),
            selection_strategy="reference_aware_diversity",
            final_count=1,
            reserve_count=1,
        )
    )
    assert len(result.reserve_ids) == 1
    assert result.decision.selected_count == 1
    assert set(result.reserve_ids).isdisjoint(result.decision.selected_ids())
    assert set(result.decision.decisions.loc[list(result.reserve_ids), "selection_status"]) == {
        "RESERVE"
    }


def test_pipeline_persists_overlapping_zone_allocation(tmp_path):
    candidates_path = tmp_path / "candidates.csv"
    pd.DataFrame(
        [("C1", "CCO"), ("C2", "CCC"), ("C3", "CCCC")],
        columns=["ID", "SMILES"],
    ).to_csv(candidates_path, index=False)
    result = run(
        RunConfig(
            sources=(SourceFile(candidates_path, ColumnMapping("SMILES", "ID")),),
            profile_ids=("lipinski",),
            alert_catalogs=(),
            n_jobs=1,
            final_count=2,
            reserve_count=1,
            zones=(
                {"zone_id": "polar", "name": "Polar", "expression": "mol_wt >= 0", "quota": 1},
                {"zone_id": "small", "name": "Small", "expression": "mol_wt <= 100", "quota": 1},
            ),
            database_path=tmp_path / "zones.sqlite",
        )
    )
    assert result.zone_allocation is not None
    assert len(result.zone_allocation.memberships) >= 2
    assert len(result.reserve_ids) == 1
