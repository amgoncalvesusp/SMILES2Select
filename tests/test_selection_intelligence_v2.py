"""Phases 4, 6 and 7: chemical space, rescue and constrained final selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smiles2select.chemical_space import (
    clustering,
    density_tiles,
    pca_projection,
    projection_cache,
    umap_projection,
)
from smiles2select.chemical_space.density_tiles import Viewport
from smiles2select.chemistry.fingerprints import FingerprintConfig
from smiles2select.selection_intelligence import rescue
from smiles2select.selection_intelligence.basket import SelectionBasket
from smiles2select.selection_intelligence.constrained_selection import (
    SelectionConstraints,
    SelectionOutcome,
    Strategy,
    order_candidates,
    select,
)
from smiles2select.selection_intelligence.selection_explanations import (
    MANUAL_COLUMNS,
    explanation_table,
    final_alerts,
    manual_decisions,
    summary_rows,
)
from smiles2select.selection_intelligence.similarity_index import SimilarityIndex
from smiles2select.selection_intelligence.states import ChemicalStatus, MoleculeState

pytestmark = pytest.mark.unit

FAMILY = [
    "c1ccccc1C",
    "c1ccccc1CC",
    "c1ccccc1CCC",
    "c1ccncc1C",
    "c1ccncc1CC",
    "C1CCCCC1C",
]


def frame(**columns) -> pd.DataFrame:
    data = pd.DataFrame(columns)
    data.index = pd.RangeIndex(1, len(data) + 1, name="record_id")
    return data


@pytest.fixture
def descriptors() -> pd.DataFrame:
    return frame(
        mol_wt=[180.0, 350.0, 420.0, 260.0, 300.0, 500.0],
        rdkit_wlogp=[1.0, 2.0, 3.0, 1.5, 2.5, 4.0],
        tpsa=[40.0, 60.0, 80.0, 50.0, 70.0, 30.0],
        qed=[0.9, 0.7, 0.5, 0.8, 0.6, 0.3],
        canonical_smiles=FAMILY,
    )


@pytest.fixture
def candidates() -> pd.DataFrame:
    return frame(
        pareto_rank=[1, 1, 2, 2, 3, 3],
        crowding_distance=[9.0, 2.0, 5.0, 1.0, 4.0, 3.0],
        robustness_score=[0.8, 0.6, 0.7, 0.2, 0.9, 0.1],
        qed=[0.9, 0.7, 0.5, 0.8, 0.6, 0.3],
        murcko_scaffold=["A", "A", "A", "B", "B", "C"],
        cluster_id=[1, 1, 2, 2, 3, 3],
    )


# --- phase 4: chemical space -------------------------------------------------


def test_pca_is_deterministic(descriptors):
    features = ["mol_wt", "rdkit_wlogp", "tpsa", "qed"]
    first = pca_projection.project(descriptors, features)
    second = pca_projection.project(descriptors, features)
    pd.testing.assert_frame_equal(first.coordinates, second.coordinates)


def test_pca_reports_its_features_and_variance(descriptors):
    projection = pca_projection.project(descriptors, ["mol_wt", "tpsa"])
    assert projection.method == "pca"
    assert projection.features == ("mol_wt", "tpsa")
    assert 0 < sum(projection.explained_variance) <= 1.0000001
    assert "pca::" in projection.projection_id


def test_pca_standardises_so_one_descriptor_cannot_dominate(descriptors):
    """Without scaling, MW (hundreds) would swamp QED (0-1)."""
    matrix, _, _ = pca_projection.standardise(descriptors[["mol_wt", "qed"]])
    assert abs(matrix.mean()) < 1e-9
    assert matrix.std(axis=0) == pytest.approx([1.0, 1.0])


def test_standardise_flags_imputed_rows():
    data = pd.DataFrame(
        {
            "mol_wt": [100.0, np.nan, 200.0],
            "qed": [0.5, 0.6, 0.7],
        },
        index=[1, 2, 3],
    )
    _, index, imputed = pca_projection.standardise(data)
    assert imputed.tolist() == [False, True, False]
    assert imputed.index.tolist() == [1, 2, 3]


def test_projection_describe_reports_imputation():
    data = pd.DataFrame(
        {
            "mol_wt": [100.0, np.nan, 200.0, 150.0],
            "qed": [0.5, 0.6, 0.7, 0.8],
        },
        index=[1, 2, 3, 4],
    )
    projection = pca_projection.project(data, ["mol_wt", "qed"])
    desc = dict(projection.describe())
    assert desc.get("moléculas com descritor imputado") == 1


def test_pca_needs_at_least_one_known_descriptor(descriptors):
    with pytest.raises(ValueError, match="none of the requested descriptors"):
        pca_projection.project(descriptors, ["not_a_descriptor"])


def test_umap_descriptors_rejects_missing_columns(descriptors):
    with pytest.raises(ValueError, match="none of the requested descriptors"):
        umap_projection.project_descriptors(descriptors, ["not_a_descriptor"])


def test_projection_rows_carry_optional_clusters(descriptors):
    projection = pca_projection.project(descriptors, ["mol_wt", "tpsa"])
    labels = pd.Series([1, 1, 2, 2, 3, 3], index=descriptors.index, dtype="Int64")
    rows = projection.rows_for_storage(labels)
    assert set(rows[0]) == {"record_id", "x", "y", "cluster_id"}
    assert rows[0]["cluster_id"] == 1


def test_method_choice_explains_itself():
    small = projection_cache.choose_method(1000)
    assert small.method == "pca"
    assert "deterministic" in small.reason

    large = projection_cache.choose_method(400_000)
    assert large.method in projection_cache.available_methods()
    assert large.reason


def test_projection_cache_reuses_an_identical_map(descriptors):
    cache = projection_cache.ProjectionCache()
    first = cache.project_descriptors(descriptors, ["mol_wt", "tpsa"])
    assert cache.project_descriptors(descriptors, ["mol_wt", "tpsa"]) is first
    assert len(cache) == 1


def test_projection_cache_notices_changed_data(descriptors):
    cache = projection_cache.ProjectionCache()
    first = cache.project_descriptors(descriptors, ["mol_wt", "tpsa"])
    changed = descriptors.assign(mol_wt=descriptors["mol_wt"] * 2)
    assert cache.project_descriptors(changed, ["mol_wt", "tpsa"]) is not first


def test_stored_coordinates_come_back_as_float32():
    loaded = projection_cache.load_coordinates(
        [{"record_id": 1, "x": 0.5, "y": -0.25, "cluster_id": 3}]
    )
    assert loaded["x"].dtype == np.float32
    assert loaded.loc[1, "cluster_id"] == 3


def test_clustering_groups_the_chemotypes(descriptors):
    result = clustering.cluster(FAMILY, descriptors.index, cutoff=0.5)
    assert result.cluster_count >= 2
    assert result.labels.notna().all()
    assert dict(result.summary_rows())["clusters"] == result.cluster_count


def test_cluster_coverage_and_gaps(descriptors):
    result = clustering.cluster(FAMILY, descriptors.index, cutoff=0.5)
    first_member = int(result.labels.index[0])
    coverage = clustering.coverage(result.labels, [first_member])
    assert coverage["represented_clusters"] == 1
    assert coverage["coverage"] < 1.0
    assert clustering.unrepresented(result.labels, [first_member])


def test_cluster_cutoff_must_be_a_fraction(descriptors):
    with pytest.raises(ValueError):
        clustering.cluster(FAMILY, descriptors.index, cutoff=1.5)


def test_viewport_culls_to_what_is_visible():
    coordinates = pd.DataFrame(
        {"x": [0.0, 5.0, 10.0], "y": [0.0, 5.0, 10.0]},
        index=pd.Index([1, 2, 3], name="record_id"),
    )
    assert list(density_tiles.visible(coordinates, Viewport(-1, 6, -1, 6))) == [1, 2]
    assert len(density_tiles.visible(coordinates, Viewport.around(coordinates))) == 3


def test_density_switches_on_above_the_threshold():
    assert density_tiles.should_aggregate(50_000)
    assert not density_tiles.should_aggregate(100)


def test_tiles_report_selected_share_and_gaps():
    coordinates = pd.DataFrame(
        {"x": np.linspace(0, 1, 20), "y": np.linspace(0, 1, 20)},
        index=pd.RangeIndex(1, 21, name="record_id"),
    )
    grid = density_tiles.tiles(coordinates, selected_ids=pd.Index([1, 2, 3]), resolution=4)
    assert grid["molecules"].sum() == 20
    assert grid["selected"].sum() == 3
    assert not density_tiles.uncovered_regions(grid, min_molecules=1).empty


def test_lasso_selects_the_points_inside_a_polygon():
    coordinates = pd.DataFrame(
        {"x": [0.5, 5.0, 0.1], "y": [0.5, 5.0, 0.1]},
        index=pd.Index([1, 2, 3], name="record_id"),
    )
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert set(density_tiles.lasso_contains(coordinates, square)) == {1, 3}
    assert density_tiles.lasso_contains(coordinates, square[:2]).empty


# --- phase 6: rescue ---------------------------------------------------------


def test_similarity_index_ranks_by_exact_tanimoto(descriptors):
    index = SimilarityIndex(FAMILY, descriptors.index, FingerprintConfig())
    neighbours = index.query("c1ccccc1C", top=3, minimum_similarity=0.1, exclude_ids=[1])
    assert neighbours
    assert 1 not in {item.record_id for item in neighbours}
    scores = [item.tanimoto for item in neighbours]
    assert scores == sorted(scores, reverse=True)


def test_similarity_index_reports_its_settings(descriptors):
    settings = dict(SimilarityIndex(FAMILY, descriptors.index).settings())
    assert settings["final metric"] == "Exact Tanimoto"
    assert settings["indexed molecules"] == len(FAMILY)


def test_analogues_only_come_from_the_approved_set(descriptors):
    approved = [2, 3, 4]
    index = rescue.build_index(descriptors, approved)
    found = rescue.find_analogues(1, FAMILY[0], index, descriptors, minimum_similarity=0.1)
    assert found
    assert {item.analog_id for item in found} <= set(approved)


def test_analogue_comparison_reports_property_deltas(descriptors):
    index = rescue.build_index(descriptors, [2])
    found = rescue.find_analogues(1, FAMILY[0], index, descriptors, minimum_similarity=0.1)
    assert found[0].deltas["mol_wt"] == pytest.approx(170.0)
    assert "Δmol_wt" in found[0].describe_deltas()


def test_rescue_table_uses_the_export_columns(descriptors):
    index = rescue.build_index(descriptors, [2, 3])
    found = rescue.find_analogues(1, FAMILY[0], index, descriptors, minimum_similarity=0.1)
    assert set(rescue.rescue_table(found).columns) == set(rescue.RESCUE_COLUMNS)
    assert rescue.rescue_table([]).empty


def test_neighbourhood_edges_shrink_with_similarity(descriptors):
    index = rescue.build_index(descriptors, [2, 3])
    found = rescue.find_analogues(1, FAMILY[0], index, descriptors, minimum_similarity=0.1)
    graph = rescue.neighbourhood(found)
    assert (graph["distance"] == (1.0 - graph["tanimoto"]).round(4)).all()


def test_common_substructure_is_computed_for_one_pair():
    assert rescue.common_substructure("c1ccccc1CC", "c1ccccc1CCC")
    assert rescue.common_substructure("not_a_smiles", "c1ccccc1") is None


# --- phase 7: constrained selection -----------------------------------------


def test_selection_stops_at_the_requested_count(candidates):
    outcome = select(candidates, SelectionConstraints(target_count=3))
    assert outcome.count == 3
    assert all("final count reached" in reason for reason in outcome.rejections.values())


def test_scaffold_quota_is_respected(candidates):
    outcome = select(candidates, SelectionConstraints(max_per_scaffold=1))
    scaffolds = candidates.loc[list(outcome.selected_ids), "murcko_scaffold"]
    assert scaffolds.nunique() == len(scaffolds)
    assert max(outcome.scaffold_usage.values()) == 1


def test_cluster_quota_is_respected(candidates):
    outcome = select(candidates, SelectionConstraints(max_per_cluster=1))
    assert max(outcome.cluster_usage.values()) == 1


def test_pinned_molecules_survive_the_quotas(candidates):
    """A human decision outranks the automatic ranking."""
    outcome = select(
        candidates, SelectionConstraints(target_count=2, max_per_scaffold=1), pinned_ids=[6]
    )
    assert 6 in outcome.selected_ids
    assert outcome.reasons[6] == ["pinned molecule"]


def test_excluded_molecules_never_return(candidates):
    outcome = select(candidates, SelectionConstraints(), excluded_ids=[1, 2])
    assert 1 not in outcome.selected_ids
    assert 2 not in outcome.selected_ids


def test_shortfall_is_reported_not_hidden(candidates):
    constraints = SelectionConstraints(target_count=10, max_per_scaffold=1)
    outcome = select(candidates, constraints)
    assert outcome.shortfall(constraints) == 10 - outcome.count
    assert any("Missing" in warning for warning in outcome.warnings(constraints))


def test_minimum_scaffolds_produces_a_warning(candidates):
    constraints = SelectionConstraints(target_count=2, min_scaffolds=5)
    outcome = select(candidates, constraints)
    assert any("scaffolds" in warning for warning in outcome.warnings(constraints))


def test_strategies_order_differently(candidates):
    pareto_first = list(order_candidates(candidates, Strategy.PARETO_FIRST).index)
    diversity_first = list(order_candidates(candidates, Strategy.DIVERSITY_FIRST).index)
    assert pareto_first != diversity_first
    assert pareto_first[0] in {1, 2}


def test_missing_columns_do_not_break_a_strategy():
    minimal = frame(qed=[0.5, 0.9])
    assert select(minimal, SelectionConstraints(target_count=1)).count == 1


def test_every_selected_molecule_carries_a_reason(candidates):
    outcome = select(candidates, SelectionConstraints(target_count=3))
    assert all(outcome.reasons[record_id] for record_id in outcome.selected_ids)
    assert explanation_table(outcome, candidates.index)["reason"].ne("-").all()


def test_summary_counts_scaffolds_and_clusters(candidates):
    constraints = SelectionConstraints(target_count=3)
    rows = dict(summary_rows(select(candidates, constraints), constraints))
    assert rows["final selected"] == 3
    assert rows["scaffolds covered"] >= 1


def test_final_alerts_flag_a_dominant_scaffold():
    outcome = SelectionOutcome(selected_ids=(1, 2, 3), scaffold_usage={"A": 3})
    alerts = final_alerts(outcome, SelectionConstraints(target_count=3))
    assert any("one scaffold" in alert for alert in alerts)


def test_final_alerts_flag_manual_overrides():
    basket = SelectionBasket([MoleculeState(1, ChemicalStatus.AUTO_FAIL)])
    basket.add_to_final([1], reason="análogo aprovado")
    alerts = final_alerts(
        SelectionOutcome(selected_ids=(1,)), SelectionConstraints(), basket=basket
    )
    assert any("despite" in alert for alert in alerts)


def test_manual_decisions_come_from_the_history():
    basket = SelectionBasket([MoleculeState(1), MoleculeState(2)])
    basket.add_to_final([1], reason="Pareto 1")
    table = manual_decisions(basket)
    assert list(table.columns) == MANUAL_COLUMNS
    assert table.iloc[0]["New_Status"] == "FINAL_SELECTED"
    assert table.iloc[0]["Timestamp"].endswith("+00:00")
