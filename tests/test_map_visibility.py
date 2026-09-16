"""Final molecules must survive projection and display sampling unchanged."""

import os
from types import SimpleNamespace

import pandas as pd
import pytest

from smiles2select.chemical_space.layers import progressive_layer
from smiles2select.chemistry.fingerprints import FingerprintConfig

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
from PySide6.QtWidgets import QApplication  # noqa: E402

from smiles2select.gui.workspace.map_compute import compute_map  # noqa: E402
from smiles2select.gui.workspace.views import ChemicalSpaceView  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_projection_preserves_finals_outside_context_sample():
    candidates = pd.DataFrame({"mol_wt": range(6000), "tpsa": range(6000)})
    run = SimpleNamespace(config=SimpleNamespace(fingerprint_config=FingerprintConfig()))
    selected = list(range(5500, 6000))
    result, _ = compute_map(candidates, run, "property_pca", False, selected_ids=selected)
    again, _ = compute_map(candidates.iloc[::-1], run, "property_pca", False, selected_ids=selected)
    assert set(selected) <= set(result.projection.coordinates.index)
    assert len(result.projection.coordinates) == 5000
    pd.testing.assert_frame_equal(result.projection.coordinates, again.projection.coordinates)
    metadata = result.recipe_block()["parameters"]["display_sampling"]
    assert metadata["seed"] == 42
    assert metadata["selected_projected"] == 500
    assert metadata["candidate_total"] == 6000


def test_projection_limit_never_truncates_final_selection():
    candidates = pd.DataFrame({"mol_wt": range(5002), "tpsa": range(5002)})
    run = SimpleNamespace(config=SimpleNamespace(fingerprint_config=FingerprintConfig()))
    result, _ = compute_map(candidates, run, "property_pca", False, selected_ids=candidates.index)
    assert len(result.projection.coordinates) == len(candidates)


def test_progressive_layer_preserves_finals_beyond_display_limit():
    coordinates = pd.DataFrame({"x": range(30), "y": range(30)})
    layer = progressive_layer(coordinates, selected_ids=pd.Index([25, 29]), max_points=5)
    assert {25, 29} <= set(layer.points.index)
    pd.testing.assert_frame_equal(layer.points, coordinates.loc[layer.points.index])
    many = progressive_layer(coordinates, selected_ids=pd.Index(range(10, 30)), max_points=5)
    assert set(range(10, 30)) <= set(many.points.index)


def test_finals_have_separate_overlay_and_selection_only_preserves_positions(qapp):
    view = ChemicalSpaceView()
    coordinates = pd.DataFrame({"x": [0.0, 0.0, 2.0], "y": [1.0, 1.0, 3.0]}, index=[1, 2, 3])
    view.set_points(coordinates, selected_ids=[1, 2])
    assert [point.data() for point in view.selected_scatter.points()] == [1, 2]
    assert view.selected_scatter.zValue() > view.scatter.zValue()
    assert "2 / 2" in view.selection_summary.text()
    assert "centroid" in view.legend_note.text().lower()
    view.selection_only.setChecked(True)
    assert not view.scatter.isVisible()
    assert view.selected_scatter.isVisible()
    assert [(p.pos().x(), p.pos().y()) for p in view.selected_scatter.points()] == [(0, 1), (0, 1)]
    view.close()


def test_click_overlap_offers_each_molecule_for_inspection(qapp):
    view = ChemicalSpaceView()
    coordinates = pd.DataFrame({"x": [0.0, 0.0], "y": [1.0, 1.0]}, index=[1, 2])
    view.set_points(coordinates, selected_ids=[2])
    clicked = []
    view.point_clicked.connect(clicked.append)
    view._on_click(view.selected_scatter, list(view.selected_scatter.points()))
    actions = view._overlap_menu.actions()
    assert len(actions) == 2
    actions[0].trigger()
    actions[1].trigger()
    assert set(clicked) == {1, 2}
    view._overlap_menu.close()
    view.close()


def test_empty_update_clears_final_overlay(qapp):
    view = ChemicalSpaceView()
    view.set_points(pd.DataFrame({"x": [0], "y": [0]}, index=[1]), selected_ids=[1])
    view.set_points(pd.DataFrame(columns=["x", "y"]))
    assert len(view.selected_scatter.points()) == 0
    view.close()


def test_projection_with_reference_overlay_keeps_final_ids():
    candidates = pd.DataFrame({"mol_wt": range(5010), "tpsa": range(5010)})
    reference = SimpleNamespace(
        library_id="known", valid=pd.DataFrame({"canonical_smiles": ["CCO"]}, index=[91])
    )
    run = SimpleNamespace(
        config=SimpleNamespace(fingerprint_config=FingerprintConfig()),
        reference_libraries=[reference],
    )
    result, references = compute_map(candidates, run, "property_pca", True, selected_ids=[5009])
    assert 5009 in result.projection.coordinates.index
    assert references.index.tolist() == ["reference::known::91"]
    assert result.recipe_block()["parameters"]["display_sampling"]["selected_projected"] == 1


def test_stale_projection_reports_missing_finals(qapp):
    view = ChemicalSpaceView()
    view.set_points(pd.DataFrame({"x": [0], "y": [0]}, index=[1]), selected_ids=[1, 2])
    assert "1 / 2" in view.selection_summary.text()
    assert "recompute" in view.selection_summary.text()
    view.close()


def test_selection_only_overlap_does_not_offer_hidden_context(qapp):
    view = ChemicalSpaceView()
    view.set_points(pd.DataFrame({"x": [0, 0], "y": [0, 0]}, index=[1, 2]), selected_ids=[2])
    clicked = []
    view.point_clicked.connect(clicked.append)
    view.selection_only.setChecked(True)
    view._on_click(view.selected_scatter, view.selected_scatter.points())
    assert clicked == [2]
    assert view._overlap_menu is None
    view.close()


def test_selection_only_lasso_ignores_hidden_context(qapp):
    view = ChemicalSpaceView()
    view.set_points(pd.DataFrame({"x": [0, 0], "y": [0, 0]}, index=[1, 2]), selected_ids=[2])
    view.selection_only.setChecked(True)
    view._lasso = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    assert view.finish_lasso() == [2]
    view.close()
