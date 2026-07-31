"""Phases 5 and 8: the workspace window and the selection export."""

from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtWidgets import QApplication  # noqa: E402

from smiles2select.export import selection_export  # noqa: E402
from smiles2select.gui.workspace.panels import BasketPanel, InspectorPanel  # noqa: E402
from smiles2select.gui.workspace.views import ChemicalSpaceView, ParetoView  # noqa: E402
from smiles2select.gui.workspace.workspace_window import WorkspaceWindow  # noqa: E402
from smiles2select.io.importer import ColumnMapping, SourceFile  # noqa: E402
from smiles2select.pipeline.config import RunConfig  # noqa: E402
from smiles2select.pipeline.runner import run  # noqa: E402
from smiles2select.selection_intelligence.basket import SelectionBasket  # noqa: E402
from smiles2select.selection_intelligence.states import ChemicalStatus, MoleculeState  # noqa: E402
from tests.conftest import REFERENCE_SMILES  # noqa: E402

pytestmark = pytest.mark.integration

LIBRARY = [
    ("MOL001", REFERENCE_SMILES["aspirin"]),
    ("MOL002", REFERENCE_SMILES["caffeine"]),
    ("MOL003", REFERENCE_SMILES["ibuprofen"]),
    ("MOL004", REFERENCE_SMILES["paracetamol"]),
    ("MOL005", REFERENCE_SMILES["chalcone"]),
    ("MOL006", REFERENCE_SMILES["long_alkane"]),
]


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    directory = tmp_path_factory.mktemp("workspace")
    csv = directory / "library.csv"
    pd.DataFrame(LIBRARY, columns=["ID", "SMILES"]).to_csv(csv, index=False)
    return run(
        RunConfig(
            sources=(
                SourceFile(path=csv, mapping=ColumnMapping(smiles="SMILES", molecule_id="ID")),
            ),
            profile_ids=("lipinski", "veber"),
            alert_catalogs=(),
            compute_sa=True,
            n_jobs=1,
            chunk_size=6,
        )
    )


@pytest.fixture
def window(qapp, result):
    window = WorkspaceWindow(result)
    yield window
    window.close()


# --- views -------------------------------------------------------------------


def test_scatter_carries_record_ids(qapp):
    view = ChemicalSpaceView()
    coordinates = pd.DataFrame(
        {"x": [0.0, 1.0], "y": [0.0, 1.0]}, index=pd.Index([7, 9], name="record_id")
    )
    view.set_points(coordinates, selected_ids=[9])
    assert [point.data() for point in view.scatter.points()] == [7, 9]


def test_empty_coordinates_draw_nothing(qapp):
    view = ChemicalSpaceView()
    view.set_points(pd.DataFrame(columns=["x", "y"]))
    assert len(view.scatter.points()) == 0


def test_lasso_reports_the_enclosed_molecules(qapp):
    view = ChemicalSpaceView()
    coordinates = pd.DataFrame(
        {"x": [0.5, 9.0], "y": [0.5, 9.0]}, index=pd.Index([1, 2], name="record_id")
    )
    view.set_points(coordinates)
    view._lasso = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assert view.finish_lasso() == [1]
    assert view._lasso == []


def test_a_short_lasso_selects_nothing(qapp):
    view = ChemicalSpaceView()
    view.set_points(pd.DataFrame({"x": [0.0], "y": [0.0]}, index=pd.Index([1], name="record_id")))
    view._lasso = [(0.0, 0.0), (1.0, 1.0)]
    assert view.finish_lasso() == []


def test_pareto_view_plots_two_objectives(qapp):
    view = ParetoView()
    values = pd.DataFrame(
        {"qed": [0.9, 0.2], "mol_wt": [300.0, 500.0]}, index=pd.Index([1, 2], name="record_id")
    )
    view.show_objectives(values, "qed", "mol_wt", pd.Series([1, 2], index=values.index))
    assert len(view.scatter.points()) == 2


# --- panels ------------------------------------------------------------------


def test_inspector_shows_what_exists_and_omits_the_rest(qapp):
    panel = InspectorPanel()
    descriptors = pd.DataFrame(
        {"molecule_id": ["MOL001"], "mol_wt": [180.16]}, index=pd.Index([1], name="record_id")
    )
    panel.show_molecule(1, descriptors, {"Pareto rank": 1})
    text = panel.details.toPlainText()
    assert "MW: 180.160" in text
    assert "Pareto rank: 1" in text
    assert "TPSA" not in text


def test_inspector_reports_an_unknown_record(qapp):
    panel = InspectorPanel()
    panel.show_molecule(99, pd.DataFrame(index=pd.Index([1], name="record_id")))
    assert "não encontrado" in panel.title.text()


def test_basket_panel_lists_only_decided_molecules(qapp):
    panel = BasketPanel()
    basket = SelectionBasket([MoleculeState(1), MoleculeState(2)])
    basket.add_to_final([1])
    panel.refresh(basket)
    assert panel.table.rowCount() == 1
    assert panel.undo_button.isEnabled()
    assert not panel.redo_button.isEnabled()


# --- workspace ---------------------------------------------------------------


def test_workspace_seeds_the_basket_from_the_chemistry(window, result):
    """Every record keeps the verdict the rules gave it."""
    assert len(window.basket) == result.total_records
    assert {state.chemical_status for state in window.basket.states()} <= set(ChemicalStatus)


def test_workspace_computes_pareto_on_open(window):
    assert window.pareto is not None
    assert set(window.pareto.table.columns) >= {"pareto_rank", "distance_to_ideal"}


def test_identical_objectives_are_refused(window):
    window.second_objective.setCurrentText(window.first_objective.currentText())
    assert window.pareto is None
    assert "diferentes" in window.warnings.text()


def test_clicking_a_point_fills_the_inspector(window):
    record_id = int(window.candidates.index[0])
    window._show_molecule(record_id)
    assert window.inspector.record_id == record_id
    assert "Status químico" in window.inspector.details.toPlainText()


def test_auto_selection_respects_the_target(window):
    window.target_count.setValue(2)
    window._auto_select()
    assert window.basket.counters().final_selected == 2


def test_undo_and_redo_walk_the_selection(window):
    window.target_count.setValue(2)
    window._auto_select()
    window._undo()
    assert window.basket.counters().final_selected == 0
    window._redo()
    assert window.basket.counters().final_selected == 2


def test_scaffold_quota_narrows_the_automatic_selection(window):
    window.target_count.setValue(6)
    window.per_scaffold.setValue(1)
    window._auto_select()
    scaffolds = window.candidates.loc[list(window.basket.final_ids()), "murcko_scaffold"]
    assert scaffolds.nunique() == len(scaffolds)


def test_candidates_carry_scaffolds_and_clusters(window):
    assert "murcko_scaffold" in window.candidates.columns
    assert "cluster_id" in window.candidates.columns


# --- export ------------------------------------------------------------------


def test_export_writes_the_sheets_and_the_recipe(window, tmp_path):
    import openpyxl

    window.target_count.setValue(3)
    window._auto_select()
    workbook, recipe = selection_export.export(window.build_artifacts(), tmp_path / "sessao.xlsx")

    names = set(openpyxl.load_workbook(workbook).sheetnames)
    assert {"SELECTION_SUMMARY", "FINAL_SELECTED", "MANUAL_DECISIONS", "SELECTION_RECIPE"} <= names
    assert recipe.exists()
    assert recipe.name.endswith(".selection.json")


def test_final_selected_sheet_carries_the_decision_columns(window):
    window.target_count.setValue(2)
    window._auto_select()
    sheet = selection_export.final_selected_sheet(window.build_artifacts())
    assert {"Selection_Status", "Selection_Origin", "Selection_Reason", "Pinned"} <= set(
        sheet.columns
    )
    assert sheet["Selection_Reason"].ne("-").all()


def test_summary_sheet_reports_counters_and_constraints(window):
    window.target_count.setValue(2)
    window._auto_select()
    summary = selection_export.selection_summary_sheet(window.build_artifacts())
    assert set(summary["section"]) >= {"cesta", "seleção", "restrições"}


def test_recipe_records_the_objectives(window):
    artifacts = window.build_artifacts()
    assert len(artifacts.recipe.objectives) == 2
    assert artifacts.recipe.input_hash
