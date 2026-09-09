"""Beginner workspace and bounded display regressions with real Qt widgets."""

from dataclasses import replace

import pandas as pd

from smiles2select.gui.workspace.panels import BasketPanel, InspectorPanel
from smiles2select.gui.workspace.workspace_window import WorkspaceWindow
from smiles2select.selection_intelligence.basket import SelectionBasket
from smiles2select.selection_intelligence.states import ChemicalStatus, MoleculeState
from tests import test_workspace

qapp = test_workspace.qapp
result = test_workspace.result
window = test_workspace.window


def test_essential_controls_visible_advanced_collapsed_on_notebook(window, qapp):
    window.resize(1280, 800)
    window.show()
    qapp.processEvents()
    assert window.target_count.maximum() >= 3_000_000
    assert window.select_button.isVisible()
    assert window.controls_scroll.widgetResizable()
    assert not window.advanced_content.isVisible()
    window.advanced_toggle.click()
    qapp.processEvents()
    assert window.advanced_content.isVisible()
    assert window.controls_scroll.verticalScrollBar().maximum() > 0
    assert "qed" in window.criteria_summary.text().lower()


def test_inspector_renders_one_structure_and_clears_missing_record(qapp):
    panel = InspectorPanel()
    data = pd.DataFrame({"canonical_smiles": ["CCO"]}, index=[1])
    panel.show_molecule(1, data)
    assert not panel.structure.pixmap().isNull()
    panel.show_molecule(2, data)
    assert panel.structure.pixmap().isNull()


def test_basket_caps_display_without_truncating_selection(qapp):
    basket = SelectionBasket(MoleculeState(i, ChemicalStatus.AUTO_PASS) for i in range(1200))
    basket.add_to_final(list(range(1200)))
    panel = BasketPanel()
    panel.refresh(basket)
    assert panel.table.rowCount() <= 500
    assert "1200" in panel.display_notice.text()
    assert len(basket.final_ids()) == 1200


def test_large_workspace_does_not_start_exact_pareto(window, monkeypatch):
    window.candidates = pd.concat([window.candidates] * 1001, ignore_index=True)
    monkeypatch.setattr(window.ranker, "rank", lambda *a: (_ for _ in ()).throw(
        AssertionError("Exact Pareto must not run automatically on large libraries")
    ))
    window._recompute_pareto()
    assert window.pareto is None
    assert "large" in window.performance_notice.text().lower()


def test_original_count_reduction_is_not_a_chemical_failure(qapp, result):
    selected_id = int(result.decision.selected_ids()[0])
    reduced = result.decision.decisions.assign(selected=False)
    altered = replace(result, decision=replace(result.decision, decisions=reduced))
    workspace = WorkspaceWindow(altered)
    assert workspace.basket.state(selected_id).chemical_status.passed
    workspace._show_molecule(selected_id)
    assert "later allocation" in workspace.inspector.details.toPlainText()
    workspace.close()
