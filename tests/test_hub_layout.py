"""The Hub reorganizes existing decisions without changing their scientific state."""

from __future__ import annotations

import time
from concurrent.futures import CancelledError
from threading import Event

import pandas as pd
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QComboBox, QFormLayout, QLabel, QPushButton, QWidget

from smiles2select.gui.workspace import criteria_state
from smiles2select.gui.workspace.panels import InspectorPanel
from tests import test_workspace
from tests.test_workspace_model_decision import _proposal

qapp = test_workspace.qapp
result = test_workspace.result
window = test_workspace.window

pytestmark = pytest.mark.integration
METHODS = ("chemical", "model", "contextual", "tolerance")


def _choose(window, method, qapp):
    index = window.method_selector.findData(method)
    assert index >= 0, f"The Hub must expose the {method} method"
    window.method_selector.setCurrentIndex(index)
    qapp.processEvents()


def _shown_inside(window, widget):
    return widget.isVisible() and window.rect().contains(
        widget.rect().translated(widget.mapTo(window, widget.rect().topLeft()))
    )


def _pump_until(qapp, predicate):
    deadline = time.monotonic() + 5
    while not predicate() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.005)
    qapp.processEvents()
    assert predicate(), "The workspace job did not settle"


def test_method_pages_keep_existing_panels_and_initial_selection(window, qapp, result):
    assert isinstance(window.method_selector, QComboBox)
    assert isinstance(window.method_pages, QWidget)
    assert tuple(window.method_selector.itemData(i)
                 for i in range(window.method_selector.count())) == METHODS
    assert window.method_selector.currentData() == "chemical"
    original_ids = window.basket.final_ids()
    original_criteria = criteria_state.capture(window)
    original_actions = tuple(window.basket.log.applied)
    window.show()
    panels = {"chemical": window.chemical_controls, "model": window.model_panel,
              "contextual": window.model_panel.contextual,
              "tolerance": window.model_panel.tolerance}
    for method, panel in panels.items():
        _choose(window, method, qapp)
        assert window.method_pages.isAncestorOf(panel)
        assert panel.isVisible()
        assert all(not other.isVisible() for key, other in panels.items() if key != method)
    _choose(window, "chemical", qapp)
    assert window.basket.final_ids() == original_ids == tuple(sorted(result.decision.selected_ids()))
    assert criteria_state.capture(window) == original_criteria
    assert tuple(window.basket.log.applied) == original_actions
    assert "Original pipeline" in window.selection_summary.text()
    assert not window.has_pending_criteria()


def test_method_switch_retains_inputs_and_verified_model_preview(window, qapp, tmp_path):
    panel = window.model_panel
    panel.contextual.target.setText("Target α / contextual")
    panel.tolerance.target.setText("Target β / tolerance")
    panel.contextual.report.setPlainText("Contextual preview: 2 molecules retained")
    panel.tolerance.report.setPlainText("Tolerance preview: review required")
    preview = _proposal(window, tmp_path / "preview")
    panel.accept_preview(preview)
    original_report = panel.report.toPlainText()
    original_ids = window.basket.final_ids()
    for method in (*METHODS, "model"):
        _choose(window, method, qapp)
    assert panel._preview is preview
    assert preview.root.exists()
    assert panel.adopt_button.isEnabled()
    assert panel.report.toPlainText() == original_report
    assert panel.contextual.target.text() == "Target α / contextual"
    assert panel.tolerance.target.text() == "Target β / tolerance"
    assert panel.contextual.report.toPlainText() == "Contextual preview: 2 molecules retained"
    assert panel.tolerance.report.toPlainText() == "Tolerance preview: review required"
    assert window.basket.final_ids() == original_ids


@pytest.mark.parametrize("size", [(1280, 800), (1440, 900)])
def test_selection_actions_remain_visible_while_method_form_scrolls(window, qapp, size):
    window.resize(*size)
    window.show()
    qapp.processEvents()
    assert window.width() <= size[0] and window.height() <= size[1]
    assert not window.controls_scroll.isAncestorOf(window.selection_actions)
    sticky = (window.method_selector, window.target_count, window.basket_panel.undo_button,
              window.basket_panel.redo_button, window.basket_panel.export_button,
              window.basket_panel.docking_button)
    for widget in sticky:
        assert window.selection_actions.isAncestorOf(widget)
    for method in ("model", "contextual", "tolerance", "chemical"):
        _choose(window, method, qapp)
        if method == "chemical":
            window.advanced_toggle.setChecked(True)
            qapp.processEvents()
            active_actions = (window.select_button,)
        else:
            panel = (window.model_panel if method == "model"
                     else getattr(window.model_panel, method))
            active_actions = (panel.preview_button, panel.adopt_button)
            assert not window.select_button.isVisible()
        assert all(window.selection_actions.isAncestorOf(widget) for widget in active_actions)
        scrollbar = window.controls_scroll.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        qapp.processEvents()
        assert window.controls_scroll.horizontalScrollBar().maximum() == 0
        assert all(_shown_inside(window, widget) for widget in (*sticky, *active_actions))


def test_map_controls_live_with_views_and_do_not_edit_selection_criteria(window, qapp, monkeypatch):
    window.target_count.setValue(2)
    window._auto_select()
    before = criteria_state.capture(window)
    selected = window.basket.final_ids()
    actions = tuple(window.basket.log.applied)
    assert not window.controls_scroll.isAncestorOf(window.map_controls)
    for widget in (window.view_selector, window.projection_selector, window.color_selector,
                   window.reference_overlay, window.map_button):
        assert window.map_controls.isAncestorOf(widget)
    for widget in (window.per_scaffold, window.min_scaffolds, window.first_objective):
        assert window.controls_scroll.isAncestorOf(widget)
    requested = []
    monkeypatch.setattr(window, "_start_job", lambda *args, **kwargs: requested.append(True))
    window.view_selector.setCurrentText("Pareto")
    window.projection_selector.setCurrentIndex(1)
    window.color_selector.setCurrentIndex(1)
    window.reference_overlay.setChecked(not window.reference_overlay.isChecked())
    window.map_button.click()
    qapp.processEvents()
    assert requested
    assert criteria_state.capture(window) == before
    assert window.basket.final_ids() == selected
    assert tuple(window.basket.log.applied) == actions
    assert not window.has_pending_criteria()


def test_scientific_form_labels_remain_readable_in_all_method_pages(window, qapp):
    window.resize(1280, 800)
    window.show()
    window.advanced_toggle.setChecked(True)
    pages = {"chemical": window.chemical_controls, "model": window.model_panel,
             "contextual": window.model_panel.contextual,
             "tolerance": window.model_panel.tolerance}
    for method, page in pages.items():
        _choose(window, method, qapp)
        for container in (page, window.advanced_content):
            labels = []
            for form in container.findChildren(QFormLayout):
                for row in range(form.rowCount()):
                    item = form.itemAt(row, QFormLayout.LabelRole)
                    label = item.widget() if item is not None else None
                    if isinstance(label, QLabel) and label.isVisible():
                        labels.append(label)
                        assert label.text().strip(), f"Empty scientific label in {method}"
                        assert label.width() > 0, f"Collapsed {method} label: {label.text()}"
                        assert label.height() > 0, f"Collapsed {method} label: {label.text()}"
                        assert label.parentWidget().rect().contains(label.geometry()), (
                            f"Clipped {method} label: {label.text()}"
                        )
            assert labels, f"No readable scientific form labels in {method}"


def test_pending_count_guards_both_exports_in_every_method(window, qapp, monkeypatch):
    window.target_count.setValue(2)
    window._auto_select()
    selected = window.basket.final_ids()
    window.target_count.setValue(3)
    monkeypatch.setattr(
        "smiles2select.gui.workspace.workspace_window.QFileDialog.getSaveFileName",
        lambda *args: pytest.fail("Pending criteria must block before opening an export dialog"),
    )
    for method in METHODS:
        _choose(window, method, qapp)
        assert window.has_pending_criteria()
        assert not window.basket_panel.export_button.isEnabled()
        assert not window.basket_panel.docking_button.isEnabled()
        window._export()
        assert "not been applied" in window.warnings.text()
        window._export_docking()
        assert "not been applied" in window.warnings.text()
        assert window.basket.final_ids() == selected
        assert "2 final" in window.selection_summary.text()


def test_persistent_undo_redo_restore_applied_criteria_after_method_switch(window, qapp):
    window.target_count.setValue(2)
    window._auto_select()
    first = window.build_artifacts()
    window.target_count.setValue(3)
    window.per_scaffold.setValue(1)
    window._auto_select()
    second = window.build_artifacts()
    _choose(window, "contextual", qapp)
    window.basket_panel.undo_button.click()
    assert window.constraints() == first.constraints
    assert window.basket.final_ids() == first.outcome.selected_ids
    assert not window.has_pending_criteria()
    _choose(window, "tolerance", qapp)
    window.basket_panel.redo_button.click()
    assert window.constraints() == second.constraints
    assert window.basket.final_ids() == second.outcome.selected_ids
    assert not window.has_pending_criteria()
    assert window.basket_panel.export_button.isEnabled()
    assert window.basket_panel.docking_button.isEnabled()


def test_persistent_create_button_runs_one_selection_action(window, qapp):
    _choose(window, "chemical", qapp)
    window.target_count.setValue(2)
    actions = len(window.basket.log.applied)
    window.select_button.click()
    assert len(window.basket.log.applied) == actions + 1
    assert len(window.basket.final_ids()) == 2
    assert not window.has_pending_criteria()


@pytest.mark.parametrize("finish", ["failure", "cancel"])
def test_jobs_disable_persistent_controls_and_restore_them(window, qapp, finish):
    _choose(window, "chemical", qapp)
    started, release = Event(), Event()
    selected = window.basket.final_ids()

    def compute():
        started.set()
        if not release.wait(5):
            raise RuntimeError("Test job was not released")
        if finish == "cancel":
            raise CancelledError("Hub preview cancelled")
        raise ValueError("Hub preview failed")

    window._start_job(compute, lambda _: pytest.fail("A failed preview cannot be adopted"),
                      "Computing Hub preview", cancel=release.set)
    try:
        _pump_until(qapp, started.is_set)
        for widget in (window.method_selector, window.target_count, window.select_button,
                       window.view_selector, window.projection_selector, window.map_button,
                       window.basket_panel.export_button, window.basket_panel.docking_button):
            assert not widget.isEnabled()
        if finish == "cancel":
            window.cancel_job_button.click()
        else:
            release.set()
        _pump_until(qapp, lambda: not window.is_busy)
        for widget in (window.method_selector, window.target_count, window.select_button,
                       window.view_selector, window.projection_selector, window.map_button):
            assert widget.isEnabled()
        expected = "cancelled" if finish == "cancel" else "failed"
        assert expected in window.warnings.text().lower()
        assert window.basket.final_ids() == selected
        assert not window.has_pending_criteria()
    finally:
        release.set()
        if window._job is not None:
            window._job.wait(6000)
            qapp.processEvents()


def test_inspector_sections_preserve_properties_evidence_and_literal_external_text(qapp):
    panel = InspectorPanel()
    molecule_id = '<b>Compound α</b><img src="missing">'
    evidence = '<script>alert("evidence")</script> & measured < 2 µM'
    descriptors = pd.DataFrame(
        {"molecule_id": [molecule_id], "canonical_smiles": ["CCO"], "mol_wt": [46.069],
         "qed": [0.407]}, index=[1],
    )
    extras = {"Chemical status": "AUTO_PASS", "Included in docking export": "Yes",
              "Biological evidence": evidence, "Chemical tolerance": "Review needed",
              "Future extra": "Unicode 🧪 and unknown fields remain visible"}
    panel.show_molecule(1, descriptors, extras)
    text = panel.details.toPlainText()
    for expected in (molecule_id, "SMILES: CCO", "MW: 46.069", "QED: 0.407",
                     *(f"{key}: {value}" for key, value in extras.items())):
        assert expected in text
    assert panel.title.text() == molecule_id
    assert panel.title.textFormat() == Qt.PlainText
    bold_sections = []
    block = panel.details.document().begin()
    while block.isValid():
        fragment = block.begin().fragment()
        if fragment.isValid() and fragment.charFormat().fontWeight() > QFont.Normal:
            bold_sections.append(block.text())
        block = block.next()
    assert len(bold_sections) >= 2, "Inspector information must have visible section headings"
    panel.close()


def test_inspector_missing_record_clears_structure_details_and_action_target(qapp):
    panel = InspectorPanel()
    descriptors = pd.DataFrame({"molecule_id": ["first"], "canonical_smiles": ["CCO"]}, index=[1])
    emitted = []
    for signal in (panel.shortlist_requested, panel.select_requested,
                   panel.exclude_requested, panel.pin_requested):
        signal.connect(emitted.append)
    panel.show_molecule(1, descriptors, {"Biological evidence": "Previous evidence"})
    assert not panel.structure.pixmap().isNull()
    panel.show_molecule(999, descriptors)
    assert panel.record_id is None
    assert panel.structure.pixmap().isNull()
    assert panel.details.toPlainText() == ""
    assert "not found" in panel.title.text()
    for button in panel.findChildren(QPushButton):
        button.click()
    assert emitted == []
    panel.close()
