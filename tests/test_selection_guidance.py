"""Explain real selection behavior before a beginner commits to criteria."""

from PySide6.QtWidgets import QComboBox, QDialogButtonBox

from smiles2select.gui.workspace.guided_controls import ObjectiveControls
from smiles2select.selection_intelligence.objectives import Direction
from tests import test_workspace

qapp = test_workspace.qapp
result = test_workspace.result
window = test_workspace.window


def test_help_accessible_without_advanced_controls(window, qapp):
    window.show()
    qapp.processEvents()
    assert window.selection_help_button.isVisible()
    assert not window.advanced_content.isVisible()
    window.selection_help_button.click()
    qapp.processEvents()
    dialog = window.selection_help_dialog
    assert dialog.isVisible()
    text = dialog.browser.toPlainText()
    for term in ("2,000", "weighted", "Murcko", "PCA", "SA", "NP", "Create selection"):
        assert term in text
    assert dialog.browser.verticalScrollBar().maximum() > 0
    dialog.findChild(QDialogButtonBox).rejected.emit()
    assert not dialog.isVisible()


def test_objective_explanation_tracks_descriptor_without_changing_direction(qapp):
    field = QComboBox()
    field.addItems(["qed", "sa_score", "np_score", "rdkit_wlogp", "tpsa", "mol_wt"])
    control = ObjectiveControls(field, Direction.MAXIMIZE)
    assert "drug-likeness" in control.help_label.text()
    field.setCurrentText("sa_score")
    assert "Lower" in control.help_label.text()
    assert "synthesis" in control.help_label.text()
    assert control.direction.currentData() == Direction.MAXIMIZE
    field.setCurrentText("np_score")
    assert "natural" in control.help_label.text()
    assert "activity" in control.help_label.text()
    control.close()


def test_strategy_help_discloses_large_pool_override_and_missing_scaffolds():
    from smiles2select.gui.workspace.selection_help import strategy_help_text

    text = strategy_help_text("scaffold_coverage", large_pool=True, scaffold_ready=False)
    assert "Core coverage remains active" in text
    assert "same selection" in strategy_help_text("balanced", large_pool=True)
    assert "2,000" in text
    assert "computed" in text
    assert "not guarantee" in strategy_help_text("scaffold_coverage")
