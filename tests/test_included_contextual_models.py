"""Included models can be selected without locating research output directories."""

from PySide6.QtWidgets import QMessageBox

from tests import test_workspace
from tests.test_workspace_contextual_policy import wait_preview

qapp = test_workspace.qapp
result = test_workspace.result
window = test_workspace.window


def test_included_activity_and_optional_risk_use_normal_adoption(window, monkeypatch):
    controls = window.model_panel.contextual
    before = window.basket.final_ids()
    assert controls.included_models.count() == 10
    assert controls._package is None
    index = next(i for i in range(1, controls.included_models.count())
                 if "P22303" in controls.included_models.itemText(i) and "42" in controls.included_models.itemText(i))
    controls.included_models.setCurrentIndex(index)
    assert controls.included_models.toolTip() == controls.included_models.currentText()
    assert controls.included_models.minimumSizeHint().width() < controls.included_models.fontMetrics().horizontalAdvance(controls.included_models.currentText())
    assert controls._package is None
    controls.load_included_button.click()
    assert controls._context.target == "P22303"
    assert "calibration seed 42" in controls.package_label.text()
    assert "ChEMBL37" in controls.package_label.text()
    controls.included_risk_button.click()
    assert controls._risk_package.name == "shsy5y_atp_viability_48h.json"
    window.target_count.setValue(2)
    controls.preview()
    wait_preview(window)
    assert window.basket.final_ids() == before
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Yes)
    controls.adopt()
    assert len(window.basket.final_ids()) == 2
    assert "Experimental contextual ranking" in window.selection_summary.text()
    window._undo()
    assert window.basket.final_ids() == before
