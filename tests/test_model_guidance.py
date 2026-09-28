"""Model choice explains its task and estimator before changing the basket."""

import pytest

from smiles2select.gui.workspace.model_guidance import model_guidance_text, model_label
from tests import test_workspace

qapp = test_workspace.qapp
result = test_workspace.result
window = test_workspace.window


@pytest.mark.parametrize("estimator,layout,label,tip", [
    ("tiny", "tiny_branches", "Tiny", "Compare with logistic regression"),
    ("logistic", "scalar_fingerprint", "Logistic regression", "simple supervised baseline"),
    ("gradient_boosting", "scalar", "Gradient boosting", "property and reference-context"),
    ("gradient_boosting", "scalar_fingerprint", "Gradient boosting + Morgan", "structural fingerprints"),
])
def test_each_estimator_explains_use_without_claiming_universal_activity(estimator, layout, label, tip):
    item = {"name": "package", "estimator": estimator, "input_layout": layout,
            "target_id": "Q72547_WT", "endpoint": "IC50", "threshold": 6,
            "compatible": True, "origin": "bundled", "validation_selected": False,
            "calibration_status": "calibrated", "train_reference_count": 1803}
    text = model_guidance_text(item)
    assert label in model_label(item)
    assert tip in text
    assert "Q72547_WT" in text and "IC50" in text and "pActivity >= 6" in text
    assert "experimental comparator" in text
    assert "do not guarantee activity" in text
    assert "Create selection" in text and "Adopt proposal" in text


def test_validation_choice_and_unknown_import_are_not_invented():
    chosen = {"name": "chosen", "origin": "bundled", "validation_selected": True,
              "estimator": "logistic", "input_layout": "scalar_fingerprint"}
    assert "validation-selected baseline" in model_guidance_text(chosen)
    imported = {**chosen, "origin": "user", "validation_selected": False}
    assert "Imported package" in model_guidance_text(imported)
    assert "Papyrus" not in model_guidance_text(imported)
    assert "not declared" in model_guidance_text(imported)


def test_model_help_follows_every_task_choice_without_changing_basket(window):
    panel = window.model_panel
    original = window.basket.final_ids()
    for target, endpoint in (("Q72547_WT", "IC50"), ("P0DMS8_WT", "Ki"), ("Q07869_WT", "EC50")):
        panel.target.setCurrentIndex(panel.target.findData(target))
        panel.endpoint.setCurrentIndex(panel.endpoint.findData(endpoint))
        items = [panel.models.itemData(i) for i in range(1, panel.models.count())]
        assert len([item for item in items if item.get("origin") == "bundled"]) == 4
        for index, item in enumerate(items, 1):
            panel.models.setCurrentIndex(index)
            assert target in panel.info.toPlainText()
            assert "Selection tips" in panel.info.toPlainText()
            assert panel.guide_button.isEnabled()
            assert panel.preview_button.isEnabled() == item["compatible"]
            panel.guide_button.click()
            assert panel._guide_dialog.text.toPlainText() == model_guidance_text(item)
            panel._guide_dialog.close()
            assert window.basket.final_ids() == original
