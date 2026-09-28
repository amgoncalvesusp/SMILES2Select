"""Real bundled ONNX model through the Qt workspace and portable export."""

from __future__ import annotations

import json
import time

import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from s2s_decision.artifacts import read_bundle
from smiles2select.export.selection_export import export
from smiles2select.gui.workspace.model_decision import package_hashes
from smiles2select.gui.workspace.workspace_window import WorkspaceWindow
from tests import test_workspace

qapp = test_workspace.qapp
result = test_workspace.result
window = test_workspace.window


@pytest.mark.parametrize("estimator,layout", [
    ("tiny", "tiny_branches"),
    ("logistic", "scalar_fingerprint"),
    ("gradient_boosting", "scalar"),
    ("gradient_boosting", "scalar_fingerprint"),
])
def test_real_model_workspace_preview_adopt_save_reopen_export(
    window, tmp_path, monkeypatch, estimator, layout,
):
    panel = window.model_panel
    panel.target.setCurrentIndex(panel.target.findData("Q72547_WT"))
    panel.endpoint.setCurrentIndex(panel.endpoint.findData("IC50"))
    model_index = next(
        index for index in range(1, panel.models.count())
        if panel.models.itemData(index)["origin"] == "bundled"
        and panel.models.itemData(index)["estimator"] == estimator
        and panel.models.itemData(index)["input_layout"] == layout
    )
    panel.models.setCurrentIndex(model_index)
    model = panel.models.currentData()
    assert model["compatible"]
    hashes = package_hashes(model["path"])

    window.target_count.setValue(2)
    original = window.basket.final_ids()
    actions = len(window.basket.log.applied)
    panel.preview()
    deadline = time.monotonic() + 90
    while window.is_busy and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)
    QApplication.processEvents()
    assert not window.is_busy, "real ONNX preview exceeded 90 seconds"
    assert panel._preview is not None, panel.report.toPlainText()
    assert window.basket.final_ids() == original

    preview = panel._preview
    summary = json.loads((preview.root / "comparison.json").read_text(encoding="utf-8"))
    scored = read_bundle(preview.root / "scored")
    expected_ids = tuple(summary["proposed_ids"])
    assert summary["ranking_mode"] == "experimental_model"
    assert len(expected_ids) == 2
    assert set(expected_ids) <= set(scored.records.record_id)
    assert scored.manifest["model_sha256"] == hashes["model.onnx"]
    assert scored.manifest["model_manifest_sha256"] == hashes["manifest.json"]
    assert scored.manifest["reference_records_sha256"] == hashes["references/records.jsonl"]
    assert scored.records.priority_score.notna().any()

    monkeypatch.setattr(
        "smiles2select.gui.workspace.model_decision.QMessageBox.question",
        lambda *_args: QMessageBox.StandardButton.Yes,
    )
    panel.adopt()
    assert len(window.basket.log.applied) == actions + 1, panel.report.toPlainText()
    assert set(window.basket.final_ids()) == set(expected_ids)
    assert window.basket.log.applied[-1].source == "experimental_model"
    window._undo()
    assert window.basket.final_ids() == original
    window._redo()
    assert set(window.basket.final_ids()) == set(expected_ids)

    session = window.save_session(tmp_path / "model.s2s.sqlite")
    restored = WorkspaceWindow.from_session(session)
    try:
        assert set(restored.basket.final_ids()) == set(expected_ids)
        assert restored.basket.log.applied[-1].source == "experimental_model"
        artifacts = restored.build_artifacts()
        assert artifacts.recipe.strategy == "experimental_model"
        assert artifacts.recipe.provenance["model_sha256"] == hashes["model.onnx"]
        assert artifacts.model_scores is not None
        assert set(artifacts.model_scores.index) == set(scored.records.record_id)

        workbook, recipe = export(artifacts, tmp_path / "model-selection.xlsx")
        final = pd.read_excel(workbook, sheet_name="FINAL_SELECTED").set_index("record_id")
        scores = pd.read_excel(workbook, sheet_name="MODEL_SCORES").set_index("record_id")
        assert set(final.index) == set(expected_ids)
        assert set(scores.index) == set(scored.records.record_id)
        assert set(scores.index[scores.Final_Selected]) == set(expected_ids)
        source = window.result.descriptors
        for record_id, row in final.iterrows():
            assert row["ID"] == source.loc[record_id, "molecule_id"]
            assert row["Canonical_SMILES"] == source.loc[record_id, "canonical_smiles"]
            assert pd.notna(row["Model_Priority_Score"])
        exported_recipe = json.loads(recipe.read_text(encoding="utf-8"))
        assert exported_recipe["model"]["model_sha256"] == hashes["model.onnx"]
        assert exported_recipe["model"]["model_manifest_sha256"] == hashes["manifest.json"]
        assert exported_recipe["model"]["reference_records_sha256"] == hashes[
            "references/records.jsonl"
        ]
    finally:
        restored.close()
