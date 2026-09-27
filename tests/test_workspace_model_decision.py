"""A model proposal joins the native basket only after explicit verification."""

from __future__ import annotations

import shutil
import time
from concurrent.futures import CancelledError
from dataclasses import asdict
from pathlib import Path
from threading import Event

from PySide6.QtWidgets import QApplication, QMessageBox

from s2s_decision.artifacts import write_bundle, write_json
from s2s_decision.decision import _freeze_files
from s2s_decision.schema import FeatureSet
from s2s_decision.session_adapter import snapshot_from_workspace
from smiles2select.gui.workspace import criteria_state
from smiles2select.gui.workspace.model_decision import ModelPreview, package_hashes
from smiles2select.gui.workspace.workspace_window import WorkspaceWindow
from tests import test_workspace

qapp = test_workspace.qapp
result = test_workspace.result
window = test_workspace.window


def _proposal(window, root: Path, warnings=()) -> ModelPreview:
    snapshot = snapshot_from_workspace(window.result, window.candidates, window.basket, window.constraints())
    root.mkdir()
    ids = list(snapshot._records.record_id)
    chosen = ids[-2:]
    frame = snapshot._records.copy(deep=True)
    frame["priority_score"] = [index / len(ids) for index in range(len(ids))]
    frame["calibration_status"] = "calibrated"
    frame["is_final"] = frame.record_id.isin(chosen)
    frame["selection_reason"] = frame.is_final.map({True: "model rank", False: "not selected"})
    source = {"session_revision": snapshot.revision, "input_scope": "workspace_snapshot"}
    model_root = root.parent / "model"
    (model_root / "references").mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "model.onnx", "references/manifest.json",
                 "references/records.jsonl"):
        (model_root / name).write_text(name, encoding="utf-8")
    hashes = package_hashes(model_root)
    scored = {**source, "model_sha256": hashes["model.onnx"],
              "model_manifest_sha256": hashes["manifest.json"],
              "reference_records_sha256": hashes["references/records.jsonl"],
              "calibration_status": "calibrated",
              "task": {"target_id": "target-1", "endpoint": "IC50", "threshold": 6.0}}
    proposed = {"constraints": asdict(snapshot.constraints), "final_ids": chosen,
                "pinned_ids": list(snapshot.pinned_ids), "excluded_ids": list(snapshot.excluded_ids),
                "scaffold_counts": {}, "cluster_counts": {}, "warnings": list(warnings),
                "input_provenance": scored}
    write_bundle(FeatureSet(frame, source), root / "prepared")
    write_bundle(FeatureSet(frame, scored), root / "scored")
    write_bundle(FeatureSet(frame, proposed), root / "proposed")
    model = {"path": str(model_root), "name": "Demo model", "compatible": True,
             "target_id": "target-1", "endpoint": "IC50", "threshold": 6.0,
             "estimator": "logistic", "calibration_status": "calibrated",
             "train_reference_count": 10}
    summary = {"preview_version": 1, "source": source, "model": model,
               "ranking_mode": "experimental_model", "original_ids": list(snapshot.original_ids),
               "proposed_ids": chosen, "constraints": asdict(snapshot.constraints),
               "pinned_ids": list(snapshot.pinned_ids), "excluded_ids": list(snapshot.excluded_ids),
               "warnings": list(warnings), "eligible_count": int(frame.eligible.sum()),
               "original_count": len(snapshot.original_ids), "proposed_count": len(chosen),
               "retained_ids": sorted(set(chosen) & set(snapshot.original_ids)),
               "added_ids": sorted(set(chosen) - set(snapshot.original_ids)),
               "removed_ids": sorted(set(snapshot.original_ids) - set(chosen))}
    write_json(root / "comparison.json", summary)
    (root / "comparison.csv").write_text("record_id\n", encoding="utf-8")
    write_json(root / "preview.json", {"preview_version": 1, "files": _freeze_files(root)})
    return ModelPreview(root, snapshot, criteria_state.capture(window), model, hashes)


def test_model_preview_is_explicit_one_action_and_reversible(window, tmp_path):
    before = window.basket.final_ids()
    actions = len(window.basket.log.applied)
    preview = _proposal(window, tmp_path / "preview")
    window.model_panel.accept_preview(preview)
    assert window.basket.final_ids() == before
    assert "retained" in window.model_panel.report.toPlainText().lower()
    window.model_panel.adopt()
    assert len(window.basket.log.applied) == actions + 1
    assert set(window.basket.final_ids()) == set(preview.snapshot._records.record_id.iloc[-2:])
    assert window.basket.log.applied[-1].source == "experimental_model"
    artifacts = window.build_artifacts()
    assert artifacts.recipe.strategy == "experimental_model"
    assert artifacts.recipe.provenance["model_sha256"] == preview.files["model.onnx"]
    assert artifacts.model_scores.index.is_unique
    window._undo()
    assert window.basket.final_ids() == before
    window._redo()
    assert window.build_artifacts().recipe.strategy == "experimental_model"


def test_bundled_model_scope_is_visible_before_preview(window):
    panel = window.model_panel
    panel.refresh_catalog()
    assert any(item.get("origin") == "bundled" for item in panel._catalog)
    panel.target.setCurrentIndex(panel.target.findData("Q72547_WT"))
    panel.endpoint.setCurrentIndex(panel.endpoint.findData("IC50"))
    panel.models.setCurrentIndex(1)
    text = panel.info.toPlainText()
    assert "Papyrus++ 05.7" in text
    assert "source:" in text and "SHA-256:" in text
    assert "split: scaffold" in text


def test_model_warning_requires_explicit_acknowledgement(window, tmp_path, monkeypatch):
    preview = _proposal(window, tmp_path / "preview", ["Pinned molecules exceed requested count"])
    panel = window.model_panel
    panel.accept_preview(preview)
    before = len(window.basket.log.applied)
    monkeypatch.setattr(
        "smiles2select.gui.workspace.model_decision.QMessageBox.question",
        lambda *_args: QMessageBox.StandardButton.No,
    )
    panel.adopt()
    assert len(window.basket.log.applied) == before
    monkeypatch.setattr(
        "smiles2select.gui.workspace.model_decision.QMessageBox.question",
        lambda *_args: QMessageBox.StandardButton.Yes,
    )
    panel.adopt()
    assert len(window.basket.log.applied) == before + 1


def test_model_preview_runs_in_workspace_job_without_mutating_basket(window, tmp_path, monkeypatch):
    seed = _proposal(window, tmp_path / "seed")
    panel = window.model_panel
    panel._catalog = [seed.model]
    panel.target.addItem("target-1", "target-1")
    panel.target.setCurrentIndex(panel.target.count() - 1)
    panel.endpoint.setCurrentIndex(1)
    panel.models.setCurrentIndex(1)
    monkeypatch.setattr(
        "smiles2select.gui.workspace.model_decision.snapshot_from_workspace",
        lambda *args, **kwargs: seed.snapshot,
    )
    monkeypatch.setattr(
        "smiles2select.gui.workspace.model_decision.preview_session_decision",
        lambda snapshot, destination, model_dir, **_kwargs: shutil.copytree(seed.root, destination),
    )
    before = window.basket.final_ids()
    panel.preview()
    for _ in range(200):
        QApplication.processEvents()
        if not window.is_busy:
            break
        time.sleep(0.01)
    assert not window.is_busy
    assert panel.adopt_button.isEnabled()
    assert window.basket.final_ids() == before
    assert "retained" in panel.report.toPlainText().lower()
    preview_root = panel._preview.root
    panel.adopt()
    assert panel._preview is None
    assert not preview_root.exists()


def test_cancel_model_preview_keeps_basket_and_publishes_nothing(window, tmp_path, monkeypatch):
    seed = _proposal(window, tmp_path / "seed")
    panel = window.model_panel
    panel._catalog = [seed.model]
    panel.target.addItem("target-1", "target-1")
    panel.target.setCurrentIndex(panel.target.count() - 1)
    panel.endpoint.setCurrentIndex(1)
    panel.models.setCurrentIndex(1)
    started = Event()

    def wait_for_cancel(_snapshot, _destination, _model_dir, *, cancelled):
        started.set()
        for _ in range(300):
            if cancelled():
                raise CancelledError("Model preview cancelled")
            time.sleep(0.005)
        raise AssertionError("Cancel was not delivered")

    monkeypatch.setattr(
        "smiles2select.gui.workspace.model_decision.preview_session_decision", wait_for_cancel,
    )
    before = window.basket.final_ids()
    panel.preview()
    for _ in range(200):
        QApplication.processEvents()
        if started.is_set():
            break
        time.sleep(0.005)
    assert started.is_set()
    assert window.cancel_job_button.isVisible() or not window.cancel_job_button.isHidden()
    window.cancel_job_button.click()
    for _ in range(200):
        QApplication.processEvents()
        if not window.is_busy:
            break
        time.sleep(0.005)
    assert not window.is_busy
    assert panel._preview is None
    assert window.basket.final_ids() == before
    assert "cancelled" in panel.report.toPlainText().lower()


def test_model_choice_change_discards_previous_preview(window, tmp_path):
    preview = _proposal(window, tmp_path / "preview")
    panel = window.model_panel
    panel.accept_preview(preview)
    panel.models.addItem("Different model", {**preview.model, "path": "different"})
    panel.models.setCurrentIndex(panel.models.count() - 1)
    assert panel._preview is None
    assert not panel.adopt_button.isEnabled()
    assert "preview again" in panel.report.toPlainText().lower()


def test_model_adoption_rejects_changed_criteria_and_basket(window, tmp_path):
    preview = _proposal(window, tmp_path / "preview")
    window.model_panel.accept_preview(preview)
    before = window.basket.final_ids()
    window.target_count.setValue(window.target_count.value() + 1)
    window.model_panel.adopt()
    assert window.basket.final_ids() == before
    assert "changed" in window.model_panel.report.toPlainText().lower()
    window.target_count.setValue(preview.criteria.count)
    window.basket.add_to_shortlist([int(window.candidates.index[0])])
    manual = window.basket.final_ids()
    window.model_panel.adopt()
    assert window.basket.final_ids() == manual
    assert "changed" in window.model_panel.report.toPlainText().lower()


def test_model_adoption_rejects_tampered_preview_or_model(window, tmp_path):
    preview = _proposal(window, tmp_path / "preview")
    window.model_panel.accept_preview(preview)
    before = window.basket.final_ids()
    with (preview.root / "comparison.csv").open("a", encoding="utf-8") as stream:
        stream.write("tampered\n")
    window.model_panel.adopt()
    assert window.basket.final_ids() == before
    assert "checksum" in window.model_panel.report.toPlainText().lower()
    write_json(preview.root / "preview.json", {"preview_version": 1,
                                               "files": _freeze_files(preview.root)})
    (Path(preview.model["path"]) / "model.onnx").write_text("changed", encoding="utf-8")
    window.model_panel.adopt()
    assert window.basket.final_ids() == before
    assert "model" in window.model_panel.report.toPlainText().lower()


def test_saved_model_action_restores_redo_and_provenance(window, tmp_path):
    preview = _proposal(window, tmp_path / "preview")
    window.model_panel.accept_preview(preview)
    window.model_panel.adopt()
    model_ids = window.basket.final_ids()
    window._undo()
    original = window.basket.final_ids()
    path = window.save_session(tmp_path / "session.s2s.sqlite")
    restored = WorkspaceWindow.from_session(path)
    try:
        assert restored.basket.final_ids() == original
        assert restored.basket.log.can_redo
        restored._redo()
        assert restored.basket.final_ids() == model_ids
        artifacts = restored.build_artifacts()
        assert artifacts.recipe.strategy == "experimental_model"
        assert artifacts.recipe.provenance["model_sha256"] == preview.files["model.onnx"]
        assert artifacts.model_scores is not None
    finally:
        restored.close()


def test_saved_draft_criteria_remain_pending_after_open(window, tmp_path):
    requested = window.target_count.value() + 1
    window.target_count.setValue(requested)
    assert window.has_pending_criteria()
    restored = WorkspaceWindow.from_session(window.save_session(tmp_path / "draft.s2s.sqlite"))
    try:
        assert restored.target_count.value() == requested
        assert restored.has_pending_criteria()
        assert not restored.basket_panel.export_button.isEnabled()
    finally:
        restored.close()


def test_new_action_after_undo_does_not_reuse_model_metadata(window, tmp_path):
    preview = _proposal(window, tmp_path / "preview")
    window.model_panel.accept_preview(preview)
    window.model_panel.adopt()
    window._undo()
    window.basket.replace_final(window.basket.final_ids(), source="scenario:A")
    assert window._active_applied_selection() is None
    assert window.build_artifacts().model_scores is None


def test_branch_after_two_undos_prunes_abandoned_action_metadata(window, tmp_path):
    preview = _proposal(window, tmp_path / "preview")
    window.model_panel.accept_preview(preview)
    window.model_panel.adopt()
    window.target_count.setValue(2)
    window._auto_select()
    assert len(window.basket.log.applied) == 2
    window._undo()
    window._undo()
    window.target_count.setValue(3)
    window._auto_select()
    assert len(window.basket.log.applied) == 1
    for attached in (
        window._selection_outcomes, window._applied_selections,
        window._criteria_snapshots, window._model_scores_by_action,
        window._scenario_snapshots,
    ):
        assert set(attached).issubset({0})
    assert window._model_scores_by_action == {}
