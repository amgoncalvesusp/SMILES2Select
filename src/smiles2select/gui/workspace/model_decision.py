"""Model proposals inside the shared SMILES2Select selection workspace."""

from __future__ import annotations

import tempfile
from concurrent.futures import CancelledError
from dataclasses import dataclass
from pathlib import Path
from threading import Event

from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QToolButton,
    QWidget,
)

from s2s_decision.artifacts import file_hash, read_bundle
from s2s_decision.decision import preview_session_decision, verify_preview
from s2s_decision.model_catalog import import_model_package, list_catalog_models
from s2s_decision.session_adapter import SessionSnapshot, snapshot_from_workspace
from smiles2select.gui.workspace import criteria_state
from smiles2select.gui.workspace.model_guidance import (
    ModelGuideDialog,
    model_guidance_text,
    model_label,
)
from smiles2select.gui.workspace.selection_provenance import AppliedSelection
from smiles2select.selection_intelligence.basket import SelectionBasket
from smiles2select.selection_intelligence.constrained_selection import (
    SelectionOutcome,
    Strategy,
)
from smiles2select.selection_intelligence.objectives import ObjectiveSet

_MODEL_FILES = (
    "manifest.json", "model.onnx", "references/manifest.json", "references/records.jsonl",
)


def package_hashes(path: str | Path) -> dict[str, str]:
    """Bind a proposal to the exact model and reference files used for scoring."""
    root = Path(path)
    return {name: file_hash(root / name) for name in _MODEL_FILES}


@dataclass(frozen=True)
class ModelPreview:
    root: Path
    snapshot: SessionSnapshot
    criteria: criteria_state.CriteriaState
    model: dict
    files: dict[str, str]
    temporary: tempfile.TemporaryDirectory | None = None


class ModelDecisionPanel(QGroupBox):
    """Choose a task-specific model; preview before replacing the native basket."""

    def __init__(self, workspace):
        super().__init__("Prioritize by model (experimental)", workspace)
        self.workspace = workspace
        self._catalog: list[dict] = []
        self._preview: ModelPreview | None = None
        self._preview_filters: tuple[str | None, str | None] | None = None
        self._chosen_model_path: str | None = None
        self.target = QComboBox()
        self.endpoint = QComboBox()
        self.models = QComboBox()
        self.info = QPlainTextEdit()
        self.info.setReadOnly(True)
        self.info.setMaximumHeight(210)
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMaximumHeight(150)
        self.refresh_button = QPushButton("Refresh models")
        self.import_button = QPushButton("Import model folder...")
        self.guide_button = QPushButton("Model guide and selection tips...")
        self._guide_dialog = None
        self.preview_button = QPushButton("Preview model selection")
        self.adopt_button = QPushButton("Adopt proposal")
        self.adopt_button.setEnabled(False)
        self.adopt_button.setToolTip("Replace current final basket in one undoable action.")

        form = QFormLayout(self)
        form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        form.addRow("Target", self.target)
        form.addRow("Endpoint", self.endpoint)
        form.addRow("Model", self.models)
        buttons = QWidget()
        row = QHBoxLayout(buttons)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.refresh_button)
        row.addWidget(self.import_button)
        form.addRow(buttons)
        form.addRow(self.info)
        form.addRow(self.guide_button)
        form.addRow(self.preview_button)
        form.addRow(self.adopt_button)
        form.addRow(self.report)
        from .contextual_controls import ContextualControls

        self.contextual_toggle = QToolButton()
        self.contextual_toggle.setText("Contextual policy (experimental)")
        self.contextual_toggle.setCheckable(True)
        self.contextual = ContextualControls(workspace, self)
        self.contextual.hide()
        self.contextual_toggle.toggled.connect(self.contextual.setVisible)
        form.addRow(self.contextual_toggle)
        form.addRow(self.contextual)

        self.target.currentIndexChanged.connect(self._target_changed)
        self.endpoint.currentIndexChanged.connect(self._endpoint_changed)
        self.models.currentIndexChanged.connect(self._model_changed)
        self.refresh_button.clicked.connect(self.refresh_catalog)
        self.import_button.clicked.connect(self.import_model)
        self.guide_button.clicked.connect(self.show_guide)
        self.preview_button.clicked.connect(self.preview)
        self.adopt_button.clicked.connect(self.adopt)
        self.refresh_catalog()

    def refresh_catalog(self) -> None:
        try:
            catalog = list_catalog_models()
        except (OSError, ValueError, RuntimeError) as exc:
            self.report.setPlainText(f"Model catalog unavailable: {exc}")
            catalog = []
        self._catalog = catalog
        self.target.blockSignals(True)
        self.target.clear()
        self.target.addItem("Choose target", None)
        for value in sorted({str(item["target_id"]) for item in catalog if item.get("target_id")}):
            self.target.addItem(value, value)
        self.target.blockSignals(False)
        self._target_changed()
        if not catalog:
            self.info.setPlainText("No compatible models found. Import a model folder to enable proposals.")

    def _target_changed(self) -> None:
        target = self.target.currentData()
        self.endpoint.blockSignals(True)
        self.endpoint.clear()
        self.endpoint.addItem("Choose endpoint", None)
        if target:
            for value in sorted({str(item["endpoint"]) for item in self._catalog
                                 if item.get("target_id") == target and item.get("endpoint")}):
                self.endpoint.addItem(value, value)
        self.endpoint.blockSignals(False)
        self._endpoint_changed()

    def _endpoint_changed(self) -> None:
        target, endpoint = self.target.currentData(), self.endpoint.currentData()
        self.models.blockSignals(True)
        self.models.clear()
        self.models.addItem("Choose model", None)
        if target and endpoint:
            for item in self._catalog:
                if item.get("target_id") == target and item.get("endpoint") == endpoint:
                    label = model_label(item)
                    if not item["compatible"]:
                        label += " (unavailable)"
                    self.models.addItem(label, item)
                    self.models.setItemData(self.models.count() - 1,
                                            item.get("reason") or item["path"], 3)
        self.models.blockSignals(False)
        self._model_changed()

    def _model_changed(self) -> None:
        if self._preview is not None:
            self.clear_preview()
            self.report.setPlainText("Model, target or endpoint changed. Preview again.")
        item = self.models.currentData()
        self._chosen_model_path = item.get("path") if isinstance(item, dict) else None
        self.guide_button.setEnabled(isinstance(item, dict))
        self.preview_button.setEnabled(isinstance(item, dict) and bool(item.get("compatible")))
        if not item:
            self.info.setPlainText(
                "Choose target, endpoint and model. Bundled tasks offer logistic regression, "
                "two gradient-boosting variants and Tiny. All are task-specific. "
                "Chemical strategies are available under Selection strategy above."
            )
            return
        self.info.setPlainText(model_guidance_text(item))

    def show_guide(self) -> None:
        item = self.models.currentData()
        if not isinstance(item, dict):
            return
        if self._guide_dialog is not None:
            self._guide_dialog.close()
            self._guide_dialog.deleteLater()
        self._guide_dialog = ModelGuideDialog(item, self)
        self._guide_dialog.show()

    def import_model(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose ONNX model package")
        if not path:
            return
        try:
            destination = import_model_package(path)
        except (OSError, ValueError, RuntimeError) as exc:
            self.report.setPlainText(f"Model import failed: {exc}")
            return
        self.refresh_catalog()
        self.report.setPlainText(f"Imported {destination.name}. Choose target, endpoint and model.")

    def preview(self) -> None:
        model = self.models.currentData()
        if self.workspace.is_busy:
            return
        if not isinstance(model, dict) or not model.get("compatible"):
            self.report.setPlainText("Choose a compatible model for this target and endpoint.")
            return
        try:
            result = self.workspace.result
            candidates = self.workspace.candidates.copy(deep=True)
            basket = SelectionBasket(self.workspace.basket.states())
            constraints = self.workspace.constraints()
            criteria = criteria_state.capture(self.workspace)
        except (OSError, ValueError, KeyError) as exc:
            self.report.setPlainText(f"Cannot preview model selection: {exc}")
            return
        temporary = tempfile.TemporaryDirectory(
            prefix="smiles2select-model-", ignore_cleanup_errors=True,
        )
        cancelled = Event()

        def compute():
            try:
                snapshot = snapshot_from_workspace(result, candidates, basket, constraints)
                hashes = package_hashes(model["path"])
                proposal = ModelPreview(
                    Path(temporary.name) / "preview", snapshot, criteria, model, hashes, temporary,
                )
                preview_session_decision(
                    snapshot, proposal.root, model["path"], cancelled=cancelled.is_set,
                )
                verify_preview(proposal.root)
                return proposal
            except Exception as exc:
                return exc

        def completed(value):
            if isinstance(value, Exception):
                temporary.cleanup()
                self.report.setPlainText(
                    "Model preview cancelled." if isinstance(value, CancelledError)
                    else f"Model preview failed: {value}"
                )
            elif cancelled.is_set():
                temporary.cleanup()
                self.report.setPlainText("Model preview cancelled.")
            else:
                try:
                    self.accept_preview(value)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    temporary.cleanup()
                    self.report.setPlainText(f"Model preview failed: {exc}")

        self.workspace._start_job(
            compute, completed, "Computing model proposal; current basket unchanged...",
            cancel=cancelled.set,
        )

    def accept_preview(self, preview: ModelPreview) -> None:
        summary = verify_preview(preview.root)
        if (summary.get("ranking_mode") != "experimental_model"
                or summary.get("source", {}).get("session_revision") != preview.snapshot.revision
                or summary.get("model", {}).get("path") != preview.model.get("path")):
            raise ValueError("Preview does not match model or session snapshot")
        old = self._preview
        self._preview = preview
        self._preview_filters = (self.target.currentData(), self.endpoint.currentData())
        self._chosen_model_path = preview.model["path"]
        self.adopt_button.setEnabled(True)
        if old is not None and old.temporary is not None and old.temporary is not preview.temporary:
            old.temporary.cleanup()
        warnings = summary.get("warnings") or []
        self.report.setPlainText(
            f"Eligible: {summary['eligible_count']:,}; current: {summary['original_count']:,}; "
            f"proposed: {summary['proposed_count']:,}.\n"
            f"Retained: {len(summary['retained_ids']):,}; added: {len(summary['added_ids']):,}; "
            f"removed: {len(summary['removed_ids']):,}.\n"
            f"Added IDs: {summary['added_ids'][:20]}; removed IDs: {summary['removed_ids'][:20]}.\n"
            + ("Warnings: " + "; ".join(warnings) + "\n" if warnings else "")
            + "Current basket unchanged. Adopt explicitly to use this proposal."
        )

    def clear_preview(self) -> None:
        previous = self._preview
        self._preview = None
        self.adopt_button.setEnabled(False)
        if previous is not None and previous.temporary is not None:
            previous.temporary.cleanup()

    def adopt(self) -> None:
        preview = self._preview
        if preview is None or self.workspace.is_busy:
            self.report.setPlainText("Preview a compatible model before adopting.")
            return
        try:
            if criteria_state.capture(self.workspace) != preview.criteria:
                raise ValueError("Workspace criteria changed since preview; preview again")
            if self._preview_filters != (self.target.currentData(), self.endpoint.currentData()):
                raise ValueError("Target or endpoint changed since preview; preview again")
            if self._chosen_model_path != preview.model["path"]:
                raise ValueError("Selected model changed since preview; preview again")
            current = snapshot_from_workspace(
                self.workspace.result, self.workspace.candidates,
                self.workspace.basket, self.workspace.constraints(),
                session_id=preview.snapshot.session_id,
            )
            if current.revision != preview.snapshot.revision:
                raise ValueError("Workspace decisions changed since preview; preview again")
            if package_hashes(preview.model["path"]) != preview.files:
                raise ValueError("Model files changed since preview; preview again")
            summary = verify_preview(preview.root)
            if (summary.get("ranking_mode") != "experimental_model"
                    or summary.get("source", {}).get("session_revision") != preview.snapshot.revision
                    or summary.get("model", {}).get("path") != preview.model["path"]
                    or summary.get("original_ids") != list(preview.snapshot.original_ids)
                    or summary.get("pinned_ids") != list(preview.snapshot.pinned_ids)
                    or summary.get("excluded_ids") != list(preview.snapshot.excluded_ids)):
                raise ValueError("Preview does not match frozen model, basket or constraints")
            proposed = read_bundle(preview.root / "proposed")
            scored = read_bundle(preview.root / "scored")
            if (scored.manifest.get("model_sha256") != preview.files["model.onnx"]
                    or scored.manifest.get("model_manifest_sha256") != preview.files["manifest.json"]
                    or scored.manifest.get("reference_records_sha256")
                    != preview.files["references/records.jsonl"]):
                raise ValueError("Model evidence differs from scored proposal")
            scores = scored.records[["record_id", "priority_score", "calibration_status"]].copy()
            if scores.record_id.isna().any() or scores.record_id.duplicated().any():
                raise ValueError("Model scores require unique record IDs")
            scores = scores.set_index("record_id")
            if set(scores.index) != set(current._records.record_id):
                raise ValueError("Model scores do not match workspace record IDs")
            selected_ids = tuple(int(value) for value in summary["proposed_ids"])
            selected_set = set(selected_ids)
            if not selected_set.issubset(scores.index):
                raise ValueError("Proposal contains unknown record IDs")
            if not set(preview.snapshot.pinned_ids).issubset(selected_set):
                raise ValueError("Proposal drops pinned molecules")
            explanations = proposed.records.set_index("record_id")["selection_reason"]
            outcome = SelectionOutcome(
                selected_ids=selected_ids,
                reasons={rid: [str(explanations.loc[rid])] for rid in selected_ids},
                rejections={int(rid): [str(reason)] for rid, reason in explanations.items()
                            if rid not in selected_set},
                scaffold_usage=proposed.manifest.get("scaffold_counts", {}),
                cluster_usage={int(key): value for key, value in
                               proposed.manifest.get("cluster_counts", {}).items()},
                strategy=Strategy.BALANCED,
            )
            provenance = {
                **self.workspace._provenance(),
                "source": "experimental_model",
                "ranking_method": "onnx_score",
                "ranking_universe": "all eligible workspace candidates",
                "model_sha256": scored.manifest["model_sha256"],
                "model_manifest_sha256": scored.manifest["model_manifest_sha256"],
                "reference_records_sha256": scored.manifest["reference_records_sha256"],
                "target": preview.model["target_id"],
                "endpoint": preview.model["endpoint"],
                "threshold": preview.model["threshold"],
                "estimator": preview.model["estimator"],
                "calibration_status": summary.get("calibration_status"),
                "chemistry_hash": summary["source"].get("chemistry_hash"),
                "preview_sha256": file_hash(preview.root / "preview.json"),
                "session_revision": preview.snapshot.revision,
                "ranking_warnings": tuple(summary.get("warnings") or ()),
            }
            if summary.get("warnings"):
                warning_text = "\n".join(str(item) for item in summary["warnings"])
                answer = QMessageBox.question(
                    self, "Confirm model selection exceptions",
                    f"Proposal reports these exceptions:\n{warning_text}\n\nAdopt this selection?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    self.report.setPlainText("Proposal not adopted; current basket unchanged.")
                    return
            applied = AppliedSelection.capture(
                constraints=preview.snapshot.constraints, objectives=ObjectiveSet(),
                pareto=None, strategy="experimental_model",
                input_hash=provenance["input_hash"], provenance=provenance,
            )
            self.workspace._apply_selection(
                outcome, preview.snapshot.constraints, applied,
                source="experimental_model", reason="Adopted model proposal",
                model_scores=scores,
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.report.setPlainText(f"Adoption blocked: {exc}")
            return
        self.clear_preview()
        self.report.setPlainText(
            f"Model proposal adopted: {len(selected_ids):,} final molecules. "
            "Undo restores the previous basket."
        )
