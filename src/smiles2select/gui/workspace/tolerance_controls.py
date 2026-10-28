"""Optional chemical-tolerance proposals in the existing selection workspace."""

import json
import tempfile
from concurrent.futures import CancelledError
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from threading import Event

from PySide6.QtCore import QProcess, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from s2s_decision.artifacts import file_hash, read_bundle
from s2s_decision.decision import verify_preview
from smiles2select.gui.workspace import criteria_state
from smiles2select.gui.workspace.selection_provenance import AppliedSelection
from smiles2select.selection_intelligence.basket import SelectionBasket
from smiles2select.selection_intelligence.constrained_selection import SelectionOutcome, Strategy
from smiles2select.selection_intelligence.objectives import ObjectiveSet


@dataclass(frozen=True)
class TolerancePreview:
    root: Path
    snapshot: object
    criteria: object
    signature: str
    package: Path
    files: dict
    temporary: object
    summary: object = None
    records: object = None


def package_files(path):
    from s2s_decision.tolerance_decision import tolerance_package_hash

    return {"package_sha256": tolerance_package_hash(path)}


def molecule_evidence(row, context):
    fields = (
        ("tolerance_state", "Tolerance decision"),
        ("tolerance_eligible", "Tolerance eligibility"),
        ("tolerance_nonconformity", "Tolerance nonconformity"),
        ("tolerance_support", "Tolerance support"),
        ("tolerance_evidence", "Tolerance rule and alert evidence"),
        ("tolerance_applied_context", "Tolerance applied context"),
        ("tolerance_fallback", "Tolerance fallback"),
        ("selection_reason", "Selection explanation"),
    )
    result = {label: row[field] for field, label in fields if field in row}
    return {**result, "Tolerance requested context": context,
            "Tolerance interpretation": "Chemical compatibility with observed training actives. "
            "This score does not establish activity or safety."}


class ToleranceControls(QWidget):
    """Keep the basket unchanged until a verified proposal is explicitly adopted."""

    def __init__(self, workspace, parent=None):
        super().__init__(parent)
        self.workspace = workspace
        self._package = self._profile = self._preview = self._studio = None
        form = QFormLayout(self)
        form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        note = QLabel("Learn chemical compatibility from observed actives. This experimental "
                      "workflow needs broader independent validation. N and scaffold/cluster "
                      "quotas use the existing selection controls.")
        note.setWordWrap(True)
        form.addRow(note)
        self.workflow = QComboBox()
        self.workflow.addItems(["General", "Target-conditioned", "Custom"])
        form.addRow("Profile use", self.workflow)
        self.included_profiles = QComboBox()
        self.included_profiles.addItem("Choose an included experimental profile", None)
        included = Path(str(files("s2s_decision").joinpath("bundled_tolerance_models")))
        for path in sorted(included.glob("*.json")):
            self.included_profiles.addItem(path.stem.replace("_", " "), str(path))
        self.load_included_button = QPushButton("Load included tolerance profile")
        self.load_included_button.setEnabled(False)
        self.included_profiles.currentIndexChanged.connect(lambda: self.load_included_button.setEnabled(
            bool(self.included_profiles.currentData())))
        self.load_included_button.clicked.connect(self.load_included)
        form.addRow("Included profiles (experimental)", self.included_profiles)
        form.addRow(self.load_included_button)
        self.load_button = QPushButton("Load tolerance package...")
        self.load_button.clicked.connect(self.browse_package)
        form.addRow(self.load_button)
        self.package_label = QLabel("No tolerance package loaded")
        self.package_label.setWordWrap(True)
        form.addRow(self.package_label)
        self.target, self.species, self.endpoint = (QLineEdit() for _ in range(3))
        for label, widget in (("Target identifier", self.target), ("Species", self.species),
                              ("Endpoint", self.endpoint)):
            form.addRow(label, widget)
            widget.textChanged.connect(self.invalidate)
        self.workflow.currentTextChanged.connect(self._workflow_changed)
        self.train_button = QPushButton("Validate data / train custom profile...")
        self.train_button.clicked.connect(self.train_custom)
        form.addRow(self.train_button)
        self.criteria = QTableWidget(0, 4)
        self.criteria.setHorizontalHeaderLabels(["Reconsider", "Kind", "Criterion", "Restriction"])
        self.criteria.setMaximumHeight(210)
        self.criteria.setEditTriggers(QTableWidget.NoEditTriggers)
        self.criteria.itemChanged.connect(self.invalidate)
        form.addRow("Explicit criteria for reconsideration", self.criteria)
        self.calibrated_criteria_button = QPushButton("Use the package's calibrated criteria")
        self.calibrated_criteria_button.setEnabled(False)
        self.calibrated_criteria_button.clicked.connect(self.use_calibrated_criteria)
        form.addRow(self.calibrated_criteria_button)
        self.preview_button = QPushButton("Preview chemical tolerance")
        self.adopt_button = QPushButton("Adopt tolerance proposal")
        self.preview_button.setEnabled(False)
        self.adopt_button.setEnabled(False)
        self.preview_button.clicked.connect(self.preview)
        self.adopt_button.clicked.connect(self.adopt)
        form.addRow(self.preview_button)
        form.addRow(self.adopt_button)
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMaximumHeight(170)
        form.addRow(self.report)
        self.review = QTableWidget(0, 3)
        self.review.setHorizontalHeaderLabels(["Record", "Tolerance state", "Evidence"])
        self.review.setMaximumHeight(180)
        self.review.setEditTriggers(QTableWidget.NoEditTriggers)
        self.review.cellClicked.connect(self.inspect_row)
        form.addRow("Molecule evidence (first 200; export contains all)", self.review)
        self._populate_criteria()
        self._workflow_changed()

    def _populate_criteria(self):
        policy = self.workspace.result.config.policy
        entries = []
        for profile in self.workspace.result.profiles:
            hard = policy.roles.get(profile.id) == "exclusion"
            for rule in profile.rules:
                locked = hard or rule.is_substructure
                entries.append(("rule", f"rule:{profile.id}:{rule.id}",
                                f"{profile.name}: {rule.describe()}", profile.id, locked))
        from s2s_decision.tolerance_features import catalog_manifest

        for catalog_id in self.workspace.result.config.alert_catalogs:
            for row in catalog_manifest(catalog_id)["entries"]:
                entries.append(("alert", row["criterion_id"],
                                f"{catalog_id}: {row['alert_name']}", None, False))
        self.criteria.setRowCount(len(entries))
        for position, (kind, identity, label, profile, locked) in enumerate(entries):
            check = QTableWidgetItem()
            check.setData(Qt.UserRole, (identity, profile))
            check.setFlags(Qt.ItemIsUserCheckable | (Qt.ItemIsEnabled if not locked else Qt.NoItemFlags))
            check.setCheckState(Qt.Unchecked)
            check.setToolTip(f"{identity}\n{label}\n" + ("Mandatory restriction remains binding."
                if locked else "Reconsider this criterion only when the profile supplies supported evidence."))
            self.criteria.setItem(position, 0, check)
            for column, value in enumerate((kind, label, "Locked" if locked else "Optional"), 1):
                item = QTableWidgetItem(value)
                item.setToolTip(f"{identity}\n{label}")
                self.criteria.setItem(position, column, item)

    def reconsidered_criteria(self):
        return tuple(sorted(self.criteria.item(row, 0).data(Qt.UserRole)[0]
                            for row in range(self.criteria.rowCount())
                            if self.criteria.item(row, 0).checkState() == Qt.Checked))

    def revisited_profiles(self):
        return tuple(sorted({self.criteria.item(row, 0).data(Qt.UserRole)[1]
                             for row in range(self.criteria.rowCount())
                             if self.criteria.item(row, 0).checkState() == Qt.Checked
                             and self.criteria.item(row, 0).data(Qt.UserRole)[1]}))

    def revisited_alert_entries(self):
        return tuple(value for value in self.reconsidered_criteria() if value.startswith("alert:"))

    def _workflow_changed(self, *_):
        specific = self.workflow.currentText() != "General"
        for widget in (self.target, self.species, self.endpoint):
            widget.setEnabled(specific)
        self.invalidate()

    def context(self):
        if self.workflow.currentText() == "General":
            if self._profile and any(self._profile.get("context", {}).get(key)
                                     for key in ("target_id", "task_id")):
                raise ValueError("This package is task-specific; choose Target-conditioned or Custom")
            return {}
        values = {"target_id": self.target.text().strip(), "species": self.species.text().strip(),
                  "endpoint": self.endpoint.text().strip()}
        if self.workflow.currentText() == "Target-conditioned" and not all(values.values()):
            raise ValueError("Target-conditioned use requires target, species and endpoint")
        return {key: value for key, value in values.items() if value}

    def signature(self):
        return json.dumps({"context": self.context(), "workflow": self.workflow.currentText(),
                           "criteria": self.reconsidered_criteria(), "package": str(self._package)},
                          sort_keys=True)

    def browse_package(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load tolerance package manifest or profile", "",
                                            "Tolerance JSON (*.json)")
        if path:
            try:
                chosen = Path(path)
                self.load_package(chosen.parent if chosen.name == "manifest.json"
                    or chosen.name == "profile.json" and (chosen.parent / "manifest.json").is_file() else chosen)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.report.setPlainText(f"Cannot load tolerance package: {exc}")

    def load_package(self, path):
        from s2s_decision.tolerance_profile import read_tolerance_package

        profile = read_tolerance_package(path)
        self.invalidate()
        self._package, self._profile = Path(path).resolve(), profile
        context = profile.get("context", {})
        for widget, key in ((self.target, "target_id"), (self.species, "species"), (self.endpoint, "endpoint")):
            widget.setText(str(context.get(key) or ""))
        counts, policy = profile["training_counts"], profile.get("policy", {})
        support = profile.get("support_settings", {})
        scope = policy.get("scope", {})
        self.calibrated_criteria_button.setEnabled(bool(scope.get("criteria")))
        if any(context.get(key) for key in ("target_id", "task_id")):
            self.workflow.setCurrentText("Target-conditioned")
        self.package_label.setText(
            f"{self._package.name}: {profile['mode']} / recipe {profile['recipe']}\n"
            f"Training: {counts.get('rows', 0)} rows, {counts.get('identities', 0)} identities, "
            f"{counts.get('scaffolds', 0)} scaffolds, {counts.get('documents', 0)} documents, "
            f"{counts.get('assays', 0)} assays, {counts.get('tasks', 0)} tasks; pActivity >= 6.5.\n"
            f"Support gates: {support.get('min_identities', 'unknown')} identities / "
            f"{support.get('min_scaffolds', 'unknown')} scaffolds / "
            f"{support.get('min_documents', 'unknown')} documents.\n"
            f"Calibration scope: {len(scope.get('criteria', []))} criteria; "
            f"context {scope.get('context_mode', 'not calibrated')}.\n"
            f"Policy: {policy.get('status', 'uncalibrated')}. "
            "Uncalibrated or unsupported cases require review and are not automatically rescued."
        )
        self.preview_button.setEnabled(True)

    def load_included(self):
        if path := self.included_profiles.currentData():
            try:
                self.load_package(path)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.report.setPlainText(f"Cannot load included tolerance profile: {exc}")

    def use_calibrated_criteria(self):
        if self._profile is None:
            self.report.setPlainText("Load a calibrated tolerance package first.")
            return
        required = set(self._profile.get("policy", {}).get("scope", {}).get("criteria", ()))
        available = {self.criteria.item(row, 0).data(Qt.UserRole)[0]: self.criteria.item(row, 0)
                     for row in range(self.criteria.rowCount())}
        unavailable = required - {identity for identity, item in available.items()
                                  if item.flags() & Qt.ItemIsEnabled}
        if unavailable:
            self.report.setPlainText(f"Calibration covers {len(unavailable)} unavailable or locked criteria. "
                "Use a matching screening run or calibrate a package for this run's criteria. "
                "The existing threshold cannot be transferred silently.")
            return
        self.criteria.blockSignals(True)
        for identity, item in available.items():
            item.setCheckState(Qt.Checked if identity in required else Qt.Unchecked)
        self.criteria.blockSignals(False)
        self.invalidate()
        self.report.setPlainText("Loaded the package's calibrated criteria. The requested context must also "
                                 "match calibration; inspect preview evidence for scope mismatches.")

    def train_custom(self):
        from s2s_decision.gui import Studio

        if self._studio is None:
            self._studio = Studio(self)
        self._studio.mode.setCurrentText("Chemical tolerance")
        self._studio.action.setCurrentText("tolerance-train")
        self._studio.fields["target"].setText(self.target.text())
        self._studio.fields["species"].setText(self.species.text() or "unknown")
        self._studio.fields["reconsider-criteria"].setText(" ".join(self.reconsidered_criteria()))
        endpoint = self._studio.fields["endpoint"]
        if endpoint.findText(self.endpoint.text()) >= 0:
            endpoint.setCurrentText(self.endpoint.text())
        self._studio.show()
        self._studio.raise_()

    @property
    def training_busy(self):
        return self._studio is not None and self._studio.process.state() != QProcess.NotRunning

    def invalidate(self, *_):
        previous, self._preview = self._preview, None
        self.adopt_button.setEnabled(False)
        self.review.setRowCount(0)
        if previous is not None:
            previous.temporary.cleanup()
            self.report.setPlainText("Context or criteria changed. Preview again before adopting.")

    def preview(self):
        if self.workspace.is_busy or self._package is None:
            return
        try:
            context, criteria, signature = self.context(), self.reconsidered_criteria(), self.signature()
            editable = criteria_state.capture(self.workspace)
            result = self.workspace.result
            cache_path = getattr(result.config, "cache_path", None)
            candidates = self.workspace.candidates.copy(deep=True)
            basket = SelectionBasket(self.workspace.basket.states())
            constraints = self.workspace.constraints()
            profiles, alerts = self.revisited_profiles(), self.revisited_alert_entries()
            package, files = self._package, package_files(self._package)
        except (ValueError, TypeError, OSError) as exc:
            self.report.setPlainText(f"Cannot preview tolerance: {exc}")
            return
        self.invalidate()
        temporary = tempfile.TemporaryDirectory(prefix="s2s-tolerance-", ignore_cleanup_errors=True)
        cancelled = Event()

        def compute():
            from s2s_decision.tolerance_decision import (
                preview_tolerance_decision,
                tolerance_snapshot_from_workspace,
            )

            try:
                snapshot = tolerance_snapshot_from_workspace(result, candidates, basket, constraints,
                    revisited_profiles=profiles, revisited_alert_entries=alerts,
                    reconsidered_criteria=criteria)
                root = Path(temporary.name) / "preview"
                preview_tolerance_decision(snapshot, root, package, context,
                                          reconsidered_criteria=criteria, cancelled=cancelled.is_set,
                                          cache_path=cache_path)
                summary = verify_preview(root)
                records = read_bundle(root / "proposed").records.set_index("record_id")
                return TolerancePreview(root, snapshot, editable, signature, package, files,
                                        temporary, summary, records)
            except Exception as exc:
                return exc

        def complete(value):
            if cancelled.is_set() or isinstance(value, Exception):
                temporary.cleanup()
                self.report.setPlainText("Tolerance preview cancelled." if cancelled.is_set()
                    or isinstance(value, CancelledError) else f"Tolerance preview failed: {value}")
                return
            try:
                if self.signature() != value.signature or criteria_state.capture(self.workspace) != value.criteria:
                    raise ValueError("Context or workspace criteria changed; preview again")
                self.accept_preview(value)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                temporary.cleanup()
                self.report.setPlainText(f"Tolerance preview failed: {exc}")

        self.workspace._start_job(compute, complete, "Computing chemical tolerance; basket unchanged...",
                                  cancel=cancelled.set)

    def accept_preview(self, preview):
        summary = preview.summary
        if summary.get("ranking_mode") != "chemical_tolerance" or summary["source"].get("session_revision") != preview.snapshot.revision:
            raise ValueError("Tolerance proposal does not match workspace snapshot")
        self.invalidate()
        self._preview = preview
        self.adopt_button.setEnabled(True)
        rows = preview.records
        states = rows.tolerance_state.value_counts().to_dict()
        self.report.setPlainText(
            f"Requested {preview.snapshot.constraints.target_count}; proposed {summary['proposed_count']}; "
            f"eligible {summary['eligible_count']}.\nStates: {json.dumps(states, sort_keys=True)}\n"
            + "\n".join(summary.get("warnings", [])) + "\nBasket unchanged until explicit adoption."
        )
        self.review.setRowCount(min(200, len(rows)))
        for position, row in enumerate(rows.head(200).reset_index().to_dict("records")):
            for column, value in enumerate((row["record_id"], row["tolerance_state"],
                                             row.get("tolerance_evidence", ""))):
                self.review.setItem(position, column, QTableWidgetItem(str(value)))

    def inspect_row(self, row, _column):
        item = self.review.item(row, 0)
        if item is not None and self._preview is not None:
            record_id = int(item.text())
            rows, summary = self._preview.records, self._preview.summary
            self.workspace._show_molecule(record_id)
            self.workspace.inspector.show_molecule(record_id, self.workspace.candidates,
                molecule_evidence(rows.loc[record_id], summary.get("context", {})))

    def adopt(self):
        preview = self._preview
        if preview is None or self.workspace.is_busy:
            return
        try:
            if self.signature() != preview.signature or criteria_state.capture(self.workspace) != preview.criteria:
                raise ValueError("Context or workspace criteria changed; preview again")
            context, criteria = self.context(), self.reconsidered_criteria()
            states = tuple(self.workspace.basket.states())
            candidates = self.workspace.candidates.copy(deep=True)
            result, constraints = self.workspace.result, self.workspace.constraints()
            cache_path = getattr(result.config, "cache_path", None)
            profiles, alerts = self.revisited_profiles(), self.revisited_alert_entries()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.report.setPlainText(f"Cannot adopt tolerance proposal: {exc}")
            return
        cancelled = Event()

        def compute():
            from s2s_decision.tolerance_decision import (
                tolerance_snapshot_from_workspace,
                verify_tolerance_decision,
            )

            try:
                if cancelled.is_set():
                    raise CancelledError("Tolerance adoption cancelled")
                current = tolerance_snapshot_from_workspace(result, candidates, SelectionBasket(states), constraints,
                    revisited_profiles=profiles, revisited_alert_entries=alerts,
                    reconsidered_criteria=criteria, session_id=preview.snapshot.session_id)
                if current.revision != preview.snapshot.revision:
                    raise ValueError("Workspace decisions changed; preview again")
                if package_files(preview.package) != preview.files:
                    raise ValueError("Tolerance package changed; preview again")
                summary, proposed = verify_tolerance_decision(current, preview.root, preview.package,
                    context, reconsidered_criteria=criteria, cache_path=cache_path)
                if cancelled.is_set():
                    raise CancelledError("Tolerance adoption cancelled")
                if package_files(preview.package) != preview.files:
                    raise ValueError("Tolerance package changed during verification; preview again")
                return current, summary, proposed
            except Exception as exc:
                return exc

        def complete(value):
            try:
                if cancelled.is_set() or isinstance(value, CancelledError):
                    self.report.setPlainText("Tolerance adoption cancelled; basket unchanged.")
                    return
                if isinstance(value, Exception):
                    raise value
                if (self.signature() != preview.signature or criteria_state.capture(self.workspace) != preview.criteria
                        or tuple(self.workspace.basket.states()) != states
                        or self.workspace.result is not result or not self.workspace.candidates.equals(candidates)):
                    raise ValueError("Workspace changed during verification; preview again")
                self._apply_verified(preview, *value)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.report.setPlainText(f"Cannot adopt tolerance proposal: {exc}")

        self.workspace._start_job(compute, complete, "Verifying chemical tolerance; basket unchanged...",
                                  cancel=cancelled.set)

    def _apply_verified(self, preview, current, summary, proposed):
        try:
            records = proposed.records.set_index("record_id")
            selected = tuple(map(int, summary["proposed_ids"]))
            if not set(current.pinned_ids) <= set(selected) or set(current.excluded_ids) & set(selected):
                raise ValueError("Tolerance proposal conflicts with manual decisions")
            if QMessageBox.question(self.workspace, "Review chemical tolerance proposal",
                "\n".join(summary.get("warnings", [])) + "\n\nAdopt this experimental selection?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                return
            if package_files(preview.package) != preview.files:
                raise ValueError("Tolerance package changed before adoption; preview again")
            explanations = records.selection_reason.astype(str).to_dict()
            outcome = SelectionOutcome(selected,
                reasons={rid: [explanations[rid]] for rid in selected},
                rejections={rid: [why] for rid, why in explanations.items() if rid not in selected},
                scaffold_usage=proposed.manifest.get("scaffold_counts", {}),
                cluster_usage={int(key): value for key, value in proposed.manifest.get("cluster_counts", {}).items()},
                strategy=Strategy.BALANCED)
            provenance = {**self.workspace._provenance(), "source": "chemical_tolerance",
                "ranking_method": "qed_with_chemical_tolerance", "context": summary.get("context", {}),
                "tolerance_settings": summary.get("tolerance_settings", {}),
                "model_sha256": summary["model"]["package_sha256"], "preview_sha256": file_hash(preview.root / "preview.json"),
                "session_revision": current.revision, "ranking_warnings": tuple(summary.get("warnings", [])),
                "revisited_profiles": list(self.revisited_profiles()),
                "revisited_alert_entries": list(self.revisited_alert_entries()),
                "revisited_criteria": list(self.reconsidered_criteria()),
                "profile_status": self._profile.get("policy", {}).get("status"),
                "training_counts": self._profile.get("training_counts", {}),
                "profile_sha256": self._profile.get("profile_sha256"),
                "tolerance_profile": {key: self._profile[key] for key in (
                    "schema", "mode", "recipe", "context", "support_settings", "policy",
                    "profile_sha256") if key in self._profile}}
            applied = AppliedSelection.capture(constraints=current.constraints, objectives=ObjectiveSet(),
                pareto=None, strategy="chemical_tolerance", provenance=provenance,
                input_hash=provenance["input_hash"])
            evidence = records.drop(columns=[name for name in records if name.startswith("source__")
                or name == "fingerprint_hex"], errors="ignore")
            self.workspace._apply_selection(outcome, current.constraints, applied,
                source="chemical_tolerance", reason="Explicitly adopted chemical tolerance with recorded evidence",
                model_scores=evidence)
            self.report.setPlainText(f"Tolerance proposal adopted: {len(selected)} molecules. Undo restores prior basket.")
            self.adopt_button.setEnabled(False)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.report.setPlainText(f"Cannot adopt tolerance proposal: {exc}")
