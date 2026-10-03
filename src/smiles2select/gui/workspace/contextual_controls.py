"""Contextual policy controls inside the existing Chemical Space Hub."""

import hashlib
import json
import tempfile
from concurrent.futures import CancelledError
from dataclasses import asdict, dataclass, replace
from importlib.resources import files
from pathlib import Path
from threading import Event

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from s2s_decision.artifacts import file_hash, read_bundle
from s2s_decision.contextual_decision import (
    load_policy_package,
    preview_contextual_decision,
    verify_contextual_decision,
)
from s2s_decision.contextual_policy import FeatureAction, ProfileAction
from s2s_decision.decision import verify_preview
from s2s_decision.risk_model import load_risk_bundle
from s2s_decision.session_adapter import snapshot_from_workspace
from smiles2select.alerts.rdkit_catalogs import get_catalog
from smiles2select.gui.workspace import criteria_state
from smiles2select.gui.workspace.selection_provenance import AppliedSelection
from smiles2select.profiles.loader import builtin_registry
from smiles2select.selection_intelligence.basket import SelectionBasket
from smiles2select.selection_intelligence.constrained_selection import SelectionOutcome, Strategy
from smiles2select.selection_intelligence.objectives import ObjectiveSet


@dataclass(frozen=True)
class ContextualPreview:
    root: Path
    snapshot: object
    criteria: object
    signature: str
    package: Path
    package_hash: str
    temporary: object


def molecule_evidence(row, context, risk_evidence=None):
    """Same readable evidence before adoption and in the restored workspace."""
    fields = (
        ("activity_score", "Activity score (package endpoint)"),
        ("activity_prediction_set", "Activity prediction set"),
        ("risk_prediction_status", "Risk evidence"),
        ("risk_score", "Measured-risk model score"),
        ("risk_prediction_set", "Risk prediction set"),
        ("risk_in_domain", "Within risk model applicability domain"),
        ("in_domain", "Within model applicability domain"),
        ("max_training_similarity", "Maximum training similarity"),
        ("policy_explanation", "Policy explanation"),
        ("policy_warnings", "Review warnings"),
        ("rule_evidence", "Rules and magnitudes"),
        ("alert_details", "Matched structural patterns"),
        ("alert_evidence", "Alert support"),
        ("activity_contributions", "Learned score contributions"),
    )
    result = {label: row[field] for field, label in fields if field in row}
    for label, value in tuple(result.items()):
        if value is None or isinstance(value, float) and value != value:
            result[label] = "Unknown"
        elif isinstance(value, float):
            result[label] = f"{value:.4f}"
    for field, label, identifier in (("rule_evidence", "Rules and magnitudes", "rule_id"),
                                     ("alert_evidence", "Alert support", "alert_id")):
        if field in row:
            result[label] = "\n" + "\n".join(_feature_evidence(item, identifier) for item in row[field])
    if isinstance(row.get("alert_details"), (list, tuple)):
        result["Matched structural patterns"] = "; ".join(
            f"{item['catalog_id']}: {item['alert_name']}" for item in row.alert_details)
    return {**result, "Decision context": context,
            **({"Risk assay context": risk_evidence.get("context"),
                "Risk evidence source": risk_evidence.get("source")} if risk_evidence else {})}


def _feature_evidence(item, identifier):
    parts = [str(item.get(identifier, item.get("feature_id", "pattern")))]
    if "observed" in item:
        parts.append(f"value {item['observed']}; limits {item.get('lower_limit')} .. {item.get('upper_limit')}")
        excess = item.get("normalized_excess")
        parts.append("excess unknown" if excess is None else f"normalized excess {excess:.3f}")
    parts.append(f"action {item.get('effective_action', item.get('action', 'inform'))}")
    evidence = item.get("evidence")
    if evidence:
        parts.append(f"{evidence['origin']}; support {evidence['support_n']} molecules "
            f"({evidence['positive_n']} positive, {evidence['negative_n']} negative), "
            f"{evidence['scaffolds']} scaffolds, {evidence['documents']} documents")
    else:
        parts.append("no learned per-pattern support")
    return " | ".join(parts)


class ContextualControls(QWidget):
    """One opt-in proposal; native basket remains authoritative and reversible."""

    def __init__(self, workspace, parent=None):
        super().__init__(parent)
        self.workspace, self._package, self._context, self._preview = workspace, None, None, None
        self._risk_package = None
        form = QFormLayout(self)
        form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        self.included_models = QComboBox()
        self.included_models.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.included_models.setMinimumContentsLength(18)
        self.included_models.currentTextChanged.connect(self.included_models.setToolTip)
        self.included_models.addItem("Choose an included contextual model", None)
        included = Path(str(files("s2s_decision").joinpath("bundled_contextual_models")))
        names = {"P00918": "Carbonic anhydrase II", "P22303": "Acetylcholinesterase", "P56817": "BACE1"}
        for path in sorted(included.glob("*-seed*.json")):
            task, seed = path.stem.rsplit("-seed", 1)
            target, _, endpoint = task.partition("_HUMAN_")
            self.included_models.addItem(
                f"{names.get(target, target)} ({target}) / {endpoint} / calibration seed {seed}", str(path))
        self.load_included_button = QPushButton("Load included contextual model")
        self.load_included_button.setEnabled(False)
        self.included_models.currentIndexChanged.connect(
            lambda: self.load_included_button.setEnabled(bool(self.included_models.currentData())))
        self.load_included_button.clicked.connect(self.load_included)
        form.addRow("Included contextual model (experimental)", self.included_models)
        form.addRow(self.load_included_button)
        note = QLabel("Three target/endpoint tasks, one logistic architecture. Seeds 42, 43 and 44 vary calibration partitioning; they are not nine different architectures.")
        note.setWordWrap(True)
        form.addRow(note)
        self.load_button = QPushButton("Load contextual model JSON...")
        self.load_button.clicked.connect(self.browse_package)
        form.addRow(self.load_button)
        self.package_label = QLabel("No contextual package loaded")
        self.package_label.setWordWrap(True)
        form.addRow(self.package_label)
        self.target, self.species, self.endpoint, self.assay = (QLineEdit() for _ in range(4))
        for widget in (self.target, self.species, self.endpoint):
            widget.setReadOnly(True)
        self.stage = QComboBox()
        self.stage.addItems(["hit_finding", "lead", "fragment"])
        for label, widget in (("Target", self.target), ("Species", self.species),
                              ("Endpoint", self.endpoint), ("Stage", self.stage),
                              ("Assay context", self.assay)):
            form.addRow(label, widget)
        self.profiles = QLineEdit()
        self.profiles.setPlaceholderText("Comma-separated existing profile IDs")
        self.profiles.setText(", ".join(profile.id for profile in workspace.result.profiles
                                       if profile.id in {"lipinski", "veber"}))
        self.profiles.setToolTip(
            "Only listed profile filters are revisited. Other required profiles, explicit "
            "exclusions, custom expressions and reference restrictions remain binding."
        )
        form.addRow("Revisit these profile filters", self.profiles)
        self.alert_action = QComboBox()
        self.alert_action.addItems(["warn", "inform", "penalize", "exclude"])
        form.addRow("Unspecified alert action", self.alert_action)
        self.risk_button = QPushButton("Load measured-risk model JSON (optional)...")
        self.risk_button.clicked.connect(self.browse_risk)
        self._included_risk = included / "risk" / "shsy5y_atp_viability_48h.json"
        self.included_risk_button = QPushButton("Load included SH-SY5Y viability\nrisk model (optional)")
        self.included_risk_button.setEnabled(self._included_risk.is_file())
        self.included_risk_button.clicked.connect(self.load_included_risk)
        self.risk_clear = QPushButton("Clear risk model")
        self.risk_clear.clicked.connect(self.clear_risk)
        self.risk_label = QLabel("No measured-risk model; risk remains unknown")
        self.risk_label.setWordWrap(True)
        for widget in (self.included_risk_button, self.risk_button, self.risk_clear, self.risk_label):
            form.addRow(widget)
        self.risk_weight, self.risk_cutoff = QDoubleSpinBox(), QDoubleSpinBox()
        for widget in (self.risk_weight, self.risk_cutoff):
            widget.setRange(0, 1)
            widget.setSingleStep(.05)
        self.risk_cutoff.setValue(.5)
        self.risk_cutoff.setEnabled(False)
        self.risk_exclude = QCheckBox("Exclude at chosen measured-risk score (explicit constraint)")
        self.risk_exclude.toggled.connect(self.risk_cutoff.setEnabled)
        self.risk_exclude.toggled.connect(self.invalidate)
        form.addRow("Risk ranking penalty weight", self.risk_weight)
        form.addRow(self.risk_exclude)
        form.addRow("Risk exclusion cutoff", self.risk_cutoff)
        self.actions = QTableWidget(0, 5)
        self.actions.setHorizontalHeaderLabels(["Kind", "Feature / pattern", "User action", "Penalty", "Allowed violations"])
        self.actions.setMaximumHeight(160)
        self.action_add, self.action_remove = QPushButton("Add explicit action"), QPushButton("Remove selected action")
        self.action_add.clicked.connect(lambda: self.add_action())
        self.action_remove.clicked.connect(self.remove_action)
        form.addRow("Advanced: explicit actions override learned advice", self.actions)
        form.addRow(self.action_add)
        form.addRow(self.action_remove)
        self.required_smarts, self.excluded_smarts, self.preferred_smarts = (
            QPlainTextEdit() for _ in range(3)
        )
        for label, widget in (("Required SMARTS (optional)", self.required_smarts),
                              ("Excluded SMARTS (optional)", self.excluded_smarts),
                              ("Preferred SMARTS (optional)", self.preferred_smarts)):
            widget.setMaximumHeight(55)
            widget.setPlaceholderText("One validated SMARTS per line; leave blank to disable")
            form.addRow(label, widget)
        self.notice = QLabel(
            "Experimental policy. Activity evidence does not establish absence of interference "
            "or toxicity. Changing stage or assay may require review. Quantity and diversity "
            "use the workspace controls above."
        )
        self.notice.setWordWrap(True)
        form.addRow(self.notice)
        self.review_count = QSpinBox()
        self.review_count.setRange(0, 1_000_000)
        self.review_count.setSpecialValueText("No separate review subset")
        self.review_count.valueChanged.connect(self.invalidate)
        form.addRow("Additional review budget (informative candidates)", self.review_count)
        self.preview_button, self.adopt_button = QPushButton("Preview contextual policy"), QPushButton("Adopt contextual proposal")
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
        self.review.setHorizontalHeaderLabels(["Record", "Action", "Evidence / review"])
        self.review.setMaximumHeight(180)
        self.review.setEditTriggers(QTableWidget.NoEditTriggers)
        self.review.cellClicked.connect(self.inspect_row)
        form.addRow("Review queue (first 200; export contains all)", self.review)
        for widget in (self.stage, self.alert_action):
            widget.currentIndexChanged.connect(self.invalidate)
        for widget in (self.assay, self.profiles):
            widget.textChanged.connect(self.invalidate)
        for widget in (self.required_smarts, self.excluded_smarts, self.preferred_smarts):
            widget.textChanged.connect(self.invalidate)
        for widget in (self.risk_weight, self.risk_cutoff):
            widget.valueChanged.connect(self.invalidate)

    def add_action(self, *, kind="profile", identifier=None, action="warn", penalty=.1, max_violations=None):
        self.invalidate()
        row = self.actions.rowCount()
        self.actions.insertRow(row)
        kinds, features, actions, weight = QComboBox(), QComboBox(), QComboBox(), QDoubleSpinBox()
        kinds.addItems(["profile", "rule", "alert"])
        actions.addItems(["inform", "warn", "penalize", "exclude"])
        weight.setRange(0, 1)
        weight.setSingleStep(.05)
        weight.setValue(penalty)
        actions.setCurrentText(action)
        tolerance = QSpinBox()
        tolerance.setRange(-1, 99)
        tolerance.setSpecialValueText("Native tolerance")
        tolerance.setValue(-1 if max_violations is None else max_violations)
        for column, widget in enumerate((kinds, features, actions, weight, tolerance)):
            self.actions.setCellWidget(row, column, widget)

        def choices():
            features.blockSignals(True)
            features.clear()
            tolerance.setEnabled(kinds.currentText() == "profile")
            registry = builtin_registry()
            if kinds.currentText() == "profile":
                for profile in registry.all():
                    features.addItem(profile.name, profile.id)
            elif kinds.currentText() == "rule":
                seen = set()
                for profile in registry.all():
                    for rule in profile.rules:
                        if rule.id not in seen:
                            features.addItem(f"{profile.name}: {rule.id} ({rule.descriptor})", rule.id)
                            seen.add(rule.id)
            else:
                for catalog_id in ("pains", "brenk"):
                    catalog = get_catalog(catalog_id)
                    for position in range(catalog.GetNumEntries()):
                        name = catalog.GetEntryWithIdx(position).GetDescription()
                        identity = f"{catalog_id}:" + hashlib.sha256(name.encode()).hexdigest()[:20]
                        features.addItem(f"{catalog_id}: {name}", identity)
            features.blockSignals(False)
            self.invalidate()

        kinds.currentIndexChanged.connect(choices)
        kinds.setCurrentText(kind)
        choices()
        if identifier is not None:
            index = features.findData(identifier)
            if index < 0:
                self.actions.removeRow(row)
                raise ValueError(f"Unknown {kind} feature: {identifier}")
            features.setCurrentIndex(index)
        for widget in (features, actions):
            widget.currentIndexChanged.connect(self.invalidate)
        weight.valueChanged.connect(self.invalidate)
        tolerance.valueChanged.connect(self.invalidate)

    def remove_action(self):
        row = self.actions.currentRow()
        if row >= 0:
            self.actions.removeRow(row)
            self.invalidate()

    def action_overrides(self):
        result = {"profiles": [], "rule_actions": [], "alert_actions": []}
        seen = set()
        for row in range(self.actions.rowCount()):
            kind = self.actions.cellWidget(row, 0).currentText()
            identity = self.actions.cellWidget(row, 1).currentData()
            if (kind, identity) in seen:
                raise ValueError(f"Duplicate {kind} action: {identity}")
            seen.add((kind, identity))
            action = self.actions.cellWidget(row, 2).currentText()
            penalty = self.actions.cellWidget(row, 3).value()
            if kind == "profile":
                tolerance = self.actions.cellWidget(row, 4).value()
                result["profiles"].append(ProfileAction(identity, action=action, penalty=penalty,
                    max_violations=tolerance if tolerance >= 0 else None))
            else:
                result[f"{kind}_actions"].append(FeatureAction(identity, action=action,
                                                              penalty=penalty, origin="user"))
        return {key: tuple(value) for key, value in result.items()}

    def browse_package(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load contextual model", "", "Contextual model (*.json)")
        if path:
            try:
                self.load_package(path)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                self.report.setPlainText(f"Cannot load contextual model: {exc}")

    def load_included(self):
        path = self.included_models.currentData()
        if path:
            try:
                self.load_package(path)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                self.report.setPlainText(f"Cannot load included contextual model: {exc}")

    def load_included_risk(self):
        try:
            self.load_risk(self._included_risk)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.report.setPlainText(f"Cannot load included risk model: {exc}")

    def load_package(self, path):
        bundle, context = load_policy_package(path)
        self.invalidate()
        self._package, self._context = Path(path).resolve(), context
        for widget, value in ((self.target, context.target), (self.species, context.species),
                              (self.endpoint, context.endpoint), (self.assay, context.assay_context)):
            widget.setText(value)
        self.stage.setCurrentText(context.stage)
        self.package_label.setText(
            f"{self._package.name}\nModel {context.model_version}; data {context.source_version}; "
            f"chemistry {context.chemistry_version}\n"
            f"{bundle.get('family', 'undeclared estimator')}; calibration seed {bundle.get('calibration_seed', 'not declared')}. "
            f"Source: {bundle.get('source', 'not declared')}.\n"
            f"Activity label: {context.endpoint} {context.activity_relation} {context.activity_threshold} {context.activity_unit}. "
            "Calibration on the source study does not establish calibrated probabilities in a new library."
        )
        self.preview_button.setEnabled(True)

    def context(self):
        if self._context is None:
            raise ValueError("Load a contextual model first")
        return replace(self._context, stage=self.stage.currentText(), assay_context=self.assay.text().strip())

    def browse_risk(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load measured-risk model", "", "Risk model (*.json)")
        if path:
            try:
                self.load_risk(path)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                self.report.setPlainText(f"Cannot load risk model: {exc}")

    def load_risk(self, path):
        bundle = load_risk_bundle(path)
        self.invalidate()
        self._risk_package = Path(path).resolve()
        self.risk_label.setText(f"Endpoint: {bundle['risk_scope']}. Applicability unknown; not general safety.")

    def clear_risk(self):
        self.invalidate()
        self._risk_package = None
        self.risk_weight.setValue(0)
        self.risk_exclude.setChecked(False)
        self.risk_label.setText("No measured-risk model; risk remains unknown")

    def overrides(self):
        return {
            **self.action_overrides(),
            "default_alert_action": self.alert_action.currentText(),
            "risk_weight": self.risk_weight.value(),
            "risk_exclude_at": self.risk_cutoff.value() if self.risk_exclude.isChecked() else None,
            **{name: tuple(value.strip() for value in widget.toPlainText().splitlines() if value.strip())
               for name, widget in (("required_smarts", self.required_smarts),
                                    ("excluded_smarts", self.excluded_smarts),
                                    ("preferred_smarts", self.preferred_smarts))},
        }

    def revisited_profiles(self):
        return tuple(sorted({value.strip() for value in self.profiles.text().split(",") if value.strip()}))

    def signature(self):
        return json.dumps({"context": asdict(self.context()), "overrides": self.overrides(),
            "profiles": self.revisited_profiles(), "package": str(self._package),
            "risk_package": str(self._risk_package), "review_count": self.review_count.value()},
            sort_keys=True, default=asdict)

    def invalidate(self, *_):
        previous, self._preview = self._preview, None
        self.adopt_button.setEnabled(False)
        self.review.setRowCount(0)
        if previous is not None:
            previous.temporary.cleanup()
            self.report.setPlainText("Context or policy changed. Preview again before adopting.")

    def preview(self):
        if self.workspace.is_busy or self._package is None:
            return
        try:
            context, overrides, signature = self.context(), self.overrides(), self.signature()
            result, candidates = self.workspace.result, self.workspace.candidates.copy(deep=True)
            basket = SelectionBasket(self.workspace.basket.states())
            constraints, criteria = self.workspace.constraints(), criteria_state.capture(self.workspace)
            revisited, package = self.revisited_profiles(), self._package
            risk, review_count = self._risk_package, self.review_count.value()
        except (ValueError, TypeError) as exc:
            self.report.setPlainText(f"Cannot preview contextual policy: {exc}")
            return
        self.invalidate()
        temporary = tempfile.TemporaryDirectory(prefix="s2s-context-", ignore_cleanup_errors=True)
        cancel = Event()

        def compute():
            try:
                snapshot = snapshot_from_workspace(result, candidates, basket, constraints,
                    candidate_scope="all_valid", revisited_profiles=revisited)
                preview = ContextualPreview(Path(temporary.name) / "preview", snapshot, criteria,
                    signature, package, file_hash(package), temporary)
                preview_contextual_decision(snapshot, preview.root, package, context,
                    overrides=overrides, cancelled=cancel.is_set, risk_path=risk, review_count=review_count)
                return preview
            except Exception as exc:
                return exc

        def complete(value):
            if isinstance(value, Exception) or cancel.is_set():
                temporary.cleanup()
                self.report.setPlainText("Contextual preview cancelled." if cancel.is_set()
                    or isinstance(value, CancelledError) else f"Contextual preview failed: {value}")
                return
            if self.signature() != value.signature:
                temporary.cleanup()
                self.report.setPlainText("Context changed during computation. Preview again.")
                return
            try:
                self.accept_preview(value)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                temporary.cleanup()
                self.report.setPlainText(f"Contextual preview failed: {exc}")

        self.workspace._start_job(compute, complete, "Computing contextual policy; basket unchanged...", cancel=cancel.set)

    def accept_preview(self, preview):
        summary = verify_preview(preview.root)
        if summary.get("ranking_mode") != "contextual_policy" or summary["source"].get("session_revision") != preview.snapshot.revision:
            raise ValueError("Contextual proposal does not match workspace snapshot")
        self.invalidate()
        self._preview = preview
        self.adopt_button.setEnabled(True)
        self.report.setPlainText(
            f"Requested {preview.snapshot.constraints.target_count}; proposed {summary['proposed_count']}; "
            f"eligible {summary['eligible_count']}. Retained {len(summary['retained_ids'])}, "
            f"added {len(summary['added_ids'])}, removed {len(summary['removed_ids'])}.\n"
            + "\n".join(summary.get("warnings", [])) + "\nBasket unchanged until explicit adoption."
        )
        rows = read_bundle(preview.root / "proposed").records.sort_values("information_rank")
        rows = rows.loc[rows.in_information_queue if "in_information_queue" in rows else rows.review_required]
        self.review.setRowCount(min(200, len(rows)))
        for position, row in enumerate(rows.head(200).to_dict("records")):
            for column, value in enumerate((row["record_id"], "Review" if row.get("review_required") else "Rank",
                                             row.get("policy_explanation", row.get("selection_reason", "")))):
                self.review.setItem(position, column, QTableWidgetItem(str(value)))

    def inspect_row(self, row, _column):
        item = self.review.item(row, 0)
        if item is not None:
            record_id = int(item.text())
            self.workspace._show_molecule(record_id)
            if self._preview is not None:
                records = read_bundle(self._preview.root / "proposed").records.set_index("record_id")
                summary = verify_preview(self._preview.root)
                self.workspace.inspector.show_molecule(record_id, self.workspace.candidates,
                    molecule_evidence(records.loc[record_id], summary["context"],
                                      summary["policy_settings"].get("risk_evidence")))

    def adopt(self):
        preview = self._preview
        if preview is None or self.workspace.is_busy:
            return
        try:
            if self.signature() != preview.signature or criteria_state.capture(self.workspace) != preview.criteria:
                raise ValueError("Context or workspace criteria changed; preview again")
            current = snapshot_from_workspace(self.workspace.result, self.workspace.candidates,
                self.workspace.basket, self.workspace.constraints(), session_id=preview.snapshot.session_id,
                candidate_scope="all_valid", revisited_profiles=self.revisited_profiles())
            if current.revision != preview.snapshot.revision:
                raise ValueError("Workspace decisions changed; preview again")
            if file_hash(preview.package) != preview.package_hash:
                raise ValueError("Contextual model changed; preview again")
            summary, proposed = verify_contextual_decision(current, preview.root, preview.package,
                self.context(), overrides=self.overrides(), risk_path=self._risk_package,
                review_count=self.review_count.value())
            records = proposed.records.set_index("record_id")
            selected = tuple(map(int, summary["proposed_ids"]))
            if not set(current.pinned_ids) <= set(selected) or set(current.excluded_ids) & set(selected):
                raise ValueError("Contextual proposal conflicts with manual decisions")
            if QMessageBox.question(self, "Review contextual proposal",
                "\n".join(summary["warnings"]) + "\n\nAdopt this experimental selection?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                return
            explanations = records.selection_reason.astype(str).to_dict()
            outcome = SelectionOutcome(selected,
                reasons={rid: [explanations[rid]] for rid in selected},
                rejections={rid: [why] for rid, why in explanations.items() if rid not in selected},
                scaffold_usage=proposed.manifest.get("scaffold_counts", {}),
                cluster_usage={int(key): value for key, value in proposed.manifest.get("cluster_counts", {}).items()},
                strategy=Strategy.BALANCED)
            provenance = {**self.workspace._provenance(), "source": "contextual_policy",
                "ranking_method": "contextual_logistic_policy", "context": summary["context"],
                "evidence_context": summary["evidence_context"], "policy_settings": summary["policy_settings"],
                "risk_model": summary["risk_model"], "information_queue": summary["information_queue"],
                "model_sha256": preview.package_hash, "preview_sha256": file_hash(preview.root / "preview.json"),
                "session_revision": preview.snapshot.revision, "ranking_warnings": tuple(summary["warnings"]),
                "candidate_scope": "all_valid", "revisited_profiles": list(self.revisited_profiles())}
            applied = AppliedSelection.capture(constraints=current.constraints, objectives=ObjectiveSet(),
                pareto=None, strategy="contextual_policy", provenance=provenance,
                input_hash=provenance["input_hash"])
            evidence = records.drop(columns=[name for name in records if name.startswith("source__")
                or name == "fingerprint_hex"], errors="ignore")
            self.workspace._apply_selection(outcome, current.constraints, applied,
                source="contextual_policy", reason="Explicitly adopted contextual policy with recorded review",
                model_scores=evidence)
            self.report.setPlainText(f"Contextual proposal adopted: {len(selected)} molecules. Undo restores prior basket.")
            self.adopt_button.setEnabled(False)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.report.setPlainText(f"Cannot adopt contextual proposal: {exc}")
