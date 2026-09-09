"""Independent scenario previews, explicit adoption, and reproducible study files."""

from dataclasses import replace
from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
)

from smiles2select.gui.workspace.guided_controls import StrategyCombo
from smiles2select.gui.workspace.jobs import ComputationJob
from smiles2select.selection_intelligence.constrained_selection import Strategy
from smiles2select.selection_intelligence.scenario_io import load_study, save_study
from smiles2select.selection_intelligence.scenarios import (
    ScenarioSpec,
    compare_scenarios,
    evaluate_scenario,
    stability_table,
)
from smiles2select.selection_intelligence.states import SelectionOrigin


class ScenarioDialog(QDialog):
    """Small drafts and snapshots; all heavy evaluation happens outside the GUI thread."""

    def __init__(self, workspace):
        super().__init__(workspace)
        self.workspace = workspace
        self.snapshots = {}
        self._draft_thresholds = {}
        self._job = None
        self._stability = None
        self.setWindowTitle("3. Compare alternatives before adopting")
        self.resize(760, 720)
        self.setModal(True)
        layout = QVBoxLayout(self)
        help_text = QLabel(
            "1. Preview A. 2. Duplicate A as B and change quantity, strategy or a threshold. "
            "3. Compare and adopt explicitly. Previews never change the current selection. "
            "Stability measures agreement between scenarios, not probability of biological activity."
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        form = QFormLayout()
        self.slot = QComboBox()
        self.slot.addItems(["A", "B"])
        self.count = QSpinBox()
        self.count.setRange(1, 10_000_000)
        self.count.setValue(workspace.target_count.value())
        self.strategy = StrategyCombo()
        self.strategy.setCurrentText(workspace.strategy.currentText())
        form.addRow("Scenario", self.slot)
        form.addRow("Number of molecules", self.count)
        form.addRow("Strategy", self.strategy)
        self.rule = QComboBox()
        for profile in workspace.result.profiles:
            for rule in profile.rules:
                if not rule.is_substructure and isinstance(rule.threshold, (int, float, tuple, list)):
                    self.rule.addItem(f"{profile.label}: {rule.descriptor} {rule.operator}", rule)
        self.low, self.high = QDoubleSpinBox(), QDoubleSpinBox()
        for widget in (self.low, self.high):
            widget.setRange(-1_000_000, 1_000_000)
            widget.setDecimals(3)
        self.high.setPrefix("Upper: ")
        form.addRow("Optional threshold to change", self.rule)
        threshold_row = QHBoxLayout()
        threshold_row.addWidget(self.low)
        threshold_row.addWidget(self.high)
        self.apply_threshold = QPushButton("Set in draft")
        self.apply_threshold.clicked.connect(self._set_threshold)
        threshold_row.addWidget(self.apply_threshold)
        form.addRow(threshold_row)
        self.draft_label = QLabel("Thresholds: original screening policy")
        self.draft_label.setWordWrap(True)
        form.addRow(self.draft_label)
        layout.addLayout(form)
        self.rule.currentIndexChanged.connect(self._rule_changed)
        self.slot.currentTextChanged.connect(self._load_slot)
        self._rule_changed()
        buttons = QHBoxLayout()
        self.preview_button = QPushButton("Preview scenario")
        self.duplicate_button = QPushButton("Duplicate A as B")
        self.compare_button = QPushButton("Compare + stability")
        self.preview_button.clicked.connect(self.preview)
        self.duplicate_button.clicked.connect(self.duplicate_a)
        self.compare_button.clicked.connect(self.compare)
        for button in (self.preview_button, self.duplicate_button, self.compare_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        layout.addWidget(self.progress)
        self.report = QTextEdit()
        self.report.setReadOnly(True)
        layout.addWidget(self.report, 1)
        self.justification = QLineEdit()
        self.justification.setPlaceholderText(
            "Written justification required if adopting molecules rejected by original screening"
        )
        layout.addWidget(self.justification)
        bottom = QHBoxLayout()
        self.adopt_button = QPushButton("Adopt previewed scenario")
        self.save_button = QPushButton("Save study...")
        self.load_button = QPushButton("Open study...")
        self.export_button = QPushButton("Export stability CSV...")
        self.adopt_button.clicked.connect(self.adopt)
        self.save_button.clicked.connect(self.save)
        self.load_button.clicked.connect(self.load)
        self.export_button.clicked.connect(self.export_stability)
        for button in (self.adopt_button, self.save_button, self.load_button, self.export_button):
            bottom.addWidget(button)
        layout.addLayout(bottom)
        self._actions = (self.preview_button, self.duplicate_button, self.compare_button,
                         self.adopt_button, self.save_button, self.load_button, self.export_button)

    @property
    def is_busy(self):
        return self._job is not None

    def _rule_changed(self):
        rule = self.rule.currentData()
        if rule is None:
            return
        value = self._draft_thresholds.get(rule.id, rule.threshold)
        interval = isinstance(value, (tuple, list))
        self.low.setValue(value[0] if interval else value)
        self.high.setVisible(interval)
        if interval:
            self.high.setValue(value[1])

    def _set_threshold(self):
        rule = self.rule.currentData()
        if rule is None:
            return
        value = ((self.low.value(), self.high.value())
                 if isinstance(rule.threshold, (tuple, list)) else self.low.value())
        self._draft_thresholds = {**self._draft_thresholds, rule.id: value}
        self.draft_label.setText(f"Draft overrides: {self._draft_thresholds}. Preview to evaluate.")

    def _load_slot(self):
        snapshot = self.snapshots.get(self.slot.currentText())
        self._draft_thresholds = dict(snapshot.spec.thresholds) if snapshot else {}
        if snapshot:
            self.count.setValue(snapshot.spec.constraints.target_count or 50)
            self.strategy.setCurrentText(snapshot.spec.strategy.value)
        self.draft_label.setText(f"Draft overrides: {self._draft_thresholds or 'none'}")
        self._rule_changed()

    def duplicate_a(self):
        if "A" not in self.snapshots:
            self.report.setPlainText("Preview A first.")
            return
        self._stability = None
        source = self.snapshots["A"]
        self.snapshots = {**self.snapshots, "B": replace(source, spec=replace(source.spec, name="B"))}
        self.slot.setCurrentText("B")
        self._load_slot()
        self.report.setPlainText("B copied from A. Change the draft, then preview B.")

    def capture_spec(self):
        workspace = self.workspace
        previous = self.snapshots.get(self.slot.currentText())
        return ScenarioSpec(
            self.slot.currentText(), dict(self._draft_thresholds),
            previous.spec.objectives if previous else tuple(workspace.objectives()),
            replace(previous.spec.constraints if previous else workspace.constraints(),
                    target_count=self.count.value()),
            Strategy(self.strategy.currentData()),
        )

    def preview(self):
        try:
            spec = self.capture_spec()
        except Exception as exc:
            self.report.setPlainText(str(exc))
            return
        workspace = self.workspace
        # Captured IDs are immutable; background work does not inspect live widgets/basket.
        flags = self._manual_flags()
        self._start(lambda: evaluate_scenario(workspace.result, workspace.candidates, spec, **flags),
                    self.accept_snapshot)

    def _manual_flags(self):
        workspace = self.workspace
        universe = workspace.candidates.index
        return {
            "pinned_ids": tuple(s.record_id for s in workspace.basket.states()
                                if s.pinned and s.record_id in universe
                                and (s.chemical_status.passed or s.is_selected)),
            "excluded_ids": tuple(rid for rid in workspace.basket.excluded_ids() if rid in universe),
            "rescued_ids": tuple(s.record_id for s in workspace.basket.overrides()
                                 if s.is_selected and s.record_id in universe),
        }

    def accept_snapshot(self, snapshot):
        self.snapshots = {**self.snapshots, snapshot.spec.name: snapshot}
        self._stability = None
        self.report.setPlainText(
            f"Scenario {snapshot.spec.name}: {len(snapshot.eligible_ids):,} chemically eligible; "
            f"{len(snapshot.outcome.selected_ids):,} selected.\n"
            f"Ranking method: {snapshot.provenance.get('ranking_method', 'unavailable')}\n"
            f"Threshold overrides: {dict(snapshot.spec.thresholds) or 'none'}\n"
            + "\n".join(snapshot.warnings)
            + "\nCurrent selection unchanged. Adopt explicitly to use this result."
        )

    def compare(self):
        if "A" not in self.snapshots or "B" not in self.snapshots:
            self.report.setPlainText("Preview both A and B first.")
            return
        snapshots = (self.snapshots["A"], self.snapshots["B"])
        self._start(lambda: (compare_scenarios(*snapshots), stability_table(snapshots)),
                    self._show_comparison)

    def _show_comparison(self, value):
        comparison, self._stability = value
        stable = int((self._stability["selection_frequency"] == 1).sum())
        self.report.setPlainText(
            f"A to B: {len(comparison.common_ids):,} retained; "
            f"{len(comparison.entered_ids):,} entered; {len(comparison.left_ids):,} left.\n"
            f"Overlap (Jaccard): {comparison.jaccard:.1%}\n"
            f"Molecular cores gained: {len(comparison.gained_scaffolds):,}; "
            f"lost: {len(comparison.lost_scaffolds):,}.\n"
            f"Stable core: {stable:,} selected in every evaluated scenario.\n"
            "Frequency measures sensitivity to these choices, not biological activity. "
            "Pinned/rescued/excluded flags are identified separately in CSV.\n\n"
            + self._stability.head(30).to_string()
            + "\nDisplay: first 30 records. CSV contains all records."
        )

    def adopt(self):
        snapshot = self.snapshots.get(self.slot.currentText())
        if snapshot is None:
            self.report.setPlainText("Preview this scenario first.")
            return
        workspace = self.workspace
        selected = snapshot.outcome.selected_ids
        flags_changed = any(set(ids) != set(getattr(snapshot, name))
                            for name, ids in self._manual_flags().items())
        retained_pins = {rid for rid in workspace.basket.pinned_ids()
                         if workspace.basket.state(rid).is_selected}
        if flags_changed or not retained_pins.issubset(selected):
            self.report.setPlainText(
                "Manual decisions changed since this preview. Preview again before adopting; "
                "currently pinned choices must be represented explicitly."
            )
            return
        if self.capture_spec() != snapshot.spec:
            self.report.setPlainText("Draft criteria changed. Preview again before adopting.")
            return
        rejected = [rid for rid in selected if not workspace.basket.state(rid).chemical_status.passed]
        reason = self.justification.text().strip()
        if rejected and not reason:
            self.report.setPlainText(
                f"{len(rejected)} selected molecules failed the original screening. "
                "Write the scientific justification below before adopting the changed policy."
            )
            return
        try:
            workspace.basket.replace_final(
                selected, origin=SelectionOrigin.AUTOMATIC, source=f"scenario:{snapshot.spec.name}",
                reason=f"Adopted scenario {snapshot.spec.name}. {reason}",
            )
        except Exception as exc:
            self.report.setPlainText(str(exc))
            return
        index = len(workspace.basket.log.applied) - 1
        workspace._selection_outcomes[index] = snapshot.outcome
        workspace._scenario_snapshots[index] = snapshot
        workspace.target_count.setValue(snapshot.spec.constraints.target_count or len(selected))
        workspace.strategy.setCurrentText(snapshot.spec.strategy.value)
        workspace.refresh()
        self.report.setPlainText(f"Scenario {snapshot.spec.name} adopted. Undo restores the prior selection.")

    def save(self, path=None):
        if not self.snapshots:
            self.report.setPlainText("Preview a scenario before saving.")
            return
        if not isinstance(path, (str, Path)):
            path, _ = QFileDialog.getSaveFileName(self, "Save reproducible study", "", "JSON (*.json)")
        if path:
            try:
                save_study(path, tuple(self.snapshots.values()))
                self.report.setPlainText("Study saved with frozen criteria and exact input fingerprint.")
            except Exception as exc:
                self.report.setPlainText(str(exc))

    def load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open study", "", "JSON (*.json)")
        if path:
            self._start(lambda: load_study(path, self.workspace.result, self.workspace.candidates),
                        self._loaded)

    def _loaded(self, snapshots):
        self._stability = None
        self.snapshots = {snapshot.spec.name: snapshot for snapshot in snapshots}
        self.slot.clear()
        self.slot.addItems(list(dict.fromkeys(["A", "B", *self.snapshots])))
        self._load_slot()
        self.report.setPlainText("Study replayed against the same data. Current selection unchanged.")

    def export_stability(self, path=None):
        if self._stability is None:
            self.report.setPlainText("Compare scenarios before exporting stability.")
            return
        if not isinstance(path, (str, Path)):
            path, _ = QFileDialog.getSaveFileName(self, "Export stability", "", "CSV (*.csv)")
        if path:
            table = self._stability
            self._start(lambda: table.to_csv(path, index_label="record_id"),
                        lambda _: self.report.setPlainText("Complete stability table exported."))

    def _start(self, function, completed):
        if self.is_busy:
            return
        self.progress.show()
        self.report.setPlainText("Computing from cached descriptors. Current selection unchanged.")
        for widget in (*self._actions, self.slot, self.count, self.strategy, self.apply_threshold):
            widget.setEnabled(False)
        self._job = ComputationJob(function, self)
        self._job.completed.connect(completed)
        self._job.failed.connect(self.report.setPlainText)
        self._job.finished.connect(self._finished)
        self._job.start()

    def _finished(self):
        job, self._job = self._job, None
        self.progress.hide()
        for widget in (*self._actions, self.slot, self.count, self.strategy, self.apply_threshold):
            widget.setEnabled(True)
        if job is not None:
            job.deleteLater()

    def reject(self):
        if not self.is_busy:
            super().reject()

    def closeEvent(self, event):
        if self.is_busy:
            event.ignore()
        else:
            super().closeEvent(event)
