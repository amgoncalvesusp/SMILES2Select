"""Step 5: selection policy, alerts and outputs."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from smiles2select.alerts.policies import ACTION_LABELS
from smiles2select.alerts.rdkit_catalogs import CATALOG_LABELS, available_catalogs
from smiles2select.decision.explanations import policy_sentence, restrictiveness_warning
from smiles2select.decision.expression_parser import ExpressionError, compile_expression
from smiles2select.gui.pages.base import WizardPage
from smiles2select.pipeline import presets
from smiles2select.profiles.loader import builtin_registry
from smiles2select.scores.qed import QedSelection

QED_MODES = {
    "Compute only": "compute",
    "Rank": "rank",
    "Top percentile": "top_percentile",
    "Minimum threshold": "threshold",
}
MODE_LABELS = {value: label for label, value in QED_MODES.items()}
ACTION_ORDER = ("inform", "warn", "penalize", "exclude")


class PolicyPage(WizardPage):
    title = "5. Selection policy"
    subtitle = "Define how profiles, QED and alerts combine in the final decision."

    def __init__(self, state) -> None:
        super().__init__(state)

        self.consensus_spin = QSpinBox()
        self.consensus_spin.setRange(0, 20)
        self.consensus_spin.setSpecialValueText("(no consensus)")
        self.consensus_spin.valueChanged.connect(self._apply)

        self.expression_edit = QLineEdit()
        self.expression_edit.setPlaceholderText(
            "(lipinski AND veber) AND qed >= 0.50 AND NOT pains"
        )
        self.expression_edit.editingFinished.connect(self._apply)

        self.qed_mode = QComboBox()
        self.qed_mode.addItems(QED_MODES.keys())
        self.qed_mode.setCurrentText(MODE_LABELS.get(state.qed.mode, "Rank"))
        self.qed_mode.currentTextChanged.connect(self._apply)

        self.qed_value = QDoubleSpinBox()
        self.qed_value.setRange(0.0, 100.0)
        self.qed_value.setSingleStep(0.05)
        self.qed_value.setValue(0.5)
        self.qed_value.valueChanged.connect(self._apply)

        decision_group = QGroupBox("Decision")
        decision_form = QFormLayout(decision_group)
        decision_form.addRow("Approve in at least N consensus profiles:", self.consensus_spin)
        decision_form.addRow("Custom expression:", self.expression_edit)
        decision_form.addRow("QED:", self.qed_mode)
        decision_form.addRow("Value (threshold or percentile):", self.qed_value)

        self.alert_table = QTableWidget(0, 3)
        self.alert_table.setHorizontalHeaderLabels(["Catalog", "Use", "Action"])
        self.alert_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.alert_table.horizontalHeader().setStretchLastSection(True)
        self._catalog_boxes: dict[str, QCheckBox] = {}
        self._action_combos: dict[str, QComboBox] = {}
        self._build_alert_rows()

        alert_group = QGroupBox("Structural alerts")
        alert_layout = QVBoxLayout(alert_group)
        alert_layout.addWidget(self.alert_table)

        self.database_edit = QLineEdit()
        self.excel_edit = QLineEdit()
        self.detailed_box = QCheckBox("Detailed export (one sheet per broken rule)")
        self.detailed_box.setChecked(True)
        self.detailed_box.stateChanged.connect(self._apply)

        output_group = QGroupBox("Outputs")
        output_form = QFormLayout(output_group)
        output_form.addRow("SQLite database:", _with_browse(self.database_edit, self._pick_database))
        output_form.addRow("Excel report:", _with_browse(self.excel_edit, self._pick_excel))
        output_form.addRow(self.detailed_box)

        self.explanation = QLabel("")
        self.explanation.setWordWrap(True)
        self.explanation.setStyleSheet("background: #eef3f8; padding: 8px; border-radius: 4px;")

        self.warning = QLabel("")
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet("color: #a33;")

        save_button = QPushButton("Save preset...")
        load_button = QPushButton("Load preset...")
        save_button.clicked.connect(self._save_preset)
        load_button.clicked.connect(self._load_preset)
        preset_buttons = QHBoxLayout()
        preset_buttons.addWidget(save_button)
        preset_buttons.addWidget(load_button)
        preset_buttons.addStretch(1)

        self.body.addWidget(decision_group)
        self.body.addWidget(alert_group, stretch=1)
        self.body.addWidget(output_group)
        self.body.addLayout(preset_buttons)
        self.body.addWidget(self.explanation)
        self.body.addWidget(self.warning)

    def _save_preset(self) -> None:
        """Store the current choices; files and output paths are not included."""
        path, _ = QFileDialog.getSaveFileName(self, "Save preset", "", "Preset (*.json)")
        if not path:
            return
        self._apply()
        try:
            presets.save(self.state.to_preset(name=Path(path).stem), path)
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def _load_preset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load preset", "", "Preset (*.json)")
        if not path:
            return
        try:
            preset = presets.load(path)
            self.state.apply_preset(preset, builtin_registry().ids())
        except (presets.PresetError, ValueError) as exc:
            QMessageBox.critical(self, "Invalid preset", str(exc))
            return
        self._refresh_widgets_from_state()
        QMessageBox.information(
            self,
            "Preset loaded",
            f"'{preset.name}' applied. The selected files were kept.",
        )

    def _refresh_widgets_from_state(self) -> None:
        """Push a loaded preset back into the widgets."""
        self.qed_mode.setCurrentText(MODE_LABELS.get(self.state.qed.mode, "Rank"))
        value = self.state.qed.threshold or self.state.qed.percentile
        if value is not None:
            self.qed_value.setValue(float(value))
        self.expression_edit.setText(self.state.expression or "")
        self.detailed_box.setChecked(self.state.detailed_export)
        for catalog_id, box in self._catalog_boxes.items():
            box.setChecked(catalog_id in self.state.active_catalogs)
        for catalog_id, combo in self._action_combos.items():
            combo.setCurrentText(ACTION_LABELS[self.state.alert_actions.get(catalog_id, "warn")])
        self._refresh_explanation()

    def _build_alert_rows(self) -> None:
        catalogs = available_catalogs()
        self.alert_table.setRowCount(len(catalogs))
        for row, catalog_id in enumerate(catalogs):
            self.alert_table.setItem(
                row, 0, QTableWidgetItem(CATALOG_LABELS.get(catalog_id, catalog_id))
            )

            use_box = QCheckBox()
            use_box.setChecked(catalog_id in self.state.active_catalogs)
            use_box.stateChanged.connect(self._apply)
            self._catalog_boxes[catalog_id] = use_box
            self.alert_table.setCellWidget(row, 1, use_box)

            combo = QComboBox()
            combo.addItems([ACTION_LABELS[action] for action in ACTION_ORDER])
            combo.setCurrentText(ACTION_LABELS[self.state.alert_actions.get(catalog_id, "warn")])
            combo.currentTextChanged.connect(self._apply)
            self._action_combos[catalog_id] = combo
            self.alert_table.setCellWidget(row, 2, combo)

    def _pick_database(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Run database", "", "SQLite (*.sqlite)")
        if path:
            self.database_edit.setText(path)
            self._apply()

    def _pick_excel(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Excel report", "", "Excel (*.xlsx)")
        if path:
            self.excel_edit.setText(path)
            self._apply()

    def _apply(self) -> None:
        self.state.active_catalogs = [
            catalog_id for catalog_id, box in self._catalog_boxes.items() if box.isChecked()
        ]
        label_to_action = {label: action for action, label in ACTION_LABELS.items()}
        for catalog_id, combo in self._action_combos.items():
            self.state.alert_actions[catalog_id] = label_to_action[combo.currentText()]

        mode = QED_MODES[self.qed_mode.currentText()]
        value = self.qed_value.value()
        self.state.qed = QedSelection(
            mode=mode,
            threshold=value if mode == "threshold" else None,
            percentile=value if mode == "top_percentile" else None,
        )

        self.state.consensus_min_pass = self.consensus_spin.value() or None
        self.state.expression = self.expression_edit.text().strip() or None
        self.state.detailed_export = self.detailed_box.isChecked()
        self.state.database_path = (
            Path(self.database_edit.text()) if self.database_edit.text() else None
        )
        self.state.excel_path = Path(self.excel_edit.text()) if self.excel_edit.text() else None

        self._refresh_explanation()

    def _refresh_explanation(self) -> None:
        try:
            policy = self.state.build_policy()
        except ValueError as exc:
            self.explanation.setText("")
            self.warning.setText(str(exc))
            return
        self.explanation.setText(policy_sentence(policy))
        self.warning.setText(restrictiveness_warning(policy) or "")

    def on_enter(self) -> None:
        consensus = [pid for pid, role in self.state.roles.items() if role == "consensus"]
        self.consensus_spin.setMaximum(max(1, len(consensus)))
        if self.state.consensus_min_pass:
            self.consensus_spin.setValue(self.state.consensus_min_pass)
        self._apply()

    def validate(self) -> str | None:
        if self.state.expression:
            try:
                compile_expression(self.state.expression)
            except ExpressionError as exc:
                return f"Invalid expression: {exc}"
        problems = self.state.validation_errors()
        return "\n".join(problems) if problems else None


def _with_browse(edit: QLineEdit, handler) -> QWidget:
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    button = QPushButton("Browse...")
    button.clicked.connect(handler)
    layout.addWidget(edit, stretch=1)
    layout.addWidget(button)
    return container
