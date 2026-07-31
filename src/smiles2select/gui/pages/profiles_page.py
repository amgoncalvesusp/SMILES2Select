"""Step 4: choose profiles and their roles."""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QWidget,
)

from smiles2select.gui.pages.base import WizardPage
from smiles2select.pipeline.runner import run as run_pipeline
from smiles2select.pipeline.sampling import analyse
from smiles2select.profiles.loader import builtin_registry
from smiles2select.profiles.registry import Profile

ROLE_VALUES = {
    "Obrigatório": "mandatory",
    "Consenso": "consensus",
    "Informativo": "informative",
    "Ranqueamento": "ranking",
    "Advertência": "warning",
    "Exclusão": "exclusion",
}
VALUE_ROLES = {value: label for label, value in ROLE_VALUES.items()}

STRICT = "Nenhuma violação"
ONE_VIOLATION = "Até 1 violação"


class ProfilesPage(WizardPage):
    title = "4. Perfis"
    subtitle = (
        "Escolha os perfis e o papel de cada um. Perfis informativos são calculados e "
        "aparecem no relatório, mas não excluem moléculas."
    )

    def __init__(self, state) -> None:
        super().__init__(state)
        self.registry = builtin_registry()
        self.profiles: tuple[Profile, ...] = self.registry.all()

        self.table = QTableWidget(len(self.profiles), 5)
        self.table.setHorizontalHeaderLabels(
            ["Perfil", "Usar", "Papel", "Configuração", "Espaço químico"]
        )
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self._use_boxes: dict[str, QCheckBox] = {}
        self._role_combos: dict[str, QComboBox] = {}
        self._policy_combos: dict[str, QComboBox] = {}
        self._build_rows()

        self.sample_size = QSpinBox()
        self.sample_size.setRange(50, 100000)
        self.sample_size.setValue(1000)
        self.sample_size.setSingleStep(50)
        analyse_button = QPushButton("Analisar amostra")
        analyse_button.clicked.connect(self._analyse_sample)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Moléculas na amostra:"))
        controls.addWidget(self.sample_size)
        controls.addWidget(analyse_button)
        controls.addStretch(1)

        self.report = QTextEdit()
        self.report.setReadOnly(True)
        self.report.setMaximumHeight(220)
        self.report.setPlaceholderText(
            "A análise de amostra mostra o impacto de cada perfil sobre os seus próprios dados. "
            "Ela não escolhe regras por você."
        )

        self.body.addWidget(self.table, stretch=1)
        self.body.addLayout(controls)
        self.body.addWidget(self.report)

    def _build_rows(self) -> None:
        for row, profile in enumerate(self.profiles):
            name = QTableWidgetItem(profile.name)
            name.setToolTip(profile.notes or profile.name)
            self.table.setItem(row, 0, name)

            use_box = QCheckBox()
            use_box.setChecked(profile.id in self.state.roles)
            use_box.stateChanged.connect(lambda _checked, pid=profile.id: self._sync(pid))
            self._use_boxes[profile.id] = use_box
            self.table.setCellWidget(row, 1, _centered(use_box))

            role_combo = QComboBox()
            role_combo.addItems(ROLE_VALUES.keys())
            role_combo.setCurrentText(
                VALUE_ROLES.get(self.state.roles.get(profile.id, "informative"), "Informativo")
            )
            role_combo.currentTextChanged.connect(lambda _text, pid=profile.id: self._sync(pid))
            self._role_combos[profile.id] = role_combo
            self.table.setCellWidget(row, 2, role_combo)

            policy_combo = QComboBox()
            policy_combo.addItems([STRICT, ONE_VIOLATION])
            policy_combo.setCurrentText(
                ONE_VIOLATION if profile.pass_policy.get("type") == "max_violations" else STRICT
            )
            policy_combo.currentTextChanged.connect(
                lambda text, pid=profile.id: self._set_pass_policy(pid, text)
            )
            self._policy_combos[profile.id] = policy_combo
            self.table.setCellWidget(row, 3, policy_combo)

            space = QTableWidgetItem(profile.category)
            space.setToolTip(profile.notes or "")
            self.table.setItem(row, 4, space)

    def _sync(self, profile_id: str) -> None:
        if self._use_boxes[profile_id].isChecked():
            role = ROLE_VALUES[self._role_combos[profile_id].currentText()]
            self.state.set_role(profile_id, role)
        else:
            self.state.set_role(profile_id, None)

        consensus = [pid for pid, role in self.state.roles.items() if role == "consensus"]
        if consensus and self.state.consensus_min_pass is None:
            self.state.consensus_min_pass = max(1, len(consensus) - 1)
        elif not consensus:
            self.state.consensus_min_pass = None

    def _set_pass_policy(self, profile_id: str, text: str) -> None:
        """Switch a profile between the strict and the tolerant reading."""
        profile = self.registry.get(profile_id)
        policy = (
            {"type": "max_violations", "value": 1}
            if text == ONE_VIOLATION
            else {"type": "all_rules"}
        )
        self.registry.add(profile.with_pass_policy(policy), overwrite=True)
        self.profiles = self.registry.all()

    def _analyse_sample(self) -> None:
        """Run the selected profiles over a sample and report their impact."""
        problem = self.validate()
        if problem:
            QMessageBox.warning(self, "Configuração incompleta", problem)
            return
        try:
            config = self.state.build_config()
        except ValueError as exc:
            QMessageBox.warning(self, "Configuração incompleta", str(exc))
            return

        sample_config = replace(
            config,
            max_records=self.sample_size.value(),
            database_path=None,
            excel_path=None,
            n_jobs=1,
        )

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            result = run_pipeline(sample_config)
            report = analyse(result.evaluation, list(self.state.selected_profiles()))
        except Exception as exc:
            QMessageBox.critical(self, "Falha na análise", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.report.setPlainText(_format_report(report, result))

    def on_enter(self) -> None:
        for profile_id, box in self._use_boxes.items():
            box.setChecked(profile_id in self.state.roles)

    def validate(self) -> str | None:
        if not self.state.roles:
            return "Selecione pelo menos um perfil."
        return None


def _centered(widget: QWidget) -> QWidget:
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.addWidget(widget)
    layout.setAlignment(Qt.AlignCenter)
    layout.setContentsMargins(0, 0, 0, 0)
    return container


def _format_report(report, result) -> str:
    lines = [f"Amostra analisada: {report.sample_size} moléculas válidas.", ""]
    lines.append("Impacto isolado (aprovação de cada perfil sozinho):")
    for row in report.per_profile.itertuples():
        lines.append(f"  {row.profile_id:<16} {row.passed:>6}  ({row.pass_rate * 100:.1f}%)")

    lines.append("")
    lines.append("Impacto acumulado (interseção na ordem listada):")
    for row in report.cumulative.itertuples():
        lines.append(f"  + {row.profile_id:<14} {row.surviving:>6}  ({row.pass_rate * 100:.1f}%)")

    lines.append("")
    lines.append(f"Perfis mais restritivos: {', '.join(report.most_restrictive())}")

    if not report.only_one_rule.empty:
        lines.append("")
        lines.append("Moléculas reprovadas por uma única regra:")
        for row in report.only_one_rule.head(8).itertuples():
            lines.append(f"  {row.failure_code:<18} {row.sole_failures}")

    if not report.recovered_with_one_violation.empty:
        lines.append("")
        lines.append("Recuperadas se o perfil tolerar uma violação:")
        for row in report.recovered_with_one_violation.itertuples():
            lines.append(f"  {row.profile_id:<16} {row.recovered}")

    lines.append("")
    lines.append(
        f"Na amostra, {result.decision.selected_count} de {result.evaluated_count} moléculas "
        "seriam selecionadas com a política atual."
    )
    return "\n".join(lines)
