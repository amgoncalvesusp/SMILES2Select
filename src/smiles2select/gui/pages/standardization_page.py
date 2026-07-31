"""Step 3: standardization profile."""

from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QGroupBox, QLabel, QVBoxLayout

from smiles2select.chemistry.standardization import StandardizationConfig
from smiles2select.gui.pages.base import WizardPage

OPTIONS = (
    ("cleanup", "Limpeza RDKit (sanitização, normalização)", True),
    ("remove_salts", "Remover sais e contraíons (manter o fragmento principal)", True),
    ("neutralize", "Neutralizar cargas", False),
    ("canonical_tautomer", "Canonicalizar tautômero (mais lento)", False),
    ("remove_stereo", "Remover estereoquímica", False),
)


class StandardizationPage(WizardPage):
    title = "3. Padronização"
    subtitle = (
        "A padronização define a estrutura sobre a qual todos os descritores serão calculados. "
        "Alterá-la invalida qualquer cache anterior, porque muda os valores calculados."
    )

    def __init__(self, state) -> None:
        super().__init__(state)
        self.checkboxes: dict[str, QCheckBox] = {}

        group = QGroupBox("Etapas")
        layout = QVBoxLayout(group)
        for field_name, label, default in OPTIONS:
            checkbox = QCheckBox(label)
            checkbox.setChecked(getattr(state.standardization, field_name, default))
            checkbox.stateChanged.connect(self._apply)
            self.checkboxes[field_name] = checkbox
            layout.addWidget(checkbox)

        self.fingerprint = QLabel("")
        self.fingerprint.setStyleSheet("color: #666; font-size: 11px;")
        self.fingerprint.setWordWrap(True)

        self.body.addWidget(group)
        self.body.addWidget(self.fingerprint)
        self.body.addStretch(1)
        self._apply()

    def _apply(self) -> None:
        self.state.standardization = StandardizationConfig(
            **{name: box.isChecked() for name, box in self.checkboxes.items()}
        )
        self.fingerprint.setText(
            "Hash da padronização (usado no cache e no relatório): "
            f"{self.state.standardization.fingerprint()[:16]}…"
        )

    def on_enter(self) -> None:
        self._apply()
