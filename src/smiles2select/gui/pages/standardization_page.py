"""Step 3: standardization profile."""

from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QGroupBox, QLabel, QVBoxLayout

from smiles2select.chemistry.standardization import StandardizationConfig
from smiles2select.gui.pages.base import WizardPage

OPTIONS = (
    ("cleanup", "RDKit cleanup (sanitization, normalization)", True),
    ("remove_salts", "Remove salts and counterions (keep the principal fragment)", True),
    ("neutralize", "Neutralize charges", False),
    ("canonical_tautomer", "Canonicalize tautomers (slower)", False),
    ("remove_stereo", "Remove stereochemistry", False),
)


class StandardizationPage(WizardPage):
    title = "3. Standardization"
    subtitle = (
        "Standardization defines the structure on which all descriptors are calculated. "
        "Changing it invalidates previous caches because it changes the values."
    )

    def __init__(self, state) -> None:
        super().__init__(state)
        self.checkboxes: dict[str, QCheckBox] = {}

        group = QGroupBox("Steps")
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
            "Standardization hash (used by the cache and report): "
            f"{self.state.standardization.fingerprint()[:16]}…"
        )

    def on_enter(self) -> None:
        self._apply()
