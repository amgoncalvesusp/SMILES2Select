"""Shared page scaffolding."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QScrollArea, QVBoxLayout, QWidget

from smiles2select.gui.state import WizardState


class WizardPage(QWidget):
    """Base class for the seven steps.

    Subclasses fill ``self.body``; navigation calls :meth:`validate` before
    moving forward and :meth:`on_enter` after arriving.
    """

    title = ""
    subtitle = ""

    def __init__(self, state: WizardState) -> None:
        super().__init__()
        self.state = state

        heading = QLabel(self.title)
        heading.setStyleSheet("font-size: 18px; font-weight: 600;")
        description = QLabel(self.subtitle)
        description.setWordWrap(True)
        description.setStyleSheet("color: #444;")

        content = QWidget()
        self.body = QVBoxLayout(content)
        self.body.setContentsMargins(0, 0, 0, 0)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setWidget(content)

        layout = QVBoxLayout(self)
        layout.addWidget(heading)
        if self.subtitle:
            layout.addWidget(description)
        layout.addWidget(self.scroll_area, stretch=1)

    def validate(self) -> str | None:
        """Return a message blocking navigation, or None when the step is complete."""
        return None

    def on_enter(self) -> None:
        """Refresh the page with whatever earlier steps have set."""
