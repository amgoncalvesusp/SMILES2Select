"""Read-only comparison of two live workspace records and their applied evidence."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from .active_criteria import ranking_source
from .panels import InspectorPanel


class MoleculeComparisonDialog(QDialog):
    """Hold A explicitly while B follows the workspace's inspected record."""

    def __init__(self, workspace) -> None:
        super().__init__(workspace)
        self.workspace = workspace
        self.reference_id: int | None = None
        self.current_id: int | None = workspace.inspector.record_id
        self.setWindowTitle("Compare molecules")
        self.setModal(False)

        self.status = QLabel()
        self.status.setTextFormat(Qt.PlainText)
        self.status.setWordWrap(True)
        self.reference_panel = InspectorPanel(read_only=True)
        self.current_panel = InspectorPanel(read_only=True)
        columns = QGridLayout()
        for column, (heading, panel) in enumerate((
            ("A — kept reference", self.reference_panel),
            ("B — currently inspected", self.current_panel),
        )):
            panel.title.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            label = QLabel(heading)
            label.setTextFormat(Qt.PlainText)
            columns.addWidget(label, 0, column)
            columns.addWidget(panel, 1, column)
            columns.setColumnStretch(column, 1)

        self.keep_reference_button = QPushButton("Keep current molecule as A")
        self.keep_reference_button.clicked.connect(self._keep_reference)
        close = QDialogButtonBox(QDialogButtonBox.Close)
        close.rejected.connect(self.close)
        layout = QVBoxLayout(self)
        layout.addWidget(self.status)
        layout.addWidget(self.keep_reference_button)
        layout.addLayout(columns, stretch=1)
        layout.addWidget(close)
        self.refresh()

        available = workspace.screen().availableGeometry()
        width = min(760, workspace.width(), available.width() - 32)
        height = min(600, workspace.height(), available.height() - 64)
        self.resize(width, height)
        parent_frame = workspace.frameGeometry()
        self.move(
            max(available.left() + 16, min(parent_frame.right() - width + 1,
                                           available.right() - width + 1 - 16)),
            max(available.top(), min(parent_frame.top() + 32,
                                     available.bottom() - height + 1)),
        )

    def set_current_id(self, record_id: int | None) -> None:
        self.current_id = record_id
        self.refresh()

    def _keep_reference(self) -> None:
        if self.workspace.is_busy or self.current_id not in self.workspace.candidates.index:
            return
        self.reference_id = self.current_id
        self.refresh()

    def _show_record(self, panel: InspectorPanel, record_id: int) -> None:
        scrollbar = panel.details.verticalScrollBar()
        position = scrollbar.value() if panel.record_id == record_id else 0
        extras = {key: value for key, value in self.workspace._molecule_extras(record_id).items()
                  if key not in {"Pareto rank", "Map point"}}
        panel.show_molecule(record_id, self.workspace.candidates, extras)
        scrollbar.setValue(position)

    def refresh(self) -> None:
        candidates = self.workspace.candidates
        if self.reference_id not in candidates.index:
            self.reference_id = None
        if self.current_id not in candidates.index:
            self.current_id = None
        if self.reference_id is None:
            self.reference_panel.clear("Keep a molecule as A to start comparing")
        else:
            self._show_record(self.reference_panel, self.reference_id)
        if self.current_id is None:
            self.current_panel.clear("Inspect a molecule in the workspace")
        elif self.current_id == self.reference_id:
            self.current_panel.clear("A and B are the same record. Inspect another molecule to compare.")
        else:
            self._show_record(self.current_panel, self.current_id)
        self.keep_reference_button.setEnabled(
            self.current_id is not None and not self.workspace.is_busy
        )
        self.status.setText(
            f"Applied method: {ranking_source(self.workspace)}. "
            "Read-only comparison; keeping A does not select, pin or adopt a proposal. "
            "Preview evidence remains in the main inspector."
        )
