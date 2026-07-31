"""Inspector and basket panels.

The inspector answers "what is this molecule?"; the basket answers "what have I
decided so far?". Both read from the same state the views draw, so nothing on
screen can disagree with anything else.
"""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from smiles2select.selection_intelligence.basket import SelectionBasket

INSPECTOR_FIELDS = (
    ("molecule_id", "ID"),
    ("canonical_smiles", "SMILES"),
    ("mol_wt", "MW"),
    ("rdkit_wlogp", "WLOGP"),
    ("tpsa", "TPSA"),
    ("qed", "QED"),
    ("sa_score", "SA"),
    ("murcko_scaffold", "Scaffold"),
)


class InspectorPanel(QWidget):
    """Everything known about one molecule, plus the decision buttons."""

    shortlist_requested = Signal(int)
    select_requested = Signal(int)
    exclude_requested = Signal(int)
    pin_requested = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.record_id: int | None = None

        self.title = QLabel("Nenhuma molécula selecionada")
        self.title.setStyleSheet("font-weight: 600;")
        self.details = QTextEdit()
        self.details.setReadOnly(True)

        buttons = QHBoxLayout()
        for label, signal in (
            ("Shortlist", self.shortlist_requested),
            ("Selecionar", self.select_requested),
            ("Excluir", self.exclude_requested),
            ("Fixar", self.pin_requested),
        ):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, emitter=signal: self._emit(emitter))
            buttons.addWidget(button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.title)
        layout.addWidget(self.details, stretch=1)
        layout.addLayout(buttons)

    def _emit(self, signal) -> None:
        if self.record_id is not None:
            signal.emit(self.record_id)

    def show_molecule(
        self,
        record_id: int,
        descriptors: pd.DataFrame,
        extras: dict[str, object] | None = None,
    ) -> None:
        """Fill the panel; missing properties are simply omitted."""
        self.record_id = record_id
        if record_id not in descriptors.index:
            self.title.setText(f"Registro {record_id} não encontrado")
            self.details.setPlainText("")
            return

        row = descriptors.loc[record_id]
        lines = [f"record_id: {record_id}"]
        for column, label in INSPECTOR_FIELDS:
            if column in descriptors.columns and not pd.isna(row.get(column)):
                value = row[column]
                lines.append(
                    f"{label}: {value:.3f}" if isinstance(value, float) else f"{label}: {value}"
                )
        for label, value in (extras or {}).items():
            lines.append(f"{label}: {value}")

        self.title.setText(str(row.get("molecule_id", record_id)))
        self.details.setPlainText("\n".join(lines))


class BasketPanel(QWidget):
    """Counters, the current decisions, and undo/redo."""

    undo_requested = Signal()
    redo_requested = Signal()
    auto_select_requested = Signal()
    export_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.counters = QLabel("")
        self.counters.setStyleSheet("font-weight: 600;")
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["ID", "Status", "Origem", "Nota"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)

        self.undo_button = QPushButton("Desfazer")
        self.redo_button = QPushButton("Refazer")
        auto_button = QPushButton("Completar automaticamente")
        export_button = QPushButton("Exportar seleção...")
        self.undo_button.clicked.connect(self.undo_requested)
        self.redo_button.clicked.connect(self.redo_requested)
        auto_button.clicked.connect(self.auto_select_requested)
        export_button.clicked.connect(self.export_requested)

        buttons = QHBoxLayout()
        buttons.addWidget(self.undo_button)
        buttons.addWidget(self.redo_button)
        buttons.addStretch(1)
        buttons.addWidget(auto_button)
        buttons.addWidget(export_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.counters)
        layout.addWidget(self.table, stretch=1)
        layout.addLayout(buttons)

    def refresh(self, basket: SelectionBasket, identifiers: pd.Series | None = None) -> None:
        """Redraw the counters and the decided molecules."""
        counters = basket.counters()
        self.counters.setText(
            " | ".join(f"{label}: {value}" for label, value in counters.as_rows())
        )
        self.undo_button.setEnabled(basket.log.can_undo)
        self.redo_button.setEnabled(basket.log.can_redo)

        decided = [state for state in basket.states() if state.origin is not None]
        self.table.setRowCount(len(decided))
        for row, state in enumerate(decided):
            label = (
                str(identifiers.get(state.record_id, state.record_id))
                if identifiers is not None
                else str(state.record_id)
            )
            self.table.setItem(row, 0, QTableWidgetItem(label))
            self.table.setItem(row, 1, QTableWidgetItem(state.selection_status.value))
            self.table.setItem(
                row, 2, QTableWidgetItem(state.origin.value if state.origin else "-")
            )
            self.table.setItem(row, 3, QTableWidgetItem(state.note or ""))
