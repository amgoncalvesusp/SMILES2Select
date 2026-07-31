"""Step 2: map the SMILES and identifier columns."""

from __future__ import annotations

from PySide6.QtWidgets import QAbstractItemView, QComboBox, QLabel, QTableWidget, QTableWidgetItem

from smiles2select.gui.pages.base import WizardPage
from smiles2select.io.importer import SourceReadError, guess_mapping, preview_columns

NO_ID = "(gerar automaticamente)"


class ColumnsPage(WizardPage):
    title = "2. Colunas"
    subtitle = (
        "Indique qual coluna contém o SMILES e, se houver, qual contém o identificador. "
        "Sem identificador, um código sequencial é gerado."
    )

    def __init__(self, state) -> None:
        super().__init__(state)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Arquivo", "Aba", "Coluna SMILES", "Coluna ID"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.message = QLabel("")
        self.message.setStyleSheet("color: #a33;")
        self.message.setWordWrap(True)

        self.body.addWidget(self.table, stretch=1)
        self.body.addWidget(self.message)

    def on_enter(self) -> None:
        problems: list[str] = []
        self.table.setRowCount(len(self.state.files))
        for row, selection in enumerate(self.state.files):
            try:
                columns = preview_columns(selection.path, selection.sheet)
            except SourceReadError as exc:
                problems.append(str(exc))
                columns = []

            if columns and not selection.smiles_column:
                guessed = guess_mapping(columns)
                selection.smiles_column = guessed.smiles
                selection.id_column = guessed.molecule_id

            self.table.setItem(row, 0, QTableWidgetItem(selection.path.name))
            self.table.setItem(row, 1, QTableWidgetItem(selection.sheet or "-"))
            self.table.setCellWidget(row, 2, self._smiles_combo(row, columns))
            self.table.setCellWidget(row, 3, self._id_combo(row, columns))
        self.message.setText("\n".join(problems))

    def _smiles_combo(self, row: int, columns: list[str]) -> QComboBox:
        combo = QComboBox()
        combo.addItems([str(column) for column in columns])
        current = self.state.files[row].smiles_column
        if current:
            combo.setCurrentText(current)
        combo.currentTextChanged.connect(
            lambda value, index=row: setattr(self.state.files[index], "smiles_column", value)
        )
        return combo

    def _id_combo(self, row: int, columns: list[str]) -> QComboBox:
        combo = QComboBox()
        combo.addItem(NO_ID)
        combo.addItems([str(column) for column in columns])
        current = self.state.files[row].id_column
        combo.setCurrentText(current if current else NO_ID)
        combo.currentTextChanged.connect(
            lambda value, index=row: setattr(
                self.state.files[index], "id_column", None if value == NO_ID else value
            )
        )
        return combo

    def validate(self) -> str | None:
        missing = [
            selection.path.name for selection in self.state.files if not selection.smiles_column
        ]
        if missing:
            return f"Defina a coluna de SMILES para: {', '.join(missing)}"
        return None
