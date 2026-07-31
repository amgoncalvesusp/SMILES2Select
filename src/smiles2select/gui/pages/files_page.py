"""Step 1: choose the input files."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
)

from smiles2select.gui.pages.base import WizardPage
from smiles2select.gui.state import FileSelection
from smiles2select.io.importer import SourceReadError, sheet_names

FILE_FILTER = "Bibliotecas (*.csv *.tsv *.txt *.smi *.smiles *.xlsx *.xlsm *.xls);;Todos (*)"


class FilesPage(WizardPage):
    title = "1. Arquivos"
    subtitle = (
        "Selecione uma ou mais bibliotecas de SMILES. Planilhas com várias abas são "
        "adicionadas uma linha por aba."
    )

    def __init__(self, state) -> None:
        super().__init__(state)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Arquivo", "Aba", "Caminho"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)

        add_button = QPushButton("Adicionar arquivos...")
        remove_button = QPushButton("Remover selecionado")
        add_button.clicked.connect(self._add_files)
        remove_button.clicked.connect(self._remove_selected)

        buttons = QHBoxLayout()
        buttons.addWidget(add_button)
        buttons.addWidget(remove_button)
        buttons.addStretch(1)

        self.body.addWidget(self.table, stretch=1)
        self.body.addLayout(buttons)

    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Selecionar bibliotecas", "", FILE_FILTER)
        for raw_path in paths:
            path = Path(raw_path)
            try:
                sheets = sheet_names(path)
            except SourceReadError as exc:
                QMessageBox.warning(self, "Arquivo ilegível", str(exc))
                continue
            if sheets:
                for sheet in sheets:
                    self.state.files.append(FileSelection(path=path, sheet=sheet))
            else:
                self.state.files.append(FileSelection(path=path))
        self.on_enter()

    def _remove_selected(self) -> None:
        rows = {item.row() for item in self.table.selectedIndexes()}
        for index in sorted(rows, reverse=True):
            if 0 <= index < len(self.state.files):
                self.state.files.pop(index)
        self.on_enter()

    def on_enter(self) -> None:
        self.table.setRowCount(len(self.state.files))
        for row, selection in enumerate(self.state.files):
            self.table.setItem(row, 0, QTableWidgetItem(selection.path.name))
            self.table.setItem(row, 1, QTableWidgetItem(selection.sheet or "-"))
            self.table.setItem(row, 2, QTableWidgetItem(str(selection.path)))

    def validate(self) -> str | None:
        if not self.state.files:
            return "Selecione pelo menos um arquivo."
        return None
