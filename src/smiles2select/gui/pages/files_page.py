"""Step 1: choose the input files."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
)

from smiles2select.gui.pages.base import WizardPage
from smiles2select.gui.state import FileSelection
from smiles2select.io.importer import SourceReadError, sheet_names

FILE_FILTER = "Libraries (*.csv *.tsv *.txt *.smi *.smiles *.xlsx *.xlsm *.xls);;All (*)"


class FilesPage(WizardPage):
    title = "1. Files"
    subtitle = (
        "Select one or more SMILES libraries. Workbooks with multiple sheets add one row per sheet."
    )

    def __init__(self, state) -> None:
        super().__init__(state)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["File", "Sheet", "Path"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.empty_hint = QLabel(
            "Start with a CSV, TSV or Excel file containing a SMILES column and, "
            "optionally, molecule IDs. Add your file below, then choose the columns. "
            "Existing criteria can be reviewed at each step before processing."
        )
        self.empty_hint.setWordWrap(True)

        add_button = QPushButton("Add files...")
        remove_button = QPushButton("Remove selected")
        add_button.clicked.connect(self._add_files)
        remove_button.clicked.connect(self._remove_selected)

        buttons = QHBoxLayout()
        buttons.addWidget(add_button)
        buttons.addWidget(remove_button)
        buttons.addStretch(1)

        self.body.addWidget(self.empty_hint)
        self.body.addWidget(self.table, stretch=1)
        self.body.addLayout(buttons)

    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Select libraries", "", FILE_FILTER)
        for raw_path in paths:
            path = Path(raw_path)
            try:
                sheets = sheet_names(path)
            except SourceReadError as exc:
                QMessageBox.warning(self, "Unreadable file", str(exc))
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
        self.empty_hint.setVisible(not self.state.files)
        self.table.setRowCount(len(self.state.files))
        for row, selection in enumerate(self.state.files):
            self.table.setItem(row, 0, QTableWidgetItem(selection.path.name))
            self.table.setItem(row, 1, QTableWidgetItem(selection.sheet or "-"))
            self.table.setItem(row, 2, QTableWidgetItem(str(selection.path)))

    def validate(self) -> str | None:
        if not self.state.files:
            return "Select at least one file."
        return None
