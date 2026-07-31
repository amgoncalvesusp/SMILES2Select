"""Step 7: results."""

from __future__ import annotations

import pandas as pd
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from smiles2select.export.excel import export_frame
from smiles2select.gui import charts
from smiles2select.gui.pages.base import WizardPage

VIEWS = ("Selecionados", "Excluídos", "Todos")
MAX_TABLE_ROWS = 5000


class ResultsPage(WizardPage):
    title = "7. Resultados"
    subtitle = "Comparação entre perfis, distribuições e a tabela final."

    def __init__(self, state) -> None:
        super().__init__(state)
        self._result = None
        self._frame = pd.DataFrame()

        self.headline = QLabel("Nenhuma execução concluída.")
        self.headline.setStyleSheet("font-size: 14px; font-weight: 600;")

        self.tabs = QTabWidget()
        self.profile_canvas = charts.Canvas()
        self.violation_canvas = charts.Canvas()
        self.intersection_canvas = charts.Canvas()
        self.distribution_canvas = charts.Canvas()
        self.tabs.addTab(self.profile_canvas, "Aprovação por perfil")
        self.tabs.addTab(self.violation_canvas, "Violações")
        self.tabs.addTab(self.intersection_canvas, "Interseção")
        self.tabs.addTab(self._distribution_tab(), "Distribuições")

        self.view_combo = QComboBox()
        self.view_combo.addItems(VIEWS)
        self.view_combo.currentTextChanged.connect(self._refresh_table)
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filtrar por ID ou SMILES...")
        self.filter_edit.textChanged.connect(self._refresh_table)

        filters = QHBoxLayout()
        filters.addWidget(QLabel("Exibir:"))
        filters.addWidget(self.view_combo)
        filters.addWidget(self.filter_edit, stretch=1)

        self.table = QTableWidget(0, 0)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table_note = QLabel("")
        self.table_note.setStyleSheet("color: #666; font-size: 11px;")

        self.body.addWidget(self.headline)
        self.body.addWidget(self.tabs, stretch=2)
        self.body.addLayout(filters)
        self.body.addWidget(self.table, stretch=3)
        self.body.addWidget(self.table_note)

    def _distribution_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        self.descriptor_combo = QComboBox()
        self.descriptor_combo.addItems(["mol_wt", "rdkit_wlogp", "tpsa", "qed"])
        self.descriptor_combo.currentTextChanged.connect(self._refresh_distribution)
        layout.addWidget(self.descriptor_combo)
        layout.addWidget(self.distribution_canvas, stretch=1)
        return container

    def show_result(self, result) -> None:
        """Populate the page after a run finishes."""
        self._result = result
        self.headline.setText(
            f"{result.decision.selected_count} selecionadas de {result.total_records} registros "
            f"({result.invalid_count} inválidos, {result.duplicate_count} duplicatas, "
            f"{result.evaluated_count} avaliados)."
        )

        charts.profile_bars(self.profile_canvas, result.profile_summary())
        charts.violation_counts(self.violation_canvas, result.evaluation.failures)
        charts.intersection_matrix(
            self.intersection_canvas,
            result.evaluation.status,
            [profile.id for profile in result.profiles],
        )
        self._refresh_distribution()

        frame = export_frame(result)
        frame.insert(0, "record_id", frame.index)
        self._frame = frame
        self._refresh_table()

    def _refresh_distribution(self) -> None:
        if self._result is None:
            return
        charts.descriptor_histogram(
            self.distribution_canvas,
            self._result.descriptors,
            self.descriptor_combo.currentText(),
            self._result.decision.decisions["selected"],
        )

    def _refresh_table(self) -> None:
        if self._frame.empty:
            return
        view = self.view_combo.currentText()
        frame = self._frame
        if view == "Selecionados":
            frame = frame[frame["Final_Status"] == "SELECTED"]
        elif view == "Excluídos":
            frame = frame[frame["Final_Status"] != "SELECTED"]

        text = self.filter_edit.text().strip().lower()
        if text:
            haystack = (
                frame["ID"].astype(str).str.lower()
                + " "
                + frame["Canonical_SMILES"].astype(str).str.lower()
            )
            frame = frame[haystack.str.contains(text, regex=False, na=False)]

        shown = frame.head(MAX_TABLE_ROWS)
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(shown))
        self.table.setColumnCount(len(shown.columns))
        self.table.setHorizontalHeaderLabels([str(column) for column in shown.columns])
        for row, (_, record) in enumerate(shown.iterrows()):
            for column, value in enumerate(record):
                self.table.setItem(row, column, QTableWidgetItem(_format(value)))
        self.table.setSortingEnabled(True)

        if len(frame) > MAX_TABLE_ROWS:
            self.table_note.setText(
                f"Exibindo {MAX_TABLE_ROWS} de {len(frame)} linhas. "
                "O arquivo Excel contém o conjunto completo."
            )
        else:
            self.table_note.setText(f"{len(frame)} linhas.")


def _format(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
