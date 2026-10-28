"""Virtual, read-only candidate rows linked by the pipeline's record IDs."""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import (
    QAbstractTableModel,
    QItemSelectionModel,
    QModelIndex,
    QSignalBlocker,
    Qt,
    Signal,
)
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableView

from smiles2select.selection_intelligence.basket import SelectionBasket

COLUMNS = (
    ("record_id", "Record ID"),
    ("molecule_id", "Molecule ID"),
    ("final_for_export", "Final for export"),
    ("chemical_status", "Chemical status"),
    ("selection_status", "Selection status"),
    ("mol_wt", "MW"),
    ("rdkit_wlogp", "WLOGP"),
    ("tpsa", "TPSA"),
    ("qed", "QED"),
    ("sa_score", "SA"),
    ("canonical_smiles", "SMILES"),
)
STATE_COLUMNS = ("final_for_export", "chemical_status", "selection_status")
NUMERIC_COLUMNS = ("record_id", "mol_wt", "rdkit_wlogp", "tpsa", "qed", "sa_score")


class CandidateTableModel(QAbstractTableModel):
    """Read cells lazily from a shared frame, storing only the row ordering."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._frame = pd.DataFrame()
        self._basket: SelectionBasket | None = None
        self._row_ids = self._frame.index
        self._sort_column = -1
        self._sort_order = Qt.AscendingOrder

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._row_ids)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            if 0 <= section < len(COLUMNS):
                return COLUMNS[section][1]
        return None

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def data(self, index, role=Qt.DisplayRole):
        record_id = self.record_id(index)
        if record_id is None or role != Qt.DisplayRole:
            return None
        value = self._value(record_id, COLUMNS[index.column()][0])
        return "" if pd.isna(value) else str(value)

    def record_id(self, index: QModelIndex) -> int | None:
        if (not index.isValid() or index.model() is not self
                or not 0 <= index.row() < len(self._row_ids)):
            return None
        return int(self._row_ids[index.row()])

    def index_for_record(self, record_id: int) -> QModelIndex:
        try:
            return self.index(int(self._row_ids.get_loc(record_id)), 0)
        except KeyError:
            return QModelIndex()

    def refresh(self, candidates: pd.DataFrame, basket: SelectionBasket) -> None:
        if candidates is not self._frame:
            self.beginResetModel()
            self._frame = candidates
            self._basket = basket
            self._row_ids = self._ordered_ids()
            self.endResetModel()
            return
        self._basket = basket
        if self._sort_column >= 0 and COLUMNS[self._sort_column][0] in STATE_COLUMNS:
            self.sort(self._sort_column, self._sort_order)
        if len(self._row_ids):
            self.dataChanged.emit(
                self.index(0, 2), self.index(len(self._row_ids) - 1, 4), [Qt.DisplayRole]
            )

    def sort(self, column, order=Qt.AscendingOrder) -> None:
        if not -1 <= column < len(COLUMNS):
            return
        self._sort_column, self._sort_order = column, order
        row_ids = self._ordered_ids()
        if row_ids.equals(self._row_ids):
            return
        self.layoutAboutToBeChanged.emit()
        previous = self.persistentIndexList()
        records = [self.record_id(index) for index in previous]
        self._row_ids = row_ids
        updated = [self.index(int(row_ids.get_loc(record_id)), index.column())
                   for record_id, index in zip(records, previous)]
        self.changePersistentIndexList(previous, updated)
        self.layoutChanged.emit()

    def _value(self, record_id: int, key: str):
        if key == "record_id":
            return record_id
        if key in STATE_COLUMNS:
            if self._basket is None or record_id not in self._basket:
                return None
            state = self._basket.state(record_id)
            if key == "final_for_export":
                return "Yes" if state.is_selected else "No"
            return getattr(state, key).value
        return self._frame.at[record_id, key] if key in self._frame.columns else None

    def _ordered_ids(self) -> pd.Index:
        if self._sort_column < 0 or self._frame.empty:
            return self._frame.index
        key = COLUMNS[self._sort_column][0]
        if key in STATE_COLUMNS or key == "record_id":
            values = pd.Series(
                (self._value(int(record_id), key) for record_id in self._frame.index),
                index=self._frame.index,
            )
        elif key in self._frame.columns:
            values = self._frame[key]
        else:
            return self._frame.index
        values = (pd.to_numeric(values, errors="coerce") if key in NUMERIC_COLUMNS
                  else values.astype("string"))
        return values.sort_values(
            ascending=self._sort_order == Qt.AscendingOrder, kind="stable", na_position="last"
        ).index


class CandidateTableView(QTableView):
    """Inspect candidate rows without making selection decisions."""

    molecule_requested = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.table_model = CandidateTableModel(self)
        self.setModel(self.table_model)
        self._sort_blocker = None
        self.table_model.layoutAboutToBeChanged.connect(self._block_sort_selection)
        self.table_model.layoutChanged.connect(self._unblock_sort_selection)
        self.selectionModel().currentRowChanged.connect(self._inspect_row)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.setWordWrap(False)
        self.verticalHeader().hide()
        self.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.verticalHeader().setDefaultSectionSize(26)
        widths = (90, 140, 120, 130, 170, 110, 110, 110, 130, 110, 360)
        for column, width in enumerate(widths):
            self.setColumnWidth(column, width)
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
        self.setSortingEnabled(True)

    def refresh(self, candidates: pd.DataFrame, basket: SelectionBasket) -> None:
        current = self.currentIndex()
        record_id = self.table_model.record_id(current)
        horizontal_position = self.horizontalScrollBar().value()
        with QSignalBlocker(self.selectionModel()):
            self.table_model.refresh(candidates, basket)
            if record_id is not None:
                index = self.table_model.index_for_record(record_id)
                self._set_inspected_index(index.siblingAtColumn(current.column()))
            self.horizontalScrollBar().setValue(horizontal_position)

    def set_inspected_id(self, record_id: int | None) -> None:
        current = self.currentIndex()
        if record_id is not None and self.table_model.record_id(current) == record_id:
            return
        index = (self.table_model.index_for_record(record_id) if record_id is not None
                 else QModelIndex())
        if current.isValid():
            index = index.siblingAtColumn(current.column())
        self._set_inspected_index(index)

    def _set_inspected_index(self, index: QModelIndex) -> None:
        horizontal_position = self.horizontalScrollBar().value()
        with QSignalBlocker(self.selectionModel()):
            self.selectionModel().setCurrentIndex(
                index, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows
            )
            if index.isValid():
                self.scrollTo(index)
            self.horizontalScrollBar().setValue(horizontal_position)

    def _inspect_row(self, current, _previous) -> None:
        record_id = self.table_model.record_id(current)
        if record_id is not None:
            self.molecule_requested.emit(record_id)

    def _block_sort_selection(self) -> None:
        self._sort_blocker = QSignalBlocker(self.selectionModel())

    def _unblock_sort_selection(self) -> None:
        if self._sort_blocker is not None:
            self._sort_blocker.unblock()
            self._sort_blocker = None
