"""Linked scatter views: chemical space and Pareto.

Both are the same widget with different axes, because they answer the same
question from different angles and must behave identically: hover previews,
click commits, lasso selects in bulk.

Rendered with pyqtgraph rather than a vector canvas: a vector renderer draws one
element per molecule and stops being interactive long before the library sizes
this tool targets.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QCursor
from PySide6.QtWidgets import QCheckBox, QLabel, QMenu, QVBoxLayout, QWidget

from smiles2select.chemical_space.density_tiles import lasso_contains
from smiles2select.chemical_space.layers import progressive_layer

#: Colour-blind safe ramp; never a rainbow scale, which encodes no order.
RANK_COLOURS = ("#1b6ca8", "#4c9f70", "#d9a441", "#c9772f", "#a33a3a")
UNSELECTED = QColor("#b8c4cc")
SELECTED = QColor("#12304a")


class ScatterView(QWidget):
    """A scatter plot whose points carry record ids.

    Emits ``point_clicked`` for a single commitment and ``region_selected`` for
    a lasso, so the workspace can treat the two differently: one opens the
    inspector, the other proposes a bulk action.
    """

    point_clicked = Signal(int)
    region_selected = Signal(object)

    def __init__(self, x_label: str = "x", y_label: str = "y", parent=None) -> None:
        super().__init__(parent)
        pg.setConfigOptions(antialias=False, background="w", foreground="#333")

        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", x_label)
        self.plot.setLabel("left", y_label)
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.scatter = pg.ScatterPlotItem(size=7, pen=None, hoverable=True)
        self.scatter.sigClicked.connect(self._on_click)
        self.plot.addItem(self.scatter)
        self.selected_scatter = pg.ScatterPlotItem(
            size=13,
            symbol="s",
            brush=pg.mkBrush("#e69f00"),
            pen=pg.mkPen(SELECTED, width=2),
            hoverable=True,
        )
        self.selected_scatter.setZValue(10)
        self.selected_scatter.sigClicked.connect(self._on_click)
        self.plot.addItem(self.selected_scatter)
        self.reference_scatter = pg.ScatterPlotItem(
            size=10, symbol="t", brush=pg.mkBrush("#a33a3a"), pen=pg.mkPen("#6e2222", width=1)
        )
        self.reference_scatter.setZValue(2)
        self.plot.addItem(self.reference_scatter)
        self.density_scatter = pg.ScatterPlotItem(pen=None)
        self.density_scatter.setZValue(-1)
        self.plot.addItem(self.density_scatter)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.legend_note = QLabel(
            "■ Gold square: final molecule for export · ● Circle: candidate context · "
            "▲ Red triangle: reference · Pale bubbles: aggregate density, not centroids."
        )
        self.legend_note.setWordWrap(True)
        self.selection_summary = QLabel("Projected: 0 · Final molecules shown: 0 / 0")
        self.selection_summary.setWordWrap(True)
        self.selection_only = QCheckBox("Show final molecules only")
        self.selection_only.toggled.connect(self._set_selection_only)
        self.overlap_note = QLabel(
            "Each square is an actual final molecule. Points may overlap: click to choose "
            "a record for inspection. Coordinates are never shifted for display."
        )
        self.overlap_note.setWordWrap(True)
        layout.addWidget(self.legend_note)
        layout.addWidget(self.selection_summary)
        layout.addWidget(self.selection_only)
        layout.addWidget(self.plot)
        layout.addWidget(self.overlap_note)

        self._coordinates = pd.DataFrame(columns=["x", "y"])
        self._lasso: list[tuple[float, float]] = []
        self._lasso_item = None
        self._selected_ids: set[int] = set()
        self._overlap_menu = None
        self.plot.scene().sigMouseClicked.connect(self._on_scene_click)

    # -- data ---------------------------------------------------------------

    def set_points(
        self,
        coordinates: pd.DataFrame,
        selected_ids: Sequence[int] = (),
        colour_by: pd.Series | None = None,
    ) -> None:
        """Draw actual final molecules above context without moving coordinates."""
        self._coordinates = coordinates
        self._selected_ids = {int(value) for value in selected_ids}
        selected_coordinates = coordinates.loc[coordinates.index.isin(self._selected_ids)]
        self.selection_summary.setText(
            f"Projected: {len(coordinates):,} · Final molecules shown: "
            f"{len(selected_coordinates):,} / {len(self._selected_ids):,}"
            + (
                " · Some finals are outside this projection; recompute the map."
                if len(selected_coordinates) < len(self._selected_ids)
                else ""
            )
        )
        self.selected_scatter.setData(
            x=selected_coordinates["x"].to_numpy(dtype=float),
            y=selected_coordinates["y"].to_numpy(dtype=float),
            data=[int(value) for value in selected_coordinates.index],
        )
        if coordinates.empty:
            self.scatter.setData([])
            self.density_scatter.setData([])
            return

        layer = progressive_layer(coordinates, selected_ids=pd.Index(selected_ids))
        display = layer.points
        selected = {int(value) for value in selected_ids}
        brushes, pens = [], []
        for record_id in display.index:
            brushes.append(pg.mkBrush(self._colour_for(record_id, colour_by)))
            pens.append(
                pg.mkPen(SELECTED, width=2) if int(record_id) in selected else pg.mkPen(None)
            )

        self.scatter.setData(
            x=display["x"].to_numpy(dtype=float),
            y=display["y"].to_numpy(dtype=float),
            brush=brushes,
            pen=pens,
            data=[int(record_id) for record_id in display.index],
        )
        if layer.aggregated:
            self.density_scatter.setData(
                x=layer.density["x"].to_numpy(dtype=float),
                y=layer.density["y"].to_numpy(dtype=float),
                size=(6 + 2 * layer.density["molecules"].clip(upper=100).pow(0.5)).to_numpy(),
                brush=[pg.mkBrush("#d7e3ea") for _ in range(len(layer.density))],
            )
        else:
            self.density_scatter.setData([])

    def _set_selection_only(self, enabled: bool) -> None:
        self.scatter.setVisible(not enabled)
        self.density_scatter.setVisible(not enabled)
        self.reference_scatter.setVisible(not enabled)

    def set_reference_points(self, coordinates: pd.DataFrame | None) -> None:
        """Show reference compounds as a separate triangular overlay layer."""
        if coordinates is None or coordinates.empty:
            self.reference_scatter.setData([])
            return
        self.reference_scatter.setData(
            x=coordinates["x"].to_numpy(dtype=float),
            y=coordinates["y"].to_numpy(dtype=float),
            data=[str(record_id) for record_id in coordinates.index],
        )

    def _colour_for(self, record_id, colour_by: pd.Series | None) -> QColor:
        if colour_by is None or record_id not in colour_by.index:
            return UNSELECTED
        value = colour_by.loc[record_id]
        if pd.isna(value):
            return UNSELECTED
        return QColor(RANK_COLOURS[min(int(value) - 1, len(RANK_COLOURS) - 1)])

    # -- interaction --------------------------------------------------------

    def _on_click(self, _scatter, points, event=None) -> None:
        """Offer every hit/identical-position record, including covered context."""
        if not len(points) or (event is not None and event.modifiers() & Qt.ControlModifier):
            return
        hit_points = list(points)
        if event is not None:
            for layer in (self.scatter, self.selected_scatter):
                if layer.isVisible():
                    hit_points.extend(layer.pointsAt(layer.mapFromScene(event.scenePos())))
        positions = {(point.pos().x(), point.pos().y()) for point in hit_points}
        overlaps = pd.MultiIndex.from_frame(self._coordinates[["x", "y"]]).isin(positions)
        record_ids = {int(value) for value in self._coordinates.index[overlaps]}
        if self.selection_only.isChecked():
            record_ids &= self._selected_ids
        if len(record_ids) == 1:
            self.point_clicked.emit(next(iter(record_ids)))
            return
        if self._overlap_menu is not None:
            self._overlap_menu.close()
            self._overlap_menu.deleteLater()
        self._overlap_menu = QMenu(self)
        for record_id in sorted(record_ids):
            status = "final for export" if record_id in self._selected_ids else "context"
            action = self._overlap_menu.addAction(f"Record {record_id} — {status}")
            action.triggered.connect(
                lambda _checked=False, value=record_id: self.point_clicked.emit(value)
            )
        self._overlap_menu.popup(QCursor.pos())

    def _on_scene_click(self, event) -> None:
        """Ctrl-click builds a lasso; a plain click closes it."""
        position = self.plot.plotItem.vb.mapSceneToView(event.scenePos())
        if event.modifiers() & Qt.ControlModifier:
            self._lasso.append((float(position.x()), float(position.y())))
            self._draw_lasso()
        elif self._lasso:
            self.finish_lasso()

    def _draw_lasso(self) -> None:
        if self._lasso_item is None:
            self._lasso_item = self.plot.plot(pen=pg.mkPen("#a33a3a", width=1.5))
        points = np.array(self._lasso + self._lasso[:1])
        self._lasso_item.setData(points[:, 0], points[:, 1])

    def finish_lasso(self) -> list[int]:
        """Close the polygon and report the molecules inside it."""
        polygon = np.array(self._lasso)
        self.clear_lasso()
        if len(polygon) < 3 or self._coordinates.empty:
            return []
        eligible = (
            self._coordinates.loc[self._coordinates.index.isin(self._selected_ids)]
            if self.selection_only.isChecked()
            else self._coordinates
        )
        inside = [int(value) for value in lasso_contains(eligible, polygon)]
        self.region_selected.emit(inside)
        return inside

    def clear_lasso(self) -> None:
        self._lasso = []
        if self._lasso_item is not None:
            self.plot.removeItem(self._lasso_item)
            self._lasso_item = None


class ChemicalSpaceView(ScatterView):
    """The chemical space map."""

    def __init__(self, parent=None) -> None:
        super().__init__("Component 1", "Component 2", parent)


class ParetoView(ScatterView):
    """Two objectives against each other, coloured by front."""

    def __init__(self, parent=None) -> None:
        super().__init__("Objective 1", "Objective 2", parent)

    def show_objectives(
        self,
        values: pd.DataFrame,
        first: str,
        second: str,
        ranks: pd.Series,
        selected_ids: Sequence[int] = (),
    ) -> None:
        """Plot two objectives, colouring each point by its Pareto front."""
        coordinates = pd.DataFrame(
            {"x": values[first].astype(float), "y": values[second].astype(float)},
            index=values.index,
        )
        self.plot.setLabel("bottom", first)
        self.plot.setLabel("left", second)
        self.set_points(coordinates, selected_ids, ranks)
