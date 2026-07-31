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
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QVBoxLayout, QWidget

from smiles2select.chemical_space.density_tiles import lasso_contains

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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot)

        self._coordinates = pd.DataFrame(columns=["x", "y"])
        self._lasso: list[tuple[float, float]] = []
        self._lasso_item = None
        self.plot.scene().sigMouseClicked.connect(self._on_scene_click)

    # -- data ---------------------------------------------------------------

    def set_points(
        self,
        coordinates: pd.DataFrame,
        selected_ids: Sequence[int] = (),
        colour_by: pd.Series | None = None,
    ) -> None:
        """Draw the points, marking the selection with a heavier outline.

        Selection is encoded by outline as well as fill, so it survives being
        printed in greyscale or read by someone who cannot distinguish the
        colours.
        """
        self._coordinates = coordinates
        if coordinates.empty:
            self.scatter.setData([])
            return

        selected = {int(value) for value in selected_ids}
        brushes, pens = [], []
        for record_id in coordinates.index:
            brushes.append(pg.mkBrush(self._colour_for(record_id, colour_by)))
            pens.append(
                pg.mkPen(SELECTED, width=2) if int(record_id) in selected else pg.mkPen(None)
            )

        self.scatter.setData(
            x=coordinates["x"].to_numpy(dtype=float),
            y=coordinates["y"].to_numpy(dtype=float),
            brush=brushes,
            pen=pens,
            data=[int(record_id) for record_id in coordinates.index],
        )

    def _colour_for(self, record_id, colour_by: pd.Series | None) -> QColor:
        if colour_by is None or record_id not in colour_by.index:
            return UNSELECTED
        value = colour_by.loc[record_id]
        if pd.isna(value):
            return UNSELECTED
        return QColor(RANK_COLOURS[min(int(value) - 1, len(RANK_COLOURS) - 1)])

    # -- interaction --------------------------------------------------------

    def _on_click(self, _scatter, points) -> None:
        if points:
            self.point_clicked.emit(int(points[0].data()))

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
        inside = [int(value) for value in lasso_contains(self._coordinates, polygon)]
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
        super().__init__("Componente 1", "Componente 2", parent)


class ParetoView(ScatterView):
    """Two objectives against each other, coloured by front."""

    def __init__(self, parent=None) -> None:
        super().__init__("objetivo 1", "objetivo 2", parent)

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
