"""Matplotlib canvases embedded in the results screen.

Charts answer the questions the tables cannot: which filter removed most
molecules, where the property distributions sit, and how the selected and
excluded sets differ.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QSizePolicy

DESCRIPTOR_LABELS = {
    "mol_wt": "MW (Da)",
    "rdkit_wlogp": "WLOGP",
    "tpsa": "TPSA (Å²)",
    "qed": "QED",
}


class Canvas(FigureCanvasQTAgg):
    """A figure that resizes with its container."""

    def __init__(self, width: float = 5.0, height: float = 3.2, dpi: int = 100) -> None:
        self.figure = Figure(figsize=(width, height), dpi=dpi, layout="constrained")
        super().__init__(self.figure)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def clear(self):
        self.figure.clear()
        return self.figure.add_subplot(111)


def profile_bars(canvas: Canvas, summary: pd.DataFrame) -> None:
    """Approval rate per profile."""
    axes = canvas.clear()
    if summary.empty:
        axes.set_title("Sem dados")
        canvas.draw_idle()
        return
    axes.barh(summary["profile_id"], summary["percentage"], color="#3b6ea5")
    axes.set_xlabel("Approval (%)")
    axes.set_xlim(0, 100)
    axes.set_title("Approval by profile")
    for index, value in enumerate(summary["percentage"]):
        axes.text(min(value + 1, 96), index, f"{value:.1f}%", va="center", fontsize=8)
    canvas.draw_idle()


def descriptor_histogram(
    canvas: Canvas, descriptors: pd.DataFrame, column: str, selected_mask: pd.Series | None = None
) -> None:
    """Distribution of one descriptor, selected versus excluded."""
    axes = canvas.clear()
    label = DESCRIPTOR_LABELS.get(column, column)
    values = pd.to_numeric(descriptors.get(column), errors="coerce").dropna()
    if values.empty:
        axes.set_title(f"{label}: sem dados")
        canvas.draw_idle()
        return

    if selected_mask is not None:
        mask = selected_mask.reindex(values.index).fillna(False).astype(bool)
        axes.hist(
            [values[mask], values[~mask]],
            bins=25,
            stacked=True,
            label=["Selected", "Excluded"],
            color=["#3b6ea5", "#c9772f"],
        )
        axes.legend(fontsize=8)
    else:
        axes.hist(values, bins=25, color="#3b6ea5")
    axes.set_xlabel(label)
    axes.set_ylabel("Molecules")
    axes.set_title(f"Distribution of {label}")
    canvas.draw_idle()


def docking_budget_histogram(canvas: Canvas, structures: pd.Series) -> None:
    """How many 3D structures each selected molecule would cost to prepare.

    Logarithmic x-axis: the counts are powers of two, so a linear axis would
    crush everything below the worst offender into a single bar.
    """
    axes = canvas.clear()
    values = pd.to_numeric(structures, errors="coerce").dropna()
    values = values[values > 0]
    if values.empty:
        axes.set_title("No preparability data")
        canvas.draw_idle()
        return

    bins = np.logspace(0, np.log10(max(float(values.max()), 2.0)), 20)
    axes.hist(values, bins=bins, color="#3b6ea5")
    axes.set_xscale("log")
    axes.set_xlabel("Estimated 3D structures per molecule")
    axes.set_ylabel("Molecules")
    axes.set_title("Docking preparation cost")
    canvas.draw_idle()


def violation_counts(canvas: Canvas, failures: pd.DataFrame, top: int = 15) -> None:
    """How often each rule is broken."""
    axes = canvas.clear()
    if failures.empty:
        axes.set_title("No violations recorded")
        canvas.draw_idle()
        return
    counted = failures.groupby("failure_code").size().sort_values(ascending=True).tail(top)
    axes.barh(counted.index, counted.to_numpy(), color="#c9772f")
    axes.set_xlabel("Molecules")
    axes.set_title("Violations by rule")
    canvas.draw_idle()


def intersection_matrix(canvas: Canvas, status: pd.DataFrame, profile_ids: list[str]) -> None:
    """Pairwise co-approval between profiles.

    Cell (i, j) is the share of molecules approved by profile i that profile j
    also approves, which is what shows whether two filters overlap or cut in
    different directions.
    """
    axes = canvas.clear()
    if status.empty or not profile_ids:
        axes.set_title("Sem dados")
        canvas.draw_idle()
        return

    passes = status[[f"{profile_id}__passed" for profile_id in profile_ids]].astype(bool)
    size = len(profile_ids)
    matrix = [[0.0] * size for _ in range(size)]
    for row, first in enumerate(profile_ids):
        base = passes[f"{first}__passed"]
        denominator = int(base.sum())
        for column, second in enumerate(profile_ids):
            both = int((base & passes[f"{second}__passed"]).sum())
            matrix[row][column] = 100.0 * both / denominator if denominator else 0.0

    image = axes.imshow(matrix, cmap="Blues", vmin=0, vmax=100)
    axes.set_xticks(range(size), profile_ids, rotation=45, ha="right", fontsize=8)
    axes.set_yticks(range(size), profile_ids, fontsize=8)
    for row in range(size):
        for column in range(size):
            axes.text(
                column,
                row,
                f"{matrix[row][column]:.0f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black" if matrix[row][column] < 60 else "white",
            )
    axes.set_title("Intersection between profiles (% of row)")
    canvas.figure.colorbar(image, ax=axes, shrink=0.8)
    canvas.draw_idle()
