"""Density tiles and viewport culling.

At a distant zoom, half a million individual points is neither drawable nor
readable: they overlap into a single dark mass. Aggregating into tiles shows
where the library actually is, and - more usefully - which regions the current
selection covers and which it misses.

At close zoom the raw points are sent instead; these helpers decide which ones
are even inside the viewport.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Above this many visible points, the map should aggregate rather than plot.
DENSITY_THRESHOLD = 20_000

_TILE_COLUMNS = ["x", "y", "molecules", "selected", "selected_share", "x_index", "y_index"]


@dataclass(frozen=True)
class Viewport:
    """Visible rectangle of the map."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains(self, coordinates: pd.DataFrame) -> pd.Series:
        return coordinates["x"].between(self.x_min, self.x_max) & coordinates["y"].between(
            self.y_min, self.y_max
        )

    @classmethod
    def around(cls, coordinates: pd.DataFrame, margin: float = 0.05) -> Viewport:
        """Viewport covering everything, with a small margin."""
        if coordinates.empty:
            return cls(-1.0, 1.0, -1.0, 1.0)
        x_min, x_max = float(coordinates["x"].min()), float(coordinates["x"].max())
        y_min, y_max = float(coordinates["y"].min()), float(coordinates["y"].max())
        pad_x = (x_max - x_min) * margin or 1.0
        pad_y = (y_max - y_min) * margin or 1.0
        return cls(x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y)


def visible(coordinates: pd.DataFrame, viewport: Viewport) -> pd.Index:
    """Record ids inside the viewport."""
    if coordinates.empty:
        return coordinates.index
    return coordinates.index[viewport.contains(coordinates)]


def should_aggregate(visible_count: int, threshold: int = DENSITY_THRESHOLD) -> bool:
    return visible_count > threshold


def tiles(
    coordinates: pd.DataFrame,
    selected_ids: pd.Index | None = None,
    resolution: int = 60,
) -> pd.DataFrame:
    """Grid of counts, with the share of each tile that is selected.

    The selected share is what turns a density plot into a coverage map: a
    dense tile with no selection is a region the final set ignores.
    """
    if coordinates.empty:
        return pd.DataFrame(columns=_TILE_COLUMNS)
    if resolution < 1:
        raise ValueError("resolution must be positive")

    x_values = coordinates["x"].to_numpy(dtype=float)
    y_values = coordinates["y"].to_numpy(dtype=float)
    x_edges = np.linspace(x_values.min(), x_values.max(), resolution + 1)
    y_edges = np.linspace(y_values.min(), y_values.max(), resolution + 1)

    x_index = np.clip(np.digitize(x_values, x_edges) - 1, 0, resolution - 1)
    y_index = np.clip(np.digitize(y_values, y_edges) - 1, 0, resolution - 1)

    frame = pd.DataFrame(
        {
            "x_index": x_index,
            "y_index": y_index,
            "selected": (
                coordinates.index.isin(selected_ids) if selected_ids is not None else False
            ),
        },
        index=coordinates.index,
    )
    grouped = frame.groupby(["x_index", "y_index"]).agg(
        molecules=("selected", "size"), selected=("selected", "sum")
    )
    if grouped.empty:
        return pd.DataFrame(columns=_TILE_COLUMNS)

    grouped = grouped.reset_index()
    centres_x = (x_edges[:-1] + x_edges[1:]) / 2
    centres_y = (y_edges[:-1] + y_edges[1:]) / 2
    grouped["x"] = centres_x[grouped["x_index"].to_numpy()]
    grouped["y"] = centres_y[grouped["y_index"].to_numpy()]
    grouped["selected_share"] = (grouped["selected"] / grouped["molecules"]).round(4)
    return grouped


def uncovered_regions(tile_frame: pd.DataFrame, min_molecules: int = 5) -> pd.DataFrame:
    """Populated tiles with no selected molecule - the gaps worth explaining."""
    if tile_frame.empty:
        return tile_frame
    gaps = tile_frame[(tile_frame["molecules"] >= min_molecules) & (tile_frame["selected"] == 0)]
    return gaps.sort_values("molecules", ascending=False)


def over_represented(tile_frame: pd.DataFrame, share: float = 0.5) -> pd.DataFrame:
    """Tiles where most of the molecules were selected."""
    if tile_frame.empty:
        return tile_frame
    return tile_frame[tile_frame["selected_share"] >= share].sort_values(
        "selected", ascending=False
    )


def lasso_contains(coordinates: pd.DataFrame, polygon: np.ndarray) -> pd.Index:
    """Record ids inside a lasso polygon.

    Ray casting, so the polygon may be concave - a lasso drawn by hand almost
    always is.
    """
    if coordinates.empty or len(polygon) < 3:
        return pd.Index([], name=coordinates.index.name)

    x_values = coordinates["x"].to_numpy(dtype=float)
    y_values = coordinates["y"].to_numpy(dtype=float)
    inside = np.zeros(len(coordinates), dtype=bool)

    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        straddles = (current_y > y_values) != (previous_y > y_values)
        with np.errstate(divide="ignore", invalid="ignore"):
            crossing_x = (previous_x - current_x) * (y_values - current_y) / (
                previous_y - current_y
            ) + current_x
        inside ^= straddles & (x_values < crossing_x)
        previous_x, previous_y = current_x, current_y

    return coordinates.index[inside]
