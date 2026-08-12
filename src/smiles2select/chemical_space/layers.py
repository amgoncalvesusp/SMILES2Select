"""Display-layer policies for the Chemical Space Hub."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from smiles2select.chemical_space.density_tiles import tiles


@dataclass(frozen=True)
class ProgressiveLayer:
    """A bounded point layer plus optional density tiles."""

    points: pd.DataFrame
    density: pd.DataFrame
    aggregated: bool
    total_points: int


def progressive_layer(
    coordinates: pd.DataFrame,
    *,
    selected_ids: pd.Index | None = None,
    max_points: int = 20_000,
    density_threshold: int = 20_000,
    resolution: int = 60,
) -> ProgressiveLayer:
    """Return deterministic display data without changing scientific data.

    Points are sampled by stable row order when zoomed out; the complete
    population remains available for lasso/coverage calculations.  Above the
    threshold, density tiles are supplied as a second, aggregate layer.
    """
    if max_points < 1 or density_threshold < 1:
        raise ValueError("max_points and density_threshold must be positive")
    if coordinates.empty:
        return ProgressiveLayer(coordinates.copy(), pd.DataFrame(), False, 0)
    display = coordinates.iloc[:max_points].copy()
    aggregated = len(coordinates) > density_threshold
    density = tiles(coordinates, selected_ids, resolution) if aggregated else pd.DataFrame()
    return ProgressiveLayer(display, density, aggregated, len(coordinates))
