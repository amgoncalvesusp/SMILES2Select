"""Display-layer policies for the Chemical Space Hub."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from smiles2select.chemical_space.density_tiles import tiles


@dataclass(frozen=True)
class ProgressiveLayer:
    """A bounded context layer preserving all finals, plus density tiles."""

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

    Context is sampled by stable row order when zoomed out; every selected
    molecule is retained even if the selection exceeds max_points. The complete
    population remains available for lasso/coverage calculations.  Above the
    threshold, density tiles are supplied as a second, aggregate layer.
    """
    if max_points < 1 or density_threshold < 1:
        raise ValueError("max_points and density_threshold must be positive")
    if coordinates.empty:
        return ProgressiveLayer(coordinates.copy(), pd.DataFrame(), False, 0)
    selected_mask = coordinates.index.isin(selected_ids if selected_ids is not None else [])
    final = coordinates.loc[selected_mask]
    context = coordinates.loc[~selected_mask].iloc[: max(0, max_points - len(final))]
    retained = final.index.union(context.index, sort=False)
    display = coordinates.loc[coordinates.index.isin(retained)].copy()
    aggregated = len(coordinates) > density_threshold
    density = tiles(coordinates, selected_ids, resolution) if aggregated else pd.DataFrame()
    return ProgressiveLayer(display, density, aggregated, len(coordinates))
