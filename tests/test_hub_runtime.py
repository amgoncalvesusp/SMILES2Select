"""Runtime contracts for cancellation and progressive Hub rendering."""

from __future__ import annotations

import pandas as pd
import pytest

from smiles2select.chemical_space.layers import progressive_layer
from smiles2select.pipeline.cancellation import CancellationToken, RunCancelled


def test_cancellation_token_is_cooperative_and_idempotent():
    token = CancellationToken()
    assert not token.cancelled
    token.cancel()
    token.cancel()
    assert token.cancelled
    with pytest.raises(RunCancelled):
        token.raise_if_cancelled()


def test_progressive_layer_preserves_population_and_builds_density_tiles():
    coordinates = pd.DataFrame(
        {"x": range(25), "y": range(25)}, index=pd.Index(range(25), name="record_id")
    )
    layer = progressive_layer(
        coordinates,
        selected_ids=pd.Index([0, 1]),
        max_points=5,
        density_threshold=10,
        resolution=5,
    )
    assert layer.total_points == 25
    assert len(layer.points) == 5
    assert layer.aggregated
    assert not layer.density.empty
