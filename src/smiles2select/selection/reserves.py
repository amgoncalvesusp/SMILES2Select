"""Reserve-set helpers kept separate from final selection decisions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd

from smiles2select.chemical_space.zones import SelectionZone
from smiles2select.selection.zone_allocator import ZoneAllocationResult, allocate_zones


def select_final_and_reserve(
    candidates: pd.DataFrame | Iterable[object],
    zones: Iterable[SelectionZone],
    *,
    final_count: int,
    reserve_count: int,
    ranking: Mapping[object, float] | None = None,
) -> ZoneAllocationResult:
    """Convenience API for the Hub's two-output selection contract."""

    return allocate_zones(
        candidates,
        zones,
        final_count=final_count,
        reserve_count=reserve_count,
        ranking=ranking,
    )
