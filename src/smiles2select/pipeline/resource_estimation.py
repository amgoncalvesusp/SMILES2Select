"""How many workers to launch, bounded by memory rather than just CPU count.

``n_jobs=-1`` hands joblib every core with no regard for RAM: with several
worker processes each holding RDKit and its own copy of the chunk in flight,
that is exactly the setting that produces OOM under load. This module is what
``RunConfig``'s default (auto) resolves through instead.
"""

from __future__ import annotations

import os

try:
    import psutil
except ImportError:  # pragma: no cover - exercised only in environments missing psutil
    psutil = None  # type: ignore[assignment]

_GIB = 1024**3

#: Conservative Windows defaults from the diagnostics spec, keyed by total RAM
#: in GiB. Used only when there is no measured per-worker memory estimate to
#: divide by; these are floors to start from, not limits to enforce forever.
_RAM_TABLE = (
    (8, 2),
    (16, 4),
    (32, 6),
    (64, 8),
)
_RAM_TABLE_FLOOR = 2


def physical_cpu_count() -> int:
    if psutil is not None:
        count = psutil.cpu_count(logical=False)
        if count:
            return int(count)
    return os.cpu_count() or 1


def _ram_table_cap(total_bytes: int) -> int:
    total_gib = total_bytes / _GIB
    cap = _RAM_TABLE_FLOOR
    for threshold, workers in _RAM_TABLE:
        if total_gib >= threshold:
            cap = workers
    return cap


def estimate_worker_count(
    user_limit: int | None = None,
    *,
    reserve_bytes: int | None = None,
    estimated_peak_per_worker_bytes: int | None = None,
) -> int:
    """Workers to launch, bounded by CPU count, available RAM, and the caller's
    own limit. Never returns joblib's "all cores" semantics: the memory cap
    always participates in the ``min()``.

    ``reserve_bytes`` defaults to the spec's floor: the larger of 2 GiB or 25%
    of total RAM, held back for the OS, the GUI, SQLite and the export step.
    """
    cpu_cap = max(1, physical_cpu_count() - 1)

    if psutil is not None:
        memory = psutil.virtual_memory()
        reserve = (
            reserve_bytes if reserve_bytes is not None else max(2 * _GIB, int(0.25 * memory.total))
        )
        usable = max(0, memory.available - reserve)
        if estimated_peak_per_worker_bytes:
            memory_cap = max(1, usable // estimated_peak_per_worker_bytes)
        else:
            memory_cap = _ram_table_cap(memory.total)
    else:
        # psutil missing: fall back to the smallest table tier rather than
        # guessing at a memory figure we cannot measure.
        memory_cap = _RAM_TABLE_FLOOR

    candidates = [cpu_cap, memory_cap]
    if user_limit is not None and user_limit > 0:
        candidates.append(user_limit)
    return max(1, min(candidates))
