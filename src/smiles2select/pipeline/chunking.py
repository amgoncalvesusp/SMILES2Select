"""Molecule chunks: the unit of work handed to a worker, and how it is sized.

A chunk never carries an RDKit object, a DataFrame or anything beyond
primitives - a worker builds its own ``Mol`` from the SMILES string and
discards it when the chunk is done. Chunk boundaries become permanent the
moment a chunk is registered in the checkpoint (see ``checkpoint_store``): the
adaptive sizer only ever decides the size of the *next* chunk, never resizes
one already handed out.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

#: Bounds from the diagnostics spec: below 25, per-task overhead dominates;
#: above 2000, a single crash or slow record wastes too much recomputation.
DEFAULT_MIN_CHUNK_SIZE = 25
DEFAULT_MAX_CHUNK_SIZE = 2000
DEFAULT_INITIAL_CHUNK_SIZE = 500


@dataclass(frozen=True, slots=True)
class MoleculeChunk:
    """A slice of ``(record_id, smiles)`` pairs processed by one worker call."""

    chunk_id: int
    records: tuple[tuple[int, str], ...]

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("a chunk must hold at least one record")

    @property
    def first_input_order(self) -> int:
        return self.records[0][0]

    @property
    def last_input_order(self) -> int:
        return self.records[-1][0]

    def __len__(self) -> int:
        return len(self.records)


def build_chunks(
    pending: Sequence[tuple[int, str]], chunk_size: int, *, start_chunk_id: int = 0
) -> list[MoleculeChunk]:
    """Fixed-size chunking, used by the safe-diagnostics mode and by tests.

    The adaptive path in normal runs uses ``ChunkPlanner`` instead, which
    sizes each chunk lazily rather than all at once.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    return [
        MoleculeChunk(
            chunk_id=start_chunk_id + offset, records=tuple(pending[index : index + chunk_size])
        )
        for offset, index in enumerate(range(0, len(pending), chunk_size))
    ]


def split_chunk(chunk: MoleculeChunk) -> tuple[MoleculeChunk, MoleculeChunk]:
    """Bisect a chunk for crash isolation.

    The halves keep the parent's ``chunk_id`` for logging only - isolation
    sub-chunks are never registered in the checkpoint as first-class chunks,
    so there is no id collision to worry about.
    """
    if len(chunk.records) < 2:
        raise ValueError("cannot split a chunk with fewer than two records")
    middle = len(chunk.records) // 2
    left = MoleculeChunk(chunk_id=chunk.chunk_id, records=chunk.records[:middle])
    right = MoleculeChunk(chunk_id=chunk.chunk_id, records=chunk.records[middle:])
    return left, right


@dataclass(frozen=True)
class AdaptiveChunkSizer:
    """Pure sizing policy: given how the last chunk went, how big should the
    next one be? Never mutates anything itself - ``ChunkPlanner`` holds the
    running size and calls this on each observation.
    """

    minimum: int = DEFAULT_MIN_CHUNK_SIZE
    maximum: int = DEFAULT_MAX_CHUNK_SIZE
    fast_threshold_s: float = 0.25
    slow_threshold_s: float = 5.0
    memory_growth_bytes: int = 200 * 1024 * 1024

    def next_size(
        self, current_size: int, *, duration_s: float, rss_delta_bytes: int | None
    ) -> int:
        if rss_delta_bytes is not None and rss_delta_bytes > self.memory_growth_bytes:
            return self._clamp(current_size // 2)
        if duration_s > self.slow_threshold_s:
            return self._clamp(current_size // 2)
        if duration_s < self.fast_threshold_s:
            return self._clamp(int(current_size * 1.5) + 1)
        return self._clamp(current_size)

    def after_worker_crash(self, current_size: int) -> int:
        return self._clamp(current_size // 2)

    def _clamp(self, size: int) -> int:
        return max(self.minimum, min(self.maximum, size))


@dataclass
class ChunkPlanner:
    """Slices ``pending`` into chunks one at a time, sized by ``sizer``.

    Deliberately stateful (a running cursor and current size) rather than
    returning a new immutable planner per step: with up to half a million
    records to slice, an allocation per chunk for no behavioural gain is not
    worth it. Treat it like an iterator, not a value object.
    """

    pending: Sequence[tuple[int, str]]
    start_offset: int
    starting_chunk_id: int
    initial_size: int
    sizer: AdaptiveChunkSizer = field(default_factory=AdaptiveChunkSizer)

    def __post_init__(self) -> None:
        self._cursor = self.start_offset
        self._current_size = max(self.sizer.minimum, self.initial_size)
        self._next_id = self.starting_chunk_id

    def has_more(self) -> bool:
        return self._cursor < len(self.pending)

    def next_chunk(self) -> MoleculeChunk:
        size = max(1, self._current_size)
        records = tuple(self.pending[self._cursor : self._cursor + size])
        chunk = MoleculeChunk(chunk_id=self._next_id, records=records)
        self._cursor += len(records)
        self._next_id += 1
        return chunk

    def observe(self, *, duration_s: float, rss_delta_bytes: int | None) -> None:
        self._current_size = self.sizer.next_size(
            self._current_size, duration_s=duration_s, rss_delta_bytes=rss_delta_bytes
        )

    def shrink_after_crash(self) -> None:
        self._current_size = self.sizer.after_worker_crash(self._current_size)

    @property
    def current_size(self) -> int:
        return self._current_size
