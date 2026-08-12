"""Ties the pure pieces together into the actual per-run execution loop.

Plan a window of chunks (at most ``n_jobs``), run it with crash isolation,
commit each finished chunk to the checkpoint before asking the planner for
more. This is what bounds memory (only one window's worth of chunks is ever
in flight), guarantees no completed work is lost (each chunk is durable
before the next window is planned), and lets the adaptive sizer react
(``observe``/``shrink_after_crash``) between windows.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from smiles2select.pipeline.checkpoint_store import ChunkCheckpointStore
from smiles2select.pipeline.chunking import AdaptiveChunkSizer, ChunkPlanner, MoleculeChunk
from smiles2select.pipeline.crash_isolation import stream_chunks
from smiles2select.pipeline.diagnostics import log_memory_sample
from smiles2select.pipeline.workers import RecordResult

ProgressCallback = Callable[[int, int, str], None]
ChunkRunner = Callable[[MoleculeChunk], list[RecordResult]]


@dataclass(frozen=True)
class StreamingReport:
    results: list[RecordResult]
    isolated_chunk_count: int
    crashed_record_count: int


def run_streaming(
    pending: Sequence[tuple[int, str]],
    run_chunk: ChunkRunner,
    *,
    checkpoint: ChunkCheckpointStore,
    n_jobs: int,
    initial_chunk_size: int,
    log_directory: str,
    log_memory: bool,
    report: ProgressCallback,
    already_done: int,
    total: int,
) -> StreamingReport:
    planner = ChunkPlanner(
        pending=pending,
        start_offset=checkpoint.planned_record_count(),
        starting_chunk_id=checkpoint.next_chunk_id(),
        initial_size=checkpoint.last_chunk_size() or initial_chunk_size,
        sizer=AdaptiveChunkSizer(),
    )

    results = list(checkpoint.completed_results())
    done_count = already_done + len(results)
    isolated_chunks = 0
    crashed_records = 0
    report(done_count, total, "Computing descriptors")

    while planner.has_more():
        window: list[MoleculeChunk] = []
        while planner.has_more() and len(window) < max(1, n_jobs):
            window.append(planner.next_chunk())
        checkpoint.register_chunks(window)

        for outcome in stream_chunks(window, run_chunk, n_jobs=n_jobs, log_directory=log_directory):
            checkpoint.commit_chunk(outcome.chunk_id, outcome.results)
            results.extend(outcome.results)
            done_count += len(outcome.results)
            report(done_count, total, "Computing descriptors")

            if outcome.isolated:
                checkpoint.mark_isolating(outcome.chunk_id)
                isolated_chunks += 1
                crashed_records += sum(
                    1 for result in outcome.results if result.error_code == "WORKER_NATIVE_CRASH"
                )
                planner.shrink_after_crash()
            else:
                # Memory-growth-triggered shrinking is intentionally not wired
                # in here: worker RSS is only observable from inside the
                # worker (see diagnostics.log_memory_sample below), and this
                # loop runs in the parent. Timing and crash signals already
                # drive the sizer; a future pass could feed the logged RSS
                # samples back in as a third signal.
                planner.observe(duration_s=outcome.duration_s, rss_delta_bytes=None)

            if log_memory:
                log_memory_sample(
                    log_directory,
                    chunk_id=outcome.chunk_id,
                    phase="chunk_complete",
                    record_count=len(outcome.results),
                )

    return StreamingReport(
        results=results, isolated_chunk_count=isolated_chunks, crashed_record_count=crashed_records
    )
