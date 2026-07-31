"""Crash-tolerant chunk execution.

A worker dying (segfault, OOM-killer, native abort) surfaces to the parent as
``BrokenProcessPool`` - joblib's loky backend raises the same family for
exactly this reason. When it happens mid-stream, only the chunks that were
in flight in that batch are affected: everything already queued behind them
still runs, and the batch that died gets bisected down to the one record that
kills the worker, which is reported instead of silently dropped.

This is what keeps one bad molecule (or one moment of memory pressure) from
ending a run over hundreds of thousands of records.
"""

from __future__ import annotations

import multiprocessing
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass

from joblib import Parallel, delayed, parallel_config

from smiles2select.pipeline.chunking import MoleculeChunk, split_chunk
from smiles2select.pipeline.diagnostics import initialize_worker
from smiles2select.pipeline.workers import RecordResult

#: What a dead worker looks like from the parent's side. loky's
#: TerminatedWorkerError subclasses this for compatibility with stdlib
#: ProcessPoolExecutor, so catching the base class covers both the streaming
#: path (loky, via joblib) and the isolation path (ProcessPoolExecutor).
_WORKER_DEATH_EXCEPTIONS = (BrokenProcessPool,)

#: 2**12 single-record isolation attempts is far past any real chunk size;
#: this is a backstop against runaway recursion, not a realistic depth.
_MAX_ISOLATION_DEPTH = 12

ChunkRunner = Callable[[MoleculeChunk], list[RecordResult]]


@dataclass(frozen=True)
class ChunkOutcome:
    chunk_id: int
    results: tuple[RecordResult, ...]
    duration_s: float
    isolated: bool = False


def stream_chunks(
    chunks: Sequence[MoleculeChunk],
    run_chunk: ChunkRunner,
    *,
    n_jobs: int,
    log_directory: str,
) -> Iterator[ChunkOutcome]:
    """Run every chunk, yielding each as soon as it finishes.

    ``chunks`` is one window (at most ``n_jobs`` chunks) - the caller is
    responsible for pulling the next window only after this one drains, which
    is what keeps memory bounded: at most ``n_jobs`` chunks are ever in
    flight, matching ``pre_dispatch``.
    """
    remaining = list(chunks)
    effective_jobs = max(1, n_jobs)

    while remaining:
        batch = remaining[:effective_jobs]
        rest = remaining[effective_jobs:]
        done_ids: set[int] = set()

        try:
            with parallel_config(
                backend="loky",
                n_jobs=effective_jobs,
                inner_max_num_threads=1,
                initializer=initialize_worker,
                initargs=(log_directory,),
            ):
                stream = Parallel(
                    return_as="generator_unordered", pre_dispatch=effective_jobs, batch_size=1
                )(delayed(_timed_run)(run_chunk, chunk) for chunk in batch)
                for chunk_id, results, duration in stream:
                    done_ids.add(chunk_id)
                    yield ChunkOutcome(
                        chunk_id=chunk_id, results=tuple(results), duration_s=duration
                    )
        except _WORKER_DEATH_EXCEPTIONS:
            pass  # whichever chunks in `batch` never reached done_ids get isolated below

        for chunk in batch:
            if chunk.chunk_id in done_ids:
                continue
            start = time.monotonic()
            isolated_results = _isolate_chunk(chunk, run_chunk, log_directory)
            yield ChunkOutcome(
                chunk_id=chunk.chunk_id,
                results=tuple(isolated_results),
                duration_s=time.monotonic() - start,
                isolated=True,
            )

        remaining = rest


def _timed_run(
    run_chunk: ChunkRunner, chunk: MoleculeChunk
) -> tuple[int, list[RecordResult], float]:
    start = time.monotonic()
    results = run_chunk(chunk)
    return chunk.chunk_id, results, time.monotonic() - start


def _isolate_chunk(
    chunk: MoleculeChunk, run_chunk: ChunkRunner, log_directory: str, *, depth: int = 0
) -> list[RecordResult]:
    """Bisect a crashing chunk down to the one record that kills the worker.

    Retries the whole (still >1 record) chunk once, alone, before bisecting:
    if that retry succeeds, the original crash was pressure from running
    several chunks at once, not a specific poison molecule, and bisecting
    further would just be wasted subprocess spawns.
    """
    if len(chunk.records) == 1:
        return _run_isolated(chunk, run_chunk, log_directory)
    if depth >= _MAX_ISOLATION_DEPTH:
        return _native_crash_results(chunk)

    try:
        return _run_isolated_or_raise(chunk, run_chunk, log_directory)
    except _WORKER_DEATH_EXCEPTIONS:
        left, right = split_chunk(chunk)
        return _isolate_chunk(left, run_chunk, log_directory, depth=depth + 1) + _isolate_chunk(
            right, run_chunk, log_directory, depth=depth + 1
        )


def _run_isolated(
    chunk: MoleculeChunk, run_chunk: ChunkRunner, log_directory: str
) -> list[RecordResult]:
    try:
        return _run_isolated_or_raise(chunk, run_chunk, log_directory)
    except _WORKER_DEATH_EXCEPTIONS:
        return _native_crash_results(chunk)


def _run_isolated_or_raise(
    chunk: MoleculeChunk, run_chunk: ChunkRunner, log_directory: str
) -> list[RecordResult]:
    """Run one chunk in a single, guaranteed-fresh subprocess.

    Uses ``ProcessPoolExecutor`` directly rather than joblib: joblib's
    ``n_jobs=1`` can fall back to running in-process for simple cases, which
    would mean a segfault here kills the orchestrator itself instead of being
    contained. A dedicated one-worker pool per attempt is slower but actually
    isolates the crash, which is the entire point of this function.
    """
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=1, mp_context=context, initializer=initialize_worker, initargs=(log_directory,)
    ) as executor:
        future = executor.submit(run_chunk, chunk)
        return future.result()


def _native_crash_results(chunk: MoleculeChunk) -> list[RecordResult]:
    return [
        RecordResult(
            record_id=record_id,
            valid=False,
            invalid_reason="processing this structure terminated a worker process",
            standardized_smiles="",
            canonical_smiles="",
            descriptors={},
            substructure_flags={},
            alert_rows=[],
            error_code="WORKER_NATIVE_CRASH",
        )
        for record_id, _ in chunk.records
    ]
