"""Resilience of the parallel chemistry pipeline: chunking, memory-aware
worker sizing, checkpointed resume, and crash isolation.

The crash tests spawn real OS processes and kill them with ``os._exit`` -
that is the only way to exercise "a worker died mid-chunk" for real, since no
``try/except`` can catch a process actually terminating. They are slower than
the rest of the suite (subprocess start-up cost) but are the only tests that
prove the isolation code path actually contains a dead worker instead of
losing the whole run.
"""

from __future__ import annotations

import sqlite3

import pytest

from smiles2select.pipeline import resource_estimation
from smiles2select.pipeline._crash_simulation import POISON_SMILES, crash_on_poison
from smiles2select.pipeline.checkpoint_store import ChunkCheckpointStore
from smiles2select.pipeline.chunking import (
    AdaptiveChunkSizer,
    ChunkPlanner,
    MoleculeChunk,
    build_chunks,
    split_chunk,
)
from smiles2select.pipeline.crash_isolation import stream_chunks
from smiles2select.pipeline.diagnostics import log_memory_sample
from smiles2select.pipeline.streaming import run_streaming
from smiles2select.pipeline.workers import RecordResult

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


def test_build_chunks_covers_every_record_without_overlap():
    pending = [(i, f"C{i}") for i in range(1, 11)]
    chunks = build_chunks(pending, chunk_size=3)
    assert [len(chunk) for chunk in chunks] == [3, 3, 3, 1]
    covered = [record_id for chunk in chunks for record_id, _ in chunk.records]
    assert covered == list(range(1, 11))


def test_split_chunk_bisects_evenly_and_keeps_the_parent_chunk_id():
    chunk = MoleculeChunk(chunk_id=5, records=((1, "A"), (2, "B"), (3, "C"), (4, "D")))
    left, right = split_chunk(chunk)
    assert left.records == ((1, "A"), (2, "B"))
    assert right.records == ((3, "C"), (4, "D"))
    assert left.chunk_id == right.chunk_id == 5


def test_split_chunk_rejects_a_single_record():
    with pytest.raises(ValueError, match="fewer than two"):
        split_chunk(MoleculeChunk(chunk_id=0, records=((1, "A"),)))


def test_sizer_grows_on_a_fast_chunk_and_shrinks_on_a_slow_one():
    sizer = AdaptiveChunkSizer(minimum=10, maximum=1000)
    assert sizer.next_size(100, duration_s=0.05, rss_delta_bytes=None) > 100
    assert sizer.next_size(100, duration_s=10.0, rss_delta_bytes=None) < 100
    assert sizer.next_size(100, duration_s=1.0, rss_delta_bytes=None) == 100


def test_sizer_shrinks_on_memory_growth_even_if_the_chunk_was_fast():
    sizer = AdaptiveChunkSizer(minimum=10, maximum=1000, memory_growth_bytes=1_000)
    assert sizer.next_size(100, duration_s=0.01, rss_delta_bytes=2_000) == 50


def test_sizer_never_exceeds_its_bounds():
    sizer = AdaptiveChunkSizer(minimum=10, maximum=20)
    assert sizer.next_size(20, duration_s=0.01, rss_delta_bytes=None) == 20
    assert sizer.next_size(10, duration_s=10.0, rss_delta_bytes=None) == 10


def test_sizer_after_worker_crash_halves_and_respects_the_floor():
    sizer = AdaptiveChunkSizer(minimum=10, maximum=1000)
    assert sizer.after_worker_crash(100) == 50
    assert sizer.after_worker_crash(15) == 10


def test_chunk_planner_slices_lazily_and_advances_ids():
    pending = [(i, f"C{i}") for i in range(1, 21)]
    planner = ChunkPlanner(
        pending=pending,
        start_offset=0,
        starting_chunk_id=0,
        initial_size=5,
        sizer=AdaptiveChunkSizer(minimum=2, maximum=50),
    )
    first = planner.next_chunk()
    assert first.chunk_id == 0
    assert [r for r, _ in first.records] == list(range(1, 6))

    planner.observe(duration_s=0.01, rss_delta_bytes=None)  # fast: next chunk grows
    second = planner.next_chunk()
    assert second.chunk_id == 1
    assert len(second) > 5


def test_chunk_planner_resumes_from_an_offset_without_reslicing_earlier_records():
    pending = [(i, f"C{i}") for i in range(1, 21)]
    planner = ChunkPlanner(
        pending=pending,
        start_offset=12,
        starting_chunk_id=3,
        initial_size=4,
        sizer=AdaptiveChunkSizer(),
    )
    chunk = planner.next_chunk()
    assert chunk.chunk_id == 3
    assert chunk.first_input_order == 13  # pending is 0-indexed; record_id 13 sits at offset 12


def test_chunk_planner_exhausts_exactly_at_the_end():
    pending = [(1, "A"), (2, "B"), (3, "C")]
    # minimum=1 here: the real default (25) would swallow all 3 records into
    # one chunk, which is correct in production but defeats this test's point.
    planner = ChunkPlanner(
        pending=pending,
        start_offset=0,
        starting_chunk_id=0,
        initial_size=2,
        sizer=AdaptiveChunkSizer(minimum=1, maximum=50),
    )
    planner.next_chunk()
    assert planner.has_more()
    planner.next_chunk()
    assert not planner.has_more()


# --------------------------------------------------------------------------
# Memory-aware worker count
# --------------------------------------------------------------------------


def test_estimate_worker_count_never_returns_every_core_by_default(monkeypatch):
    monkeypatch.setattr(resource_estimation, "physical_cpu_count", lambda: 64)
    if resource_estimation.psutil is not None:
        monkeypatch.setattr(
            resource_estimation.psutil,
            "virtual_memory",
            lambda: type(
                "VM",
                (),
                {
                    "total": 256 * resource_estimation._GIB,
                    "available": 200 * resource_estimation._GIB,
                },
            )(),
        )
    workers = resource_estimation.estimate_worker_count(None)
    assert workers < 64  # never "every core"


def test_estimate_worker_count_respects_an_explicit_user_limit(monkeypatch):
    monkeypatch.setattr(resource_estimation, "physical_cpu_count", lambda: 32)
    if resource_estimation.psutil is not None:
        monkeypatch.setattr(
            resource_estimation.psutil,
            "virtual_memory",
            lambda: type(
                "VM",
                (),
                {
                    "total": 256 * resource_estimation._GIB,
                    "available": 200 * resource_estimation._GIB,
                },
            )(),
        )
    assert resource_estimation.estimate_worker_count(3) == 3


def test_estimate_worker_count_falls_back_when_psutil_is_unavailable(monkeypatch):
    monkeypatch.setattr(resource_estimation, "psutil", None)
    monkeypatch.setattr(resource_estimation, "physical_cpu_count", lambda: 16)
    assert resource_estimation.estimate_worker_count(None) == resource_estimation._RAM_TABLE_FLOOR


def test_estimate_worker_count_is_at_least_one_even_on_a_single_core_low_memory_box(monkeypatch):
    monkeypatch.setattr(resource_estimation, "physical_cpu_count", lambda: 1)
    monkeypatch.setattr(resource_estimation, "psutil", None)
    assert resource_estimation.estimate_worker_count(None) >= 1


# --------------------------------------------------------------------------
# Checkpoint store
# --------------------------------------------------------------------------


def _stub_result(record_id: int) -> RecordResult:
    return RecordResult(
        record_id=record_id,
        valid=True,
        invalid_reason=None,
        standardized_smiles=f"C{record_id}",
        canonical_smiles=f"C{record_id}",
        descriptors={"mol_wt": float(record_id)},
        substructure_flags={},
        alert_rows=[],
    )


def test_checkpoint_commit_is_visible_after_reopen_with_the_same_signature(tmp_path):
    path = tmp_path / "run.checkpoint.sqlite"
    chunk = MoleculeChunk(chunk_id=0, records=((1, "A"), (2, "B")))

    store = ChunkCheckpointStore(path, input_hash="h1", config_hash="c1")
    store.register_chunks([chunk])
    store.commit_chunk(0, [_stub_result(1), _stub_result(2)])
    store.close()

    resumed = ChunkCheckpointStore(path, input_hash="h1", config_hash="c1")
    assert resumed.planned_record_count() == 2
    assert resumed.completed_chunk_count() == 1
    assert {r.record_id for r in resumed.completed_results()} == {1, 2}
    resumed.close()


def test_checkpoint_starts_fresh_when_the_input_hash_changes(tmp_path):
    path = tmp_path / "run.checkpoint.sqlite"
    chunk = MoleculeChunk(chunk_id=0, records=((1, "A"),))

    store = ChunkCheckpointStore(path, input_hash="h1", config_hash="c1")
    store.register_chunks([chunk])
    store.commit_chunk(0, [_stub_result(1)])
    store.close()

    different_input = ChunkCheckpointStore(path, input_hash="h2", config_hash="c1")
    assert different_input.planned_record_count() == 0
    assert different_input.completed_results() == []
    different_input.close()


def test_checkpoint_starts_fresh_when_the_config_hash_changes(tmp_path):
    path = tmp_path / "run.checkpoint.sqlite"
    chunk = MoleculeChunk(chunk_id=0, records=((1, "A"),))

    store = ChunkCheckpointStore(path, input_hash="h1", config_hash="c1")
    store.register_chunks([chunk])
    store.commit_chunk(0, [_stub_result(1)])
    store.close()

    different_config = ChunkCheckpointStore(path, input_hash="h1", config_hash="c2")
    assert different_config.planned_record_count() == 0
    different_config.close()


def test_checkpoint_prunes_a_registered_but_never_committed_chunk(tmp_path):
    """Simulates a crash between register_chunks and commit_chunk: the row
    exists but was never marked COMPLETED, so it must not count as done."""
    path = tmp_path / "run.checkpoint.sqlite"
    chunk = MoleculeChunk(chunk_id=0, records=((1, "A"), (2, "B")))

    store = ChunkCheckpointStore(path, input_hash="h1", config_hash="c1")
    store.register_chunks([chunk])  # never committed - process "died" here
    store.close()

    resumed = ChunkCheckpointStore(path, input_hash="h1", config_hash="c1")
    assert resumed.planned_record_count() == 0
    assert resumed.next_chunk_id() == 0  # chunk id 0 is free to reuse
    resumed.close()


def test_checkpoint_refuses_to_delete_a_file_that_is_not_its_own(tmp_path):
    path = tmp_path / "not_a_checkpoint.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE something_else (x INTEGER)")
    connection.commit()
    connection.close()

    with pytest.raises(ValueError, match="refusing to overwrite"):
        ChunkCheckpointStore(path, input_hash="h1", config_hash="c1")


# --------------------------------------------------------------------------
# Diagnostics logging (called directly - safe because it never touches
# sys.stderr; that redirect only happens inside initialize_worker, which
# real worker processes call, never the test process itself)
# --------------------------------------------------------------------------


def test_log_memory_sample_writes_one_json_line(tmp_path):
    import json

    log_dir = tmp_path / "logs"
    log_memory_sample(str(log_dir), chunk_id=7, phase="chunk_complete", record_count=3)
    files = list(log_dir.glob("worker_*.memory.log"))
    assert len(files) == 1
    line = files[0].read_text(encoding="utf-8").strip()
    payload = json.loads(line)
    assert payload["chunk_id"] == 7
    assert payload["record_count"] == 3
    assert "rss_bytes" in payload


# --------------------------------------------------------------------------
# Real crash isolation - spawns actual OS processes
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_stream_chunks_isolates_a_crashing_record_and_keeps_the_rest(tmp_path):
    pending = [(1, "A"), (2, "B"), (3, POISON_SMILES), (4, "D"), (5, "E")]
    chunks = build_chunks(pending, chunk_size=2)
    log_dir = tmp_path / "logs"

    outcomes = list(stream_chunks(chunks, crash_on_poison, n_jobs=2, log_directory=str(log_dir)))
    all_results = [result for outcome in outcomes for result in outcome.results]

    assert len(all_results) == 5  # nothing lost - the run continued past the crash
    crashed = [result for result in all_results if result.error_code == "WORKER_NATIVE_CRASH"]
    assert [result.record_id for result in crashed] == [3]
    ok_ids = {result.record_id for result in all_results if result.error_code is None}
    assert ok_ids == {1, 2, 4, 5}
    assert any(outcome.isolated for outcome in outcomes)  # the crashing chunk was bisected
    assert list(log_dir.glob("worker_*.fault.log"))  # a real worker ran initialize_worker


@pytest.mark.slow
def test_run_streaming_checkpoints_survive_a_crashing_record(tmp_path):
    """End-to-end: run_streaming over a real crash, then confirm every
    non-poison record is present exactly once in the final report."""
    pending = (
        [(i, f"C{i}") for i in range(1, 5)]
        + [(5, POISON_SMILES)]
        + [(i, f"C{i}") for i in range(6, 9)]
    )
    checkpoint = ChunkCheckpointStore(
        tmp_path / "run.checkpoint.sqlite", input_hash="h", config_hash="c"
    )
    report_events = []
    try:
        report = run_streaming(
            pending,
            crash_on_poison,
            checkpoint=checkpoint,
            n_jobs=2,
            initial_chunk_size=2,
            log_directory=str(tmp_path / "logs"),
            log_memory=True,
            report=lambda done, total, stage: report_events.append((done, total)),
            already_done=0,
            total=len(pending),
        )
    finally:
        checkpoint.close()

    assert len(report.results) == 8
    assert report.crashed_record_count == 1
    ids = {result.record_id for result in report.results}
    assert ids == {1, 2, 3, 4, 5, 6, 7, 8}
    crashed = [result for result in report.results if result.error_code == "WORKER_NATIVE_CRASH"]
    assert [result.record_id for result in crashed] == [5]
    assert report_events  # progress was reported incrementally, not just once at the end


def test_run_streaming_resumes_after_an_unrelated_failure_without_duplicating_work(tmp_path):
    """Not a process crash - an ordinary exception raised partway through, to
    prove the checkpoint alone (not the crash-isolation path) is what
    prevents lost or duplicated results on a second attempt."""
    # 30 records: with the sizer's real floor of 25, the first chunk swallows
    # 25 of them in one call, leaving a second chunk (and a second worker
    # call) for the remainder - which is where this test makes it fail.
    pending = [(i, f"C{i}") for i in range(1, 31)]
    calls = {"count": 0}

    def flaky_run_chunk(chunk):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("simulated interruption")
        return [_stub_result(record_id) for record_id, _ in chunk.records]

    def stable_run_chunk(chunk):
        return [_stub_result(record_id) for record_id, _ in chunk.records]

    checkpoint_path = tmp_path / "run.checkpoint.sqlite"
    log_dir = str(tmp_path / "logs")

    checkpoint = ChunkCheckpointStore(checkpoint_path, input_hash="h", config_hash="c")
    with pytest.raises(RuntimeError):
        run_streaming(
            pending,
            flaky_run_chunk,
            checkpoint=checkpoint,
            n_jobs=1,
            initial_chunk_size=2,
            log_directory=log_dir,
            log_memory=False,
            report=lambda *a: None,
            already_done=0,
            total=len(pending),
        )
    checkpoint.close()
    assert calls["count"] == 2  # confirms the crash landed where this test expects

    resumed_checkpoint = ChunkCheckpointStore(checkpoint_path, input_hash="h", config_hash="c")
    report = run_streaming(
        pending,
        stable_run_chunk,
        checkpoint=resumed_checkpoint,
        n_jobs=1,
        initial_chunk_size=2,
        log_directory=log_dir,
        log_memory=False,
        report=lambda *a: None,
        already_done=0,
        total=len(pending),
    )
    resumed_checkpoint.close()

    assert len(report.results) == 30
    assert {r.record_id for r in report.results} == set(range(1, 31))
