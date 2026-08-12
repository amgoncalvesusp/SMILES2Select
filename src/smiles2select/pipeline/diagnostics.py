"""Per-worker fault and memory diagnostics.

A ``try/except`` inside a worker cannot catch a segfault - the process just
dies. ``faulthandler`` is the only thing that leaves a trace when that
happens, and it has to be armed *inside every worker*, not just the main
process, which is why ``initialize_worker`` exists as a joblib/loky
initializer rather than something set up once at import time.
"""

from __future__ import annotations

import faulthandler
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

try:
    import psutil
except ImportError:  # pragma: no cover - exercised only in environments missing psutil
    psutil = None  # type: ignore[assignment]

#: Kept open for the worker's lifetime - faulthandler writes to this file
#: descriptor even after the frame that opened it has returned, so closing it
#: early would make the crash log silently empty.
_worker_fault_log: TextIO | None = None


@dataclass(frozen=True)
class ParallelDiagnosticsConfig:
    """Safe-diagnostics mode: force the conservative settings used to reproduce
    a failure deterministically instead of chasing it under full parallelism.
    """

    enabled: bool = False
    force_sequential: bool = False
    worker_count: int | None = None
    chunk_size: int = 100
    log_memory: bool = True
    enable_faulthandler: bool = True
    keep_worker_logs: bool = True
    log_directory: Path | None = None

    def resolved_n_jobs(self, requested: int) -> int:
        if self.force_sequential:
            return 1
        if self.worker_count is not None:
            return max(1, self.worker_count)
        return requested

    def resolved_chunk_size(self, requested: int) -> int:
        return self.chunk_size if self.enabled else requested


def initialize_worker(log_directory: str) -> None:
    """Runs once per worker process (loky ``initializer``): faulthandler + a
    per-PID log file that also catches anything the worker prints to stderr.
    """
    global _worker_fault_log
    directory = Path(log_directory)
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / f"worker_{os.getpid()}.fault.log"
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")
    _worker_fault_log = log_file
    sys.stderr = log_file
    faulthandler.enable(file=log_file, all_threads=True)
    print(f"worker initialized: pid={os.getpid()}", file=log_file, flush=True)


def worker_rss_bytes() -> int | None:
    """Current process RSS, or ``None`` when psutil is unavailable.

    Not ``tracemalloc``: it only tracks Python-level allocations and misses
    the native memory RDKit and its dependencies hold, which is most of it.
    """
    if psutil is None:
        return None
    return int(psutil.Process(os.getpid()).memory_info().rss)


def system_available_bytes() -> int | None:
    if psutil is None:
        return None
    return int(psutil.virtual_memory().available)


def log_memory_sample(
    log_directory: str,
    *,
    chunk_id: int,
    phase: str,
    record_count: int,
    extra: dict[str, Any] | None = None,
) -> None:
    """Append one JSON line of memory telemetry, keyed by this process's PID."""
    directory = Path(log_directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"worker_{os.getpid()}.memory.log"
    sample = {
        "pid": os.getpid(),
        "chunk_id": chunk_id,
        "phase": phase,
        "record_count": record_count,
        "rss_bytes": worker_rss_bytes(),
        "available_bytes": system_available_bytes(),
    }
    if extra:
        sample.update(extra)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


def default_diagnostics_config() -> ParallelDiagnosticsConfig:
    return ParallelDiagnosticsConfig()


def safe_mode_config(log_directory: Path | str | None = None) -> ParallelDiagnosticsConfig:
    """The diagnostic-safe preset: n_jobs=1, small chunks, everything logged."""
    return ParallelDiagnosticsConfig(
        enabled=True,
        force_sequential=True,
        chunk_size=100,
        log_memory=True,
        enable_faulthandler=True,
        keep_worker_logs=True,
        log_directory=Path(log_directory) if log_directory is not None else None,
    )
