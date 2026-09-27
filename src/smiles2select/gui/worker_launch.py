"""Launch Decision CLI jobs from source installs and frozen desktop bundles."""

from __future__ import annotations

import sys
from collections.abc import Sequence

_WORKER_FLAG = "--s2s-worker"
_DECISION_MODULE = "s2s_decision"
_TRAINING_COMMANDS = frozenset({"train", "train-baseline", "benchmark"})


def worker_command(
    module: str = _DECISION_MODULE,
    args: Sequence[str] = (),
    *,
    python_executable: str | None = None,
) -> tuple[str, list[str]]:
    """Return QProcess program and arguments for an installed Decision CLI."""
    if module != _DECISION_MODULE:
        raise ValueError(f"Unsupported worker module: {module}")
    if python_executable is not None:
        if not python_executable.strip():
            raise ValueError("Training Python executable is empty")
        return python_executable, ["-m", module, *args]
    if getattr(sys, "frozen", False):
        if args and args[0] in _TRAINING_COMMANDS:
            raise RuntimeError("Training requires a configured Python runtime outside the desktop bundle")
        return sys.executable, [_WORKER_FLAG, "-m", module, *args]
    return sys.executable, ["-m", module, *args]


def worker_dispatch(argv: Sequence[str]) -> int | None:
    """Handle frozen worker mode before importing Qt or constructing a window."""
    if not argv or argv[0] != _WORKER_FLAG:
        return None
    if len(argv) < 3 or tuple(argv[1:3]) != ("-m", _DECISION_MODULE):
        print("ERROR: invalid Decision worker command", file=sys.stderr)
        return 2
    if getattr(sys, "frozen", False) and len(argv) > 3 and argv[3] in _TRAINING_COMMANDS:
        print("ERROR: training requires a configured external Python runtime", file=sys.stderr)
        return 2
    from s2s_decision.cli import main

    return main(list(argv[3:]))
