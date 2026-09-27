"""The packaged application must run Decision jobs without starting another GUI."""

from __future__ import annotations

import sys

import pytest

from smiles2select.gui import app
from smiles2select.gui.worker_launch import worker_command, worker_dispatch


def test_source_worker_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert worker_command("s2s_decision", ["models", "--output", "result.json"]) == (
        sys.executable,
        ["-m", "s2s_decision", "models", "--output", "result.json"],
    )


def test_frozen_worker_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert worker_command("s2s_decision", ["preview"]) == (
        sys.executable,
        ["--s2s-worker", "-m", "s2s_decision", "preview"],
    )


def test_external_training_runtime_stays_python_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert worker_command("s2s_decision", ["train"], python_executable="python311") == (
        "python311",
        ["-m", "s2s_decision", "train"],
    )


def test_frozen_training_requires_external_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    with pytest.raises(RuntimeError, match="configured Python runtime"):
        worker_command("s2s_decision", ["train"])


def test_dispatch_calls_cli_without_qt(monkeypatch: pytest.MonkeyPatch) -> None:
    from s2s_decision import cli

    received = []
    monkeypatch.setattr(cli, "main", lambda args: received.append(args) or 7)
    assert worker_dispatch(["--s2s-worker", "-m", "s2s_decision", "models"]) == 7
    assert received == [["models"]]
    assert worker_dispatch(["--s2s-worker", "-m", "unknown"]) == 2
    assert received == [["models"]]


def test_frozen_dispatch_rejects_training(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert worker_dispatch(["--s2s-worker", "-m", "s2s_decision", "train"]) == 2


def test_app_dispatches_before_window(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "worker_dispatch", lambda args: 7)
    assert app.main(["--s2s-worker", "-m", "s2s_decision", "models"]) == 7
