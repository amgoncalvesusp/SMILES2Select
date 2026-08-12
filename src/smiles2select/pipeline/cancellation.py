"""Cooperative cancellation for long-running pipeline and Hub jobs."""

from __future__ import annotations

from threading import Event


class RunCancelled(RuntimeError):
    """Raised at a safe checkpoint after the user requests cancellation."""


class CancellationToken:
    """Thread-safe flag shared by a GUI worker and the pipeline."""

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise RunCancelled("operation cancelled by the user")
