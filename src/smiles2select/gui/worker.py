"""Background execution.

The pipeline runs in a QThread so the interface keeps responding while a large
library is processed. Progress arrives as signals; the result object is handed
over once, at the end.
"""

from __future__ import annotations

import traceback

from PySide6.QtCore import QThread, Signal

from smiles2select.export import excel
from smiles2select.pipeline.config import RunConfig
from smiles2select.pipeline.runner import RunResult, run


class RunWorker(QThread):
    """Runs one :class:`RunConfig` and reports progress."""

    progress = Signal(int, int, str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, config: RunConfig, parent=None) -> None:
        super().__init__(parent)
        self._config = config

    def run(self) -> None:  # QThread entry point
        try:
            result: RunResult = run(self._config, self._report)
            if self._config.excel_path is not None:
                self.progress.emit(0, 1, "Gerando Excel")
                excel.export(
                    result,
                    self._config.excel_path,
                    excel.ExportOptions(detailed=self._config.detailed_export),
                )
            self.completed.emit(result)
        except Exception as exc:  # surfaced in the interface, never swallowed
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")

    def _report(self, done: int, total: int, stage: str) -> None:
        self.progress.emit(done, total, stage)
