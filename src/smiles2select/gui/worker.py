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
from smiles2select.pipeline.runner import (
    RunResult,
    resolve_checkpoint_path,
    resolve_log_directory,
    run,
)


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
            self.failed.emit(self._failure_message(exc))

    def _report(self, done: int, total: int, stage: str) -> None:
        self.progress.emit(done, total, stage)

    def _failure_message(self, exc: Exception) -> str:
        """A dead worker or an OOM should not read as an unreadable Python
        traceback: name what happened, where the completed work and the
        per-worker crash logs are, and how to continue (run again - the
        checkpoint at the same path resumes automatically).
        """
        checkpoint_path = resolve_checkpoint_path(self._config)
        log_directory = resolve_log_directory(self._config)
        preserved = (
            "sim, no checkpoint" if checkpoint_path.exists() else "nenhum bloco concluído ainda"
        )
        return (
            f"O processamento foi interrompido: {exc}\n\n"
            f"Resultados já concluídos preservados: {preserved}\n"
            f"Checkpoint: {checkpoint_path}\n"
            f"Logs por worker (falhas nativas e memória): {log_directory}\n\n"
            "Execute novamente com a mesma configuração para retomar a partir "
            "do último bloco concluído, em vez de recomeçar do zero.\n\n"
            f"Detalhe técnico:\n{traceback.format_exc()}"
        )
