"""Step 6: execution."""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
)

from smiles2select.gui.pages.base import WizardPage
from smiles2select.gui.worker import RunWorker


class RunPage(WizardPage):
    title = "6. Processing"
    subtitle = "Review the summary and run. Calculation happens in the background."

    run_finished = Signal(object)

    def __init__(self, state) -> None:
        super().__init__(state)
        self._worker: RunWorker | None = None

        self.jobs_spin = QSpinBox()
        self.jobs_spin.setRange(-1, 128)
        self.jobs_spin.setValue(-1)
        # -1 no longer means "every core": it is resolved from available
        # memory and CPU count at run time (see pipeline.resource_estimation),
        # specifically to avoid the OOM risk of always maxing out n_jobs.
        self.jobs_spin.setSpecialValueText("automatic (memory + CPU)")

        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(100, 100000)
        self.chunk_spin.setSingleStep(100)
        self.chunk_spin.setValue(2000)

        settings = QGroupBox("Processing resources")
        form = QFormLayout(settings)
        form.addRow("Processes:", self.jobs_spin)
        form.addRow("Molecules per batch:", self.chunk_spin)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(240)

        self.progress = QProgressBar()
        self.progress.setFormat("%v / %m")
        self.stage_label = QLabel("Ready to run.")

        self.start_button = QPushButton("Run")
        self.start_button.clicked.connect(self._start)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)

        buttons = QHBoxLayout()
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch(1)

        self.body.addWidget(settings)
        self.body.addWidget(self.summary)
        self.body.addLayout(buttons)
        self.body.addWidget(self.stage_label)
        self.body.addWidget(self.progress)
        self.body.addStretch(1)

    def on_enter(self) -> None:
        problems = self.state.validation_errors()
        if problems:
            self.summary.setPlainText(
                "Pending issues:\n" + "\n".join(f"  - {problem}" for problem in problems)
            )
            self.start_button.setEnabled(False)
            return
        self.start_button.setEnabled(True)
        config = self.state.build_config()
        lines = [f"{key}: {value}" for key, value in config.summary_rows()]
        lines.append(f"files: {len(config.sources)}")
        self.summary.setPlainText("\n".join(lines))

    def _start(self) -> None:
        try:
            config = self.state.build_config()
        except ValueError as exc:
            QMessageBox.warning(self, "Incomplete configuration", str(exc))
            return

        config = replace(config, n_jobs=self.jobs_spin.value(), chunk_size=self.chunk_spin.value())

        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setRange(0, 0)
        self._worker = RunWorker(config, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.completed.connect(self._on_completed)
        self._worker.failed.connect(self._on_failed)
        self._worker.canceled.connect(self._on_canceled)
        self._worker.start()

    def _on_progress(self, done: int, total: int, stage: str) -> None:
        self.stage_label.setText(stage)
        if total > 0:
            self.progress.setRange(0, total)
            self.progress.setValue(done)
        else:
            self.progress.setRange(0, 0)

    def _on_completed(self, result: object) -> None:
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.stage_label.setText("Completed.")
        self.run_finished.emit(result)

    def _on_failed(self, message: str) -> None:
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.stage_label.setText("Failed.")
        QMessageBox.critical(self, "Execution failed", message)

    def _cancel(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self.cancel_button.setEnabled(False)
            self.stage_label.setText("Cancellation requested; finishing the current batch...")
            self._worker.cancel()

    def _on_canceled(self) -> None:
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.stage_label.setText("Cancelled. The checkpoint can be resumed.")

    def validate(self) -> str | None:
        problems = self.state.validation_errors()
        return "\n".join(problems) if problems else None
