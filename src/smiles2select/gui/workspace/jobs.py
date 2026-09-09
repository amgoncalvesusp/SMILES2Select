"""Owned Qt jobs: worker computation never reads or writes live widgets."""

from PySide6.QtCore import QThread, Signal


class ComputationJob(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, function, parent=None):
        super().__init__(parent)
        self.function = function

    def run(self):
        try:
            value = self.function()
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.completed.emit(value)
