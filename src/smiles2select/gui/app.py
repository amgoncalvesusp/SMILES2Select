"""GUI entry point."""

from __future__ import annotations

import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from smiles2select.app_metadata import APP_NAME, app_icon_path
from smiles2select.gui.main_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    application = QApplication(argv if argv is not None else sys.argv)
    application.setApplicationName(APP_NAME)
    application.setWindowIcon(QIcon(str(app_icon_path())))
    window = MainWindow()
    window.show()
    return application.exec()



if __name__ == "__main__":
    # Required before anything else touches multiprocessing: this is the
    # PyInstaller entrypoint (packaging/smiles2select.spec), and on Windows a
    # frozen build re-executes this same module in every worker process. Without
    # freeze_support(), that re-execution would open a second GUI window in
    # each worker instead of running as a plain worker.
    from multiprocessing import freeze_support

    freeze_support()
    raise SystemExit(main())
