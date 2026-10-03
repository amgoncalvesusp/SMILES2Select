"""GUI entry point."""

from __future__ import annotations

import sys

from smiles2select.gui.desktop_self_check import desktop_resources_dispatch
from smiles2select.gui.worker_launch import worker_dispatch


def main(argv: list[str] | None = None) -> int:
    from multiprocessing import freeze_support

    freeze_support()
    arguments = list(sys.argv[1:] if argv is None else argv)
    worker_status = worker_dispatch(arguments)
    if worker_status is not None:
        return worker_status
    desktop_status = desktop_resources_dispatch(arguments)
    if desktop_status is not None:
        return desktop_status

    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from smiles2select.app_metadata import APP_NAME, app_icon_path
    from smiles2select.gui.main_window import MainWindow

    application = QApplication(argv if argv is not None else sys.argv)
    application.setApplicationName(APP_NAME)
    application.setWindowIcon(QIcon(str(app_icon_path())))
    window = MainWindow()
    window.show()
    return application.exec()



if __name__ == "__main__":
    raise SystemExit(main())
