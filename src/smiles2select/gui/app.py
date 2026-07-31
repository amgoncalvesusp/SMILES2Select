"""GUI entry point."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from smiles2select.app_metadata import APP_NAME
from smiles2select.gui.main_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    application = QApplication(argv if argv is not None else sys.argv)
    application.setApplicationName(APP_NAME)
    window = MainWindow()
    window.show()
    return application.exec()


def workspace_main(argv: list[str] | None = None) -> int:
    """Open the wizard with the workspace reachable from the results screen.

    The workspace needs a finished run to work on, so it cannot be the first
    screen: this entry point starts the normal flow, and the results step hands
    the run over.
    """
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
