"""Main window: the seven-step wizard.

Files -> Columns -> Standardization -> Profiles -> Policy -> Processing ->
Results. The step list stays visible so the user always knows where they are
and can go back without losing what they have chosen.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from smiles2select.app_metadata import APP_NAME, APP_VERSION, DISCLAIMER, app_icon_path
from smiles2select.gui.pages.columns_page import ColumnsPage
from smiles2select.gui.pages.files_page import FilesPage
from smiles2select.gui.pages.policy_page import PolicyPage
from smiles2select.gui.pages.profiles_page import ProfilesPage
from smiles2select.gui.pages.results_page import ResultsPage
from smiles2select.gui.pages.run_page import RunPage
from smiles2select.gui.pages.standardization_page import StandardizationPage
from smiles2select.gui.state import WizardState
from smiles2select.gui.workspace.workspace_window import WorkspaceWindow

STEPS = (
    "1. Files",
    "2. Columns",
    "3. Standardization",
    "4. Profiles",
    "5. Selection policy",
    "6. Processing",
    "7. Results",
)


class MainWindow(QMainWindow):
    """Hosts the wizard pages and the navigation between them."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.setWindowIcon(QIcon(str(app_icon_path())))
        self.resize(1180, 780)

        self.state = WizardState()
        self._workspace_window: WorkspaceWindow | None = None
        self.pages = QStackedWidget()
        self.steps = QListWidget()
        self.steps.setMaximumWidth(200)
        self.steps.setSelectionMode(QListWidget.NoSelection)
        for label in STEPS:
            self.steps.addItem(QListWidgetItem(label))

        self._page_widgets = [
            FilesPage(self.state),
            ColumnsPage(self.state),
            StandardizationPage(self.state),
            ProfilesPage(self.state),
            PolicyPage(self.state),
            RunPage(self.state),
            ResultsPage(self.state),
        ]
        for page in self._page_widgets:
            self.pages.addWidget(page)

        self._page_widgets[5].run_finished.connect(self._on_run_finished)
        self._page_widgets[6].workspace_requested.connect(self._open_workspace)

        self.back_button = QPushButton("Back")
        self.next_button = QPushButton("Next")
        self.back_button.clicked.connect(lambda: self._go(self.pages.currentIndex() - 1))
        self.next_button.clicked.connect(lambda: self._go(self.pages.currentIndex() + 1))

        disclaimer = QLabel(DISCLAIMER)
        disclaimer.setWordWrap(True)
        disclaimer.setStyleSheet("color: #555; font-size: 11px;")

        navigation = QHBoxLayout()
        navigation.addWidget(disclaimer, stretch=1)
        navigation.addWidget(self.back_button)
        navigation.addWidget(self.next_button)

        content = QVBoxLayout()
        content.addWidget(self.pages, stretch=1)
        content.addLayout(navigation)

        layout = QHBoxLayout()
        layout.addWidget(self.steps)
        layout.addLayout(content, stretch=1)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self._highlight_step(0)

    def _go(self, index: int) -> None:
        if not 0 <= index < self.pages.count():
            return
        if index > self.pages.currentIndex():
            problem = self.pages.currentWidget().validate()
            if problem:
                QMessageBox.warning(self, "Check this step", problem)
                return
        self.pages.setCurrentIndex(index)
        self.pages.currentWidget().on_enter()
        self._highlight_step(index)

    def _highlight_step(self, index: int) -> None:
        for position in range(self.steps.count()):
            item = self.steps.item(position)
            font = item.font()
            font.setBold(position == index)
            item.setFont(font)
            item.setForeground(Qt.black if position <= index else Qt.gray)
        self.back_button.setEnabled(index > 0)
        self.next_button.setEnabled(index < self.pages.count() - 1)

    def _on_run_finished(self, result: object) -> None:
        if self._workspace_window is not None:
            self._workspace_window.close()
            self._workspace_window = None
        self._page_widgets[6].show_result(result)
        self._go(6)

    def _open_workspace(self, result: object) -> None:
        """Open the interactive chemical-space session for a finished run."""
        try:
            workspace = WorkspaceWindow(result, parent=self)
        except Exception as exc:
            QMessageBox.critical(self, "Chemical Space Hub", str(exc))
            return
        self._workspace_window = workspace
        workspace.show()
        workspace.raise_()
        workspace.activateWindow()
