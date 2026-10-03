"""Offline methods shared by the main window, workspace and training tools."""

import re
from importlib.resources import files

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices, QTextCursor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

EXPERIMENTAL_NOTICE = (
    "Experimental AI selection and training. These models need broader independent validation. "
    "Scores support prioritization; they do not establish biological activity or safety."
)


def experimental_notice(parent=None):
    label = QLabel(EXPERIMENTAL_NOTICE, parent)
    label.setWordWrap(True)
    label.setStyleSheet("background: #fff3cd; color: #503d00; padding: 7px;")
    return label


def methods_text():
    return files("smiles2select").joinpath("assets", "methods.md").read_text(encoding="utf-8")


class OfflineBrowser(QTextBrowser):
    """No document resource may trigger network access or local-file reads."""

    def loadResource(self, resource_type, name):
        return None


class MethodsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.browser = OfflineBrowser()
        self.browser.setOpenLinks(False)
        self.browser.setOpenExternalLinks(False)
        self.browser.anchorClicked.connect(self.open_link)
        self.sections = QListWidget()
        self.sections.setMaximumWidth(260)
        self.sections.setMinimumWidth(170)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Find in Methods")
        self.search.returnPressed.connect(self.find_next)
        button = QPushButton("Find next")
        button.clicked.connect(self.find_next)
        self.search_status = QLabel()
        try:
            source = methods_text()
        except (OSError, UnicodeError) as exc:
            source = "# Methods unavailable\nThe packaged Methods document could not be read."
            self.search_status.setText(str(exc))
        self.browser.setMarkdown(source)
        self.sections.addItems(re.findall(r"^##\s+(.+?)\s*$", source, re.MULTILINE))
        self.sections.currentTextChanged.connect(self.go_to_section)
        search_row = QHBoxLayout()
        search_row.addWidget(self.search, 1)
        search_row.addWidget(button)
        search_row.addWidget(self.search_status)
        body = QHBoxLayout()
        body.addWidget(self.sections)
        body.addWidget(self.browser, 1)
        layout = QVBoxLayout(self)
        layout.addLayout(search_row)
        layout.addLayout(body, 1)

    def go_to_section(self, title):
        block = self.browser.document().begin()
        while block.isValid():
            if block.blockFormat().headingLevel() == 2 and block.text() == title:
                cursor = QTextCursor(block)
                cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
                self.browser.setTextCursor(cursor)
                self.browser.ensureCursorVisible()
                return
            block = block.next()

    def find_next(self):
        term = self.search.text().strip()
        self.search_status.clear()
        if not term:
            return
        if not self.browser.find(term):
            self.browser.moveCursor(QTextCursor.Start)
            if not self.browser.find(term):
                self.search_status.setText("No match")

    def open_link(self, url):
        if url.isRelative() and url.fragment() and not url.path():
            self.browser.scrollToAnchor(url.fragment())
        elif url.scheme().lower() in {"https", "http"} and url.host():
            QDesktopServices.openUrl(QUrl(url))


def show_methods(owner):
    dialog = getattr(owner, "_methods_dialog", None)
    if dialog is None:
        dialog = QDialog(owner)
        dialog.setWindowTitle("Methods — SMILES2Select")
        dialog.resize(1000, 740)
        layout = QVBoxLayout(dialog)
        layout.addWidget(experimental_notice(dialog))
        dialog.methods_page = MethodsPage(dialog)
        layout.addWidget(dialog.methods_page, 1)
        close = QDialogButtonBox(QDialogButtonBox.Close)
        close.rejected.connect(dialog.close)
        layout.addWidget(close)
        owner._methods_dialog = dialog
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()


def install_methods_action(owner):
    owner.methods_action = owner.menuBar().addAction("Methods")
    owner.methods_action.triggered.connect(lambda: show_methods(owner))
