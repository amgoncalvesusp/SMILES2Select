"""Visible software credits retain both authors and their affiliations."""

from PySide6.QtWidgets import QApplication, QMessageBox

from smiles2select.app_metadata import APP_VERSION
from smiles2select.gui.main_window import MainWindow


def test_about_menu_shows_both_authors(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    captured = {}
    monkeypatch.setattr(QMessageBox, "about", lambda _parent, title, text:
                        captured.update(title=title, text=text))
    try:
        action = next((item for item in window.menuBar().actions()
                       if item.text() == "About SMILES2Select"), None)
        assert action is not None, "Software authors must be accessible in the GUI"
        action.trigger()
        app.processEvents()
        assert APP_VERSION in captured["text"]
        assert "Adriano Marques Gonçalves" in captured["text"]
        assert "Universidade de Araraquara (UNIARA)" in captured["text"]
        assert "Daniel Andres Grajales Ruiz" in captured["text"]
        assert "IQ-UNESP" in captured["text"]
    finally:
        window.close()
