"""Offline Methods remains available before inputs and from model tools."""

from PySide6.QtCore import QUrl

from smiles2select.gui.main_window import MainWindow
from smiles2select.gui.pages import methods_page
from tests import test_gui

qapp = test_gui.qapp


def test_methods_tab_access_without_input_preserves_wizard(qapp, monkeypatch):
    monkeypatch.setattr(methods_page, "methods_text", lambda: "# Methods\n\n## Preparation\nOffline methods.\n\n## Models\nExperimental training.")
    window = MainWindow()
    window.show()
    window.methods_action.trigger()
    qapp.processEvents()
    assert window.navigation.currentWidget() is window.methods_page
    assert window.pages.count() == 7
    assert not window.state.files
    assert window.experimental_notice.isVisible()
    assert "broader independent validation" in window.experimental_notice.text()
    assert "Offline methods" in window.methods_page.browser.toPlainText()
    window.navigation.setCurrentIndex(0)
    assert window.pages.currentIndex() == 0
    window.close()


def test_methods_sections_search_and_safe_links(qapp, monkeypatch):
    monkeypatch.setattr(methods_page, "methods_text", lambda: "# Methods\n\nModels appear below.\n\n## Preparation\nSMILES details.\n\n## Models\nCalibration and limitations.")
    page = methods_page.MethodsPage()
    assert page.sections.count() == 2
    page.sections.setCurrentRow(1)
    assert page.browser.textCursor().selectedText() == "Models"
    assert page.browser.textCursor().blockFormat().headingLevel() == 2
    page.search.setText("Calibration")
    page.find_next()
    assert page.browser.textCursor().selectedText() == "Calibration"
    page.find_next()
    assert page.browser.textCursor().selectedText() == "Calibration"
    page.search.setText("absent value")
    page.find_next()
    assert "No match" in page.search_status.text()
    page.search.clear()
    page.find_next()
    opened = []
    monkeypatch.setattr(methods_page.QDesktopServices, "openUrl", lambda url: opened.append(url.toString()))
    for value in ("file:///secret", "javascript:alert(1)", "ftp://example.org/a", "relative.md"):
        page.open_link(QUrl(value))
    assert not opened
    page.open_link(QUrl("https://example.org/method"))
    assert opened == ["https://example.org/method"]
    assert page.browser.loadResource(2, QUrl("https://example.org/pixel.png")) is None
    page.close()


def test_packaged_methods_and_missing_resource_message(qapp, monkeypatch):
    assert "# Methods" in methods_page.methods_text()
    monkeypatch.setattr(methods_page, "methods_text", lambda: (_ for _ in ()).throw(OSError("Missing resource")))
    page = methods_page.MethodsPage()
    assert "Methods unavailable" in page.browser.toPlainText()
    assert "Missing resource" in page.search_status.text()
    page.go_to_section("absent")
    page.open_link(QUrl("#local-section"))
    page.close()


def test_studio_has_methods_and_persistent_training_notice(qapp):
    from s2s_decision.gui import Studio

    window = Studio()
    window.show()
    window.methods_action.trigger()
    qapp.processEvents()
    assert window.experimental_notice.isVisible()
    assert "training" in window.experimental_notice.text().lower()
    assert window._methods_dialog.isVisible()
    window.close()
