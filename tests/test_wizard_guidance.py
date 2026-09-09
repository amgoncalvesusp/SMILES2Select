"""Beginner guidance must not silently change scientific settings."""

from PySide6.QtGui import QCloseEvent

from tests import test_gui

qapp = test_gui.qapp
window = test_gui.window


def test_policy_advanced_controls_are_optional_and_preserve_values(window):
    page = window.pages.widget(4)
    page.on_enter()
    before = window.state.build_policy()
    assert page.advanced_options.isHidden()
    page.advanced_toggle.click()
    assert not page.advanced_options.isHidden()
    page.expression_edit.setText("lipinski AND veber")
    page.expression_edit.editingFinished.emit()
    page.advanced_toggle.click()
    assert window.state.expression == "lipinski AND veber"
    assert window.state.build_policy().roles == before.roles
    assert "lipinski" in page.explanation.text().lower()


def test_loaded_advanced_expression_remains_discoverable(window):
    page = window.pages.widget(4)
    window.state.expression = "lipinski AND veber"
    page._refresh_widgets_from_state()
    assert page.advanced_toggle.isChecked()
    assert not page.advanced_options.isHidden()


def test_wizard_pages_scroll_and_initial_guidance_identifies_next_step(window):
    assert all(window.pages.widget(i).scroll_area.widgetResizable() for i in range(7))
    files = window.pages.widget(0)
    assert not files.empty_hint.isHidden()
    assert "SMILES" in files.empty_hint.text()
    assert "Step 1 of 7" in window.progress_label.text()
    assert "Columns" in window.next_button.text()


def test_main_window_refuses_close_while_processing(window, monkeypatch):
    class Worker:
        def isRunning(self):
            return True

    page = window.pages.widget(5)
    page._worker = Worker()
    monkeypatch.setattr(
        "smiles2select.gui.main_window.QMessageBox.information", lambda *args: None
    )
    event = QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted()
    page._worker = None
