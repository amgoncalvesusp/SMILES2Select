"""Offscreen checks for the thin CLI desktop client."""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QProcess
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from s2s_decision.cli import parser
from s2s_decision.gui import Studio
from smiles2select.app_metadata import APP_VERSION


@pytest.fixture
def studio():
    app = QApplication.instance() or QApplication([])
    window = Studio()
    yield window
    window.close()
    app.processEvents()


def choose(window, mode, action):
    window.mode.setCurrentText(mode)
    window.action.setCurrentText(action)


def test_commands_match_cli_and_hide_irrelevant_controls(studio, tmp_path):
    for mode, actions in studio.MODES.items():
        for action in actions:
            choose(studio, mode, action)
            studio.input_path.setText(str(tmp_path / "source"))
            studio.output_path.setText(str(tmp_path / "new output"))
            studio.model_path.setText(str(tmp_path / "model"))
            studio.fields["target"].setText("P12345")
            parsed = parser().parse_args(studio.command_arguments()[2:])
            assert parsed.command == action
            assert studio.fields["epochs"].isEnabled() == (action == "train")
            assert studio.fields["n"].isEnabled() == (action in {"select", "benchmark", "evaluate"})
            assert studio.fields["resume"].isEnabled() == (action == "train")
    choose(studio, "Selection", "select")
    studio.fields["pins"].setText("1, 3")
    assert parser().parse_args(studio.command_arguments()[2:]).pins == [1, 3]
    studio.fields["pins"].setText("1; rm")
    with pytest.raises(ValueError):
        studio.command_arguments()


def test_failure_and_finish_restore_controls(studio):
    studio.set_running(True)
    assert not studio.run_button.isEnabled()
    studio.process_error(QProcess.ProcessError.FailedToStart)
    assert studio.run_button.isEnabled()
    assert not studio.cancel_button.isEnabled()
    studio.set_running(True)
    studio.process_finished(2, QProcess.ExitStatus.NormalExit)
    assert "Failed" in studio.status.text()
    assert studio.run_button.isEnabled()


def test_prepare_runs_real_cli_and_preserves_existing_output(studio, tmp_path, monkeypatch):
    monkeypatch.setenv(
        "PYTHONPATH",
        str(Path("src").resolve()),
    )
    source = tmp_path / "molecules.csv"
    source.write_text("smiles,id\nCCO,ethanol\nCC,ethane\n", encoding="utf-8")
    output = tmp_path / "prepared"
    choose(studio, "Selection", "prepare")
    studio.input_path.setText(str(source))
    studio.output_path.setText(str(output))
    studio.run_command()
    deadline = time.monotonic() + 60
    while (
        studio.process.state() != QProcess.ProcessState.NotRunning and time.monotonic() < deadline
    ):
        QApplication.processEvents()
        time.sleep(0.01)
    QApplication.processEvents()
    assert studio.process.state() == QProcess.ProcessState.NotRunning
    assert "Completed" in studio.status.text(), studio.log.toPlainText()
    assert (output / "manifest.json").exists()
    studio.run_command()
    assert "already exists" in studio.log.toPlainText()
    assert studio.process.state() == QProcess.ProcessState.NotRunning


def test_browse_uses_file_or_directory_dialog(studio, tmp_path, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: ("source.csv", ""))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: ("report.json", ""))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args: str(tmp_path))
    choose(studio, "Selection", "prepare")
    studio.browse_input()
    studio.browse_output()
    assert studio.input_path.text() == "source.csv"
    assert Path(studio.output_path.text()) == tmp_path / "s2s_prepare"
    choose(studio, "Benchmark", "evaluate")
    studio.browse_input()
    studio.browse_model()
    studio.browse_output()
    assert studio.input_path.text() == studio.model_path.text() == str(tmp_path)
    assert studio.output_path.text() == "report.json"


def test_close_and_cancel_wait_for_subprocess(studio, monkeypatch):
    import sys

    studio.set_running(True)
    studio.process.start(sys.executable, ["-c", "import time; time.sleep(30)"])
    assert studio.process.waitForStarted(5000)
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.No)
    event = QCloseEvent()
    studio.closeEvent(event)
    assert not event.isAccepted()
    assert not studio.cancelled
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    studio.closeEvent(event)
    deadline = time.monotonic() + 10
    while (
        studio.process.state() != QProcess.ProcessState.NotRunning and time.monotonic() < deadline
    ):
        QApplication.processEvents()
        time.sleep(0.01)
    QApplication.processEvents()
    assert studio.process.state() == QProcess.ProcessState.NotRunning
    assert studio.cancelled and studio.close_after_finish
    assert "partial artifacts retained" in studio.status.text()


def test_missing_paths_and_model_are_rejected(studio, tmp_path):
    studio.run_command()
    assert "required" in studio.log.toPlainText()
    studio.input_path.setText(str(tmp_path / "missing.csv"))
    studio.output_path.setText(str(tmp_path / "new"))
    studio.run_command()
    assert "existing table file" in studio.log.toPlainText()
    choose(studio, "Selection", "predict")
    with pytest.raises(ValueError, match="Model directory"):
        studio.command_arguments()


def test_frozen_training_without_external_python_does_not_start(studio, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    source = tmp_path / "dataset"
    source.mkdir()
    choose(studio, "Training", "train")
    studio.input_path.setText(str(source))
    studio.output_path.setText(str(tmp_path / "model"))
    studio.run_command()
    assert studio.process.state() == QProcess.ProcessState.NotRunning
    assert "configured Python runtime" in studio.log.toPlainText()
    assert studio.run_button.isEnabled()


def test_studio_can_be_child_of_main_window(studio):
    assert studio.windowTitle() == "SMILES2Select — Advanced model tools"


def test_frozen_training_uses_validated_external_python(studio, tmp_path, monkeypatch):
    import s2s_decision.gui as gui_module

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    source = tmp_path / "dataset"
    source.mkdir()
    executable = tmp_path / "python.exe"
    executable.touch()
    choose(studio, "Training", "train")
    studio.input_path.setText(str(source))
    studio.output_path.setText(str(tmp_path / "model"))
    studio.training_python.setText(str(executable))
    checked = []
    started = []
    monkeypatch.setattr(gui_module, "_verify_training_python", checked.append)
    monkeypatch.setattr(studio.process, "start", lambda program, args: started.append((program, args)))
    studio.run_command()
    assert checked == [str(executable)]
    assert started[0][0] == str(executable)
    assert started[0][1][:3] == ["-m", "s2s_decision", "train"]


@pytest.mark.parametrize(
    ("report", "message"),
    [
        ({"python": [3, 10], "version": APP_VERSION, "missing": []}, "Python 3.11"),
        ({"python": [3, 11], "version": "0.0.0", "missing": []}, APP_VERSION),
        ({"python": [3, 11], "version": APP_VERSION, "missing": ["onnx"]}, "onnx"),
    ],
)
def test_training_runtime_preflight_rejects_incompatible_python(
    tmp_path, monkeypatch, report, message
):
    import s2s_decision.gui as gui_module

    executable = tmp_path / "python.exe"
    executable.touch()
    monkeypatch.setattr(
        gui_module.subprocess,
        "run",
        lambda *args, **kwargs: type("Result", (), {"returncode": 0, "stdout": json.dumps(report)})(),
    )
    with pytest.raises(ValueError, match=message):
        gui_module._verify_training_python(str(executable))
