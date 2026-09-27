"""User decisions remain explicit, previewed, and invalidated when inputs change."""

import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QProcess
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication

from s2s_decision.decision_gui import DecisionWindow


@pytest.fixture
def window():
    app = QApplication.instance() or QApplication([])
    view = DecisionWindow()
    yield view
    if view.process.state() != QProcess.ProcessState.NotRunning:
        view.process.kill()
        view.process.waitForFinished(5000)
    view.close()
    app.processEvents()


def wait(view):
    deadline = time.monotonic() + 60
    while view.process.state() != QProcess.ProcessState.NotRunning and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)
    QApplication.processEvents()
    assert view.process.state() == QProcess.ProcessState.NotRunning


def test_requires_explicit_mode_and_invalidates_preview(window, tmp_path, monkeypatch):
    source = tmp_path / "source.csv"
    source.write_text("smiles,id\nCCO,ethanol\n", encoding="utf-8")
    window.input_path.setText(str(source))
    window.preview()
    assert "Escolha" in window.status.text()
    assert not window.export_button.isEnabled()
    preview = tmp_path / "preview"
    preview.mkdir()
    (preview / "comparison.json").write_text(
        json.dumps(
            {
                "original_count": None,
                "proposed_count": 1,
                "calibration_status": "not_applicable",
                "ranking_mode": "chemical_only",
                "warnings": [],
                "model": None,
            }
        ),
        encoding="utf-8",
    )
    (preview / "comparison.csv").write_text(
        "record_id,molecule_id,original_selected,proposed_selected,change,priority_score\n"
        "1,ethanol,,True,unknown_original,0.5\n",
        encoding="utf-8",
    )
    window.load_preview(preview)
    assert window.export_button.isEnabled()
    assert "desconhecida" in window.summary.text()
    assert window.table.rowCount() == 1
    opened = []
    monkeypatch.setattr(
        QDesktopServices, "openUrl", lambda url: opened.append(url.toLocalFile()) or True
    )
    assert window.comparison_button.isEnabled()
    window.comparison_button.click()
    assert Path(opened[0]) == preview / "comparison.csv"
    window.n.setValue(3)
    assert not window.export_button.isEnabled()
    assert not window.comparison_button.isEnabled()
    assert window.preview_dir is None


def test_discovery_filters_and_keeps_incompatible_model_unavailable(window):
    window.populate_models(
        [
            {
                "path": "ok",
                "name": "Modelo A",
                "estimator": "logistic",
                "target_id": "P1",
                "endpoint": "IC50",
                "compatible": True,
                "reason": None,
            },
            {
                "path": "bad",
                "name": "Modelo B",
                "estimator": "tiny",
                "target_id": "P2",
                "endpoint": "Ki",
                "compatible": False,
                "reason": "chemistry mismatch",
            },
        ]
    )
    assert window.model_choice.count() == 4
    assert not window.model_choice.model().item(3).isEnabled()
    window.target_filter.setText("P1")
    assert window.model_choice.count() == 3
    window.model_choice.setCurrentIndex(2)
    assert window.model_choice.currentData() == "ok"
    window.target_filter.setText("P2")
    assert window.model_choice.currentData() is None


def test_invalid_ids_do_not_start_process(window, tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("smiles,id\nCCO,ethanol\n", encoding="utf-8")
    window.input_path.setText(str(source))
    window.model_choice.setCurrentIndex(1)
    window.pins.setText("1; rm")
    window.preview()
    assert window.process.state() == QProcess.ProcessState.NotRunning
    assert "inteiros" in window.status.text()


def test_real_preview_requires_separate_export(window, tmp_path, monkeypatch):
    monkeypatch.setenv(
        "PYTHONPATH",
        str(Path("src").resolve()),
    )
    source = tmp_path / "source.csv"
    source.write_text("smiles,id\nCCO,ethanol\nCC,ethane\nCCCC,butane\n", encoding="utf-8")
    window.input_path.setText(str(source))
    window.model_choice.setCurrentIndex(1)
    window.n.setValue(2)
    window.pins.setText("1")
    window.exclude.setText("2")
    window.preview()
    wait(window)
    assert window.export_button.isEnabled(), window.log.toPlainText()
    assert window.table.rowCount() == 3
    assert window.preview_dir.is_dir()
    destination = tmp_path / "adopted"
    assert not destination.exists()
    window.export_to(destination)
    wait(window)
    assert "exportada" in window.status.text(), window.log.toPlainText()
    assert destination.is_dir()
    assert (destination / "comparison.json").is_file()
    adopted = json.loads((destination / "selection.json").read_text(encoding="utf-8"))
    assert set(adopted["preview"]["proposed_ids"]) == {1, 3}
    window.export_to(destination)
    assert "existe" in window.status.text()


def test_scan_empty_model_library_uses_real_cli(window, tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(Path("src").resolve()))
    window.model_root.setText(str(tmp_path))
    window.scan_models()
    wait(window)
    assert "Busca concluída" in window.status.text(), window.log.toPlainText()
    assert window.model_choice.count() == 2
    assert window.model_choice.currentData() is None
    assert not window.export_button.isEnabled()


def test_frozen_worker_launch_uses_internal_dispatch(window, tmp_path, monkeypatch):
    import sys

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    calls = []
    monkeypatch.setattr(window.process, "start", lambda program, args: calls.append((program, args)))
    window.start("models", tmp_path / "models.json", ["--input", str(tmp_path)])
    assert calls == [
        (
            sys.executable,
            [
                "--s2s-worker",
                "-m",
                "s2s_decision",
                "models",
                "--input",
                str(tmp_path),
                "--output",
                str(tmp_path / "models.json"),
            ],
        )
    ]


def test_failure_cancel_and_bad_report_restore_controls(window, tmp_path):
    import sys

    window.set_running(True)
    window.process_error(QProcess.ProcessError.FailedToStart)
    assert window.preview_button.isEnabled()
    window.finished(2, QProcess.ExitStatus.NormalExit)
    assert "falhou" in window.status.text()
    window.operation = "models"
    window.operation_output = tmp_path / "models.json"
    window.operation_output.write_text('{"models": "invalid"}', encoding="utf-8")
    window.finished(0, QProcess.ExitStatus.NormalExit)
    assert "inválido" in window.status.text()
    window.set_running(True)
    window.process.start(sys.executable, ["-c", "import time; time.sleep(30)"])
    assert window.process.waitForStarted(5000)
    window.cancel()
    wait(window)
    assert "interrompida" in window.status.text()
    assert window.preview_button.isEnabled()
    assert not window.cancel_button.isEnabled()
