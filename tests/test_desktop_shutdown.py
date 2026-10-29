"""Permanent Qt disposal finishes before the application and interpreter exit."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pyqtgraph as pg
import pytest
import shiboken6
from PySide6.QtWidgets import QApplication, QWidget

from smiles2select.gui import desktop_self_check

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def test_permanent_disposal_closes_primary_and_orphan_plots_before_destroy(qapp, monkeypatch):
    existing = set(qapp.topLevelWidgets())
    owners = [QWidget(), QWidget()]
    plots = [pg.PlotWidget(owner) for owner in owners]
    events = []
    original = pg.PlotWidget.close

    def closed(plot):
        events.append((id(plot), "close"))
        return original(plot)

    monkeypatch.setattr(pg.PlotWidget, "close", closed)
    for plot in plots:
        identity = id(plot)
        plot.destroyed.connect(lambda _object=None, key=identity: events.append((key, "destroy")))
    desktop_self_check._dispose_widget(owners[0], existing)
    for plot in plots:
        assert [(key, event) for key, event in events if key == id(plot)] == [
            (id(plot), "close"), (id(plot), "destroy"),
        ]
        assert not shiboken6.isValid(plot)
    assert set(qapp.topLevelWidgets()) == existing


@pytest.mark.parametrize("already_closed", [False, True])
def test_permanent_disposal_is_idempotent_for_detached_plots(qapp, already_closed):
    existing = set(qapp.topLevelWidgets())
    plot = pg.PlotWidget()
    if already_closed:
        plot.close()
    desktop_self_check._dispose_widget(plot, existing)
    desktop_self_check._dispose_widget(plot, existing)
    assert not shiboken6.isValid(plot)
    assert set(qapp.topLevelWidgets()) == existing


def _run(arguments, timeout=180):
    process = subprocess.run(
        [sys.executable, "-X", "faulthandler", *arguments], cwd=ROOT,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen",
             "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])},
        capture_output=True, text=True, timeout=timeout, check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert "shared QObject was deleted directly" not in process.stderr
    return process


def test_qt_signal_emission_preserves_bool_references_and_exits_normally():
    process = _run(["-c", dedent("""
        import sys
        from PySide6.QtCore import QObject, Signal

        class Sender(QObject):
            ping = Signal()

        sender = Sender()
        sender.ping.emit()
        before = sys.getrefcount(True)
        for _ in range(100):
            assert sender.ping.emit() is True
        after = sys.getrefcount(True)
        assert after == before, (before, after)
        print('signal-refcount-clean', flush=True)
    """)])
    assert "signal-refcount-clean" in process.stdout


def test_fixed_desktop_diagnostic_exits_normally_after_hub_disposal(tmp_path):
    report_path = tmp_path / "desktop-resources.json"
    _run(["-m", "smiles2select.gui.app", "--verify-desktop-resources", "--output", str(report_path)])
    report = json.loads(report_path.read_text("utf-8"))
    assert report["status"] == "ok"
    comparison = report["hub_gui"]["comparison"]
    assert comparison["reference_id"] == 1 and comparison["current_id"] == 2
    assert comparison["reference_smiles"] == "CCO" and comparison["current_smiles"] == "CCN"


def test_pytest_session_disposes_closed_hub_graphs_before_normal_exit(tmp_path):
    test_path = tmp_path / "test_retained_hub.py"
    test_path.write_text(dedent("""
        from unittest.mock import patch
        from PySide6.QtWidgets import QApplication
        from smiles2select.gui import desktop_self_check

        retained = []

        def retain(widget, existing):
            widget.close()
            retained.append(widget)

        def test_retained_hub():
            with patch.object(desktop_self_check, '_dispose_widget', retain):
                assert desktop_self_check.verify_desktop_resources()['status'] == 'ok'
            assert QApplication.instance().allWidgets()
    """), encoding="utf-8")
    process = _run(["-c", dedent("""
        import sys
        import pytest
        from PySide6.QtWidgets import QApplication

        application = QApplication([])
        assert pytest.main([sys.argv[1], '-p', 'tests.conftest', '-q']) == 0
        assert QApplication.instance() is application
        assert not application.allWidgets(), 'Qt widgets survived pytest session cleanup'
        print('shutdown-clean', flush=True)
    """), str(test_path)])
    assert "shutdown-clean" in process.stdout


def test_real_main_exits_cleanly_after_hub_close_reopen_and_comparison(tmp_path):
    process = _run(["-c", dedent("""
        import sys
        import traceback
        from pathlib import Path
        from unittest.mock import patch
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication
        from smiles2select.gui import app, desktop_self_check
        from smiles2select.gui.main_window import MainWindow
        from smiles2select.io.importer import ColumnMapping, SourceFile
        from smiles2select.pipeline.config import RunConfig
        from smiles2select.pipeline.runner import run

        directory = Path(sys.argv[1])
        source = directory / 'library.csv'
        source.write_text('ID,SMILES\\n1,CCO\\n2,CCN\\n3,CC(=O)O\\n', encoding='utf-8')
        result = run(RunConfig(
            sources=(SourceFile(path=source, mapping=ColumnMapping(smiles='SMILES', molecule_id='ID')),),
            profile_ids=('lipinski', 'veber'), alert_catalogs=(), compute_sa=True,
            n_jobs=1, chunk_size=3, checkpoint_path=directory / 'checkpoint.sqlite',
        ))
        applications, failures = [], []
        show = MainWindow.show

        def exercise(window):
            application = QApplication.instance()
            applications.append(application)
            try:
                window._open_workspace(result)
                workspace = window._workspace_window
                report = desktop_self_check._check_hub_gui(workspace)
                assert report['comparison']['reference_id'] == 1
                assert report['comparison']['current_id'] == 2
                comparison = workspace._comparison_dialog
                assert workspace.close()
                assert not comparison.isVisible()
                window._open_workspace(result)
                assert window._workspace_window is workspace and workspace.isVisible()
                workspace.molecule_compare_button.click()
                assert workspace._comparison_dialog is comparison and comparison.isVisible()
                assert comparison.reference_id == 1
            except BaseException:
                failures.append(traceback.format_exc())
            finally:
                application.quit()

        def shown(window):
            show(window)
            QTimer.singleShot(0, lambda: exercise(window))

        with patch.object(MainWindow, 'show', shown):
            assert app.main([]) == 0
        assert not failures, '\\n'.join(failures)
        assert applications and QApplication.instance() is applications[0]
        assert not applications[0].allWidgets(), 'Qt widgets survived main return'
        print('main-shutdown-clean', flush=True)
    """), str(tmp_path)])
    assert "main-shutdown-clean" in process.stdout
