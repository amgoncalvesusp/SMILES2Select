"""Exercise scenario jobs through the real Qt event loop."""

import sys
import time
from dataclasses import replace

import pytest
from PySide6.QtCore import QThread

from smiles2select.gui.workspace.scenario_dialog import ScenarioDialog
from tests import test_workspace

qapp = test_workspace.qapp
result = test_workspace.result
window = test_workspace.window


@pytest.fixture
def dialog(window):
    value = ScenarioDialog(window)
    yield value
    if value._job is not None:
        value._job.wait(10000)
    value.close()


def wait_for_job(dialog, qapp):
    deadline = time.monotonic() + 10
    while dialog.is_busy and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.005)
    qapp.processEvents()
    assert not dialog.is_busy
    assert dialog.preview_button.isEnabled()
    assert dialog.progress.isHidden()


def test_completion_failure_is_reported_in_dialog(dialog, qapp, monkeypatch):
    unhandled = []
    monkeypatch.setattr(sys, "excepthook", lambda *args: unhandled.append(args))

    def broken_callback(value):
        raise ValueError("Cannot apply scenario result")

    dialog._start(lambda: 42, broken_callback)
    wait_for_job(dialog, qapp)
    assert "Cannot apply scenario result" in dialog.report.toPlainText()
    assert not unhandled


def test_success_clears_transient_progress_and_uses_gui_thread(dialog, qapp):
    callback_threads = []
    worker_threads = []

    def compute():
        worker_threads.append(QThread.currentThread() == qapp.thread())
        return 42

    def completed(value):
        callback_threads.append(QThread.currentThread() == qapp.thread())

    dialog._start(compute, completed)
    wait_for_job(dialog, qapp)
    assert worker_threads == [False]
    assert callback_threads == [True]
    assert "Computing" not in dialog.report.toPlainText()


def test_completion_preserves_scientific_warning(dialog, qapp):
    warning = "Only 2 molecules satisfy the quota; requested 5."
    dialog._start(lambda: 42, lambda _: dialog.report.setPlainText(warning))
    wait_for_job(dialog, qapp)
    assert dialog.report.toPlainText() == warning


def test_computation_failure_is_preserved_after_cleanup(dialog, qapp):
    def fail():
        raise ValueError("Molecular descriptors unavailable")

    dialog._start(fail, lambda _: pytest.fail("A failed job has no result"))
    wait_for_job(dialog, qapp)
    assert "Molecular descriptors unavailable" in dialog.report.toPlainText()


def test_adoption_restores_all_previewed_criteria_and_undo(dialog, window):
    from smiles2select.gui.workspace import criteria_state
    from smiles2select.selection_intelligence.objectives import Direction, Objective
    from smiles2select.selection_intelligence.scenarios import evaluate_scenario

    window.target_count.setValue(3)
    window._auto_select()
    original = criteria_state.capture(window)
    original_ids = window.basket.final_ids()
    spec = replace(
        dialog.capture_spec(),
        objectives=(Objective("tpsa", Direction.TARGET_RANGE, target_low=20., target_high=100.),
                    Objective("mol_wt", Direction.MINIMIZE)),
        constraints=replace(window.constraints(), target_count=2, max_per_scaffold=1,
                            max_per_cluster=2),
    )
    snapshot = evaluate_scenario(window.result, window.candidates, spec)
    dialog.accept_snapshot(snapshot)
    dialog._load_slot()
    window.per_scaffold.setValue(4)
    window.per_cluster.setValue(7)
    dialog.adopt()
    assert window.constraints() == spec.constraints
    assert tuple(window.objectives()) == spec.objectives
    assert "[20, 100]" in window.criteria_summary.text()
    assert not window.has_pending_criteria()
    assert window.build_artifacts().recipe.objectives == tuple(o.as_dict() for o in spec.objectives)
    window._undo()
    assert criteria_state.capture(window) == original
    assert window.basket.final_ids() == original_ids
    window._redo()
    assert window.constraints() == spec.constraints
    assert tuple(window.objectives()) == spec.objectives
    assert not window.has_pending_criteria()


@pytest.mark.parametrize("variant", ["weight", "extra_objective", "precision", "minimum_scaffolds"])
def test_unrepresentable_scenario_does_not_acknowledge_false_criteria(dialog, window, variant):
    from smiles2select.gui.workspace import criteria_state
    from smiles2select.selection_intelligence.objectives import Direction, Objective
    from smiles2select.selection_intelligence.scenarios import evaluate_scenario

    spec = replace(dialog.capture_spec(), constraints=replace(window.constraints(), target_count=2))
    if variant == "weight":
        spec = replace(spec, objectives=(replace(spec.objectives[0], weight=2.), spec.objectives[1]))
    elif variant == "extra_objective":
        spec = replace(spec, objectives=(*spec.objectives, Objective("tpsa")))
    elif variant == "precision":
        spec = replace(spec, objectives=(
            Objective("qed", Direction.TARGET_VALUE, target_value=.12345), spec.objectives[1]))
    else:
        spec = replace(spec, constraints=replace(spec.constraints, min_scaffolds=1))
    snapshot = evaluate_scenario(window.result, window.candidates, spec)
    dialog.accept_snapshot(snapshot)
    dialog._load_slot()
    previous = criteria_state.capture(window)
    previous_ids = window.basket.final_ids()
    previous_actions = len(window.basket.log.applied)
    dialog.adopt()
    assert "cannot represent" in dialog.report.toPlainText()
    assert criteria_state.capture(window) == previous
    assert window.basket.final_ids() == previous_ids
    assert len(window.basket.log.applied) == previous_actions
