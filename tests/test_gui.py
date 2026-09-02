"""Interface: wizard state, page construction and chart rendering.

Runs against the offscreen Qt platform, so no display is required.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from smiles2select.app_metadata import app_icon_path  # noqa: E402
from smiles2select.gui import charts  # noqa: E402
from smiles2select.gui.main_window import STEPS, MainWindow  # noqa: E402
from smiles2select.gui.state import FileSelection, WizardState  # noqa: E402
from smiles2select.pipeline.config import RunConfig  # noqa: E402
from smiles2select.pipeline.runner import run  # noqa: E402
from smiles2select.scores.qed import QedSelection  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    window = MainWindow()
    yield window
    window.close()


def test_wizard_has_the_seven_specified_steps(window):
    assert window.pages.count() == 7
    assert len(STEPS) == 7
    titles = [window.pages.widget(index).title for index in range(7)]
    assert titles == list(STEPS)


def test_application_icon_is_loaded(window):
    assert app_icon_path().is_file()
    assert not window.windowIcon().isNull()
    assert window.windowIcon().availableSizes()


def test_every_page_builds_and_can_be_entered(window):
    for index in range(window.pages.count()):
        window.pages.widget(index).on_enter()


def test_navigation_is_blocked_until_a_file_is_chosen(window, monkeypatch):
    """The warning dialog is modal, so it is stubbed out here."""
    warnings: list[str] = []
    monkeypatch.setattr(
        "smiles2select.gui.main_window.QMessageBox.warning",
        lambda _parent, _title, message: warnings.append(message),
    )
    assert window.pages.currentIndex() == 0
    window._go(1)
    assert window.pages.currentIndex() == 0  # blocked: no file selected
    assert warnings and "file" in warnings[0]


def test_state_defaults_follow_the_recommended_policy():
    policy = WizardState().build_policy()
    assert set(policy.mandatory_profiles()) == {"lipinski", "veber"}
    assert set(policy.informative_profiles()) == {"ghose", "egan", "muegge"}
    assert policy.qed.mode == "rank"
    assert policy.alert_policy.excluding_catalogs() == ()


def test_state_reports_missing_inputs():
    problems = WizardState().validation_errors()
    assert any("file" in problem for problem in problems)


def test_state_builds_a_config_once_complete(library_csv):
    state = WizardState()
    state.files.append(FileSelection(path=library_csv, smiles_column="SMILES", id_column="ID"))
    config = state.build_config()
    assert config.profile_ids == ("egan", "ghose", "lipinski", "muegge", "veber")
    assert config.policy.mandatory_profiles() == ("lipinski", "veber")


def test_removing_every_profile_is_reported(library_csv):
    state = WizardState()
    state.files.append(FileSelection(path=library_csv, smiles_column="SMILES"))
    for profile_id in list(state.roles):
        state.set_role(profile_id, None)
    assert any("profile" in problem for problem in state.validation_errors())


def test_qed_exclusion_without_qed_computation_is_reported(library_csv):
    state = WizardState()
    state.files.append(FileSelection(path=library_csv, smiles_column="SMILES"))
    state.compute_qed = False
    state.qed = QedSelection(mode="threshold", threshold=0.5)
    assert any("QED" in problem for problem in state.validation_errors())


def test_charts_render_without_data(qapp):
    canvas = charts.Canvas()
    charts.profile_bars(canvas, pd.DataFrame(columns=["profile_id", "percentage"]))
    charts.violation_counts(canvas, pd.DataFrame(columns=["failure_code"]))
    charts.intersection_matrix(canvas, pd.DataFrame(), [])
    charts.descriptor_histogram(canvas, pd.DataFrame({"mol_wt": []}), "mol_wt")


def test_charts_render_with_data(qapp):
    canvas = charts.Canvas()
    summary = pd.DataFrame({"profile_id": ["lipinski", "veber"], "percentage": [90.0, 75.0]})
    charts.profile_bars(canvas, summary)

    failures = pd.DataFrame({"failure_code": ["LIP_MW_HIGH", "LIP_MW_HIGH", "VEB_TPSA_HIGH"]})
    charts.violation_counts(canvas, failures)

    status = pd.DataFrame(
        {"lipinski__passed": [True, True, False], "veber__passed": [True, False, False]}
    )
    charts.intersection_matrix(canvas, status, ["lipinski", "veber"])

    descriptors = pd.DataFrame({"mol_wt": [180.0, 500.0, 620.0]})
    selected = pd.Series([True, True, False], index=descriptors.index)
    charts.descriptor_histogram(canvas, descriptors, "mol_wt", selected)


def test_results_page_populates_from_a_run(window, library_csv):
    state = window.state
    state.files.append(FileSelection(path=library_csv, smiles_column="SMILES", id_column="ID"))
    result = run(
        RunConfig(
            sources=tuple(selection.to_source() for selection in state.files),
            profile_ids=("lipinski", "veber"),
            alert_catalogs=("brenk",),
            n_jobs=1,
            chunk_size=4,
        )
    )

    results_page = window.pages.widget(6)
    results_page.show_result(result)
    assert "selected" in results_page.headline.text()
    assert results_page.table.rowCount() > 0


def test_results_page_exports_all_charts(window, library_csv, tmp_path):
    state = window.state
    state.files.append(FileSelection(path=library_csv, smiles_column="SMILES", id_column="ID"))
    result = run(
        RunConfig(
            sources=tuple(selection.to_source() for selection in state.files),
            profile_ids=("lipinski", "veber"),
            alert_catalogs=("brenk",),
            n_jobs=1,
            chunk_size=4,
        )
    )

    results_page = window.pages.widget(6)
    results_page.show_result(result)
    paths = results_page.export_charts(tmp_path / "charts")

    assert len(paths) == 8
    assert all(isinstance(path, Path) and path.exists() for path in paths)
    assert {path.suffix for path in paths} == {".png", ".svg"}


def test_gui_docking_preparability_run(window, tmp_path):
    csv = tmp_path / "test_mols.csv"
    pd.DataFrame(
        [
            ("MOL1", "CC(O)C(N)C=CC1CCCCC1"),
            ("MOL2", "OB(O)c1ccccc1"),
            ("MOL3", "CCO"),
        ],
        columns=["ID", "SMILES"],
    ).to_csv(csv, index=False)

    state = window.state
    state.files.append(FileSelection(path=csv, smiles_column="SMILES", id_column="ID"))
    state.n_jobs = 1
    state.chunk_size = 10

    policy_page = window.pages.widget(4)
    policy_page.on_enter()
    assert not policy_page.docking_prep_box.isChecked()

    policy_page.docking_prep_box.setChecked(True)
    assert policy_page.docking_engine_combo.isEnabled()
    assert policy_page.tautomers_box.isEnabled()
    assert state.compute_preparability

    config = state.build_config()
    assert config.compute_preparability
    assert config.docking_engine == "vina"

    result = run(config)
    assert result.preparability is not None
    assert not result.preparability.empty
    assert len(result.preparability) >= 2
