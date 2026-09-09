"""An adopted scenario must remain identifiable in the actual workbook."""

from dataclasses import replace

import pandas as pd

from smiles2select.export import selection_export
from smiles2select.selection_intelligence.scenario_io import load_study
from smiles2select.selection_intelligence.scenarios import ScenarioSpec, evaluate_scenario
from tests import test_workspace

qapp = test_workspace.qapp
result = test_workspace.result
window = test_workspace.window


def test_scenario_export_keeps_original_and_new_verdicts_distinct(window, tmp_path):
    snapshot = evaluate_scenario(window.result, window.candidates, ScenarioSpec("A"))
    window.basket.replace_final(snapshot.outcome.selected_ids)
    artifacts = replace(window.build_artifacts(), outcome=snapshot.outcome, scenario_snapshot=snapshot)
    path = tmp_path / "scenario.xlsx"
    selection_export.export(artifacts, path)
    selected = pd.read_excel(path, sheet_name="FINAL_SELECTED")
    assert selected.Final_Status.eq("SELECTED").all()
    assert selected.Scenario_Eligible.all()
    assert "Original_Screen_Status" in selected
    context = pd.read_excel(path, sheet_name="SCENARIO_CONTEXT").set_index("item")["value"]
    assert context["scenario"] == "A"
    assert context["input fingerprint"] == snapshot.data_fingerprint
    replay = load_study(path.with_suffix(".scenarios.json"), window.result, window.candidates)
    assert replay[0].outcome.selected_ids == snapshot.outcome.selected_ids


def test_final_sheet_materializes_only_requested_records(window, monkeypatch):
    selected_id = int(window.candidates.index[0])
    window.basket.add_to_final([selected_id])
    original = selection_export.export_frame
    requested = []

    def capture(result, *, record_ids=None):
        assert record_ids is not None
        requested.extend(record_ids)
        return original(result, record_ids=record_ids)

    monkeypatch.setattr(selection_export, "export_frame", capture)
    frame = selection_export.final_selected_sheet(window.build_artifacts())
    assert requested == [selected_id]
    assert frame.record_id.tolist() == requested
