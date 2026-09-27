"""Model ranking must remain distinct from the shared quota allocator."""

from dataclasses import replace

import pandas as pd

from smiles2select.export import selection_export
from smiles2select.gui.workspace.selection_provenance import AppliedSelection
from smiles2select.selection_intelligence.constrained_selection import SelectionOutcome
from smiles2select.selection_intelligence.recipes import from_dict
from tests import test_workspace

qapp = test_workspace.qapp
result = test_workspace.result
window = test_workspace.window


def test_model_adoption_export_identifies_model_and_undo_restores_original(window, tmp_path):
    original = window.build_artifacts().recipe.strategy
    ids = tuple(window.candidates.index[:2])
    applied = AppliedSelection.capture(
        constraints=window.constraints(),
        objectives=window.objectives(),
        pareto=None,
        strategy="experimental_model",
        input_hash="model-input",
        provenance={
            "source": "experimental_model",
            "ranking_method": "onnx_score",
            "model_sha256": "a" * 64,
            "model_manifest_sha256": "b" * 64,
            "reference_records_sha256": "c" * 64,
            "target": "example-target",
            "endpoint": "IC50",
        },
    )
    window._apply_selection(SelectionOutcome(selected_ids=ids), window.constraints(), applied)
    index = len(window.basket.log.applied) - 1
    window._model_scores_by_action = {
        index: pd.DataFrame(
            {"priority_score": [0.8, 0.4], "calibration_status": ["calibrated"] * 2},
            index=pd.Index(ids, name="record_id"),
        )
    }
    artifacts = window.build_artifacts()
    assert artifacts.recipe.strategy == "experimental_model"
    assert artifacts.recipe.provenance["model_sha256"] == "a" * 64
    payload = artifacts.recipe.as_dict()
    assert payload["schema_version"] == "4.0"
    assert payload["model"]["model_sha256"] == "a" * 64
    assert payload["model"]["reference_records_sha256"] == "c" * 64
    assert from_dict(payload).model == payload["model"]
    from smiles2select.selection_intelligence.recipes import compare

    changed = replace(artifacts.recipe, model={**artifacts.recipe.model,
                                               "reference_records_sha256": "d" * 64})
    assert "model or target changed" in compare(artifacts.recipe, changed)
    assert artifacts.model_scores is not None
    final = selection_export.final_selected_sheet(artifacts)
    assert final.set_index("record_id").loc[ids[0], "Model_Priority_Score"] == 0.8
    path, _ = selection_export.export(artifacts, tmp_path / "model.xlsx")
    assert "MODEL_SCORES" in pd.ExcelFile(path).sheet_names
    window._undo()
    assert window.build_artifacts().recipe.strategy == original
    assert window.build_artifacts().model_scores is None
    window._redo()
    assert window.build_artifacts().recipe.strategy == "experimental_model"
    assert window.build_artifacts().model_scores is not None
    window._undo()
    window._apply_selection(SelectionOutcome(selected_ids=ids[:1]), window.constraints())
    assert window.build_artifacts().recipe.strategy != "experimental_model"
    assert window.build_artifacts().model_scores is None
