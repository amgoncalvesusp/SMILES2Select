"""Applied criteria and data identity must survive edits to live GUI state."""

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from smiles2select.gui.workspace.selection_provenance import (
    AppliedSelection,
    input_data_hash,
    run_provenance,
)
from smiles2select.io.importer import ColumnMapping, SourceFile
from smiles2select.pipeline.config import RunConfig
from smiles2select.pipeline.runner import run
from smiles2select.selection_intelligence.constrained_selection import (
    SelectionConstraints,
    Strategy,
)
from smiles2select.selection_intelligence.objectives import Objective, ObjectiveSet
from smiles2select.selection_intelligence.pareto_ranking import rank_candidates
from smiles2select.selection_intelligence.recipes import SelectionRecipe, from_dict


def test_input_hash_identifies_data_order_columns_and_values():
    frame = pd.DataFrame({"SMILES": ["CCO", "CCN"], "score": [0.4, 0.8]}, index=[1, 2])
    original = input_data_hash(frame)
    assert original == input_data_hash(frame.copy(deep=True))
    assert len(original) == 64
    assert original != input_data_hash(frame.iloc[::-1])
    assert original != input_data_hash(frame.assign(SMILES=["CCC", "CCN"]))
    assert original != input_data_hash(frame.assign(score=[0.4000000000000001, 0.8]))
    assert original != input_data_hash(frame.rename(columns={"SMILES": "canonical_smiles"}))
    assert original != input_data_hash(frame, frame)


def test_applied_snapshot_is_deeply_detached_and_preserves_infinite_pareto_values():
    objectives = ObjectiveSet([Objective("qed")])
    pareto = rank_candidates(pd.DataFrame({"qed": [0.2, 0.9]}, index=[2, 1]), objectives)
    expected = pareto.table.copy(deep=True)
    metadata = {"algorithm": {"seed": 42}, "selected_ids": [1]}
    snapshot = AppliedSelection.capture(
        constraints=SelectionConstraints(target_count=1), objectives=objectives,
        pareto=pareto, strategy=Strategy.BALANCED, input_hash="data-hash", provenance=metadata,
    )
    pareto.table.loc[1, "crowding_distance"] = -1
    metadata["algorithm"]["seed"] = 9
    snapshot.provenance["selected_ids"].append(2)
    snapshot.objectives[0]["field"] = "mol_wt"
    returned = snapshot.pareto
    returned.table.loc[1, "pareto_rank"] = 999
    pd.testing.assert_frame_equal(snapshot.pareto.table, expected)
    assert snapshot.objectives[0]["field"] == "qed"
    assert snapshot.provenance == {"algorithm": {"seed": 42}, "selected_ids": [1]}
    with pytest.raises(FrozenInstanceError):
        snapshot.input_hash = "changed"


def test_pipeline_snapshot_has_no_unapplied_gui_objectives_or_pareto():
    snapshot = AppliedSelection.capture(
        constraints=SelectionConstraints(target_count=3), objectives=ObjectiveSet(),
        pareto=None, strategy="traditional", input_hash="data-hash",
        provenance={"source": "original_pipeline"},
    )
    assert snapshot.objectives == ()
    assert snapshot.pareto is None
    assert snapshot.strategy == "traditional"
    assert snapshot.provenance["source"] == "original_pipeline"


def test_recipe_roundtrip_preserves_final_ids_and_applied_provenance():
    recipe = SelectionRecipe(
        final_selected_ids=(7, 2), provenance={"source": "original_pipeline", "run_config": {"seed": 7}},
    )
    assert recipe.as_dict()["final_selection"]["selected_ids"] == [7, 2]
    restored = from_dict(recipe.as_dict())
    assert restored.final_selected_ids == (7, 2)
    assert restored.provenance == recipe.provenance
    assert from_dict({"schema_version": "3.0"}).final_selected_ids == ()


def test_real_run_provenance_separates_input_and_selection_data(library_csv):
    result = run(RunConfig(
        sources=(SourceFile(path=library_csv, mapping=ColumnMapping(smiles="SMILES", molecule_id="ID")),),
        profile_ids=("lipinski",), alert_catalogs=(), n_jobs=1,
    ))
    context = run_provenance(result, result.descriptors)
    changed = run_provenance(result, result.descriptors.assign(qed=0.123))
    assert context["input_hash"] == changed["input_hash"]
    assert context["selection_data_hash"] != changed["selection_data_hash"]
    assert context["run_config"]["profile_ids"] == ["lipinski"]
    assert context["profiles"][0]["rules"]
    assert context["software_versions"]["pandas"] == pd.__version__
    snapshot = AppliedSelection.capture(
        constraints=SelectionConstraints(), objectives=ObjectiveSet(), pareto=None,
        strategy="traditional", input_hash=context["input_hash"], provenance=context,
    )
    assert snapshot.provenance == context
