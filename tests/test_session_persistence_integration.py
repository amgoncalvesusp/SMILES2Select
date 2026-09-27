"""Persistence contracts for a resumable selection basket."""

from __future__ import annotations

import json
import sqlite3

import pandas as pd
import pytest

from smiles2select.gui.workspace.criteria_state import CriteriaState
from smiles2select.gui.workspace.selection_provenance import AppliedSelection
from smiles2select.io.importer import ColumnMapping, SourceFile
from smiles2select.pipeline.config import RunConfig
from smiles2select.pipeline.runner import run
from smiles2select.selection_intelligence.basket import SelectionBasket
from smiles2select.selection_intelligence.constrained_selection import (
    SelectionConstraints,
    SelectionOutcome,
    Strategy,
)
from smiles2select.selection_intelligence.objectives import Objective, ObjectiveSet
from smiles2select.selection_intelligence.pareto_ranking import ParetoResult
from smiles2select.selection_intelligence.scenarios import ScenarioSnapshot, ScenarioSpec
from smiles2select.selection_intelligence.states import ChemicalStatus, MoleculeState
from smiles2select.storage.selection_store import SelectionStore
from smiles2select.storage.session_store import _decode, _encode, load_session, save_session


def _basket(*record_ids: int) -> SelectionBasket:
    return SelectionBasket(
        (MoleculeState(record_id, ChemicalStatus.AUTO_PASS) for record_id in record_ids),
        target_count=2,
    )


@pytest.mark.integration
def test_save_restores_cursor_redo_tail_and_target_count(tmp_path):
    basket = _basket(1, 2)
    basket.add_to_final([1])
    basket.add_to_final([2])
    basket.undo()
    database = tmp_path / "selection.sqlite"

    with SelectionStore(database) as store:
        store.save_basket(basket)
    with SelectionStore(database) as store:
        restored = store.load_basket()

    assert restored.final_ids() == (1,)
    assert restored.target_count == 2
    assert [entry["applied"] for entry in restored.log.history()] == [True, False]
    assert restored.log.can_redo
    restored.redo()
    assert restored.final_ids() == (1, 2)


@pytest.mark.integration
def test_save_replaces_entire_basket_and_removes_stale_records(tmp_path):
    database = tmp_path / "selection.sqlite"
    with SelectionStore(database) as store:
        store.save_basket(_basket(1, 2, 3))
        store.save_basket(_basket(1))
        restored = store.load_basket()

    assert tuple(state.record_id for state in restored.states()) == (1,)


@pytest.mark.integration
def test_failed_save_rolls_back_state_actions_and_cursor(tmp_path):
    database = tmp_path / "selection.sqlite"
    original = _basket(1, 2)
    original.add_to_final([1])
    original.add_to_final([2])
    original.undo()
    replacement = _basket(3)
    replacement.add_to_final([3])

    with SelectionStore(database) as store:
        store.save_basket(original)
        with sqlite3.connect(database) as connection:
            connection.execute(
                "CREATE TRIGGER fail_new_action BEFORE INSERT ON selection_actions "
                "BEGIN SELECT RAISE(FAIL, 'injected failure'); END"
            )
        with pytest.raises(sqlite3.IntegrityError, match="injected failure"):
            store.save_basket(replacement)
        restored = store.load_basket()

    assert tuple(state.record_id for state in restored.states()) == (1, 2)
    assert restored.final_ids() == (1,)
    assert restored.log.can_redo
    assert restored.target_count == 2


@pytest.mark.integration
def test_legacy_actions_without_cursor_restore_as_fully_applied(tmp_path):
    database = tmp_path / "selection.sqlite"
    basket = _basket(1)
    basket.add_to_final([1])
    with SelectionStore(database) as store:
        store.save_basket(basket)
        store._connection.execute("DELETE FROM selection_session")
        store._connection.commit()
        restored = store.load_basket(target_count=7)

    assert restored.final_ids() == (1,)
    assert not restored.log.can_redo
    assert restored.target_count == 7


@pytest.mark.integration
def test_full_session_reopens_after_source_moved_and_preserves_previous_file(tmp_path):
    source = tmp_path / "molecules.csv"
    pd.DataFrame(
        {
            "ID": ["aspirin", "caffeine"],
            "SMILES": ["CC(=O)Oc1ccccc1C(=O)O", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"],
        }
    ).to_csv(source, index=False)
    result = run(
        RunConfig(
            sources=(SourceFile(source, ColumnMapping(smiles="SMILES", molecule_id="ID")),),
            profile_ids=("lipinski", "veber"),
            n_jobs=1,
            chunk_size=2,
            database_path=tmp_path / "run.sqlite",
        )
    )
    basket = _basket(1, 2)
    basket.add_to_final([1])
    basket.add_to_final([2])
    basket.undo()
    session = tmp_path / "project.s2s.sqlite"
    criteria = CriteriaState(2, "balanced", 1, 0, (("qed", 0, 0.0, 1.0),))
    outcome = SelectionOutcome((1, 2), strategy=Strategy.BALANCED)
    pareto = ParetoResult(
        table=pd.DataFrame({"pareto_rank": [1, 2]}, index=pd.Index([1, 2], name="record_id")),
        objective_fields=("qed",),
        cache_key="pareto-hash",
        objective_values=pd.DataFrame(
            {"qed": [0.9, 0.8]}, index=pd.Index([1, 2], name="record_id")
        ),
    )
    applied = AppliedSelection.capture(
        constraints=SelectionConstraints(target_count=2),
        objectives=ObjectiveSet((Objective("qed"),)),
        pareto=pareto,
        strategy="model",
        input_hash="input-hash",
        provenance={"model_sha256": "abc"},
    )
    scenario = ScenarioSnapshot(
        spec=ScenarioSpec(name="control"),
        data_fingerprint="fixture",
        universe_ids=pd.Index([1, 2], name="record_id"),
        eligible_ids=pd.Index([1, 2], name="record_id"),
        outcome=outcome,
        warnings=(),
        provenance={"source": "fixture"},
        scaffolds=pd.Series(["a", "b"], index=pd.Index([1, 2], name="record_id")),
    )
    save_session(
        session,
        result,
        basket,
        workspace_state={
            "initial_criteria": criteria,
            "draft_criteria": criteria,
            "criteria_snapshots": {1: criteria, 2: criteria},
            "applied_selections": {1: applied},
            "selection_outcomes": {1: outcome},
            "model_scores_by_action": {1: pd.DataFrame({"score": [0.9, 0.8]}, index=[1, 2])},
            "run_provenance": {"model_sha256": "abc"},
            "scenario_snapshots": {1: scenario},
        },
    )
    original_bytes = session.read_bytes()
    source.rename(tmp_path / "moved.csv")

    restored_result, restored_basket, workspace = load_session(session)

    for name in ("records", "descriptors", "alerts", "scores"):
        pd.testing.assert_frame_equal(getattr(result, name), getattr(restored_result, name))
    pd.testing.assert_frame_equal(result.evaluation.status, restored_result.evaluation.status)
    pd.testing.assert_frame_equal(result.evaluation.failures, restored_result.evaluation.failures)
    pd.testing.assert_frame_equal(result.decision.decisions, restored_result.decision.decisions)
    assert restored_result.config.sources[0].path == source
    assert restored_result.profile_summary().equals(result.profile_summary())
    assert restored_basket.final_ids() == (1,)
    assert workspace["applied_selections"][1].strategy == "model"
    assert workspace["applied_selections"][1].provenance["model_sha256"] == "abc"
    pd.testing.assert_frame_equal(workspace["applied_selections"][1].pareto.table, pareto.table)
    assert workspace["criteria_snapshots"][1] == criteria
    assert 2 not in workspace["criteria_snapshots"]
    assert workspace["draft_criteria"] == criteria
    assert workspace["selection_outcomes"][1].strategy is Strategy.BALANCED
    assert workspace["model_scores_by_action"][1].loc[1, "score"] == 0.9
    assert workspace["scenario_snapshots"][1].scaffolds.equals(scenario.scaffolds)
    restored_basket.redo()
    assert restored_basket.final_ids() == (1, 2)

    with pytest.raises(TypeError, match="RunResult"):
        save_session(session, object(), basket)
    assert session.read_bytes() == original_bytes

    save_session(session, restored_result, restored_basket, workspace_state=workspace)
    _, saved_again, repeated_workspace = load_session(session)
    assert saved_again.final_ids() == (1, 2)
    assert repeated_workspace["applied_selections"][1].strategy == "model"


@pytest.mark.integration
def test_full_session_detects_payload_tampering(tmp_path):
    source = tmp_path / "one.csv"
    pd.DataFrame({"ID": ["one"], "SMILES": ["CCO"]}).to_csv(source, index=False)
    result = run(
        RunConfig(
            sources=(SourceFile(source, ColumnMapping(smiles="SMILES", molecule_id="ID")),),
            profile_ids=("lipinski",),
            n_jobs=1,
        )
    )
    session = tmp_path / "session.sqlite"
    save_session(session, result, _basket(1))
    with sqlite3.connect(session) as connection:
        original_payload = connection.execute(
            "SELECT payload FROM session_payload WHERE session_key = 1"
        ).fetchone()[0]
        connection.execute("UPDATE session_payload SET payload = '{}' WHERE session_key = 1")

    with pytest.raises(ValueError, match="integrity"):
        load_session(session)

    with sqlite3.connect(session) as connection:
        connection.execute(
            "UPDATE session_payload SET payload = ? WHERE session_key = 1", (original_payload,)
        )
        connection.execute("UPDATE selection_state SET pinned = 1 WHERE record_id = 1")
    with pytest.raises(ValueError, match="basket integrity"):
        load_session(session)

    with sqlite3.connect(session) as connection:
        connection.execute("UPDATE selection_state SET pinned = 0 WHERE record_id = 1")
        frame_id = connection.execute(
            "SELECT frame_id FROM session_frame_rows WHERE row_number = 0 LIMIT 1"
        ).fetchone()[0]
        connection.execute(
            "UPDATE session_frame_rows SET row_json = '[1, []]' "
            "WHERE frame_id = ? AND row_number = 0",
            (frame_id,),
        )
    with pytest.raises(ValueError, match="frame integrity"):
        load_session(session)


@pytest.mark.integration
def test_session_materializes_reference_and_zone_results_without_run_database(tmp_path):
    candidates = tmp_path / "candidates.csv"
    references = tmp_path / "references.csv"
    pd.DataFrame({"ID": ["one", "two", "three"], "SMILES": ["CCO", "CCC", "CCCC"]}).to_csv(
        candidates, index=False
    )
    pd.DataFrame({"ID": ["reference"], "SMILES": ["CCO"]}).to_csv(references, index=False)
    result = run(
        RunConfig(
            sources=(SourceFile(candidates, ColumnMapping("SMILES", "ID")),),
            reference_sources=(SourceFile(references, ColumnMapping("SMILES", "ID")),),
            profile_ids=("lipinski",),
            alert_catalogs=(),
            n_jobs=1,
            final_count=2,
            reserve_count=1,
            zones=(
                {"zone_id": "polar", "name": "Polar", "expression": "mol_wt >= 0", "quota": 1},
                {"zone_id": "small", "name": "Small", "expression": "mol_wt <= 100", "quota": 1},
            ),
        )
    )
    assert result.database_path is None
    assert result.reference_similarity is not None
    assert result.zone_allocation is not None
    session = tmp_path / "project.sqlite"
    save_session(session, result, _basket(1, 2, 3))
    candidates.rename(tmp_path / "candidates.moved.csv")
    references.rename(tmp_path / "references.moved.csv")

    restored, _, _ = load_session(session)

    assert restored.database_path == session
    assert restored.reference_libraries[0].molecules.equals(result.reference_libraries[0].molecules)
    assert restored.reference_similarity.pairs.equals(result.reference_similarity.pairs)
    assert restored.zone_allocation.allocation_table.equals(result.zone_allocation.allocation_table)
    assert restored.reserve_ids == result.reserve_ids
    with sqlite3.connect(session) as connection:
        assert connection.execute("SELECT COUNT(*) FROM molecule_descriptors").fetchone()[0] == 3


@pytest.mark.integration
def test_relative_session_path_reopens_with_read_only_store(tmp_path, monkeypatch):
    source = tmp_path / "one.csv"
    pd.DataFrame({"ID": ["one"], "SMILES": ["CCO"]}).to_csv(source, index=False)
    result = run(
        RunConfig(
            sources=(SourceFile(source, ColumnMapping("SMILES", "ID")),),
            profile_ids=("lipinski",),
            n_jobs=1,
        )
    )
    monkeypatch.chdir(tmp_path)
    save_session("relative.sqlite", result, _basket(1))

    restored, basket, _ = load_session("relative.sqlite")

    assert restored.database_path == tmp_path / "relative.sqlite"
    assert basket.state(1).record_id == 1
    with SelectionStore("relative.sqlite", readonly=True) as store:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            store._connection.execute("UPDATE selection_state SET pinned = 1")


@pytest.mark.integration
def test_large_dataframe_rows_stay_out_of_manifest():
    frame = pd.DataFrame({"value": range(10_000), "text": ["x" * 100] * 10_000})
    with sqlite3.connect(":memory:") as connection:
        connection.execute(
            "CREATE TABLE session_frames (frame_id TEXT PRIMARY KEY, metadata_json TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE session_frame_rows (frame_id TEXT NOT NULL, "
            "row_number INTEGER NOT NULL, row_json TEXT NOT NULL, "
            "PRIMARY KEY (frame_id, row_number))"
        )
        reference = _encode(frame, connection)
        assert len(json.dumps(reference)) < 200
        assert connection.execute("SELECT COUNT(*) FROM session_frame_rows").fetchone()[0] == 10_000
        restored = _decode(reference, connection)

    pd.testing.assert_frame_equal(restored, frame)


def test_frame_axis_and_category_metadata_round_trip():
    frame = pd.DataFrame({"status": pd.Categorical(
        ["low", "high"], categories=["low", "high", "unused"], ordered=True,
    )}).rename_axis(index="row", columns="kind")
    with sqlite3.connect(":memory:") as connection:
        connection.execute(
            "CREATE TABLE session_frames (frame_id TEXT PRIMARY KEY, metadata_json TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE session_frame_rows (frame_id TEXT NOT NULL, "
            "row_number INTEGER NOT NULL, row_json TEXT NOT NULL, "
            "PRIMARY KEY (frame_id, row_number))"
        )
        reference = _encode(frame, connection)
        restored = _decode(reference, connection)
    pd.testing.assert_frame_equal(restored, frame)


def test_unsupported_frame_axes_fail_before_save():
    frame = pd.DataFrame({"value": [1, 2]})
    with sqlite3.connect(":memory:") as connection:
        connection.execute(
            "CREATE TABLE session_frames (frame_id TEXT PRIMARY KEY, metadata_json TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE session_frame_rows (frame_id TEXT NOT NULL, "
            "row_number INTEGER NOT NULL, row_json TEXT NOT NULL, "
            "PRIMARY KEY (frame_id, row_number))"
        )
        with pytest.raises(ValueError, match="MultiIndex"):
            _encode(frame.set_axis(pd.MultiIndex.from_tuples([("a", 1), ("b", 2)])), connection)
        with pytest.raises(ValueError, match="duplicate"):
            _encode(pd.DataFrame([[1, 2]], columns=["same", "same"]), connection)
        categorical_axis = pd.CategoricalIndex(
            ["a", "b"], categories=["a", "b", "unused"], ordered=True,
        )
        with pytest.raises(ValueError, match="CategoricalIndex"):
            _encode(frame.set_axis(categorical_axis), connection)
        with pytest.raises(ValueError, match="CategoricalIndex"):
            _encode(frame.assign(other=[3, 4]).set_axis(categorical_axis, axis="columns"), connection)
