"""Parquet export and the profile failure bitmask."""

from __future__ import annotations

import pandas as pd
import pytest

from smiles2select.export import parquet
from smiles2select.io.importer import ColumnMapping, SourceFile
from smiles2select.pipeline.config import RunConfig
from smiles2select.pipeline.runner import run
from smiles2select.rules.engine import Rule
from smiles2select.rules.evaluator import evaluate_profiles
from smiles2select.storage.sqlite_store import SqliteStore
from tests.conftest import REFERENCE_SMILES, descriptor_frame
from tests.test_rules import make_profile

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    directory = tmp_path_factory.mktemp("parquet")
    csv = directory / "library.csv"
    pd.DataFrame(
        [
            ("MOL001", REFERENCE_SMILES["aspirin"]),
            ("MOL002", REFERENCE_SMILES["long_alkane"]),
            ("MOL003", REFERENCE_SMILES["ibuprofen"]),
        ],
        columns=["ID", "SMILES"],
    ).to_csv(csv, index=False)

    return run(
        RunConfig(
            sources=(
                SourceFile(path=csv, mapping=ColumnMapping(smiles="SMILES", molecule_id="ID")),
            ),
            profile_ids=("lipinski", "veber"),
            alert_catalogs=(),
            n_jobs=1,
            chunk_size=3,
            database_path=directory / "run.sqlite",
        )
    )


def test_failure_mask_marks_the_broken_rule_positions():
    """Bit i corresponds to rule i in the profile's own order."""
    profile = make_profile(
        [
            Rule("mw", "test", "mol_wt", "<=", 500, failure_code="MW_HIGH"),
            Rule("tpsa", "test", "tpsa", "<=", 140, failure_code="TPSA_HIGH"),
            Rule("rotb", "test", "rotatable_bonds", "<=", 10, failure_code="ROTB_HIGH"),
        ],
        policy={"type": "max_violations", "value": 2},
    )
    frame = descriptor_frame(
        {
            "mol_wt": [100.0, 600.0, 600.0, 100.0],
            "tpsa": [50.0, 50.0, 50.0, 300.0],
            "rotatable_bonds": [2, 2, 20, 2],
        }
    )
    masks = evaluate_profiles(frame, [profile]).failure_mask("test").tolist()
    assert masks == [0b000, 0b001, 0b101, 0b010]


def test_failure_mask_reaches_the_database(result):
    with SqliteStore(result.database_path) as store:
        rows = store.read_frame(
            "SELECT record_id, profile_id, passed, failure_mask FROM profile_results "
            "WHERE profile_id = 'veber' ORDER BY record_id"
        )
    assert rows["failure_mask"].notna().all()
    failed = rows[rows["passed"] == 0]
    assert (failed["failure_mask"] > 0).all()
    assert (rows[rows["passed"] == 1]["failure_mask"] == 0).all()


def test_parquet_export_round_trips(result, tmp_path):
    pytest.importorskip("pyarrow")
    path = parquet.export(result, tmp_path / "all.parquet")
    frame = pd.read_parquet(path)
    assert len(frame) == result.total_records
    assert "record_id" in frame.columns
    assert "Lipinski_Status" in frame.columns


def test_parquet_can_hold_only_the_selection(result, tmp_path):
    pytest.importorskip("pyarrow")
    path = parquet.export(result, tmp_path / "selected.parquet", selected_only=True)
    frame = pd.read_parquet(path)
    assert len(frame) == result.decision.selected_count
    assert set(frame["Final_Status"]) == {"SELECTED"}
