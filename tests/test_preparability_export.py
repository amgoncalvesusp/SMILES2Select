"""Integration tests for docking preparability export (SQLite, Excel, Docking CSV)."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from smiles2select.export import docking, excel
from smiles2select.io.importer import ColumnMapping, SourceFile
from smiles2select.pipeline.config import RunConfig
from smiles2select.pipeline.runner import run

pytestmark = pytest.mark.integration

SAMPLE_ROWS = [
    ("MOL001", "CC(O)C(N)C=CC1CCCCC1"),  # 3 undefined stereocenters -> 8 structures
    ("MOL002", "OB(O)c1ccccc1"),  # Boron -> UNCOMMON_ELEMENT
    ("MOL003", "CCO"),  # Clean molecule
]


def make_config(csv, **overrides) -> RunConfig:
    settings = {
        "sources": (
            SourceFile(path=csv, mapping=ColumnMapping(smiles="SMILES", molecule_id="ID")),
        ),
        "profile_ids": ("lipinski",),
        "alert_catalogs": (),
        "n_jobs": 1,
        "chunk_size": 10,
    }
    settings.update(overrides)
    return RunConfig(**settings)


def test_sqlite_has_preparability_table(tmp_path):
    csv = tmp_path / "molecules.csv"
    pd.DataFrame(SAMPLE_ROWS, columns=["ID", "SMILES"]).to_csv(csv, index=False)
    db_path = tmp_path / "run.sqlite"

    config = make_config(csv, compute_preparability=True, database_path=db_path)
    result = run(config)
    assert result.preparability is not None
    assert not result.preparability.empty

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='preparability_flags'"
    )
    assert cursor.fetchone() is not None

    df_sql = pd.read_sql_query("SELECT * FROM preparability_flags", conn)
    conn.close()
    assert len(df_sql) >= 2
    assert "flag_id" in df_sql.columns
    assert "severity" in df_sql.columns
    assert "detail" in df_sql.columns


def test_excel_has_preparability_sheet(tmp_path):
    csv = tmp_path / "molecules.csv"
    pd.DataFrame(SAMPLE_ROWS, columns=["ID", "SMILES"]).to_csv(csv, index=False)
    excel_path = tmp_path / "report.xlsx"

    config = make_config(csv, compute_preparability=True)
    result = run(config)
    excel.export(result, excel_path)

    xls = pd.ExcelFile(excel_path)
    assert "PREPARABILITY_FLAGS" in xls.sheet_names
    df_sheet = pd.read_excel(xls, "PREPARABILITY_FLAGS")
    assert "ID" in df_sheet.columns
    assert "SMILES" in df_sheet.columns
    assert "flag_id" in df_sheet.columns
    assert "detail" in df_sheet.columns


def test_docking_csv_includes_preparability_columns(tmp_path):
    csv = tmp_path / "molecules.csv"
    pd.DataFrame(SAMPLE_ROWS, columns=["ID", "SMILES"]).to_csv(csv, index=False)
    docking_csv = tmp_path / "docking.csv"

    config = make_config(csv, compute_preparability=True)
    result = run(config)
    docking.export(result, docking_csv, docking.DockingExportOptions(selected_only=False))

    df_dock = pd.read_csv(docking_csv)
    assert "undefined_stereocenters" in df_dock.columns
    assert "estimated_3d_structures" in df_dock.columns
    assert "preparability_flags_count" in df_dock.columns

    mol1 = df_dock.loc[df_dock["access_code"] == "MOL001"].iloc[0]
    assert mol1["undefined_stereocenters"] == 3
    assert mol1["estimated_3d_structures"] == 8
