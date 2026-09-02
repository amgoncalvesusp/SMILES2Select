"""Integration tests for docking preparability in the pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from smiles2select.io.importer import ColumnMapping, SourceFile
from smiles2select.pipeline.config import RunConfig
from smiles2select.pipeline.runner import run

pytestmark = pytest.mark.integration

SAMPLE_ROWS = [
    ("MOL001", "CC(O)C(N)C=CC1CCCCC1"),  # 3 undefined stereocenters
    ("MOL002", "C[C@H](O)C(=O)O"),  # 1 defined stereocenter
    ("MOL003", "CCO"),  # simple alcohol
]


def make_config(csv: Path, **overrides) -> RunConfig:
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


def test_readiness_off_by_default(tmp_path):
    csv = tmp_path / "molecules.csv"
    pd.DataFrame(SAMPLE_ROWS, columns=["ID", "SMILES"]).to_csv(csv, index=False)
    config = make_config(csv)
    assert not config.compute_preparability
    result = run(config)
    assert "undefined_stereocenters" not in result.descriptors.columns


def test_readiness_computes_descriptors(tmp_path):
    csv = tmp_path / "molecules.csv"
    pd.DataFrame(SAMPLE_ROWS, columns=["ID", "SMILES"]).to_csv(csv, index=False)
    config = make_config(csv, compute_preparability=True, docking_engine="vina")
    result = run(config)
    assert "undefined_stereocenters" in result.descriptors.columns
    assert "defined_stereocenters" in result.descriptors.columns
    assert "fragment_count" in result.descriptors.columns
    assert "largest_ring_size" in result.descriptors.columns
    assert "amide_bond_count" in result.descriptors.columns

    row1 = result.descriptors.loc[result.descriptors["molecule_id"] == "MOL001"].iloc[0]
    assert row1["undefined_stereocenters"] == 3


def test_tautomers_require_explicit_flag(tmp_path):
    csv = tmp_path / "molecules.csv"
    pd.DataFrame(SAMPLE_ROWS, columns=["ID", "SMILES"]).to_csv(csv, index=False)

    config_without = make_config(csv, compute_preparability=True, compute_tautomers=False)
    result_without = run(config_without)
    assert "tautomer_count" not in result_without.descriptors.columns

    config_with = make_config(csv, compute_preparability=True, compute_tautomers=True)
    result_with = run(config_with)
    assert "tautomer_count" in result_with.descriptors.columns


def test_fingerprint_changes_with_readiness(tmp_path):
    csv = tmp_path / "molecules.csv"
    pd.DataFrame(SAMPLE_ROWS, columns=["ID", "SMILES"]).to_csv(csv, index=False)

    base_config = make_config(csv)
    readiness_config = make_config(csv, compute_preparability=True)
    tautomer_config = make_config(csv, compute_preparability=True, compute_tautomers=True)
    engine_config = make_config(
        csv, compute_preparability=True, compute_tautomers=True, docking_engine="gold"
    )

    hashes = {
        base_config.fingerprint(),
        readiness_config.fingerprint(),
        tautomer_config.fingerprint(),
        engine_config.fingerprint(),
    }
    assert len(hashes) == 4
