"""Tests for docking preparability evaluation."""

from __future__ import annotations

import pandas as pd
import pytest

from smiles2select.preparability.engines import load_engine
from smiles2select.preparability.evaluation import evaluate

pytestmark = pytest.mark.unit


def test_clean_molecule_raises_no_flags():
    engine = load_engine("vina")
    df = pd.DataFrame(
        [
            {
                "record_id": 1,
                "canonical_smiles": "CCO",
                "fragment_count": 1,
                "largest_ring_size": 0,
                "amide_bond_count": 0,
                "undefined_stereocenters": 0,
            }
        ]
    )
    result = evaluate(df, engine)
    assert len(result) == 0
    assert list(result.columns) == ["record_id", "flag_id", "severity", "detail"]


def test_boron_molecule_raises_uncommon_element():
    engine = load_engine("vina")
    # Phenylboronic acid: contains Boron (B), which is not in common organic set
    df = pd.DataFrame(
        [
            {
                "record_id": 2,
                "canonical_smiles": "OB(O)c1ccccc1",
                "fragment_count": 1,
                "largest_ring_size": 6,
                "amide_bond_count": 0,
                "undefined_stereocenters": 0,
            }
        ]
    )
    result = evaluate(df, engine)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["record_id"] == 2
    assert row["flag_id"] == "UNCOMMON_ELEMENT"
    assert row["severity"] == "verify"
    assert "B" in row["detail"]


def test_undefined_stereo_generates_correct_detail():
    engine = load_engine("vina")
    # CC(O)C(N)C=CC1CCCCC1 has 3 undefined stereocenters -> 2^3 = 8 structures
    df = pd.DataFrame(
        [
            {
                "record_id": 3,
                "canonical_smiles": "CC(O)C(N)C=CC1CCCCC1",
                "fragment_count": 1,
                "largest_ring_size": 6,
                "amide_bond_count": 0,
                "undefined_stereocenters": 3,
            }
        ]
    )
    result = evaluate(df, engine)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["record_id"] == 3
    assert row["flag_id"] == "UNDEFINED_STEREO"
    assert row["severity"] == "decide"
    assert "8 estruturas" in row["detail"]
    assert "3 estereocentros indefinidos" in row["detail"]
