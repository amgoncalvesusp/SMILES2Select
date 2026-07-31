"""Structural alerts: detection, actions and the flag/exclude boundary."""

from __future__ import annotations

import pandas as pd
import pytest
from rdkit import Chem

from smiles2select.alerts import rdkit_catalogs
from smiles2select.alerts.custom_smarts import CustomSmartsCatalog, SmartsAlert, alerts_from_dicts
from smiles2select.alerts.engine import AlertEngine, alerts_frame, exclusion_mask
from smiles2select.alerts.policies import AlertPolicy
from tests.conftest import REFERENCE_SMILES

pytestmark = pytest.mark.unit


def mol(name: str) -> Chem.Mol:
    return Chem.MolFromSmiles(REFERENCE_SMILES[name])


def test_pains_and_brenk_catalogs_are_available():
    available = rdkit_catalogs.available_catalogs()
    assert "pains" in available
    assert "brenk" in available


def test_brenk_flags_a_michael_acceptor():
    hits = rdkit_catalogs.match_catalog(mol("chalcone"), "brenk")
    assert any("Michael" in name for name, _ in hits)


def test_a_clean_molecule_produces_no_hits():
    assert rdkit_catalogs.match_catalog(mol("caffeine"), "pains") == []


def test_unknown_catalog_is_rejected():
    with pytest.raises(rdkit_catalogs.UnknownCatalogError):
        AlertEngine(["not_a_catalog"])


def test_alert_rows_carry_the_configured_action():
    engine = AlertEngine(["brenk"], policy=AlertPolicy())
    rows = engine.scan_to_rows(1, mol("chalcone"))
    assert rows
    assert {row["action"] for row in rows} == {"warn"}
    assert {row["catalog_id"] for row in rows} == {"brenk"}
    assert all(row["occurrence_count"] >= 1 for row in rows)


def test_warning_action_never_excludes():
    """A flagged molecule stays in the selection unless the user asks otherwise."""
    engine = AlertEngine(["brenk"], policy=AlertPolicy())
    frame = alerts_frame(engine.scan_to_rows(1, mol("chalcone")))
    index = pd.Index([1], name="record_id")
    assert exclusion_mask(frame, index, AlertPolicy()).tolist() == [False]


def test_exclude_action_removes_the_molecule():
    engine = AlertEngine(["brenk"])
    frame = alerts_frame(engine.scan_to_rows(1, mol("chalcone")))
    policy = AlertPolicy().with_action("brenk", "exclude")
    index = pd.Index([1], name="record_id")
    assert exclusion_mask(frame, index, policy).tolist() == [True]
    assert policy.excluding_catalogs() == ("brenk",)


def test_default_actions_are_warnings_for_pains_and_brenk():
    policy = AlertPolicy()
    assert policy.action_for("pains") == "warn"
    assert policy.action_for("brenk") == "warn"
    assert policy.excluding_catalogs() == ()


def test_unknown_action_is_rejected():
    with pytest.raises(ValueError):
        AlertPolicy().with_action("pains", "obliterate")


def test_custom_smarts_counts_occurrences():
    catalog = CustomSmartsCatalog(
        [SmartsAlert(id="nitro", name="Nitro group", smarts="[N+](=O)[O-]")]
    )
    hits = catalog.match(Chem.MolFromSmiles("O=[N+]([O-])c1ccc([N+](=O)[O-])cc1"))
    assert hits[0][0] == "Nitro group"
    assert hits[0][2] == 2


def test_invalid_smarts_fails_before_the_run():
    with pytest.raises(ValueError, match="invalid SMARTS"):
        alerts_from_dicts([{"id": "bad", "name": "Bad", "smarts": "C1CC"}])


def test_empty_engine_is_inactive():
    assert AlertEngine().is_active is False
