"""End-to-end pipeline behaviour."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from smiles2select.io.importer import (
    ColumnMapping,
    SourceFile,
    SourceReadError,
    guess_mapping,
    load_records,
)
from smiles2select.pipeline.config import RunConfig
from smiles2select.pipeline.runner import run
from smiles2select.pipeline.sampling import analyse
from smiles2select.storage.sqlite_store import SqliteStore
from tests.conftest import REFERENCE_SMILES

pytestmark = pytest.mark.integration

PROFILES = ("lipinski", "veber", "ghose", "egan", "muegge")

LIBRARY_ROWS = [
    ("MOL001", REFERENCE_SMILES["aspirin"]),
    ("MOL002", REFERENCE_SMILES["caffeine"]),
    ("MOL003", REFERENCE_SMILES["ibuprofen"]),
    ("MOL004", "not_a_smiles"),
    ("MOL005", REFERENCE_SMILES["aspirin"]),
    ("MOL006", REFERENCE_SMILES["long_alkane"]),
    ("MOL007", ""),
    ("MOL008", REFERENCE_SMILES["salicylic_sodium_salt"]),
    ("MOL009", REFERENCE_SMILES["chalcone"]),
]


def make_config(csv: Path, **overrides) -> RunConfig:
    settings = {
        "sources": (
            SourceFile(path=csv, mapping=ColumnMapping(smiles="SMILES", molecule_id="ID")),
        ),
        "profile_ids": PROFILES,
        "alert_catalogs": ("brenk",),
        "n_jobs": 1,
        "chunk_size": 4,
    }
    settings.update(overrides)
    return RunConfig(**settings)


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    directory = tmp_path_factory.mktemp("pipeline")
    csv = directory / "library.csv"
    pd.DataFrame(LIBRARY_ROWS, columns=["ID", "SMILES"]).to_csv(csv, index=False)
    return run(make_config(csv, database_path=directory / "run.sqlite"))


def test_counts_reconcile(result):
    """Every input row is accounted for as evaluated, invalid or duplicated."""
    assert result.total_records == 9
    assert result.invalid_count == 2
    assert result.duplicate_count == 1
    assert result.evaluated_count == (
        result.total_records - result.invalid_count - result.duplicate_count
    )


def test_invalid_rows_are_reported_not_dropped(result):
    invalid = result.descriptors[~result.descriptors["valid"].astype(bool)]
    assert set(invalid["molecule_id"]) == {"MOL004", "MOL007"}
    assert invalid["invalid_reason"].notna().all()


def test_duplicate_points_at_its_first_occurrence(result):
    duplicates = result.descriptors[result.descriptors["duplicate_of"].notna()]
    assert duplicates["molecule_id"].tolist() == ["MOL005"]
    assert duplicates["duplicate_of"].tolist() == [1]


def test_salt_is_stripped_before_descriptors(result):
    """The sodium salt is reduced to salicylic acid, so MW is the parent's."""
    row = result.descriptors[result.descriptors["molecule_id"] == "MOL008"].iloc[0]
    assert row["mol_wt"] == pytest.approx(138.12, abs=0.01)
    assert "Na" not in row["canonical_smiles"]


def test_only_needed_descriptors_are_computed(result):
    """Nothing outside the plan is calculated."""
    assert "heavy_atom_count" not in result.plan.descriptor_ids
    assert "fraction_csp3" not in result.plan.descriptor_ids
    assert "mol_wt" in result.plan.descriptor_ids


def test_lipophilic_outlier_is_excluded(result):
    decisions = result.decision.decisions
    record_id = result.descriptors.index[result.descriptors["molecule_id"] == "MOL006"][0]
    assert bool(decisions.loc[record_id, "selected"]) is False
    assert "veber" in decisions.loc[record_id, "exclusion_reasons"]


def test_alert_does_not_exclude_by_default(result):
    """The chalcone carries a Brenk alert and is still selected."""
    record_id = result.descriptors.index[result.descriptors["molecule_id"] == "MOL009"][0]
    assert record_id in set(result.alerts["record_id"])
    assert bool(result.decision.decisions.loc[record_id, "selected"]) is True


def test_database_tables_are_populated(result):
    with SqliteStore(result.database_path) as store:
        assert store.count("molecule_descriptors") == 9
        assert store.count("profile_results") == 6 * len(PROFILES)
        assert store.count("final_decisions") == 6
        assert store.count("rule_failures") > 0
        assert store.read_config()["profiles"] == ", ".join(PROFILES)


def test_rule_failures_table_is_sparse(result):
    """Only broken rules are stored, never one row per satisfied rule."""
    total_checks = result.evaluated_count * sum(len(profile.rules) for profile in result.profiles)
    assert len(result.evaluation.failures) < total_checks


def test_sample_analysis_reports_impact(result):
    report = analyse(result.evaluation, list(PROFILES))
    assert report.sample_size == result.evaluated_count
    assert len(report.per_profile) == len(PROFILES)
    assert report.cumulative["surviving"].is_monotonic_decreasing
    assert report.most_restrictive(1)


def test_max_records_limits_the_run(library_csv):
    limited = run(make_config(library_csv, max_records=3))
    assert limited.total_records == 3


def test_missing_smiles_column_is_reported(library_csv):
    source = SourceFile(path=library_csv, mapping=ColumnMapping(smiles="NOPE"))
    with pytest.raises(SourceReadError, match="not found"):
        load_records([source])


def test_records_without_id_get_generated_identifiers(tmp_path):
    path = tmp_path / "no_id.csv"
    pd.DataFrame({"SMILES": ["CCO", "CCC"]}).to_csv(path, index=False)
    records = load_records([SourceFile(path=path, mapping=guess_mapping(["SMILES"]))])
    assert records["molecule_id"].tolist() == ["REC0000001", "REC0000002"]
    assert records["source_row"].tolist() == [2, 3]
