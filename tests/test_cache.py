"""Persistent descriptor cache and its invalidation rules."""

from __future__ import annotations

import pandas as pd
import pytest

from smiles2select.chemistry.standardization import StandardizationConfig
from smiles2select.io.importer import ColumnMapping, SourceFile
from smiles2select.pipeline.config import RunConfig
from smiles2select.pipeline.runner import run
from smiles2select.pipeline.workers import RecordResult
from smiles2select.storage.descriptor_cache import DescriptorCache, rebind
from tests.conftest import REFERENCE_SMILES

pytestmark = pytest.mark.integration

PROFILES = ("lipinski", "veber")


def make_config(csv, **overrides) -> RunConfig:
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


@pytest.fixture
def library(tmp_path):
    path = tmp_path / "library.csv"
    pd.DataFrame(
        [
            ("MOL001", REFERENCE_SMILES["aspirin"]),
            ("MOL002", REFERENCE_SMILES["caffeine"]),
            ("MOL003", REFERENCE_SMILES["chalcone"]),
        ],
        columns=["ID", "SMILES"],
    ).to_csv(path, index=False)
    return path


def sample_result(smiles: str = "CCO") -> RecordResult:
    return RecordResult(
        record_id=1,
        valid=True,
        invalid_reason=None,
        standardized_smiles=smiles,
        canonical_smiles=smiles,
        descriptors={"mol_wt": 46.07, "rdkit_wlogp": -0.001},
        substructure_flags={},
        alert_rows=[{"record_id": 1, "catalog_id": "brenk", "alert_name": "example"}],
    )


def test_first_run_populates_the_cache(library, tmp_path):
    cache_path = tmp_path / "cache.sqlite"
    result = run(make_config(library, cache_path=cache_path))
    assert result.cache_stats == (0, 3)
    assert cache_path.exists()


def test_second_run_reuses_every_molecule(library, tmp_path):
    cache_path = tmp_path / "cache.sqlite"
    run(make_config(library, cache_path=cache_path))
    second = run(make_config(library, cache_path=cache_path))
    assert second.cache_stats == (3, 0)
    assert second.evaluated_count == 3
    assert second.descriptors["mol_wt"].notna().all()


def test_cached_run_matches_the_uncached_one(library, tmp_path):
    cache_path = tmp_path / "cache.sqlite"
    first = run(make_config(library, cache_path=cache_path))
    second = run(make_config(library, cache_path=cache_path))
    pd.testing.assert_series_equal(
        first.descriptors["mol_wt"], second.descriptors["mol_wt"], check_names=False
    )
    assert first.decision.selected_count == second.decision.selected_count
    assert len(first.alerts) == len(second.alerts)


def test_alerts_are_rebound_to_the_current_records(library, tmp_path):
    """Alert rows are stored without a record id and reattached on reuse."""
    cache_path = tmp_path / "cache.sqlite"
    first = run(make_config(library, cache_path=cache_path))
    second = run(make_config(library, cache_path=cache_path))
    assert set(second.alerts["record_id"]) == set(first.alerts["record_id"])
    assert second.alerts["record_id"].notna().all()


def test_changing_standardization_invalidates_the_cache(library, tmp_path):
    cache_path = tmp_path / "cache.sqlite"
    run(make_config(library, cache_path=cache_path))
    changed = run(
        make_config(
            library,
            cache_path=cache_path,
            standardization=StandardizationConfig(neutralize=True),
        )
    )
    assert changed.cache_stats == (0, 3)


def test_changing_alert_catalogs_invalidates_the_cache(library, tmp_path):
    cache_path = tmp_path / "cache.sqlite"
    run(make_config(library, cache_path=cache_path))
    changed = run(make_config(library, cache_path=cache_path, alert_catalogs=("pains",)))
    assert changed.cache_stats == (0, 3)


def test_adding_a_profile_recomputes_only_what_is_missing(library, tmp_path):
    """A cached entry lacking a descriptor is recomputed, not read half-empty."""
    cache_path = tmp_path / "cache.sqlite"
    run(make_config(library, cache_path=cache_path))
    wider = run(
        make_config(library, cache_path=cache_path, profile_ids=("lipinski", "veber", "ghose"))
    )
    assert wider.cache_stats == (0, 3)  # ghose needs mol_mr and total_atom_count
    assert wider.descriptors["mol_mr"].notna().all()

    again = run(
        make_config(library, cache_path=cache_path, profile_ids=("lipinski", "veber", "ghose"))
    )
    assert again.cache_stats == (3, 0)


def test_narrowing_the_profiles_still_hits_the_cache(library, tmp_path):
    """A superset entry serves a run that needs fewer descriptors."""
    cache_path = tmp_path / "cache.sqlite"
    run(make_config(library, cache_path=cache_path, profile_ids=("lipinski", "veber", "ghose")))
    narrower = run(make_config(library, cache_path=cache_path, profile_ids=("lipinski",)))
    assert narrower.cache_stats == (3, 0)


def test_running_without_a_cache_path_reports_no_stats(library):
    assert run(make_config(library)).cache_stats is None


def test_cache_handles_empty_and_invalid_smiles(tmp_path):
    """An empty cell arrives as NaN from pandas; it must not reach the key as a float."""
    path = tmp_path / "gaps.csv"
    pd.DataFrame(
        [
            ("MOL001", REFERENCE_SMILES["aspirin"]),
            ("MOL002", ""),
            ("MOL003", "not_a_smiles"),
        ],
        columns=["ID", "SMILES"],
    ).to_csv(path, index=False)

    cache_path = tmp_path / "cache.sqlite"
    first = run(make_config(path, cache_path=cache_path))
    assert first.invalid_count == 2

    second = run(make_config(path, cache_path=cache_path))
    assert second.cache_stats == (3, 0)
    assert second.invalid_count == 2
    reasons = second.descriptors.loc[~second.descriptors["valid"], "invalid_reason"]
    assert reasons.notna().all()


def test_incomplete_entry_is_not_served(tmp_path):
    cache = DescriptorCache(tmp_path / "cache.sqlite", StandardizationConfig(), ("brenk",))
    cache.store({1: "CCO"}, [sample_result()])
    assert cache.fetch(["CCO"], ["mol_wt"])  # present
    assert not cache.fetch(["CCO"], ["mol_wt", "qed"])  # qed missing -> recompute
    cache.close()


def test_key_changes_with_the_signature(tmp_path):
    strict = DescriptorCache(tmp_path / "a.sqlite", StandardizationConfig(), ("brenk",))
    loose = DescriptorCache(
        tmp_path / "b.sqlite", StandardizationConfig(remove_salts=False), ("brenk",)
    )
    assert strict.key_for("CCO") != loose.key_for("CCO")
    strict.close()
    loose.close()


def test_rebind_attaches_the_record_id():
    rebound = rebind(sample_result(), 42)
    assert rebound.record_id == 42
    assert rebound.alert_rows[0]["record_id"] == 42
