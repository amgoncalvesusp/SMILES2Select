"""Tolerance packages fail closed on tampering and malformed numerical data."""

import copy
import hashlib
import json

import numpy as np
import pytest

from s2s_decision.tolerance_schema import (
    profile_hash,
    read_tolerance_package,
    seal_profile,
    validate_profile,
    write_tolerance_package,
)


def fixture_profile():
    return seal_profile({
        "schema": "s2s-tolerance-profile/1", "mode": "general", "recipe": "A",
        "activity_threshold": {"pactivity": 6.5, "relation": ">=", "fit_relation": "="},
        "annotation_schema": {"schema": "s2s-tolerance-annotations/1", "chemistry_hash": "a" * 64,
                              "criterion_columns": {"rule:lipinski:mw": {"violation_column": "v", "magnitude_column": "m"}}},
        "context": {}, "support_settings": {"min_identities": 20, "min_scaffolds": 5,
            "min_documents": 3, "k": 64, "kappa": 64., "allow_fallback": True,
            "domain_threshold": 1., "quantile": .95},
        "representation": {"recipe": "A", "width": 1, "center": [0.], "scale": [1.], "fingerprint_width": 0,
                           "distance_formula": "weighted_block_sum_v1",
                           "distance_blocks": [{"name": "numeric", "kind": "robust_rms", "indices": [0], "weight": 1.}]},
        "references": [{"identity": "id", "scaffold": "ring", "document_ids": ["doc"], "assay_ids": ["assay"],
                        "contexts": [{"task_id": "task", "target_id": "target", "species": "Human", "endpoint": "IC50"}],
                        "failed": ["rule:lipinski:mw"], "unknown": [], "magnitudes": {"rule:lipinski:mw": .1},
                        "sides": {"rule:lipinski:mw": ["upper"]}, "vector": [0.], "weight": 1.}],
        "curves": {"rule:lipinski:mw": {"x": [.1], "cdf": [1.], "maximum": .1}},
        "training_counts": {"rows": 1, "identities": 1, "scaffolds": 1, "documents": 1, "assays": 1, "tasks": 1},
        "policy": {"status": "uncalibrated", "tau": None, "rescue_enabled": False},
        "provenance": {},
    })


@pytest.mark.parametrize("changes", [{"calibration_count_basis": "identity_context"},
                                  {"positive_unique_identities": 2},
                                  {"positive_input_rows": 0},
                                  {"retention_interpretation": ""}])
def test_calibration_count_basis_and_audit_are_validated(changes):
    profile = fixture_profile()
    audit = {"calibration_count_basis": "identity", "positive_count": 1, "background_count": 0,
             "positive_unique_identities": 1, "background_unique_identities": 0,
             "positive_input_rows": 2, "background_input_rows": 0,
             "retention_interpretation": "Empirical retention only"}
    invalid = seal_profile({**profile, "policy": {**profile["policy"], **audit, **changes}})
    with pytest.raises(ValueError, match="calibration audit"):
        validate_profile(invalid)


@pytest.mark.parametrize("filename", ["package", "package.json"])
def test_roundtrip_atomic_validated_package(tmp_path, filename):
    profile = fixture_profile()
    path = tmp_path / filename
    assert write_tolerance_package(profile, path) == path
    loaded = read_tolerance_package(path)
    assert loaded == profile
    assert profile_hash(loaded) == profile["profile_sha256"]
    if path.is_dir():
        with pytest.raises(ValueError, match="exists"):
            write_tolerance_package(profile, path)


def test_integrity_corruption_and_manifest_path_rejection(tmp_path):
    profile = fixture_profile()
    path = tmp_path / "package"
    write_tolerance_package(profile, path)
    original = (path / "profile.json").read_text()
    (path / "profile.json").write_text(original.replace('"k": 64', '"k": 63'))
    with pytest.raises(ValueError, match="checksum"):
        read_tolerance_package(path)
    (path / "profile.json").write_text(original)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["profile_file"] = "../outside.json"
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="profile_file"):
        read_tolerance_package(path)


@pytest.mark.parametrize("change", [
    {"schema": "activity"}, {"recipe": "E"},
    {"policy": {"status": "feasible", "tau": float("nan"), "rescue_enabled": True}},
    {"curves": {"rule:lipinski:mw": {"x": [.1, .2], "cdf": [1., .5], "maximum": .2}}},
    {"representation": {"recipe": "A", "width": 1, "center": [0.], "scale": [0.], "fingerprint_width": 0}},
    {"training_counts": {"rows": -1}},
])
def test_malformed_profile_rejected(change):
    profile = {**fixture_profile(), **change}
    with pytest.raises(ValueError):
        validate_profile(profile)


def test_hash_is_checked_and_sealing_does_not_mutate():
    profile = fixture_profile()
    before = copy.deepcopy(profile)
    sealed = seal_profile(profile)
    assert sealed == profile == before
    profile["context"]["endpoint"] = "EC50"
    with pytest.raises(ValueError, match="checksum"):
        validate_profile(profile)


def test_nested_nonfinite_and_duplicate_references_rejected():
    profile = fixture_profile()
    profile["provenance"] = {"hidden": float("inf")}
    with pytest.raises(ValueError, match="finite"):
        seal_profile(profile)
    profile = fixture_profile()
    profile["references"] += copy.deepcopy(profile["references"])
    with pytest.raises(ValueError):
        validate_profile(seal_profile(profile))


@pytest.mark.parametrize("field,value", [
    ("failed", ["unknown-rule"]), ("failed", ["rule:lipinski:mw", "rule:lipinski:mw"]),
    ("contexts", []), ("contexts", [{"endpoint": 1}]),
    ("vector", []), ("vector", ["1"]), ("weight", 0), ("scaffold", 1),
    ("sides", {"rule:lipinski:mw": ["invalid"]}),
    ("magnitudes", {"rule:lipinski:mw": -1.}),
])
def test_invalid_support_reference_fields_fail_closed(field, value):
    profile = fixture_profile()
    profile["references"][0][field] = value
    with pytest.raises(ValueError):
        validate_profile(seal_profile(profile))


def test_duplicate_json_fields_and_oversized_package_rejected(tmp_path, monkeypatch):
    from s2s_decision import tolerance_schema
    path = tmp_path / "invalid.json"
    path.write_text('{"schema": "one", "schema": "two"}')
    with pytest.raises(ValueError, match="Duplicate"):
        read_tolerance_package(path)
    monkeypatch.setattr(tolerance_schema, "MAX_PACKAGE_BYTES", 5)
    with pytest.raises(ValueError, match="large"):
        read_tolerance_package(path)
    with pytest.raises(ValueError, match="size"):
        write_tolerance_package(fixture_profile(), tmp_path / "valid.json")


def test_package_whitelist_and_nonobject_profile(tmp_path):
    root = tmp_path / "package"
    write_tolerance_package(fixture_profile(), root)
    (root / "unexpected.pkl").write_text("This is never loaded")
    with pytest.raises(ValueError, match="unsupported files"):
        read_tolerance_package(root)
    (root / "unexpected.pkl").unlink()
    (root / "profile.json").write_text("[]")
    with pytest.raises(ValueError, match="object"):
        read_tolerance_package(root)


@pytest.mark.parametrize("change", [
    {"kind": "hamming_erg"}, {"indices": [0, 0]}, {"indices": [-1]}, {"weight": 0.}, {"weight": .5},
])
def test_invalid_distance_blocks_are_rejected(change):
    profile = fixture_profile()
    profile["representation"]["distance_blocks"][0].update(change)
    with pytest.raises(ValueError):
        validate_profile(seal_profile(profile))


def test_numeric_vectors_are_checked_without_per_element_recursive_calls(monkeypatch):
    from s2s_decision import tolerance_schema
    original = tolerance_schema._finite_tree
    calls = []

    def counted(value, depth=0):
        calls.append(depth)
        return original(value, depth)

    monkeypatch.setattr(tolerance_schema, "_finite_tree", counted)
    payload = {"vectors": [[float(item) for item in range(2048)], list(range(2048))]}
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                        ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    assert profile_hash(payload) == expected
    assert len(calls) <= 10, "Large numeric vectors must not recurse once per scalar"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("kind", ["float", "mixed", "heterogeneous", "large_integer"])
def test_nonfinite_numbers_cannot_hide_inside_numeric_or_fallback_lists(bad, kind):
    values = [0.5] * 128
    if kind == "mixed":
        values[0] = 1
    elif kind == "heterogeneous":
        values[0] = {"nested": True}
    elif kind == "large_integer":
        values[0] = 10 ** 400
    values[-1] = bad
    with pytest.raises(ValueError, match="finite"):
        profile_hash({"nested": {"values": values}})


@pytest.mark.parametrize("values", [[10 ** 400] * 128, [10 ** 400, .5] * 64,
                                   [1, .5] * 64, [True, .5] * 64, []])
def test_numeric_optimization_preserves_large_integer_and_json_hash_contract(values):
    payload = {"values": values}
    before = copy.deepcopy(payload)
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                        ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    assert profile_hash(payload) == expected
    assert payload == before


@pytest.mark.parametrize("bad", [object(), np.float64(.5), np.int64(1), (.5, .6)])
def test_numpy_scalars_objects_and_tuples_remain_outside_the_json_contract(bad):
    with pytest.raises(ValueError, match="only JSON values"):
        profile_hash({"nested": [0.5] * 128 + [bad]})


def test_vector_fast_path_preserves_nesting_limit_and_revalidates_mutated_values():
    from s2s_decision import tolerance_schema
    nested = [0.5] * 128
    for _ in range(30):
        nested = {"child": nested}
    with pytest.raises(ValueError, match="nesting"):
        tolerance_schema._finite_tree(nested)
    profile = fixture_profile()
    profile["provenance"]["diagnostic"] = [0.5] * 128
    profile = seal_profile(profile)
    validate_profile(profile)
    profile["provenance"]["diagnostic"][-1] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        validate_profile(profile)
