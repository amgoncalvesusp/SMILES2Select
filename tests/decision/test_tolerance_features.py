"""Versioned chemical tolerance annotations preserve unknown observations."""

import copy

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from s2s_decision import tolerance_features as features
from smiles2select.rules.engine import Rule


def records():
    return pd.DataFrame({
        "record_id": [7, 9, 11], "model_smiles": ["CCO", "CC[N+](=O)[O-]", "invalid"],
        "identity": ["a", "b", "c"], "task_id": ["T", "T", "T"],
        "document_id": ["D1", "D2", None], "mol_wt": [600., np.nan, 50.],
    }, index=[10, 20, 30])


def test_annotation_preserves_source_and_unknowns():
    source = records()
    before = source.copy(deep=True)
    got, schema = features.annotate_tolerance(source)
    pd.testing.assert_frame_equal(source, before)
    assert got.index.equals(source.index)
    pd.testing.assert_frame_equal(got[source.columns], source)
    rule = schema["criterion_columns"]["rule:lipinski:lip_mw_max"]
    assert got.loc[10, rule["violation_column"]] == 1
    assert got.loc[10, rule["upper_excess_column"]] == 100
    assert got.loc[10, rule["magnitude_column"]] == pytest.approx(.2)
    assert np.isnan(got.loc[20, rule["violation_column"]])
    assert "rule:lipinski:lip_mw_max" in got.loc[20, "tolerance_unknown_criteria"]
    assert not got.loc[30, "tolerance_structure_valid"]
    assert len(got.loc[30, "tolerance_unknown_criteria"]) == len(schema["criterion_columns"])
    assert len([c for c in schema["criterion_columns"] if c.startswith("alert:")]) == 585
    for detail in got.loc[20, "tolerance_alert_details"]:
        assert detail["entry_hit"] is True
        assert detail["site_count"] is None
        assert detail["site_count_status"] == "not_computed"


def test_catalog_identity_preserves_homonyms_and_only_unambiguous_legacy_aliases():
    for catalog, total, names in (("zinc", 50, 49), ("chembl_dundee", 105, 87)):
        manifest = features.catalog_manifest(catalog)
        assert len(manifest["entries"]) == total
        assert len({e["alert_name"] for e in manifest["entries"]}) == names
        assert len({e["criterion_id"] for e in manifest["entries"]}) == total
        repeated = {e["alert_name"] for e in manifest["entries"]
                    if sum(x["alert_name"] == e["alert_name"] for x in manifest["entries"]) > 1}
        assert all(e["legacy_id"] is None for e in manifest["entries"] if e["alert_name"] in repeated)
        assert manifest == features.catalog_manifest(catalog)
    fresh = features.catalog_manifest("brenk")
    fresh["entries"].clear()
    assert len(features.catalog_manifest("brenk")["entries"]) == 105


def test_native_matches_keep_catalog_entry_identity():
    hits = features.match_entries(Chem.MolFromSmiles("CC[N+](=O)[O-]"), "brenk")
    assert hits
    ids = {e["criterion_id"] for e in features.catalog_manifest("brenk")["entries"]}
    assert all(h["criterion_id"] in ids for h in hits)
    assert len({h["criterion_id"] for h in hits}) == len(hits)


def test_rule_sides_strict_bounds_and_missing_values():
    rule = Rule("range", "custom", "mol_wt", "between_exclusive", [100., 200.])
    result = features.annotate_rule(pd.Series([90., 100., 200., 230., np.nan]), rule)
    assert result["violation"].iloc[:4].tolist() == [1., 1., 1., 1.]
    assert result["lower_excess"].iloc[:4].tolist() == [10., 0., 0., 0.]
    assert result["upper_excess"].iloc[:4].tolist() == [0., 0., 0., 30.]
    assert np.isnan(result["violation"].iloc[-1])
    zero = Rule("charge", "custom", "formal_charge", ">=", 0)
    with pytest.raises(ValueError, match="positive explicit scale"):
        features.annotate_rule(pd.Series([-1., 0.]), zero)
    explicit = features.annotate_rule(pd.Series([-1., 0.]), zero, scale_lower=2.)
    assert explicit["lower_normalized"].tolist() == [.5, 0.]
    category = Rule("category", "custom", "category", "in", ["a", "b"])
    categorical = features.annotate_rule(pd.Series(["a", "z", None]), category)
    assert categorical["violation"].iloc[:2].tolist() == [0., 1.]
    assert categorical["lower_excess"].isna().all()
    with pytest.raises(ValueError, match="finite"):
        features.annotate_rule(pd.Series([1.]), Rule("inf", "custom", "mol_wt", "<=", np.inf))
    with pytest.raises(ValueError, match="scale"):
        features.annotate_rule(pd.Series([1.]), zero, scale_lower="bad")


def test_frozen_recipe_rejects_mutation_and_overlapping_catalogs():
    source = records().iloc[:1]
    got, schema = features.annotate_tolerance(source, catalogs=("brenk",))
    repeat, recipe = features.annotate_tolerance(source, schema=schema)
    pd.testing.assert_frame_equal(got, repeat)
    assert recipe == schema
    bad = copy.deepcopy(schema)
    bad["criterion_columns"][next(iter(bad["criterion_columns"]))]["violation_column"] = "qed"
    with pytest.raises(ValueError, match="recipe"):
        features.annotate_tolerance(source, schema=bad)
    with pytest.raises(ValueError, match="overlap"):
        features.annotate_tolerance(source, catalogs=("pains", "pains_a"))
    with pytest.raises(ValueError, match="unique"):
        features.annotate_tolerance(pd.concat([source, source]))


@pytest.mark.parametrize("recipe,extra", [("A", 0), ("B", 2048), ("C", 315), ("D", 2048)])
def test_representations_are_finite_explicit_and_do_not_use_biological_labels(recipe, extra):
    got, schema = features.annotate_tolerance(records().assign(pActivity=9., active=True), catalogs=())
    matrix, spec = features.representation_matrix(got, schema, recipe)
    assert matrix.shape == (3, len(spec["feature_names"]))
    assert np.isfinite(matrix).all()
    assert not any("activity" in n.lower() or "active" in n.lower() for n in spec["feature_names"])
    assert spec["representation_size"] == extra
    altered = got.assign(pActivity=0., active=False)
    np.testing.assert_array_equal(matrix, features.representation_matrix(altered, schema, recipe)[0])
    if recipe == "C":
        assert spec["representation_distance"] == "euclidean_normalized"
        assert spec["representation_parameters"] == {"fuzzIncrement": .3, "minPath": 1, "maxPath": 15}


def test_annotation_is_not_dependent_on_prior_activity_annotation():
    source = records().iloc[:1].assign(alert_ids=[["legacy"]], alert__fake=1.)
    got, schema = features.annotate_tolerance(source, catalogs=())
    assert got.loc[10, "alert_ids"] == ["legacy"]
    assert got.loc[10, "alert__fake"] == 1.
    assert "alert__fake" not in schema["binary_columns"]


def test_identity_is_canonical_and_numeric_record_aliases_are_rejected():
    frame = pd.DataFrame({"record_id": [1, 2], "model_smiles": ["OCC", "CCO"]})
    got, schema = features.annotate_tolerance(frame, profiles=(), catalogs=())
    assert got.identity.nunique() == 1
    assert got.murcko_scaffold.tolist() == ["", ""]
    before = got.copy(deep=True)
    features.representation_matrix(got, schema, "D")
    pd.testing.assert_frame_equal(got, before)
    with pytest.raises(ValueError, match="unique"):
        features.annotate_tolerance(frame.assign(record_id=["1", "1.0"]))


def test_empty_annotations_keep_frozen_columns():
    empty = pd.DataFrame({"record_id": pd.Series(dtype=int), "model_smiles": pd.Series(dtype=str)})
    got, schema = features.annotate_tolerance(empty, profiles=(), catalogs=())
    matrix, spec = features.representation_matrix(got, schema, "A")
    assert matrix.shape == (0, len(spec["feature_names"]))


def test_cache_checks_inputs_before_reusing_annotations(monkeypatch):
    got, schema = features.annotate_tolerance(records().iloc[:1], catalogs=())
    original = features._criterion_annotations
    calls = []

    def watched(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(features, "_criterion_annotations", watched)
    again, _ = features.annotate_tolerance(got, schema=schema)
    assert not calls
    pd.testing.assert_frame_equal(got, again)
    changed, _ = features.annotate_tolerance(got.assign(mol_wt=400.), schema=schema)
    assert calls
    column = schema["criterion_columns"]["rule:lipinski:lip_mw_max"]["violation_column"]
    assert changed[column].iloc[0] == 0


def test_cache_identity_does_not_copy_annotation_metadata():
    annotated, _ = features.annotate_tolerance(records().iloc[:1], catalogs=())
    plain = pd.DataFrame(annotated, copy=False)
    expected = features._input_hash(plain)
    copies = []

    class WatchedMetadata(dict):
        def __deepcopy__(self, memo):
            copies.append(True)
            return dict(self)

    metadata = WatchedMetadata(annotated.attrs["tolerance_annotation_schema"])
    annotated.attrs["tolerance_annotation_schema"] = metadata
    assert features._input_hash(annotated) == expected
    assert not copies, "Cache identity should hash chemistry without copying unrelated schema"
    assert annotated.attrs["tolerance_annotation_schema"] is metadata
