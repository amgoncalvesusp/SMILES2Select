"""Phase 3: threshold indexes, sensitivity, margins, robustness, borderline, counterfactuals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smiles2select.rules.engine import Rule
from smiles2select.selection_intelligence import counterfactuals as cf
from smiles2select.selection_intelligence.borderline import (
    BorderlineSort,
    borderline_ids,
    build_panel,
    closest_rule,
    near_misses,
    rules_at_risk,
    sort_panel,
)
from smiles2select.selection_intelligence.margins import (
    MarginConfig,
    MarginScale,
    MarginStatus,
    classify,
    margin_matrix,
    margin_table,
    rows_for_storage,
    rule_margins,
    scale_for,
)
from smiles2select.selection_intelligence.robustness import (
    limiting_rule,
    most_fragile,
    robustness_table,
)
from smiles2select.selection_intelligence.sensitivity import (
    ENTERED,
    LEFT,
    STAYED_IN,
    STAYED_OUT,
    evaluate_mask,
    most_restrictive,
    retention_table,
    simulate,
)
from smiles2select.selection_intelligence.threshold_indexes import ThresholdIndexSet, build_index

pytestmark = pytest.mark.unit

MW_RULE = Rule("lip_mw_max", "lipinski", "mol_wt", "<=", 500, failure_code="LIP_MW_HIGH")
TPSA_RULE = Rule("veb_tpsa_max", "veber", "tpsa", "<=", 140, failure_code="VEB_TPSA_HIGH")
MW_MIN_RULE = Rule("gho_mw_min", "ghose", "mol_wt", ">=", 160, failure_code="GHO_MW_LOW")
RULES = {rule.id: rule for rule in (MW_RULE, TPSA_RULE, MW_MIN_RULE)}


@pytest.fixture
def descriptors() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            # Record 4 fails only on weight, so a wider MW limit can recover it.
            # Record 5 fails on both, and stays out however far MW moves.
            "mol_wt": [180.0, 350.0, 495.0, 505.0, 620.0, np.nan],
            "tpsa": [40.0, 90.0, 138.0, 100.0, 150.0, 70.0],
        }
    )
    frame.index = pd.RangeIndex(1, len(frame) + 1, name="record_id")
    return frame


@pytest.fixture
def indexes(descriptors) -> ThresholdIndexSet:
    return ThresholdIndexSet.from_rules([MW_RULE, TPSA_RULE], descriptors)


# --- threshold indexes -------------------------------------------------------


def test_index_counts_retention_by_binary_search(indexes):
    index = indexes.get("lip_mw_max")
    assert index.retention(500) == 3  # 180, 350, 495
    assert index.retention(620) == 5
    assert index.retention(100) == 0


def test_missing_descriptor_never_counts_as_retained(indexes):
    """The rule engine fails NaN closed; the slider must agree."""
    index = indexes.get("lip_mw_max")
    assert index.retention(10_000) == 5
    assert index.count == 6


def test_lower_bound_operator_counts_from_the_other_end(descriptors):
    index = build_index(MW_MIN_RULE, descriptors)
    assert index.retention(160) == 5
    assert index.retention(500) == 2  # 505, 620


def test_strict_operator_excludes_the_limit(descriptors):
    strict = Rule("strict", "p", "mol_wt", "<", 495, failure_code="X")
    assert build_index(strict, descriptors).retention(495) == 2


def test_index_mask_matches_the_retention_count(indexes, descriptors):
    index = indexes.get("lip_mw_max")
    assert int(index.mask(500, descriptors.index).sum()) == index.retention(500)


def test_rules_without_a_slider_have_no_index(descriptors):
    ranged = Rule("range", "p", "mol_wt", "between_inclusive", [160, 480], failure_code="X")
    assert build_index(ranged, descriptors) is None
    with pytest.raises(KeyError, match="no threshold index"):
        ThresholdIndexSet().get("range")


def test_retention_curve_is_monotonic_for_an_upper_bound(indexes):
    curve = indexes.get("lip_mw_max").retention_curve(points=20)
    assert curve["retained"].is_monotonic_increasing
    assert curve["retention"].max() <= 1.0


def test_change_points_find_the_steepest_part(indexes):
    points = indexes.get("lip_mw_max").change_points(top=2)
    assert len(points) == 2
    assert "slope" in points.columns


def test_recovered_between_reports_the_gain(indexes):
    index = indexes.get("lip_mw_max")
    assert index.recovered_between(500, 620) == 2
    assert index.recovered_between(620, 500) == -2


# --- sensitivity -------------------------------------------------------------


def test_moving_a_threshold_recovers_molecules(indexes, descriptors):
    result = simulate(indexes, descriptors.index, {"lip_mw_max": 550})
    counts = result.counts
    assert counts[ENTERED] == 1  # 505 Da comes back
    assert counts[LEFT] == 0
    assert result.retained == int(result.mask_before.sum()) + 1


def test_tightening_a_threshold_loses_molecules(indexes, descriptors):
    result = simulate(indexes, descriptors.index, {"lip_mw_max": 400})
    assert result.counts[LEFT] == 1  # 495 Da drops out
    assert result.counts[ENTERED] == 0


def test_transitions_cover_every_molecule(indexes, descriptors):
    result = simulate(indexes, descriptors.index, {"lip_mw_max": 550})
    assert set(result.transitions.unique()) <= {STAYED_IN, ENTERED, LEFT, STAYED_OUT}
    assert len(result.transitions) == len(descriptors)


def test_unchanged_rules_keep_their_original_threshold(indexes, descriptors):
    result = simulate(indexes, descriptors.index, {})
    assert result.counts[ENTERED] == 0
    assert result.counts[LEFT] == 0


def test_entered_and_left_ids_are_reported(indexes, descriptors):
    result = simulate(indexes, descriptors.index, {"lip_mw_max": 550})
    assert list(result.entered_ids()) == [4]
    assert list(result.left_ids()) == []


def test_mask_combines_every_indexed_rule(indexes, descriptors):
    mask = evaluate_mask(indexes, descriptors.index)
    assert mask.loc[3]  # 495 Da, TPSA 138: passes both
    assert not mask.loc[4]  # 505 Da: fails the weight rule alone
    assert not mask.loc[5]  # 620 Da, TPSA 150: fails both


def test_retention_table_covers_all_rules(indexes):
    assert set(retention_table(indexes, points=10)) == {"lip_mw_max", "veb_tpsa_max"}


def test_most_restrictive_ranks_by_retention(indexes, descriptors):
    ranked = most_restrictive(indexes, descriptors.index)
    assert list(ranked["rule_id"])[0] == "lip_mw_max"
    assert ranked["retention"].is_monotonic_increasing


def test_summary_rows_are_readable(indexes, descriptors):
    rows = dict(simulate(indexes, descriptors.index, {"lip_mw_max": 550}).summary_rows())
    assert "candidatos mantidos" in rows
    assert rows["entraram"] == 1


# --- margins -----------------------------------------------------------------


def test_upper_bound_margin_is_positive_below_the_limit(descriptors):
    margins = rule_margins(MW_RULE, descriptors, MarginConfig(scale=MarginScale.THRESHOLD))
    assert margins[margins["record_id"] == 1].iloc[0]["normalized_margin"] > 0  # 180 Da
    assert margins[margins["record_id"] == 4].iloc[0]["normalized_margin"] < 0  # 505 Da


def test_lower_bound_margin_flips_direction(descriptors):
    margins = rule_margins(MW_MIN_RULE, descriptors, MarginConfig(scale=MarginScale.THRESHOLD))
    assert margins[margins["record_id"] == 1].iloc[0]["normalized_margin"] > 0  # 180 >= 160

    tiny = descriptors.assign(mol_wt=[100.0] * len(descriptors))
    negative = rule_margins(MW_MIN_RULE, tiny, MarginConfig(scale=MarginScale.THRESHOLD))
    assert (negative["normalized_margin"] < 0).all()


def test_range_rule_uses_the_nearest_limit(descriptors):
    ranged = Rule("gho_range", "ghose", "mol_wt", "between_inclusive", [160, 480], failure_code="X")
    margins = rule_margins(ranged, descriptors, MarginConfig(scale=MarginScale.THRESHOLD))
    assert margins[margins["record_id"] == 1].iloc[0]["normalized_margin"] > 0
    assert margins[margins["record_id"] == 3].iloc[0]["normalized_margin"] < 0  # 495 > 480


@pytest.mark.parametrize(
    ("margin", "expected"),
    [
        (0.5, MarginStatus.ROBUST_PASS),
        (0.05, MarginStatus.NEAR_PASS_LIMIT),
        (-0.05, MarginStatus.NEAR_FAIL_LIMIT),
        (-0.5, MarginStatus.ROBUST_FAIL),
    ],
)
def test_margin_classification(margin, expected):
    assert classify(margin, 0.10) is expected


def test_missing_descriptor_is_a_robust_fail(descriptors):
    margins = rule_margins(MW_RULE, descriptors)
    assert margins[margins["record_id"] == 6].iloc[0]["margin_status"] == (
        MarginStatus.ROBUST_FAIL.value
    )


def test_constant_descriptor_falls_back_to_the_threshold_scale():
    values = pd.Series([100.0, 100.0, 100.0])
    assert scale_for(values, 500.0, MarginScale.IQR) == 500.0
    assert scale_for(values, 0.0, MarginScale.IQR) == 1.0


def test_borderline_fraction_must_be_a_fraction():
    with pytest.raises(ValueError):
        MarginConfig(borderline_fraction=1.5)


def test_margin_table_covers_every_rule(descriptors):
    table = margin_table([MW_RULE, TPSA_RULE], descriptors)
    assert set(table["rule_id"]) == {"lip_mw_max", "veb_tpsa_max"}
    assert len(table) == 2 * len(descriptors)


def test_margin_matrix_is_restricted_to_the_requested_records(descriptors):
    matrix = margin_matrix(margin_table([MW_RULE, TPSA_RULE], descriptors), record_ids=[1, 2])
    assert list(matrix.index) == [1, 2]
    assert set(matrix.columns) == {"lip_mw_max", "veb_tpsa_max"}


def test_storage_rows_match_the_table_columns(descriptors):
    rows = rows_for_storage(margin_table([MW_RULE], descriptors))
    assert set(rows[0]) == {
        "record_id",
        "rule_id",
        "observed_value",
        "threshold_low",
        "threshold_high",
        "normalized_margin",
        "margin_status",
    }


# --- robustness --------------------------------------------------------------


def test_robustness_is_governed_by_the_worst_rule(descriptors):
    table = robustness_table(margin_table([MW_RULE, TPSA_RULE], descriptors))
    assert table.loc[3, "minimum_rule_margin"] <= table.loc[3, "mean_rule_margin"]


def test_a_failing_molecule_scores_zero(descriptors):
    table = robustness_table(margin_table([MW_RULE, TPSA_RULE], descriptors))
    assert table.loc[4, "robustness_score"] == 0.0
    assert table.loc[4, "failed_rule_count"] >= 1


def test_a_comfortable_molecule_scores_above_a_tight_one(descriptors):
    table = robustness_table(margin_table([MW_RULE, TPSA_RULE], descriptors))
    assert table.loc[1, "robustness_score"] > table.loc[3, "robustness_score"]


def test_most_fragile_lists_only_passing_molecules(descriptors):
    fragile = most_fragile(margin_table([MW_RULE, TPSA_RULE], descriptors))
    assert (fragile["minimum_rule_margin"] >= 0).all()


def test_limiting_rule_names_the_tightest_criterion(descriptors):
    limiting = limiting_rule(margin_table([MW_RULE, TPSA_RULE], descriptors))
    assert limiting.loc[3] in {"lip_mw_max", "veb_tpsa_max"}


def test_empty_margins_produce_empty_summaries():
    empty = pd.DataFrame(columns=["record_id", "normalized_margin", "margin_status", "rule_id"])
    assert robustness_table(empty).empty
    assert limiting_rule(empty).empty


# --- borderline panel --------------------------------------------------------


def test_closest_rule_picks_the_tightest_margin(descriptors):
    nearest = closest_rule(margin_table([MW_RULE, TPSA_RULE], descriptors))
    assert set(nearest["record_id"]) == set(descriptors.index)
    assert "closest_rule" in nearest.columns


def test_borderline_ids_only_include_the_band(descriptors):
    config = MarginConfig(scale=MarginScale.THRESHOLD, borderline_fraction=0.05)
    ids = borderline_ids(margin_table([MW_RULE, TPSA_RULE], descriptors, config))
    assert 3 in ids  # 495 Da against a 500 limit
    assert 1 not in ids  # 180 Da is nowhere near


def test_panel_joins_context_and_can_be_sorted(descriptors):
    config = MarginConfig(scale=MarginScale.THRESHOLD, borderline_fraction=0.05)
    table = margin_table([MW_RULE, TPSA_RULE], descriptors, config)
    context = pd.DataFrame({"qed": [0.8, 0.6, 0.4, 0.2, 0.1, 0.0]}, index=descriptors.index)
    panel = build_panel(table, context)
    assert "qed" in panel.columns
    assert not panel.empty
    assert sort_panel(panel, BorderlineSort.TIGHTEST_MARGIN)["margin"].is_monotonic_increasing


def test_sorting_by_a_missing_column_is_a_no_op(descriptors):
    panel = build_panel(margin_table([MW_RULE], descriptors), only_borderline=False)
    assert sort_panel(panel, BorderlineSort.BEST_PARETO).equals(panel)


def test_rules_at_risk_counts_borderline_molecules(descriptors):
    config = MarginConfig(scale=MarginScale.THRESHOLD, borderline_fraction=0.10)
    at_risk = rules_at_risk(margin_table([MW_RULE, TPSA_RULE], descriptors, config))
    assert set(at_risk.columns) == {"rule_id", "borderline_molecules"}


def test_near_misses_lists_only_narrow_failures(descriptors):
    config = MarginConfig(scale=MarginScale.THRESHOLD, borderline_fraction=0.10)
    misses = near_misses(margin_table([MW_RULE], descriptors, config))
    assert (misses["margin"] < 0).all()
    assert 4 in set(misses["record_id"])  # 505 Da just above 500
    assert 5 not in set(misses["record_id"])  # 620 Da is a robust fail


# --- counterfactuals ---------------------------------------------------------


def failures_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "record_id": 4,
                "profile_id": "lipinski",
                "rule_id": "lip_mw_max",
                "failure_code": "LIP_MW_HIGH",
                "observed_value": 505.0,
                "lower_limit": np.nan,
                "upper_limit": 500.0,
            },
            {
                "record_id": 5,
                "profile_id": "lipinski",
                "rule_id": "lip_mw_max",
                "failure_code": "LIP_MW_HIGH",
                "observed_value": 620.0,
                "lower_limit": np.nan,
                "upper_limit": 500.0,
            },
            {
                "record_id": 5,
                "profile_id": "veber",
                "rule_id": "veb_tpsa_max",
                "failure_code": "VEB_TPSA_HIGH",
                "observed_value": 150.0,
                "lower_limit": np.nan,
                "upper_limit": 140.0,
            },
        ]
    )


def test_simple_counterfactual_names_the_threshold_move():
    results = cf.for_failed_rules(4, failures_frame(), RULES)
    assert len(results) == 1
    change = results[0]
    assert change.affected_rule == "lip_mw_max"
    assert change.current_threshold == 500.0
    assert change.counterfactual_threshold == 505.0
    assert change.absolute_delta == 5.0
    assert "seria aprovada" in change.describe()


def test_counterfactuals_are_ordered_by_how_small_the_change_is():
    deltas = [abs(item.absolute_delta) for item in cf.for_failed_rules(5, failures_frame(), RULES)]
    assert deltas == sorted(deltas)


def test_too_many_broken_rules_yield_no_counterfactual():
    assert cf.for_failed_rules(5, failures_frame(), RULES, max_changes=1) == []


def test_impossible_counterfactual_is_explained():
    empty_margins = pd.DataFrame(
        columns=["record_id", "normalized_margin", "rule_id", "observed_value"]
    )
    crowded = pd.concat([failures_frame()] * 2, ignore_index=True)
    assert cf.explain(5, crowded, empty_margins, RULES) == cf.NO_SIMPLE_COUNTERFACTUAL


def test_explain_returns_the_rescue_when_one_exists():
    empty_margins = pd.DataFrame(
        columns=["record_id", "normalized_margin", "rule_id", "observed_value"]
    )
    result = cf.explain(4, failures_frame(), empty_margins, RULES)
    assert isinstance(result, list)
    assert result[0].affected_rule == "lip_mw_max"


def test_passing_molecule_gets_the_fragility_counterfactual(descriptors):
    change = cf.for_passing_molecule(3, margin_table([MW_RULE, TPSA_RULE], descriptors), RULES)
    assert change is not None
    assert change.current_status == "AUTO_PASS"
    assert change.classification_after_change == "reprovada"
    assert "seria reprovada" in change.describe()


def test_violation_policy_counterfactual():
    change = cf.violation_policy_change(5, failures_frame(), "lipinski", allowed_violations=0)
    assert change is not None
    assert "violação" in change.describe()


def test_violation_policy_is_none_when_the_gap_is_too_large():
    failures = pd.concat([failures_frame()] * 3, ignore_index=True)
    assert cf.violation_policy_change(5, failures, "lipinski", allowed_violations=0) is None


def test_counterfactual_table_has_the_export_columns():
    table = cf.table(cf.for_failed_rules(4, failures_frame(), RULES))
    assert set(table.columns) == set(cf.TABLE_COLUMNS)
    assert cf.table([]).empty
