"""Independent decisions on cached descriptors; no chemistry or pairwise large ranking."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import numpy as np
import pandas as pd

from smiles2select.decision.engine import DecisionEngine
from smiles2select.pipeline.runner import RunResult
from smiles2select.rules import operators
from smiles2select.rules.evaluator import evaluate_profiles
from smiles2select.scores.consensus import score_table
from smiles2select.scores.qed import QedSelection
from smiles2select.selection_intelligence.constrained_selection import (
    SelectionConstraints,
    SelectionOutcome,
    Strategy,
    select,
)
from smiles2select.selection_intelligence.objectives import Direction, Objective, ObjectiveSet
from smiles2select.selection_intelligence.pareto_ranking import rank_candidates

EXACT_PARETO_LIMIT = 2000
EVALUATION_CHUNK_SIZE = 10000


@dataclass(frozen=True)
class ScenarioSpec:
    """Threshold overrides use existing rule IDs; descriptors are never recalculated."""

    name: str
    thresholds: Mapping[str, object] = field(default_factory=dict)
    objectives: tuple[Objective, ...] = ()
    constraints: SelectionConstraints = SelectionConstraints()
    strategy: Strategy = Strategy.BALANCED

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("scenario name is required")
        clean = {}
        for key, value in self.thresholds.items():
            if not isinstance(key, str) or not key:
                raise ValueError("threshold keys must be rule IDs")
            values = tuple(value) if isinstance(value, (list, tuple)) else (value,)
            if not values or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(item)
                for item in values
            ):
                raise ValueError("thresholds must contain finite numbers")
            clean[key] = values if isinstance(value, (list, tuple)) else value
        for name in ("target_count", "max_per_scaffold", "min_scaffolds", "max_per_cluster"):
            value = getattr(self.constraints, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        if type(self.constraints.preserve_pinned) is not bool:
            raise ValueError("preserve_pinned must be boolean")
        for objective in self.objectives:
            if (
                not isinstance(objective.field, str)
                or type(objective.enabled) is not bool
                or objective.transform != "none"
                or not isinstance(objective.direction, Direction)
            ):
                raise ValueError(
                    "objectives require a text field, boolean enabled and no transform"
                )
            for key in ("weight", "target_low", "target_high", "target_value"):
                value = getattr(objective, key)
                if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError(f"objective {key} must be finite numeric")
        ObjectiveSet(self.objectives)
        object.__setattr__(self, "thresholds", MappingProxyType(clean))
        object.__setattr__(self, "objectives", tuple(self.objectives))
        object.__setattr__(self, "strategy", Strategy(self.strategy))


@dataclass(frozen=True)
class ScenarioSnapshot:
    spec: ScenarioSpec
    data_fingerprint: str
    universe_ids: pd.Index
    eligible_ids: pd.Index
    outcome: SelectionOutcome
    warnings: tuple[str, ...]
    provenance: Mapping[str, object]
    pinned_ids: tuple[int, ...] = ()
    rescued_ids: tuple[int, ...] = ()
    excluded_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ScenarioComparison:
    common_ids: tuple[int, ...]
    entered_ids: tuple[int, ...]
    left_ids: tuple[int, ...]
    jaccard: float
    gained_scaffolds: tuple[str, ...]
    lost_scaffolds: tuple[str, ...]


def data_fingerprint(result: RunResult, candidates: pd.DataFrame) -> str:
    """Hash exact ordered input values and policy, including external objective columns."""
    digest = hashlib.sha256()
    for frame in (result.descriptors, candidates, result.alerts, result.scores):
        digest.update(json.dumps(list(frame.columns)).encode())
        # ponytail: hash one column at a time to bound scratch RAM at O(N).
        digest.update(pd.util.hash_pandas_object(frame.index).values.tobytes())
        for column in frame:
            digest.update(pd.util.hash_pandas_object(frame[column], index=False).values.tobytes())
    digest.update(repr(result.config).encode())
    digest.update(json.dumps([p.as_dict() for p in result.profiles], sort_keys=True).encode())
    return digest.hexdigest()


def _profiles(result: RunResult, spec: ScenarioSpec) -> tuple:
    all_rules = [rule for profile in result.profiles for rule in profile.rules]
    for rule_id, threshold in spec.thresholds.items():
        matches = [rule for rule in all_rules if rule.id == rule_id]
        if len(matches) != 1:
            raise ValueError(f"unknown or ambiguous rule ID: {rule_id}")
        rule = matches[0]
        if rule.is_substructure:
            raise ValueError("SMARTS changes require a new chemistry run")
        operators.apply(rule.operator, pd.Series([0.0]), threshold)
    return tuple(
        replace(
            profile,
            rules=tuple(
                replace(rule, threshold=spec.thresholds[rule.id])
                if rule.id in spec.thresholds
                else rule
                for rule in profile.rules
            ),
        )
        for profile in result.profiles
    )


def _chemistry(
    result: RunResult, candidates: pd.DataFrame, profiles: tuple, objective_fields: tuple[str, ...]
) -> tuple[pd.Series, pd.DataFrame]:
    eligible = pd.Series(False, index=candidates.index)
    score_fields = tuple(
        name
        for name in objective_fields
        if name
        in {"profile_pass_fraction", "hard_violation_count", "consensus_score", "alert_count"}
    )
    updated = pd.DataFrame(index=candidates.index, columns=list(score_fields), dtype=float)
    alert_positions = (
        candidates.index.get_indexer(result.alerts.record_id)
        if not result.alerts.empty
        else np.array([], dtype=int)
    )
    order = np.argsort(alert_positions, kind="stable")
    positions = alert_positions[order]
    ordered_alerts = result.alerts.iloc[order]
    policy = result.config.policy
    if policy.qed.mode == "top_percentile":
        if "qed" not in result.descriptors:
            raise ValueError("QED percentile requires cached QED values")
        cutoff = result.descriptors.loc[candidates.index, "qed"].quantile(
            1.0 - float(policy.qed.percentile) / 100.0
        )
        policy = replace(policy, qed=QedSelection(mode="threshold", threshold=cutoff))
    # Reuse authoritative evaluator/policy in bounded chunks, including tolerated violations.
    for start in range(0, len(candidates), EVALUATION_CHUNK_SIZE):
        index = candidates.index[start : start + EVALUATION_CHUNK_SIZE]
        descriptors = result.descriptors.loc[index]
        evaluation = evaluate_profiles(descriptors, profiles)
        scores = score_table(
            evaluation.status,
            [p.id for p in profiles],
            descriptors.get("qed"),
            result.scores.get("alert_count"),
        )
        left, right = np.searchsorted(positions, [start, start + len(index)])
        alerts = ordered_alerts.iloc[left:right]
        verdict = DecisionEngine(policy).decide(evaluation.status, scores, alerts)
        eligible.loc[index] = verdict.decisions.selected
        if score_fields:
            updated.loc[index, list(score_fields)] = scores[list(score_fields)]
    return eligible, updated


def _reference_mask(result: RunResult, index: pd.Index) -> pd.Series:
    descriptors, config = result.descriptors.loc[index], result.config
    allowed = pd.Series(True, index=index)
    if config.exclude_reference_duplicates:
        duplicate = descriptors.get("is_reference_duplicate")
        if duplicate is None and result.reference_duplicates is not None:
            duplicate = result.reference_duplicates.annotations["is_reference_duplicate"].reindex(
                index
            )
        if duplicate is None:
            raise ValueError(
                "reference duplicate annotations unavailable; rerun reference analysis"
            )
        allowed &= ~duplicate.fillna(False).astype(bool)
    if config.selection_strategy == "reference_neighborhood":
        if "max_reference_similarity" not in descriptors:
            raise ValueError("reference similarity unavailable; rerun reference analysis")
        allowed &= descriptors.max_reference_similarity.between(
            config.reference_min_similarity, config.reference_max_similarity
        )
    if config.selection_strategy in {"reference_novelty", "reference_aware_diversity"}:
        if "reference_novelty" not in descriptors:
            raise ValueError("reference novelty unavailable; rerun reference analysis")
        allowed &= descriptors.reference_novelty.notna()
    return allowed


def _rank(pool: pd.DataFrame, spec: ScenarioSpec) -> tuple[pd.DataFrame, str, tuple[str, ...]]:
    objectives = ObjectiveSet(spec.objectives)
    clean = pool.drop(
        columns=[
            c
            for c in (
                "pareto_rank",
                "crowding_distance",
                "distance_to_ideal",
                "robustness_score",
            )
            if c in pool
        ]
    )
    if spec.strategy is Strategy.SCAFFOLD_COVERAGE and "murcko_scaffold" in clean:
        clean = clean.assign(
            scaffold_size=clean.murcko_scaffold.map(clean.murcko_scaffold.value_counts(dropna=True))
        )
    if not len(objectives) or clean.empty:
        return clean, "strategy_order", ()
    if len(clean) <= EXACT_PARETO_LIMIT:
        ranking = rank_candidates(clean, objectives)
        return clean.join(ranking.table), "exact_pareto", ranking.warnings
    problems = objectives.validate(list(clean.columns))
    if problems:
        raise ValueError("; ".join(problems))
    score = pd.Series(0.0, index=clean.index)
    for objective in objectives.active:
        desirability = objective.desirability(clean[objective.field])
        score += desirability.rank(method="average", pct=True) * objective.weight
    # ponytail: O(M N log N) percentile ordering above 2000; no claimed Pareto fronts.
    ordered = clean.loc[score.sort_values(ascending=False, kind="stable").index]
    return (
        ordered,
        "weighted_percentile",
        (
            f"Above {EXACT_PARETO_LIMIT} candidates: weighted objective percentiles replace exact "
            "Pareto ranking; frequencies describe this deterministic approximation.",
        ),
    )


def evaluate_scenario(
    result: RunResult,
    candidates: pd.DataFrame,
    spec: ScenarioSpec,
    *,
    pinned_ids: Sequence[int] = (),
    excluded_ids: Sequence[int] = (),
    rescued_ids: Sequence[int] = (),
    _fingerprint: str | None = None,
) -> ScenarioSnapshot:
    """Fresh chemical eligibility then quota selection, independent of current basket."""
    if result.config.zones:
        raise ValueError("Scenario replay does not support zones; create a run without zone quotas")
    expected = result.descriptors.index[result.descriptors.evaluable.fillna(False).astype(bool)]
    if (
        not candidates.index.is_unique
        or not candidates.index.equals(expected)
        or not pd.api.types.is_integer_dtype(candidates.index.dtype)
    ):
        raise ValueError("scenarios require the complete ordered evaluable candidate universe")
    flags = tuple(
        tuple(dict.fromkeys(int(i) for i in ids)) for ids in (pinned_ids, rescued_ids, excluded_ids)
    )
    if any(not candidates.index.isin(ids).sum() == len(ids) for ids in flags):
        raise ValueError("manual flags contain records outside the candidate universe")
    chemical, updated_scores = _chemistry(
        result, candidates, _profiles(result, spec), ObjectiveSet(spec.objectives).fields()
    )
    reference = _reference_mask(result, candidates.index)
    eligible = chemical & reference
    manual = candidates.index.isin(flags[1])
    pool = candidates.loc[(eligible | (manual & reference)) & ~candidates.index.isin(flags[2])]
    if len(updated_scores.columns):
        pool = pool.assign(
            **{name: updated_scores[name].reindex(pool.index) for name in updated_scores}
        )
    ranked, method, warnings = _rank(pool, spec)
    if method == "weighted_percentile":
        # Existing selector sorts QED when Pareto is absent; drop ranking-only tie breakers.
        ranked = ranked.drop(columns=[c for c in ("qed",) if c in ranked])
    outcome = select(
        ranked, spec.constraints, spec.strategy, flags[0], flags[2], explain_rejections=False
    )
    messages = list(warnings) + outcome.warnings(spec.constraints)
    if ("murcko_scaffold" not in pool or pool.murcko_scaffold.isna().any()) and (
        spec.constraints.min_scaffolds
        or spec.constraints.max_per_scaffold
        or spec.strategy is Strategy.SCAFFOLD_COVERAGE
    ):
        raise ValueError("scaffold constraints require complete cached murcko_scaffold values")
    if spec.constraints.max_per_cluster and (
        "cluster_id" not in pool or pool.cluster_id.isna().any()
    ):
        raise ValueError("cluster quotas require complete cached cluster assignments")
    if (
        result.config.diversity_pick
        or result.config.per_scaffold_limit
        or result.config.final_count
        or result.config.selection_strategy != "traditional"
    ):
        messages.append(
            "Original post-selection strategy and limits are replaced by this scenario's "
            "strategy and constraints; reference eligibility restrictions are preserved."
        )
    return ScenarioSnapshot(
        spec,
        _fingerprint or data_fingerprint(result, candidates),
        candidates.index.copy(),
        candidates.index[eligible].copy(),
        outcome,
        tuple(messages),
        MappingProxyType(
            {
                "ranking_method": method,
                "cached_descriptors": True,
                "exact_pareto_limit": EXACT_PARETO_LIMIT,
                "manual_overrides_are_not_chemical_approval": True,
            }
        ),
        *flags,
    )


def evaluate_study(
    result: RunResult, candidates: pd.DataFrame, specs: Sequence[ScenarioSpec], **manual_flags
) -> tuple[ScenarioSnapshot, ...]:
    if not specs or len({spec.name for spec in specs}) != len(specs):
        raise ValueError("study needs scenarios with distinct names")
    fingerprint = data_fingerprint(result, candidates)
    return tuple(
        evaluate_scenario(result, candidates, spec, _fingerprint=fingerprint, **manual_flags)
        for spec in specs
    )


def compare_scenarios(a: ScenarioSnapshot, b: ScenarioSnapshot) -> ScenarioComparison:
    if a.data_fingerprint != b.data_fingerprint or not a.universe_ids.equals(b.universe_ids):
        raise ValueError("scenarios must use the same input universe, values and policy")
    first, second = set(a.outcome.selected_ids), set(b.outcome.selected_ids)
    scaffolds_a, scaffolds_b = set(a.outcome.scaffold_usage), set(b.outcome.scaffold_usage)
    return ScenarioComparison(
        tuple(sorted(first & second)),
        tuple(sorted(second - first)),
        tuple(sorted(first - second)),
        len(first & second) / len(first | second) if first | second else 1.0,
        tuple(sorted(scaffolds_b - scaffolds_a)),
        tuple(sorted(scaffolds_a - scaffolds_b)),
    )


def stability_table(snapshots: Sequence[ScenarioSnapshot]) -> pd.DataFrame:
    """Selection stability is not biological activity probability; flags stay separate."""
    if not snapshots:
        raise ValueError("stability needs at least one scenario")
    index = snapshots[0].universe_ids
    selected, eligible = np.zeros(len(index), dtype=np.int32), np.zeros(len(index), dtype=np.int32)
    flags = {name: np.zeros(len(index), dtype=bool) for name in ("pinned", "rescued", "excluded")}
    for snapshot in snapshots:
        compare_scenarios(snapshots[0], snapshot)
        selected += index.isin(snapshot.outcome.selected_ids)
        eligible += index.isin(snapshot.eligible_ids)
        for name in flags:
            flags[name] |= index.isin(getattr(snapshot, name + "_ids"))
    return pd.DataFrame(
        {
            "selected_count": selected,
            "eligible_count": eligible,
            "selection_frequency": selected / len(snapshots),
            "eligibility_frequency": eligible / len(snapshots),
            **flags,
        },
        index=index,
    )
