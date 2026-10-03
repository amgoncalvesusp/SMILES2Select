"""Auditable, opt-in contextual decisions over existing chemical feature records.

Activity, measured-risk predictions and configured chemistry constraints remain
separate. This module never trains a model or interprets missing evidence as a
negative label. Learned actions need support; user constraints stay explicit.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from numbers import Integral, Real

import numpy as np
import pandas as pd
from rdkit import Chem

from smiles2select.alerts.policies import ACTION_LABELS
from smiles2select.profiles.loader import BUILTIN_DIR, load_directory
from smiles2select.rules.evaluator import evaluate_profiles

from .schema import FeatureSet
from .selection import select_candidates

POLICY_VERSION = "s2-decision-contextual-policy/1"
STAGE_PROFILES = {"hit_finding": ("lipinski", "veber"), "lead": ("lead_like",),
                  "fragment": ("ro3_core",)}


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _number(value, name, *, maximum=None, integer=False):
    kind = Integral if integer else Real
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, kind)
            or not np.isfinite(value) or value < 0
            or (maximum is not None and value > maximum)):
        raise ValueError(f"{name} must be a finite nonnegative {'integer' if integer else 'number'}"
                         + (f" <= {maximum}" if maximum is not None else ""))


def _action(value):
    if not isinstance(value, str) or value not in ACTION_LABELS:
        raise ValueError(f"Unknown action: {value}")


@dataclass(frozen=True)
class PolicyContext:
    target: str
    species: str
    endpoint: str
    stage: str
    assay_context: str
    source_version: str
    model_version: str
    chemistry_version: str
    activity_threshold: float | None = None
    activity_unit: str = ""
    activity_relation: str = "<="

    def __post_init__(self):
        for name in ("target", "species", "endpoint", "assay_context", "source_version",
                     "model_version", "chemistry_version"):
            _text(getattr(self, name), name)
        if not isinstance(self.stage, str) or self.stage not in STAGE_PROFILES:
            raise ValueError(f"Unknown stage: {self.stage}")
        if not isinstance(self.activity_unit, str):
            raise ValueError("activity_unit must be a string")
        if self.activity_threshold is not None:
            _number(self.activity_threshold, "activity_threshold")
            _text(self.activity_unit, "activity_unit")
        elif self.activity_unit:
            raise ValueError("activity_unit requires activity_threshold")
        if self.activity_relation not in ("<", "<=", ">", ">="):
            raise ValueError("activity_relation must be <, <=, >, or >=")


@dataclass(frozen=True)
class ScoreEvidence:
    """Incoming prediction provenance; support counts do not prove calibration."""

    context: PolicyContext
    calibrated: bool = False
    calibration_method: str = ""
    calibration_n: int = 0
    negative_n: int = 0
    positive_n: int = 0
    source: str = ""
    endpoint: str = "activity"

    def __post_init__(self):
        if not isinstance(self.context, PolicyContext) or type(self.calibrated) is not bool:
            raise ValueError("ScoreEvidence needs PolicyContext and boolean calibrated")
        _text(self.endpoint, "evidence endpoint")
        for name in ("calibration_n", "negative_n", "positive_n"):
            _number(getattr(self, name), name, integer=True)
        if self.negative_n + self.positive_n > self.calibration_n:
            raise ValueError("Calibration class counts exceed calibration_n")
        if not isinstance(self.source, str) or not isinstance(self.calibration_method, str):
            raise ValueError("calibration source and method must be strings")
        if self.calibrated:
            _text(self.source, "calibration source")
            _text(self.calibration_method, "calibration_method")
            if min(self.negative_n, self.positive_n) < 1:
                raise ValueError("Calibrated classification evidence requires both classes")


@dataclass(frozen=True)
class FeatureAction:
    feature_id: str
    action: str = "warn"
    origin: str = "user"
    penalty: float = .1
    support_n: int = 0
    positive_n: int = 0
    negative_n: int = 0
    scaffolds: int = 0
    documents: int = 0
    evidence_endpoint: str = "activity"
    source: str = ""
    assays: int = 0

    def __post_init__(self):
        _text(self.feature_id, "feature_id")
        _action(self.action)
        if self.origin not in ("user", "learned"):
            raise ValueError("FeatureAction origin must be user or learned")
        _number(self.penalty, "penalty", maximum=1)
        for name in ("support_n", "positive_n", "negative_n", "scaffolds", "documents", "assays"):
            _number(getattr(self, name), name, integer=True)
        if (self.positive_n + self.negative_n > self.support_n
                or self.scaffolds > self.support_n):
            raise ValueError("FeatureAction support counts are inconsistent")
        _text(self.evidence_endpoint, "evidence_endpoint")
        if not isinstance(self.source, str):
            raise ValueError("FeatureAction source must be a string")


@dataclass(frozen=True)
class ProfileAction:
    profile_id: str
    action: str = "inform"
    max_violations: int | None = None
    penalty: float = .1

    def __post_init__(self):
        _text(self.profile_id, "profile_id")
        _action(self.action)
        _number(self.penalty, "penalty", maximum=1)
        if self.max_violations is not None:
            _number(self.max_violations, "max_violations", integer=True)


@dataclass(frozen=True)
class PolicySettings:
    n: int
    activity_evidence: ScoreEvidence | None = None
    risk_evidence: ScoreEvidence | None = None
    profiles: tuple[ProfileAction, ...] = ()
    rule_actions: tuple[FeatureAction, ...] = ()
    alert_actions: tuple[FeatureAction, ...] = ()
    default_alert_action: str = "warn"
    max_per_scaffold: int | None = None
    min_scaffolds: int | None = None
    max_per_cluster: int | None = None
    pinned_ids: tuple[int, ...] = ()
    excluded_ids: tuple[int, ...] = ()
    required_smarts: tuple[str, ...] = ()
    excluded_smarts: tuple[str, ...] = ()
    preferred_smarts: tuple[str, ...] = ()
    smarts_bonus: float = .1
    risk_weight: float = 0.
    risk_exclude_at: float | None = None

    def __post_init__(self):
        if type(self.n) is not int or self.n < 1:
            raise ValueError("n must be a positive integer")
        _action(self.default_alert_action)
        for name, kind in (("profiles", ProfileAction), ("rule_actions", FeatureAction),
                           ("alert_actions", FeatureAction), ("pinned_ids", Integral),
                           ("excluded_ids", Integral), ("required_smarts", str),
                           ("excluded_smarts", str), ("preferred_smarts", str)):
            value = getattr(self, name)
            if not isinstance(value, tuple) or any(not isinstance(item, kind) for item in value):
                raise ValueError(f"{name} must be an immutable tuple of {kind.__name__}")
        for name in ("activity_evidence", "risk_evidence"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, ScoreEvidence):
                raise ValueError(f"{name} must be ScoreEvidence")
        for name in ("risk_weight", "smarts_bonus"):
            _number(getattr(self, name), name, maximum=1)
        if self.risk_exclude_at is not None:
            _number(self.risk_exclude_at, "risk_exclude_at", maximum=1)
        if self.activity_evidence is not None and self.activity_evidence.endpoint != "activity":
            raise ValueError("activity endpoint must be declared as activity")
        if self.risk_evidence is not None and self.risk_evidence.endpoint == "activity":
            raise ValueError("risk endpoint must be distinct from activity")
        if self.risk_weight or self.risk_exclude_at is not None:
            if self.risk_evidence is None or not self.risk_evidence.source.strip():
                raise ValueError("Risk actions require separate risk evidence and source")


def _missing(value):
    return value is None or value is pd.NA or (isinstance(value, Real) and np.isnan(value))


def _nullable_number(value, name, *, probability=False):
    if _missing(value):
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must contain finite numbers or missing values")
    if probability and not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return float(value)


def _prediction_set(value, name):
    if _missing(value):
        return None
    if (not isinstance(value, (list, tuple, np.ndarray))
            or (isinstance(value, np.ndarray) and value.ndim != 1)
            or any(type(item) not in (int, np.int32, np.int64) or item not in (0, 1) for item in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"{name} must be a unique sequence of classes 0 and/or 1")
    return sorted(map(int, value))


def _validate_frame(frame):
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Contextual input must be a DataFrame or FeatureSet")
    required = {"record_id", "valid", "eligible", "identity", "murcko_scaffold", "activity_score"}
    if missing := required - set(frame):
        raise ValueError(f"Missing contextual fields: {sorted(missing)}")
    if not frame.columns.is_unique:
        raise ValueError("Contextual columns must be unique")
    ids = frame.record_id
    if ids.duplicated().any() or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral) for x in ids):
        raise ValueError("record_id values must be unique integers")
    for name in ("valid", "eligible"):
        if any(not isinstance(x, (bool, np.bool_)) for x in frame[name]):
            raise ValueError(f"{name} must contain boolean values")
    if (frame.eligible & ~frame.valid).any():
        raise ValueError("Eligible candidates cannot have invalid structures")


@lru_cache(maxsize=1)
def _builtin_profiles():
    return {profile.id: profile for profile in load_directory(BUILTIN_DIR)}


def _profiles(context, settings):
    configured = settings.profiles or tuple(ProfileAction(name) for name in STAGE_PROFILES[context.stage])
    _unique(configured, "profile_id")
    result = []
    for item in configured:
        if item.profile_id not in _builtin_profiles():
            raise ValueError(f"Unknown builtin profile: {item.profile_id}")
        profile = _builtin_profiles()[item.profile_id]
        if item.max_violations is not None:
            profile = profile.with_pass_policy({"type": "max_violations", "value": item.max_violations})
        if any(rule.is_substructure for rule in profile.rules):
            raise ValueError("Use explicit optional SMARTS constraints for substructure rules")
        result.append(profile)
    return tuple(result), configured


def _unique(items, key):
    values = [getattr(item, key) for item in items]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {key} actions")


def _rules(frame, profiles, settings):
    rules = {rule.id: rule for profile in profiles for rule in profile.rules}
    _unique(settings.rule_actions, "feature_id")
    _unique(settings.alert_actions, "feature_id")
    known = {rule.id for profile in _builtin_profiles().values() for rule in profile.rules}
    if unknown := {item.feature_id for item in settings.rule_actions} - known:
        raise ValueError(f"Unknown rule action: {sorted(unknown)}")
    if unknown := {item.feature_id for item in settings.rule_actions if item.origin == "user"} - set(rules):
        raise ValueError(f"Unknown rule action for selected profiles: {sorted(unknown)}")
    descriptors = pd.DataFrame(index=frame.record_id)
    for name in {rule.descriptor for rule in rules.values()}:
        values = frame[name] if name in frame else [None] * len(frame)
        descriptors[name] = [_nullable_number(value, name) for value in values]
    evaluation = evaluate_profiles(descriptors, profiles)
    broken = {(int(row.record_id), row.rule_id) for row in evaluation.failures.itertuples()}
    return rules, descriptors, broken


def _compiled_smarts(settings):
    result = {}
    for group in ("required", "excluded", "preferred"):
        result[group] = []
        for text in getattr(settings, f"{group}_smarts"):
            _text(text, "SMARTS")
            pattern = Chem.MolFromSmarts(text)
            if pattern is None:
                raise ValueError(f"Invalid SMARTS: {text}")
            result[group].append((text, pattern))
    return result


def _evidence_matches(evidence, context, *, risk=False):
    if evidence is None or not evidence.calibrated:
        return False
    return evidence.context.chemistry_version == context.chemistry_version if risk else evidence.context == context


def _validate_contexts(context, settings):
    evidence = settings.activity_evidence
    contract = ("target", "species", "endpoint", "activity_threshold", "activity_unit",
                "activity_relation", "model_version", "source_version", "chemistry_version")
    if evidence is not None:
        mismatches = [name for name in contract if getattr(evidence.context, name) != getattr(context, name)]
        if mismatches:
            raise ValueError(f"Incompatible activity context: {', '.join(mismatches)}")
    risk = settings.risk_evidence
    if risk is not None and risk.context.chemistry_version != context.chemistry_version:
        raise ValueError("Incompatible risk chemistry context")


def _support(action):
    return (action.support_n >= 20 and min(action.positive_n, action.negative_n) >= 5
            and action.scaffolds >= 5 and action.documents >= 3 and bool(action.source.strip()))


def _effective(action, context, settings, domains):
    if action.origin == "user":
        return action.action, None
    supported = _support(action)
    evidence = settings.activity_evidence if action.evidence_endpoint == "activity" else settings.risk_evidence
    risk = action.evidence_endpoint != "activity"
    matched = (evidence is not None and evidence.endpoint == action.evidence_endpoint
               and _evidence_matches(evidence, context, risk=risk))
    if (not supported or not matched or domains["risk" if risk else "activity"] is not True
            or (action.action == "exclude" and action.evidence_endpoint == "activity")):
        return "warn", supported
    return action.action, supported


def _apply_action(state, action, name, penalty):
    state["messages"].append(f"{name}: {action}")
    if action == "warn":
        state["warnings"].append(f"review:{name}")
    elif action == "penalize":
        state["penalty"] += penalty
    elif action == "exclude":
        state["eligible"] = False


def _score_status(row, context, settings, state):
    values = {}
    domains = {}
    for kind in ("activity", "risk"):
        score = _nullable_number(row.get(f"{kind}_score"), f"{kind}_score", probability=True)
        values[kind] = score
        evidence = getattr(settings, f"{kind}_evidence")
        prediction = _prediction_set(row.get(f"{kind}_prediction_set"), f"{kind}_prediction_set")
        if score is None:
            state["warnings"].append(f"{kind}_unknown")
        if not _evidence_matches(evidence, context, risk=kind == "risk"):
            state["warnings"].append(f"{kind}_uncalibrated")
        if kind == "activity" and evidence is not None and evidence.context != context:
            state["warnings"].append("context_mismatch")
        if prediction is None or len(prediction) != 1:
            state["warnings"].append(f"{kind}_prediction_set_" +
                                     ("unknown" if prediction is None else "empty" if not prediction else "ambiguous"))
        domain = _domain(row.get("risk_in_domain" if kind == "risk" else "in_domain"), kind)
        domains[kind] = domain
        if domain is not True:
            state["warnings"].append(("risk_" if kind == "risk" else "") +
                                     ("domain_unknown" if domain is None else "out_of_domain"))
    if context.activity_threshold is None:
        state["warnings"].append("activity_threshold_unknown")
    return values, domains


def _domain(value, kind):
    if _missing(value):
        return None
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{kind} in_domain must contain booleans or missing values")
    return bool(value)


def _rule_evidence(record_id, rules, descriptors, broken, context, settings, domains, state):
    result = []
    actions = {item.feature_id: item for item in settings.rule_actions}
    for rule in rules.values():
        value = _nullable_number(descriptors.at[record_id, rule.descriptor], rule.descriptor)
        lower, upper = rule.bounds
        excess = None if value is None else max(
            (lower - value) / (abs(lower) or 1.) if lower is not None else 0.,
            (value - upper) / (abs(upper) or 1.) if upper is not None else 0., 0.)
        violated = None if value is None else (record_id, rule.id) in broken
        action = actions.get(rule.id)
        effective, supported = _effective(action, context, settings, domains) if action else ("inform", None)
        entry = {"profile_id": rule.profile_id, "rule_id": rule.id, "descriptor": rule.descriptor,
                 "observed": value, "lower_limit": lower, "upper_limit": upper,
                 "violated": violated, "normalized_excess": excess, "effective_action": effective,
                 "supported": supported, "evidence": asdict(action) if action else None}
        result.append(entry)
        if value is None:
            state["warnings"].append(f"descriptor_missing:{rule.descriptor}")
            if action and action.origin == "user" and action.action == "exclude":
                _apply_action(state, "exclude", f"{rule.id}: unknown mandatory descriptor", 0.)
        elif violated and action:
            _apply_action(state, effective, rule.id, action.penalty)
    return result


def _profile_actions(profiles, configured, rule_evidence, state):
    result = []
    for profile, action in zip(profiles, configured):
        evidence = [item for item in rule_evidence if item["profile_id"] == profile.id]
        count = sum(item["violated"] is True for item in evidence)
        unknown = sum(item["violated"] is None for item in evidence)
        allowed = int(profile.pass_policy.get("value", 0))
        passed = False if count > allowed else None if unknown else True
        result.append({"profile_id": profile.id, "violations": count, "missing_rules": unknown,
                       "max_violations": allowed, "passed": passed, "action": action.action})
        if passed is False:
            _apply_action(state, action.action, profile.id, action.penalty)
        elif passed is None and action.action == "exclude":
            _apply_action(state, "exclude", f"{profile.id}: unknown mandatory profile", 0.)
    return result


def _alert_evidence(row, context, settings, domains, state):
    values = row.get("alert_ids")
    if _missing(values):
        state["warnings"].append("alerts_unknown")
        if settings.default_alert_action == "exclude" or any(
            item.origin == "user" and item.action == "exclude" for item in settings.alert_actions
        ):
            _apply_action(state, "exclude", "unknown mandatory alerts", 0.)
        return []
    if (not isinstance(values, (tuple, list, np.ndarray))
            or (isinstance(values, np.ndarray) and values.ndim != 1)
            or any(not isinstance(value, str) or not value for value in values)):
        raise ValueError("alert_ids must be a sequence of nonempty strings")
    actions = {item.feature_id: item for item in settings.alert_actions}
    result = []
    for name in sorted(set(values)):
        action = actions.get(name, FeatureAction(name, settings.default_alert_action))
        effective, supported = _effective(action, context, settings, domains)
        result.append({"alert_id": name, "effective_action": effective, "supported": supported,
                       "evidence": asdict(action)})
        _apply_action(state, effective, name, action.penalty)
    return result


def _smarts_evidence(row, compiled, settings, state):
    if not any(compiled.values()):
        return {}, 0.
    text = row.get("model_smiles")
    mol = Chem.MolFromSmiles(text) if isinstance(text, str) and text else None
    if mol is None:
        if row["valid"]:
            raise ValueError("SMARTS constraints require valid model_smiles")
        return {"status": "invalid_structure"}, 0.
    hits = {group: [text for text, query in patterns if mol.HasSubstructMatch(query)]
            for group, patterns in compiled.items()}
    if len(hits["required"]) != len(compiled["required"]):
        _apply_action(state, "exclude", "required_smarts_missing", 0)
    if hits["excluded"]:
        _apply_action(state, "exclude", "excluded_smarts_match", 0)
    bonus = settings.smarts_bonus if hits["preferred"] else 0.
    if bonus:
        state["messages"].append("preferred_smarts_match: configured ranking bonus, not activity evidence")
    return hits, bonus


def _row_decision(row, context, settings, rules, descriptors, broken, profiles, configured, compiled):
    state = {"eligible": bool(row["eligible"]), "warnings": [], "messages": [], "penalty": 0.}
    for action in settings.rule_actions:
        if action.feature_id not in rules:
            state["warnings"].append(f"inactive_rule_action:{action.feature_id}")
            state["messages"].append(f"{action.feature_id}: not applicable to selected stage profile")
    values, domains = _score_status(row, context, settings, state)
    rule_evidence = _rule_evidence(row["record_id"], rules, descriptors, broken, context, settings, domains, state)
    profile_evidence = _profile_actions(profiles, configured, rule_evidence, state)
    alert_evidence = _alert_evidence(row, context, settings, domains, state)
    smarts_evidence, bonus = _smarts_evidence(row, compiled, settings, state)
    if values["risk"] is not None:
        state["penalty"] += settings.risk_weight * values["risk"]
        if settings.risk_exclude_at is not None and values["risk"] >= settings.risk_exclude_at:
            _apply_action(state, "exclude", "user_risk_threshold", 0.)
    known = values["activity"] is not None
    priority = np.clip((values["activity"] or 0.) + bonus - state["penalty"], 0., 1.) if known else 0.
    warnings = sorted(set(state["warnings"]))
    discordant = known and values["activity"] >= .5 and any(item["violated"] is True for item in rule_evidence)
    uncertain = any(reason.startswith("activity_prediction_set_") for reason in warnings)
    information = sum((not known, domains["activity"] is not True, uncertain, discordant))
    information_reasons = [reason for reason in warnings if reason.startswith("activity_prediction_set_")
                           or reason in ("activity_unknown", "out_of_domain", "domain_unknown")]
    if discordant:
        information_reasons.append("activity_score_ge_0.5_with_rule_violation")
    return {"priority_score": float(priority), "eligible": state["eligible"],
            "review_required": bool(warnings), "policy_warnings": warnings,
            "policy_explanation": "; ".join(state["messages"] or ["activity ranking; no chemical action"]),
            "rule_evidence": rule_evidence, "profile_evidence": profile_evidence,
            "alert_evidence": alert_evidence, "smarts_evidence": smarts_evidence,
            "information_priority": int(information), "filter_discordant": bool(discordant),
            "information_reason": "; ".join(information_reasons)}


def _json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise ValueError(f"Provenance must be JSON-compatible, received {type(value).__name__}")


def _json_copy(value):
    return json.loads(json.dumps(value, allow_nan=False, default=_json_scalar))


def _validate_provenance(provenance, settings):
    score_context = provenance.get("score_context")
    if score_context is not None:
        if settings.activity_evidence is None or score_context != asdict(settings.activity_evidence.context):
            raise ValueError("score_context differs from declared activity evidence")
    for key, expected in (("risk_context", settings.risk_evidence),):
        if key in provenance and (expected is None or provenance[key] != asdict(expected.context)):
            raise ValueError(f"{key} differs from declared risk evidence")
    # Reject non-JSON provenance before producing a run that cannot be replayed.
    return _json_copy(provenance)


def apply_contextual_policy(frame, context: PolicyContext, settings: PolicySettings) -> FeatureSet:
    """Annotate and select absolute N without relaxing upstream hard eligibility.

    ``activity_score`` is an incoming model score, never P(advance). Missing
    activity ranks last (priority 0) with an explicit unknown flag. Optional
    ``risk_score`` requires separate endpoint/source provenance; unknown risk
    remains unknown. Returned prediction sets are incoming evidence, not fitted
    here. SMARTS requirements use ALL matches; exclusions use ANY match.
    """
    if not isinstance(context, PolicyContext) or not isinstance(settings, PolicySettings):
        raise ValueError("context and settings must be PolicyContext and PolicySettings")
    _validate_contexts(context, settings)
    records = frame.records if isinstance(frame, FeatureSet) else frame
    _validate_frame(records)
    provenance = frame.manifest if isinstance(frame, FeatureSet) else records.attrs
    provenance = _validate_provenance(provenance, settings)
    profiles, configured = _profiles(context, settings)
    rules, descriptors, broken = _rules(records, profiles, settings)
    compiled = _compiled_smarts(settings)
    decisions = [_row_decision(row, context, settings, rules, descriptors, broken, profiles, configured, compiled)
                 for row in records.to_dict("records")]
    columns = ("priority_score", "eligible", "review_required", "policy_warnings", "policy_explanation",
               "rule_evidence", "profile_evidence", "alert_evidence", "smarts_evidence",
               "information_priority", "filter_discordant", "information_reason")
    output = records.assign(**{name: [row[name] for row in decisions] for name in columns})
    output = output.assign(policy_input_eligible=records.eligible,
                           policy_hard_eligible=records.valid & records.eligible &
                           ~records.record_id.isin(settings.excluded_ids))
    ordered = output.sort_values(["information_priority", "record_id"], ascending=[False, True])
    positions = dict(zip(ordered.record_id, range(1, len(output) + 1)))
    output = output.assign(information_rank=output.record_id.map(positions))
    policy = {"version": POLICY_VERSION, "context": asdict(context), "settings": asdict(settings),
              "activity_evidence": asdict(settings.activity_evidence) if settings.activity_evidence else None,
              "risk_evidence": asdict(settings.risk_evidence) if settings.risk_evidence else None,
              "profiles": [profile.as_dict() for profile in profiles],
              "inactive_rule_actions": [asdict(item) for item in settings.rule_actions if item.feature_id not in rules],
              "support_gate": {"molecules": 20, "positive": 5, "negative": 5, "scaffolds": 5, "documents": 3},
              "risk_note": "Activity and structural alerts do not establish safety or assay reliability.",
              "stage_note": "Stage selects configured profiles; no stage-specific activity labels are invented.",
              "information_queue": "descriptive review queue; no expected information gain or experimental truth claim",
              "review_count": int(output.review_required.sum()), "source": provenance}
    result = select_candidates(FeatureSet(output, {"contextual_policy": policy}), settings.n,
                               settings.max_per_scaffold, settings.min_scaffolds, settings.max_per_cluster,
                               settings.pinned_ids, settings.excluded_ids)
    return FeatureSet(result.records, _json_copy({**result.manifest, "contextual_policy": policy}))


def select_information_queue(
    annotated: FeatureSet, n: int, max_per_scaffold=None, *, include_policy_exclusions=False,
) -> FeatureSet:
    """Build a separate, budgeted review queue; never generate experimental labels.

    Optional policy rejects enter only through explicit opt-in. Invalid records,
    upstream hard exclusions and manual excluded IDs never become queue-eligible.
    Main-basket pins do not override this separate review budget.
    """
    if not isinstance(annotated, FeatureSet) or "contextual_policy" not in annotated.manifest:
        raise ValueError("Information queue requires an annotated contextual FeatureSet")
    if type(include_policy_exclusions) is not bool:
        raise ValueError("include_policy_exclusions must be boolean")
    records = annotated.records
    required = {"information_priority", "information_reason", "policy_hard_eligible", "is_final"}
    if required - set(records):
        raise ValueError("Missing information queue annotation columns")
    for value in records.information_priority:
        _number(value, "information_priority", integer=True)
    if any(not isinstance(value, (bool, np.bool_)) for value in records.policy_hard_eligible):
        raise ValueError("policy_hard_eligible must contain boolean values")
    allowed = records.policy_hard_eligible & (records.information_priority > 0)
    if not include_policy_exclusions:
        allowed = allowed & records.eligible
    priority_max = max(1, int(records.information_priority.max())) if len(records) else 1
    prepared = records.drop(columns="pinned", errors="ignore").assign(
        main_basket_selected=records.is_final, eligible=allowed,
        priority_score=records.information_priority / priority_max,
        information_source=np.where(records.eligible, "policy_eligible", "optional_policy_exclusion"),
    )
    provenance = {"purpose": "separate information review queue; no truth labels assigned",
                  "include_policy_exclusions": include_policy_exclusions,
                  "ranking": "descriptive information_priority descending, record_id ascending",
                  "source_contextual_policy": annotated.manifest["contextual_policy"]}
    result = select_candidates(FeatureSet(prepared, provenance), n, max_per_scaffold=max_per_scaffold)
    return FeatureSet(result.records.assign(in_information_queue=result.records.is_final),
                      _json_copy({**result.manifest, "information_queue": provenance}))
