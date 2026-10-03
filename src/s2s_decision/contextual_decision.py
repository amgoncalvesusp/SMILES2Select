"""Sealed, opt-in contextual proposals on the existing workspace snapshot."""

import json
import tempfile
from concurrent.futures import CancelledError
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .artifacts import file_hash, read_bundle, spreadsheet_safe, write_bundle, write_json
from .contextual_model import load_bundle, score_frame
from .contextual_policy import (
    STAGE_PROFILES,
    FeatureAction,
    PolicyContext,
    PolicySettings,
    ProfileAction,
    ScoreEvidence,
    apply_contextual_policy,
    select_information_queue,
)
from .decision import _comparison, _freeze_files, verify_preview
from .risk_model import load_risk_bundle, predict_risk, risk_prediction_sets
from .schema import FeatureSet
from .session_adapter import reject_model_identity_collisions


def load_policy_package(path):
    """Read one validated portable model and require its complete decision context."""
    bundle = load_bundle(path)
    try:
        context = PolicyContext(**bundle["context"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Contextual package context is incomplete: {exc}") from exc
    return bundle, context


def _settings(snapshot, bundle, evidence_context, overrides, risk_evidence=None, context=None):
    allowed = {"required_smarts", "excluded_smarts", "preferred_smarts", "default_alert_action",
               "risk_weight", "risk_exclude_at", "profiles", "rule_actions", "alert_actions"}
    if set(overrides) - allowed:
        raise ValueError("Unknown contextual policy setting")
    calibration = bundle["calibration"]
    evidence = ScoreEvidence(
        context=evidence_context, calibrated=True,
        calibration_method=calibration.get("method", "monotone sigmoid"),
        calibration_n=calibration.get("n", 0), positive_n=calibration.get("positive_n", 0),
        negative_n=calibration.get("negative_n", 0), source=bundle.get("source", ""),
        endpoint="activity",
    )
    constraints = snapshot.constraints
    actions = {"rule_actions": [], "alert_actions": []}
    for support in bundle.get("support", []):
        feature = support["feature_id"]
        if feature.startswith("alert__"):
            kind, identity = "alert_actions", feature.removeprefix("alert__")
        elif feature.startswith("rule__") and feature.endswith("__violation"):
            kind, identity = "rule_actions", feature.removeprefix("rule__").removesuffix("__violation")
        else:
            continue
        actions[kind].append(FeatureAction(
            feature_id=identity, action="warn", origin="learned", evidence_endpoint="activity",
            source=support.get("source", bundle.get("source", "")),
            **{name: support.get(name, 0) for name in (
                "support_n", "positive_n", "negative_n", "scaffolds", "documents", "assays")},
        ))
    profiles = {name: ProfileAction(name) for name in STAGE_PROFILES[(context or evidence_context).stage]}
    for kind, key in (("profiles", "profile_id"), ("rule_actions", "feature_id"), ("alert_actions", "feature_id")):
        configured = overrides.get(kind, ())
        identities = [getattr(item, key) for item in configured]
        if len(identities) != len(set(identities)):
            raise ValueError(f"Duplicate {kind} override")
        if kind == "profiles":
            profiles = {**profiles, **{item.profile_id: item for item in configured}}
        else:
            actions[kind] = [item for item in actions[kind] if item.feature_id not in identities] + list(configured)
    return PolicySettings(
        n=constraints.target_count, activity_evidence=evidence, risk_evidence=risk_evidence,
        max_per_scaffold=constraints.max_per_scaffold, min_scaffolds=constraints.min_scaffolds,
        max_per_cluster=constraints.max_per_cluster,
        pinned_ids=snapshot.pinned_ids, excluded_ids=snapshot.excluded_ids,
        profiles=tuple(profiles.values()),
        **{key: tuple(value) for key, value in actions.items()},
        **{key: value for key, value in overrides.items() if key not in {"profiles", "rule_actions", "alert_actions"}},
    )


def _score(features, bundle, context):
    """Invalid rows remain present, with unmeasured scores and no inferred risk."""
    frame = features.records
    valid = frame.valid.fillna(False).astype(bool)
    if not valid.any():
        raise ValueError("No valid molecules remain for contextual scoring")
    scored = score_frame(frame.loc[valid].copy(deep=True), bundle)
    invalid = frame.loc[~valid].copy(deep=True)
    records = pd.concat([scored, invalid], ignore_index=True).set_index("record_id")
    records = records.reindex(frame.record_id).reset_index()
    records = records.assign(
        murcko_scaffold=records.session_scaffold.fillna(""), risk_score=None,
        risk_prediction_status="unmeasured", calibration_status="calibrated_for_package_context",
    )
    records.attrs["score_context"] = asdict(context)
    return FeatureSet(records, {**features.manifest, "score_context": asdict(context)})


def _risk_evidence(path, activity_context):
    if path is None:
        return None, None, None
    package_hash = file_hash(path)
    bundle = load_risk_bundle(path)
    if file_hash(path) != package_hash:
        raise ValueError("Risk model package changed during loading")
    scope = bundle["risk_scope"]
    context = PolicyContext(target=bundle.get("target", scope),
        species=bundle.get("species", "Homo sapiens" if scope.startswith(("hepg2_atp", "shsy5y")) else "cell-free"),
        endpoint=scope, stage=activity_context.stage,
        assay_context=str(bundle.get("assay_context", bundle.get("data_source", scope))),
        source_version=bundle["source_version"], model_version=package_hash,
        chemistry_version=bundle["chemistry_version"])
    counts = bundle["calibration_class_support"]
    evidence = ScoreEvidence(context=context, calibrated=True,
        calibration_method=bundle["calibrator"].get("method", "monotone_sigmoid"),
        calibration_n=int(sum(counts)), negative_n=int(counts[0]), positive_n=int(counts[1]),
        source=bundle["data_source"], endpoint=scope)
    return bundle, evidence, {"path": str(Path(path).resolve()), "package_sha256": package_hash,
                              "endpoint": scope}


def _add_risk(scored, bundle, evidence):
    if bundle is None:
        return scored
    frame = scored.records
    valid = frame.valid.fillna(False).astype(bool)
    values = predict_risk(frame.loc[valid], bundle)
    probabilities = pd.Series(values, index=frame.index[valid]).reindex(frame.index)
    sets = pd.Series(risk_prediction_sets(values, bundle), index=frame.index[valid]).reindex(frame.index)
    records = frame.assign(risk_score=probabilities, risk_prediction_set=sets,
        risk_in_domain=None, risk_prediction_status="measured_endpoint_model; domain unknown")
    return FeatureSet(records, {**scored.manifest, "risk_context": asdict(evidence.context),
                               "risk_source": evidence.source})


def _information_queue(proposed, review_count, max_per_scaffold):
    if type(review_count) is not int or review_count < 0:
        raise ValueError("review_count must be a nonnegative integer")
    if not review_count:
        return proposed, None
    queue = select_information_queue(proposed, review_count, max_per_scaffold)
    flags = queue.records.set_index("record_id").in_information_queue
    records = proposed.records.assign(in_information_queue=proposed.records.record_id.map(flags))
    metadata = {key: queue.manifest[key] for key in (
        "requested_count", "final_count", "shortfall", "final_ids", "warnings")}
    return FeatureSet(records, {**proposed.manifest, "information_queue": metadata}), metadata


def preview_contextual_decision(snapshot, output_dir, package_path, context, *, overrides=None,
                                cancelled=None, risk_path=None, review_count=0):
    """Score arbitrary molecules, freeze evidence, then select without changing the basket."""
    def check_cancelled():
        if cancelled is not None and cancelled():
            raise CancelledError("Contextual preview cancelled")

    check_cancelled()
    destination, package = Path(output_dir), Path(package_path)
    if destination.exists():
        raise ValueError(f"output already exists: {destination}")
    package_sha = file_hash(package)
    bundle, evidence_context = load_policy_package(package)
    risk_bundle, risk_evidence, risk_metadata = _risk_evidence(risk_path, context)
    settings = _settings(snapshot, bundle, evidence_context, overrides or {}, risk_evidence, context)
    prepared = snapshot.prepared()
    reject_model_identity_collisions(prepared)
    check_cancelled()
    scored = _score(prepared, bundle, evidence_context)
    scored = _add_risk(scored, risk_bundle, risk_evidence)
    check_cancelled()
    proposed = apply_contextual_policy(scored, context, settings)
    proposed, queue = _information_queue(proposed, review_count, snapshot.constraints.max_per_scaffold)
    summary_settings = asdict(settings)
    if file_hash(package) != package_sha:
        raise ValueError("Contextual model package changed during preview")
    if risk_path is not None and file_hash(risk_path) != risk_metadata["package_sha256"]:
        raise ValueError("Risk model package changed during preview")
    original, final = list(snapshot.original_ids), proposed.manifest["final_ids"]
    warnings = list(proposed.manifest.get("warnings", []))
    warnings.append("Experimental retrospective policy. " + (
        f"Risk score covers only {risk_metadata['endpoint']}; applicability is unknown, not general safety."
        if risk_metadata else "Interference and toxicity are unmeasured."))
    summary = {
        "preview_version": 1, "decision_status": "preview_not_adopted",
        "ranking_mode": "contextual_policy", "source": prepared.manifest,
        "model": {"path": str(package.resolve()), "package_sha256": package_sha},
        "risk_model": risk_metadata,
        "information_queue": queue,
        "context": asdict(context), "evidence_context": asdict(evidence_context),
        "policy_settings": summary_settings, "original_ids": original, "proposed_ids": final,
        "original_count": len(original), "proposed_count": len(final),
        "eligible_count": int(proposed.records.eligible.sum()), "total_count": len(prepared.records),
        "constraints": proposed.manifest["constraints"],
        "pinned_ids": proposed.manifest["pinned_ids"], "excluded_ids": proposed.manifest["excluded_ids"],
        "warnings": warnings,
        "retained_ids": sorted(set(final) & set(original)),
        "added_ids": sorted(set(final) - set(original)),
        "removed_ids": sorted(set(original) - set(final)),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".s2s-context-", dir=destination.parent) as temporary:
        root = Path(temporary) / "preview"
        root.mkdir()
        for name, data in (("prepared", prepared), ("scored", scored), ("proposed", proposed)):
            write_bundle(data, root / name)
        spreadsheet_safe(_comparison(proposed, original)).to_csv(root / "comparison.csv", index=False)
        write_json(root / "comparison.json", summary)
        write_json(root / "preview.json", {"preview_version": 1, "files": _freeze_files(root)})
        check_cancelled()
        root.rename(destination)
    return summary


def verify_contextual_decision(snapshot, preview_dir, package_path, context, *, overrides=None,
                               risk_path=None, review_count=0):
    """Verify package/context and replay policy decisions against frozen scored rows."""
    root, package = Path(preview_dir), Path(package_path)
    summary = verify_preview(root)
    bundle, evidence_context = load_policy_package(package)
    _risk_bundle, risk_evidence, risk_metadata = _risk_evidence(risk_path, context)
    settings = _settings(snapshot, bundle, evidence_context, overrides or {}, risk_evidence, context)
    expected_settings = json.loads(json.dumps(asdict(settings)))
    if (summary.get("ranking_mode") != "contextual_policy"
            or summary.get("context") != asdict(context)
            or summary.get("policy_settings") != expected_settings
            or summary.get("risk_model") != risk_metadata
            or summary["source"].get("session_revision") != snapshot.revision
            or summary.get("original_ids") != list(snapshot.original_ids)
            or summary.get("model") != {"path": str(package.resolve()), "package_sha256": file_hash(package)}):
        raise ValueError("Contextual proposal differs from model, policy or workspace snapshot")
    scored, proposed = read_bundle(root / "scored"), read_bundle(root / "proposed")
    replayed = apply_contextual_policy(scored, context, settings)
    replayed, queue = _information_queue(replayed, review_count, snapshot.constraints.max_per_scaffold)
    if (replayed.manifest["final_ids"] != proposed.manifest["final_ids"]
            or summary.get("information_queue") != queue
            or replayed.records.eligible.tolist() != proposed.records.eligible.tolist()):
        raise ValueError("Contextual policy replay differs from frozen proposal")
    return summary, proposed
