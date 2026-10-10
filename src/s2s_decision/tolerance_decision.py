"""Sealed chemical-tolerance proposals using the existing workspace and selector."""

from __future__ import annotations

import hashlib
import json
import tempfile
from concurrent.futures import CancelledError
from dataclasses import is_dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from rdkit import Chem

from smiles2select.alerts.engine import ALERT_COLUMNS
from smiles2select.alerts.rdkit_catalogs import available_catalogs
from smiles2select.rules.evaluator import evaluate_profiles

from .artifacts import file_hash, read_bundle, spreadsheet_safe, write_bundle, write_json
from .decision import _comparison, _freeze_files, verify_preview
from .schema import FeatureSet
from .selection import select_candidates
from .session_adapter import _frame_hash, reject_model_identity_collisions, snapshot_from_workspace
from .tolerance_profile import predict_tolerance, read_tolerance_package
from .tolerance_schema import MAX_PACKAGE_BYTES


def tolerance_package_hash(path):
    """Bind a preview to all package bytes, rejecting linked package members."""
    root = Path(path)
    if root.is_symlink():
        raise ValueError("Tolerance package cannot be a symbolic link")
    if root.is_file():
        if root.stat().st_size > MAX_PACKAGE_BYTES:
            raise ValueError("Tolerance package file too large")
        return file_hash(root)
    if not root.is_dir():
        raise ValueError("Tolerance package does not exist")
    members = list(root.iterdir())
    if {entry.name for entry in members} != {"manifest.json", "profile.json"}:
        raise ValueError("Tolerance package contains unsupported files")
    digest = hashlib.sha256()
    for entry in sorted(members):
        if entry.is_symlink():
            raise ValueError("Tolerance package cannot contain symbolic links")
        if not entry.is_file() or entry.stat().st_size > MAX_PACKAGE_BYTES:
            raise ValueError("Tolerance package file missing or too large")
        digest.update(entry.relative_to(root).as_posix().encode())
        digest.update(file_hash(entry).encode())
    return digest.hexdigest()


def _reconsider_alerts(result, entries):
    """Re-scan exact native entries; duplicate display names cannot identify exclusions."""
    if not entries:
        return result.alerts
    from .tolerance_features import catalog_manifest, match_entries

    catalogs = set()
    for identity in entries:
        parts = identity.split(":")
        if len(parts) != 5 or parts[:2] != ["alert", "v2"]:
            raise ValueError("Reconsidered alert requires a versioned entry ID")
        catalogs.add(parts[2])
    if catalogs - set(available_catalogs()):
        raise ValueError("Cannot reconsider unavailable native alert catalogs")
    known = {entry["criterion_id"] for catalog in catalogs
             for entry in catalog_manifest(catalog)["entries"]}
    if set(entries) - known:
        raise ValueError("Alert entry ID differs from the installed catalog manifest")
    retained = result.alerts.loc[~result.alerts.catalog_id.isin(catalogs)].to_dict("records") \
        if not result.alerts.empty else []
    policy = result.config.policy.alert_policy
    for record_id, row in result.descriptors.iterrows():
        if not bool(row.get("valid", False)):
            continue
        text = next((row.get(column) for column in
                     ("standardized_smiles", "canonical_smiles", "original_smiles")
                     if isinstance(row.get(column), str) and row.get(column)), None)
        mol = Chem.MolFromSmiles(text) if text else None
        if mol is None:
            raise ValueError(f"Cannot inspect alert evidence for valid record {record_id}")
        for catalog in sorted(catalogs):
            for hit in match_entries(mol, catalog):
                if hit["criterion_id"] not in entries:
                    retained.append({"record_id": record_id, "catalog_id": catalog,
                        "alert_name": hit["alert_name"],
                        "alert_description": hit["alert_description"],
                        "occurrence_count": 1, "action": policy.action_for(catalog)})
    return pd.DataFrame(retained, columns=ALERT_COLUMNS)


def tolerance_snapshot_from_workspace(result, candidates, basket, constraints, *,
                                      revisited_profiles=(), revisited_alert_entries=(),
                                      reconsidered_criteria=None, session_id=None):
    """Capture C and B separately without changing upstream policy or basket state."""
    entries = tuple(sorted(revisited_alert_entries))
    if len(entries) != len(set(entries)) or any(not isinstance(x, str) for x in entries):
        raise ValueError("Reconsidered alert entry IDs must be unique strings")
    alerts = _reconsider_alerts(result, entries)
    scoped = replace(result, alerts=alerts) if is_dataclass(result) else SimpleNamespace(
        **{**vars(result), "alerts": alerts})
    captured = snapshot_from_workspace(scoped, candidates, basket, constraints,
        session_id=session_id, candidate_scope="all_valid", revisited_profiles=revisited_profiles)
    allowed = captured._records.eligible.copy()
    if reconsidered_criteria is not None:
        chosen = set(reconsidered_criteria)
        known = {f"rule:{profile.id}:{rule.id}" for profile in result.profiles for rule in profile.rules}
        if chosen - known - set(entries):
            raise ValueError("Unknown or unauthorized reconsidered criterion")
        for profile in result.profiles:
            if profile.id not in revisited_profiles:
                continue
            rules = tuple(rule for rule in profile.rules if f"rule:{profile.id}:{rule.id}" not in chosen)
            if any(rule.is_substructure and f"rule:{profile.id}:{rule.id}" in chosen for rule in profile.rules):
                raise ValueError("Locked SMARTS rules cannot be reconsidered")
            role = result.config.policy.roles.get(profile.id)
            if rules and role == "consensus":
                raise ValueError("Cannot partially reconsider a coupled consensus rule policy")
            if rules and role in {"mandatory", "exclusion"}:
                retained = evaluate_profiles(result.descriptors, (replace(profile, rules=rules),)).passed(profile.id)
                allowed = allowed & captured._records.record_id.map(retained).fillna(False)
    states = {state.record_id: state for state in basket.states()}
    records = captured._records.assign(eligible=allowed, baseline_eligible=[
        bool(states[int(rid)].chemical_status.passed) and bool(eligible)
        for rid, eligible in zip(captured._records.record_id, allowed, strict=True)
    ])
    record_hash = _frame_hash(records)
    definitions = {f"rule:{profile.id}:{rule.id}": rule.as_dict()
                   for profile in result.profiles if profile.id in revisited_profiles
                   for rule in profile.rules}
    definitions_json = json.dumps(definitions, sort_keys=True)
    criteria_json = json.dumps(sorted(reconsidered_criteria) if reconsidered_criteria is not None else None)
    revision = hashlib.sha256(json.dumps({"snapshot": captured.revision,
        "record_hash": record_hash, "revisited_alert_entries": entries,
        "rule_definitions": definitions, "reconsidered_criteria": json.loads(criteria_json)},
        sort_keys=True).encode()).hexdigest()
    return replace(captured, _records=records, _records_hash=record_hash,
                   revision=revision, revisited_alert_entries=entries,
                   tolerance_rule_definitions_json=definitions_json,
                   tolerance_criteria_json=criteria_json)


def _settings(snapshot, criteria):
    if snapshot.tolerance_criteria_json:
        captured = json.loads(snapshot.tolerance_criteria_json)
        if criteria is None and captured is not None:
            criteria = captured
        if criteria is not None:
            expected = set(captured) if captured is not None else (
                set(json.loads(snapshot.tolerance_rule_definitions_json)) | set(snapshot.revisited_alert_entries))
            if set(criteria) != expected:
                raise ValueError("Recreate the snapshot with the requested captured criterion scope")
    if criteria is not None:
        criteria = tuple(sorted(criteria))
        if len(criteria) != len(set(criteria)) or any(not isinstance(x, str) for x in criteria):
            raise ValueError("Reconsidered criteria must be unique strings")
        for identity in criteria:
            if identity.startswith("rule:"):
                if identity.split(":")[1] not in snapshot.revisited_profiles:
                    raise ValueError("Rule criterion was not authorized in the workspace snapshot")
            elif identity not in snapshot.revisited_alert_entries:
                raise ValueError("Alert criterion was not authorized in the workspace snapshot")
    return {"reconsidered_criteria": list(criteria) if criteria is not None else None,
        "revisited_profiles": list(snapshot.revisited_profiles),
        "revisited_alert_entries": list(snapshot.revisited_alert_entries)}


def _score(snapshot, prepared, profile, context, settings, cache_path=None):
    records = prepared.records
    criteria = settings["reconsidered_criteria"]
    if criteria is None and profile.get("annotation_schema"):
        criteria = [identity for identity in profile["annotation_schema"]["criterion_columns"]
            if (identity.startswith("rule:") and identity.split(":")[1] in snapshot.revisited_profiles)
            or identity in snapshot.revisited_alert_entries]
    if profile.get("annotation_schema") and snapshot.tolerance_rule_definitions_json:
        live = json.loads(snapshot.tolerance_rule_definitions_json)
        trained = profile["annotation_schema"]["criterion_columns"]
        for identity in criteria:
            if identity.startswith("rule:") and (identity not in live or identity not in trained):
                raise ValueError(f"Unknown workspace tolerance rule: {identity}")
            if identity.startswith("rule:") and any(
                    trained.get(identity, {}).get(key) != live[identity].get(key)
                    for key in ("descriptor", "operator", "threshold", "smarts")):
                raise ValueError(f"Workspace rule differs from trained tolerance definition: {identity}")
    cache_options = {"cache_path": cache_path} if cache_path is not None else {}
    predicted = predict_tolerance(profile, records, context=context,
        baseline_eligible=records.baseline_eligible, reconsidered_criteria=criteria, **cache_options)
    if (not predicted.record_id.is_unique or set(predicted.record_id) != set(records.record_id)):
        raise ValueError("Tolerance inference changed workspace record identities")
    predicted = predicted.set_index("record_id").reindex(records.record_id).reset_index()
    hard_allowed = records.eligible.to_numpy(dtype=bool)
    baseline = records.baseline_eligible.to_numpy(dtype=bool) & hard_allowed
    eligible = (predicted.tolerance_eligible.to_numpy(dtype=bool) | baseline) & hard_allowed
    state = predicted.tolerance_state.where(hard_allowed, "excluded")
    state = state.where(~baseline, "base_eligible")
    combined = records.assign(**{name: predicted[name].to_numpy() for name in predicted
                                if name != "record_id"})
    scored = combined.assign(eligible=eligible, tolerance_eligible=eligible,
        tolerance_state=state, baseline_eligible=baseline, priority_score=records.qed.to_numpy(),
        murcko_scaffold=records.session_scaffold.fillna("").to_numpy(),
        calibration_status="not_activity_probability", score_status="chemical_tolerance")
    return FeatureSet(scored, {**prepared.manifest, "ranking_mode": "chemical_tolerance",
        "meaning": "Compatibility with observed active chemistry; not activity or safety probability"})


def preview_tolerance_decision(snapshot, output_dir, package_path, context=None, *,
                               reconsidered_criteria=None, cancelled=None, cache_path=None):
    """Freeze a tolerance proposal; adoption remains an explicit workspace operation."""
    def check():
        if cancelled and cancelled():
            raise CancelledError("Tolerance preview cancelled")

    check()
    destination = Path(output_dir)
    if destination.exists():
        raise ValueError(f"output already exists: {destination}")
    settings = _settings(snapshot, reconsidered_criteria)
    signature = tolerance_package_hash(package_path)
    profile = read_tolerance_package(package_path)
    prepared = snapshot.prepared()
    reject_model_identity_collisions(prepared)
    check()
    scored = _score(snapshot, prepared, profile, context, settings, cache_path)
    check()
    c = snapshot.constraints
    proposed = select_candidates(scored, c.target_count, c.max_per_scaffold, c.min_scaffolds,
                                 c.max_per_cluster, snapshot.pinned_ids, snapshot.excluded_ids)
    original, final = list(snapshot.original_ids), proposed.manifest["final_ids"]
    summary = {"preview_version": 1, "decision_status": "preview_not_adopted",
        "ranking_mode": "chemical_tolerance", "source": prepared.manifest,
        "model": {"path": str(Path(package_path).resolve()), "package_sha256": signature},
        "context": context or {}, "tolerance_settings": settings,
        "original_ids": original, "proposed_ids": final,
        "original_count": len(original), "proposed_count": len(final),
        "eligible_count": int(proposed.records.eligible.sum()), "total_count": len(scored.records),
        "baseline_count": int(scored.records.baseline_eligible.sum()),
        "rescue_count": int((scored.records.eligible & ~scored.records.baseline_eligible).sum()),
        "review_count": int(scored.records.tolerance_state.eq("review").sum()),
        "constraints": proposed.manifest["constraints"],
        "pinned_ids": proposed.manifest["pinned_ids"], "excluded_ids": proposed.manifest["excluded_ids"],
        "shortfall": proposed.manifest["shortfall"],
        "retained_ids": sorted(set(final) & set(original)), "added_ids": sorted(set(final) - set(original)),
        "removed_ids": sorted(set(original) - set(final)),
        "warnings": [*proposed.manifest["warnings"],
            "Experimental chemical compatibility. Activity, interference and safety are not established."],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".s2s-tolerance-", dir=destination.parent) as temporary:
        root = Path(temporary) / "preview"
        root.mkdir()
        for name, data in (("prepared", prepared), ("scored", scored), ("proposed", proposed)):
            write_bundle(data, root / name)
        comparison = _comparison(proposed, original)
        for name in proposed.records:
            if name.startswith("tolerance_") or name == "baseline_eligible":
                comparison[name] = proposed.records[name]
        spreadsheet_safe(comparison).to_csv(root / "comparison.csv", index=False)
        write_json(root / "comparison.json", summary)
        write_json(root / "preview.json", {"preview_version": 1, "files": _freeze_files(root)})
        check()
        if tolerance_package_hash(package_path) != signature:
            raise ValueError("Tolerance package changed during preview")
        root.rename(destination)
    return summary


def verify_tolerance_decision(snapshot, preview_dir, package_path, context=None, *,
                              reconsidered_criteria=None, cache_path=None):
    """Verify package, workspace and policy, then replay inference and final selection."""
    root = Path(preview_dir)
    summary = verify_preview(root)
    settings = _settings(snapshot, reconsidered_criteria)
    package = {"path": str(Path(package_path).resolve()),
               "package_sha256": tolerance_package_hash(package_path)}
    if (summary.get("ranking_mode") != "chemical_tolerance" or summary.get("model") != package
        or summary.get("context") != (context or {}) or summary.get("tolerance_settings") != settings
        or summary["source"].get("session_revision") != snapshot.revision
        or summary.get("original_ids") != list(snapshot.original_ids)):
        raise ValueError("Tolerance package, context, policy or workspace snapshot changed")
    prepared = snapshot.prepared()
    profile = read_tolerance_package(package_path)
    replayed = _score(snapshot, prepared, profile, context, settings, cache_path)
    c = snapshot.constraints
    selected = select_candidates(replayed, c.target_count, c.max_per_scaffold, c.min_scaffolds,
                                 c.max_per_cluster, snapshot.pinned_ids, snapshot.excluded_ids)
    frozen = read_bundle(root / "proposed")
    fields = ["record_id", "eligible", "tolerance_state", "tolerance_nonconformity"]
    if selected.manifest["final_ids"] != summary["proposed_ids"]:
        raise ValueError("Tolerance replay differs from frozen proposal")
    try:
        pd.testing.assert_frame_equal(selected.records[fields], frozen.records[fields],
                                      check_dtype=False, rtol=1e-12, atol=1e-12)
    except AssertionError as exc:
        raise ValueError("Tolerance replay differs from frozen proposal") from exc
    return summary, frozen
