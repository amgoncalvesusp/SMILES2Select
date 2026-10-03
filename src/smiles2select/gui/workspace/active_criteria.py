"""Readable applied evidence, never reconstructed from an unadopted model picker."""

import json
from dataclasses import replace

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QPlainTextEdit, QVBoxLayout

from smiles2select.profiles.loader import builtin_registry


def _applied(window):
    scenario, applied = window._active_snapshot(), window._active_applied_selection()
    if scenario is not None:
        return scenario.spec.constraints, scenario.spec.strategy.value, {}, tuple(
            item.as_dict() for item in scenario.spec.objectives)
    if applied is not None:
        return applied.constraints, applied.strategy, applied.provenance, applied.objectives
    return None, window.result.config.selection_strategy, {}, ()


def ranking_source(window):
    _, strategy, metadata, _ = _applied(window)
    if strategy == "experimental_model":
        return (f"Experimental model ranking: {metadata.get('estimator', 'model')} / "
                f"{metadata.get('target', 'undeclared target')} / {metadata.get('endpoint', 'undeclared endpoint')}")
    if strategy == "contextual_policy":
        context = metadata.get("context", {})
        return (f"Experimental contextual ranking: {context.get('target', 'undeclared target')} / "
                f"{context.get('endpoint', 'undeclared endpoint')} / {context.get('stage', 'undeclared stage')}")
    return f"Chemical ranking: {strategy}"


def _quota_text(constraints, config):
    scaffold = constraints.max_per_scaffold if constraints else config.per_scaffold_limit
    minimum = constraints.min_scaffolds if constraints else None
    cluster = constraints.max_per_cluster if constraints else None
    return (f"Applied quotas — maximum per core: {scaffold or 'none'}; "
            f"minimum cores: {minimum or 'none'}; maximum per cluster: {cluster or 'none'}.")


def compact(window):
    constraints, _, _, _ = _applied(window)
    return (f"{ranking_source(window)}. {_quota_text(constraints, window.result.config)} "
            f"Pins: {len(window.basket.pinned_ids())}; manual exclusions: {len(window.basket.excluded_ids())}.")


def _screening(window, metadata):
    policy = window.result.config.policy
    scenario = window._active_snapshot()
    changes = dict(scenario.spec.thresholds) if scenario else {}
    revisited = set(metadata.get("revisited_profiles", ()))
    lines = ["Screening profiles and thresholds"]
    for profile in window.result.profiles:
        role = policy.roles.get(profile.id, "not in policy")
        note = ("; explicit exclusion retained; cannot be relaxed" if role == "exclusion" else
                "; reconsidered by contextual policy") if profile.id in revisited else ""
        lines.append(f"{profile.name} [{profile.id}]: {role}; {profile.policy_label()}{note}")
        for original in profile.rules:
            rule = replace(original, threshold=changes[original.id]) if original.id in changes else original
            lines.append(f"  {rule.id}: {rule.describe()}" + (f"; SMARTS {rule.smarts}" if rule.smarts else ""))
    lines.extend([f"Consensus minimum: {policy.consensus_min_pass or 'off'}",
                  f"Custom expression: {policy.expression or 'none'}",
                  f"QED screening: {policy.qed.mode}; threshold {policy.qed.threshold}; percentile {policy.qed.percentile}"])
    lines.append("Screening alert actions")
    lines.extend(f"  {catalog}: {policy.alert_policy.action_for(catalog)}"
                 for catalog in window.result.config.alert_catalogs)
    for alert in window.result.config.custom_alerts:
        lines.append(f"  Custom SMARTS: {alert.smarts}; {policy.alert_policy.action_for('custom_smarts')}")
    return lines


def _contextual(metadata):
    context, settings = metadata.get("context", {}), metadata.get("policy_settings", {})
    lines = ["Context used by the adopted proposal"]
    lines.extend(f"  {key.replace('_', ' ')}: {value}" for key, value in context.items())
    lines.append("Profiles requested for reconsideration: " + ", ".join(metadata.get("revisited_profiles", ())))
    lines.append("Contextual profile actions (distinct from upstream screening)")
    registry = builtin_registry()
    for item in settings.get("profiles", ()):
        profile_id = item["profile_id"]
        tolerance = item.get("max_violations")
        lines.append(f"  {profile_id}: {item['action']}; allowed violations: {tolerance if tolerance is not None else 'native profile tolerance'}; penalty {item.get('penalty', 0)}")
        if profile_id in registry.ids():
            profile = registry.get(profile_id)
            lines.append(f"    Native tolerance: {profile.policy_label()}")
            lines.extend(f"    {rule.id}: {rule.describe()}" for rule in profile.rules)
    lines.append(f"Unspecified alert action: {settings.get('default_alert_action', 'warn')}")
    for kind in ("rule_actions", "alert_actions"):
        lines.append(kind.replace("_", " ").capitalize())
        lines.extend(f"  {item['feature_id']}: {item['action']}; {item.get('origin', 'user')}; penalty {item.get('penalty', 0)}; support {item.get('support_n', 0)}"
                     for item in settings.get(kind, ()))
    for kind in ("required_smarts", "excluded_smarts", "preferred_smarts"):
        lines.append(f"{kind.replace('_', ' ').capitalize()}: {', '.join(settings.get(kind, ())) or 'off'}")
    risk = settings.get("risk_evidence")
    lines.append(f"Risk penalty weight: {settings.get('risk_weight', 0)}; cutoff: {settings.get('risk_exclude_at')}")
    lines.append("Risk evidence: " + (json.dumps(risk, ensure_ascii=False, sort_keys=True) if risk else "No measured-endpoint model applied"))
    lines.append("Risk predictions describe the named assay; they do not establish safety.")
    queue = metadata.get("information_queue")
    if queue:
        lines.append("Separate review queue: " + json.dumps(queue, ensure_ascii=False, sort_keys=True))
    return lines


def details(window):
    constraints, strategy, metadata, objectives = _applied(window)
    target = constraints.target_count if constraints else window.result.config.final_count
    lines = ["Applied criteria for the current final library", ranking_source(window),
             f"Applied target: {target if target is not None else 'unlimited'}; final molecules: {len(window.basket.final_ids())}",
             f"Current requested count: {window.target_count.value()} (editable next-action control)",
             _quota_text(constraints, window.result.config),
             f"Pinned record IDs: {window.basket.pinned_ids() or 'none'}",
             f"Manually excluded record IDs: {window.basket.excluded_ids() or 'none'}"]
    if window.has_pending_criteria():
        lines.append("Edited chemical criteria are not applied. Create selection replaces the basket using those controls.")
    if strategy in {"experimental_model", "contextual_policy", "qed_only"}:
        lines.append("For this applied ranking, property objectives are ignored.")
    elif objectives:
        lines.append("Applied property objectives")
        lines.extend("  " + json.dumps(item, sort_keys=True) for item in objectives)
    else:
        lines.append("Original pipeline ranking; workspace property objectives have not been applied.")
    if strategy == "experimental_model":
        lines.extend(f"{key.replace('_', ' ')}: {metadata.get(key, 'not recorded')}" for key in (
            "target", "endpoint", "threshold", "estimator", "calibration_status", "model_sha256"))
        lines.append("The chemical-strategy dropdown and another chosen model do not change this adopted ranking.")
    if strategy == "contextual_policy":
        lines.extend(_contextual(metadata))
    lines.extend(_screening(window, metadata))
    config = window.result.config
    lines.append(f"Reference constraints: similarity {config.reference_min_similarity:g} to {config.reference_max_similarity:g}; "
                 f"exclude reference duplicates: {config.exclude_reference_duplicates}")
    lines.append("Manual decisions can change membership after ranking. Map projection and point inspection do not rank the library.")
    lines.extend(str(value) for value in metadata.get("ranking_warnings", ()))
    return "\n\n".join(lines)


def show_details(window):
    dialog = getattr(window, "_criteria_dialog", None)
    if dialog is None:
        dialog = QDialog(window)
        dialog.setWindowTitle("Applied selection criteria")
        dialog.resize(810, 660)
        layout = QVBoxLayout(dialog)
        dialog.text = QPlainTextEdit()
        dialog.text.setReadOnly(True)
        layout.addWidget(dialog.text)
        close = QDialogButtonBox(QDialogButtonBox.Close)
        close.rejected.connect(dialog.close)
        layout.addWidget(close)
        window._criteria_dialog = dialog
    dialog.text.setPlainText(details(window))
    dialog.show()
    dialog.raise_()
