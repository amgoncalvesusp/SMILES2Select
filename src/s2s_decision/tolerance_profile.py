"""Active-only chemical compatibility, support, and empirical retention policies.

These profiles describe observed chemistry. Their scores are not probabilities of
activity or safety. Calibration and biological qualification remain separate.
"""

from __future__ import annotations

import copy
import json
import math
from functools import lru_cache

import numpy as np
import pandas as pd

from .tolerance_features import annotate_tolerance, representation_matrix
from .tolerance_schema import (
    MODES,
    RECIPES,
    read_tolerance_package,
    seal_profile,
    validate_profile,
    write_tolerance_package,
)

__all__ = ["fit_tolerance", "adapt_tolerance", "predict_tolerance", "calibrate_tolerance",
           "read_tolerance_package", "write_tolerance_package"]

DEFAULT_SUPPORT = {
    "min_identities": 20, "min_scaffolds": 5, "min_documents": 3,
    "k": 64, "kappa": 64., "allow_fallback": True,
    "domain_threshold": None, "quantile": .95,
}
_CONTEXT_KEYS = ("task_id", "target_id", "species", "endpoint")


def _text(value):
    return "" if value is None or pd.isna(value) else str(value).strip()


def _evidence_id(value):
    text = _text(value)
    return "" if text.lower() in {"nan", "na", "n/a", "none", "null", "<na>"} else text


def _context_row(row):
    result = {name: _text(row.get(name, "")) for name in _CONTEXT_KEYS}
    result["task_id"] = result["task_id"] or (
        "|".join(result[key] for key in _CONTEXT_KEYS[1:]) if any(result.values()) else "general")
    return result


def _primary_training(records):
    if not isinstance(records, pd.DataFrame) or records.empty:
        raise ValueError("Tolerance training needs a nonempty dataframe")
    frame = records.copy()
    if "split" in frame:
        frame = frame.loc[frame.split.eq("fit")].copy()
    name = next((key for key in ("pactivity", "pActivity") if key in frame), None)
    if name is None:
        raise ValueError("Tolerance needs exact measured pactivity >= 6.5")
    values = pd.to_numeric(frame[name], errors="raise")
    if np.isinf(values.to_numpy(float)).any():
        raise ValueError("Tolerance activity measurements must be finite")
    exact = frame["relation"].eq("=") if "relation" in frame else pd.Series(True, index=frame.index)
    frame = frame.loc[exact & values.notna()].assign(_active=values.loc[exact & values.notna()].ge(6.5))
    # Disagreeing exact measurements within the same identity/task are not positives.
    identity = next((key for key in ("identity", "model_smiles") if key in frame), None)
    if identity and len(frame):
        tasks = frame.apply(lambda row: _context_row(row)["task_id"], axis=1)
        groups = frame.assign(_task=tasks).groupby([identity, "_task"], dropna=False)["_active"].transform("nunique")
        frame = frame.loc[groups.eq(1)]
    frame = frame.loc[frame._active].drop(columns="_active")
    if frame.empty:
        raise ValueError("No exact positive fit measurements satisfy pactivity >= 6.5")
    if "record_id" not in frame or frame.record_id.duplicated().any():
        frame = frame.assign(record_id=range(len(frame)))
    return frame.reset_index(drop=True)


def _row_pattern(row, schema):
    failed, unknown, magnitudes, sides = [], [], {}, {}
    names = schema["criterion_columns"]
    if isinstance(row.get("tolerance_failed_criteria"), list) and isinstance(row.get("tolerance_unknown_criteria"), list):
        names = dict.fromkeys(row.tolerance_failed_criteria + row.tolerance_unknown_criteria)
    for name in names:
        meta = schema["criterion_columns"][name]
        observed = row.get(meta["violation_column"], np.nan)
        if pd.isna(observed):
            unknown.append(name)
        elif bool(observed):
            failed.append(name)
            value = row.get(meta["magnitude_column"], np.nan)
            magnitude = 1. if pd.isna(value) else float(value)
            if not np.isfinite(magnitude) or magnitude < 0:
                raise ValueError("Tolerance violation magnitudes must be finite and nonnegative")
            magnitudes[name] = magnitude
            side = [direction for direction in ("lower", "upper")
                    if float(row.get(meta.get(f"{direction}_normalized_column", ""), 0.) or 0.) > 0]
            if not side:
                observation = row.get(meta.get("descriptor", ""))
                side = [direction for direction in ("lower", "upper")
                        if meta.get(f"{direction}_limit") is not None
                        and observation == meta[f"{direction}_limit"]]
            if not side:
                side = [direction for direction in ("lower", "upper") if meta.get(f"{direction}_limit") is not None]
            sides[name] = side or ["presence"]
    return failed, unknown, magnitudes, sides


def _support(refs):
    return {
        "identities": len(refs),
        "scaffolds": len({row["scaffold"] for row in refs if row["scaffold"]}),
        "documents": len({item for row in refs for item in row["document_ids"]}),
        "assays": len({item for row in refs for item in row["assay_ids"]}),
    }


def _sufficient(counts, settings):
    return all(counts[key] >= settings[f"min_{key}"] for key in ("identities", "scaffolds", "documents"))


def _balanced_weights(refs):
    """Equal task mass, then equal scaffold mass, then equal chemical identity mass."""
    if not refs:
        return []
    tasks = {}
    for index, row in enumerate(refs):
        for task in {context["task_id"] for context in row["contexts"]}:
            tasks.setdefault(task, {}).setdefault(row["scaffold"] or "<acyclic>", []).append(index)
    weights = np.zeros(len(refs), dtype=float)
    for scaffolds in tasks.values():
        for indices in scaffolds.values():
            weights[indices] += 1. / (len(tasks) * len(scaffolds) * len(indices))
    return weights.tolist()


def _curve(values, weights):
    order = np.argsort(values, kind="stable")
    x, w = np.asarray(values, float)[order], np.asarray(weights, float)[order]
    unique, index = np.unique(x, return_index=True)
    mass = np.add.reduceat(w, index)
    cumulative = np.cumsum(mass)
    return {"x": unique.tolist(), "cdf": (cumulative / cumulative[-1]).tolist(), "maximum": float(unique[-1])}


def _distance_recipe(representation, width, recipe):
    if representation.get("schema") != "s2s-tolerance-representation/1":
        groups = [{"name": "numeric", "kind": "robust_rms", "indices": list(range(width))}]
    else:
        numeric = len(representation["numeric_columns"])
        raw = numeric + len(representation["binary_columns"])
        size, offset = representation["representation_size"], representation["representation_offset"]
        if offset != 2 * raw or width != offset + size + int(size > 0):
            raise ValueError("Tolerance distance blocks conflict with representation layout")
        binary = [*range(numeric, raw), *range(raw, offset)]
        if size:
            binary.append(offset + size)
        groups = [{"name": "numeric", "kind": "robust_rms", "indices": list(range(numeric))},
                  {"name": "binary_missing", "kind": "mean_mismatch", "indices": binary}]
        if size:
            groups += [{"name": "erg" if recipe == "C" else "fingerprint",
                        "kind": "robust_rms" if recipe == "C" else "tanimoto",
                        "indices": list(range(offset, offset + size))}]
        groups = [group for group in groups if group["indices"]]
    return {"distance_formula": "weighted_block_sum_v1",
            "distance_blocks": [{**group, "weight": 1. / len(groups)} for group in groups],
            "representation_distance": "weighted_block_sum_v1",
            "normalization": "numeric/ErG median-IQR from fit only; zero IQR=1; binary unscaled"}


def _distance_matrix(refs, spec):
    train = np.asarray([row["vector"] for row in refs], dtype=np.float32)
    prepared = []
    for block in spec["distance_blocks"]:
        indices = block["indices"]
        values = train[:, indices]
        if block["kind"] == "robust_rms":
            values = (values - np.asarray(spec["center"])[indices]) / np.asarray(spec["scale"])[indices]
            norms = np.einsum("ij,ij->i", values, values)
        else:
            values = (values > 0).astype(np.float32)
            norms = values.sum(axis=1)
        prepared.append((block, values, norms))
    return prepared


def _distances(vector, refs, spec, prepared=None):
    if not refs:
        return np.empty(0)
    prepared = prepared if prepared is not None else _distance_matrix(refs, spec)
    result = np.zeros(len(refs), dtype=float)
    for block, train, norms in prepared:
        indices = block["indices"]
        candidate = np.asarray(vector)[indices]
        if block["kind"] == "robust_rms":
            candidate = (candidate - np.asarray(spec["center"])[indices]) / np.asarray(spec["scale"])[indices]
            squared = np.maximum(norms + np.dot(candidate, candidate) - 2. * (train @ candidate), 0.)
            distance = np.sqrt(squared / len(indices))
        else:
            candidate = (candidate > 0).astype(np.float32)
            intersection = train @ candidate
            if block["kind"] == "tanimoto":
                union = norms + candidate.sum() - intersection
                distance = 1. - np.divide(intersection, union, out=np.ones(len(refs)), where=union > 0)
            else:
                distance = np.maximum(norms + candidate.sum() - 2. * intersection, 0.) / len(indices)
        result += block["weight"] * distance
    return result


def _domain_threshold(refs, spec):
    if len(refs) < 2:
        return 0.
    # shortcut: deterministic 512-identity diagnostic; increase only after cost/transfer evidence.
    sample = refs[::max(1, math.ceil(len(refs) / 512))]
    prepared = _distance_matrix(refs, spec)
    positions = {row["identity"]: index for index, row in enumerate(refs)}
    values = []
    for row in sample:
        distances = _distances(row["vector"], refs, spec, prepared)
        distances[positions[row["identity"]]] = np.inf
        values.append(float(np.min(distances)))
    return float(np.quantile(values, .99))


def _make_references(frame, schema, matrix):
    refs = {}
    # A fresh wrapper shares values but omits attrs, avoiding schema deep-copies per row.
    row_frame = pd.DataFrame(frame, copy=False)
    for (_, source), vector in zip(row_frame.iterrows(), matrix, strict=True):
        identity = _text(source.get("identity")) or _text(source.get("model_smiles"))
        if not identity or not bool(source.get("tolerance_structure_valid", source.get("valid", True))):
            continue
        failed, unknown, magnitudes, sides = _row_pattern(source, schema)
        context = _context_row(source)
        documents = [_evidence_id(source.get("document_id"))] if _evidence_id(source.get("document_id")) else []
        assays = [_evidence_id(source.get("assay_id"))] if _evidence_id(source.get("assay_id")) else []
        context = {**context, "document_ids": documents, "assay_ids": assays}
        existing = refs.get(identity)
        row = {"identity": identity, "scaffold": _text(source.get("murcko_scaffold", source.get("scaffold", ""))),
               "document_ids": documents, "assay_ids": assays, "contexts": [context],
               "failed": failed, "unknown": unknown, "magnitudes": magnitudes,
               "sides": sides, "vector": [float(item) for item in vector], "weight": 1.}
        if existing:
            refs[identity] = _merge_reference(existing, row)
        else:
            refs[identity] = row
    return [refs[key] for key in sorted(refs)]


def _merge_reference(existing, row):
    if any(existing[key] != row[key] for key in ("failed", "unknown", "magnitudes", "sides", "vector", "scaffold")):
        raise ValueError("Same tolerance identity has inconsistent chemical annotations")
    return {**existing,
        "document_ids": sorted(set(existing["document_ids"] + row["document_ids"])),
        "assay_ids": sorted(set(existing["assay_ids"] + row["assay_ids"])),
        "contexts": sorted({str(item): item for item in existing["contexts"] + row["contexts"]}.values(), key=str)}


def _profile_curves(refs, schema):
    curves = {}
    for criterion in schema["criterion_columns"]:
        support = [row for row in refs if criterion in row["failed"]]
        if support:
            curves[criterion] = _curve([row["magnitudes"][criterion] for row in support],
                                       [row["weight"] for row in support])
    return curves


def fit_tolerance(records, *, mode="general", recipe="A", context=None,
                  support_settings=None, provenance=None) -> dict:
    """Fit the same active-only core for included, task, and custom workflows."""
    if mode not in MODES or recipe not in RECIPES:
        raise ValueError("Unknown tolerance mode or recipe")
    settings = {**DEFAULT_SUPPORT, **(support_settings or {})}
    if set(settings) != set(DEFAULT_SUPPORT):
        raise ValueError("Unknown tolerance support setting")
    context = copy.deepcopy(context or {})
    if not isinstance(context, dict):
        raise ValueError("Tolerance context must be an object")
    frame, schema = annotate_tolerance(_primary_training(records))
    if "tolerance_structure_valid" in frame:
        frame = frame.loc[frame.tolerance_structure_valid].reset_index(drop=True)
    matrix, representation = representation_matrix(frame, schema, recipe=recipe)
    if not len(frame) or not np.isfinite(matrix).all():
        raise ValueError("Tolerance fitting requires finite chemistry")
    center = np.median(matrix, axis=0)
    scale = np.quantile(matrix, .75, axis=0) - np.quantile(matrix, .25, axis=0)
    scale = np.where(scale > 0, scale, 1.)
    spec = {**representation, "recipe": recipe, "width": matrix.shape[1],
            "center": center.tolist(), "scale": scale.tolist(),
            "fingerprint_width": 2048 if recipe in {"B", "D"} else 0,
            **_distance_recipe(representation, matrix.shape[1], recipe)}
    refs = _make_references(frame, schema, matrix)
    if not refs:
        raise ValueError("Tolerance fitting has no valid positive references")
    weights = [1. / len(refs)] * len(refs) if mode == "fixed" else _balanced_weights(refs)
    refs = [{**row, "weight": weight} for row, weight in zip(refs, weights, strict=True)]
    if settings["domain_threshold"] is None:
        settings = {**settings, "domain_threshold": _domain_threshold(refs, spec) if mode == "contextual" else 0.}
    curves = _profile_curves(refs, schema)
    counts = {**_support(refs), "rows": len(frame),
              "tasks": len({item["task_id"] for row in refs for item in row["contexts"]})}
    profile = seal_profile({
        "schema": "s2s-tolerance-profile/1", "mode": mode, "recipe": recipe,
        "activity_threshold": {"pactivity": 6.5, "relation": ">=", "fit_relation": "="},
        "annotation_schema": schema, "context": context, "support_settings": settings,
        "representation": spec, "references": refs, "curves": curves,
        "training_counts": counts, "policy": {"status": "uncalibrated", "tau": None, "rescue_enabled": False},
        "provenance": {**(provenance or {}), "cdf_convention": "right; mass_at_or_below",
                       "activity_probability": False, "qualification": "experimental",
                       "domain_diagnostic": "99th percentile of nearest-other-identity distance; deterministic <=512 training rows"},
    })
    validate_profile(profile)
    return profile


def adapt_tolerance(base_profile, records, *, mode="hierarchical", context=None,
                    support_settings=None, provenance=None) -> dict:
    """Adapt shared fit references; never reconstruct records from hashes or activity guesses."""
    validate_profile(base_profile)
    if mode not in MODES:
        raise ValueError("Unknown tolerance adaptation mode")
    settings = {**base_profile["support_settings"], **(support_settings or {})}
    if set(settings) != set(DEFAULT_SUPPORT) or settings["domain_threshold"] != base_profile["support_settings"]["domain_threshold"]:
        raise ValueError("Adaptation must keep the frozen base domain and recognized support settings")
    schema, recipe = base_profile["annotation_schema"], base_profile["recipe"]
    frame, actual_schema = annotate_tolerance(_primary_training(records), schema=schema)
    if actual_schema != schema:
        raise ValueError("Adaptation chemistry differs from the frozen base schema")
    if "tolerance_structure_valid" in frame:
        frame = frame.loc[frame.tolerance_structure_valid].reset_index(drop=True)
    matrix, _ = representation_matrix(frame, schema, recipe=recipe)
    if matrix.shape[1] != base_profile["representation"]["width"] or not np.isfinite(matrix).all():
        raise ValueError("Adaptation representation conflicts with the frozen base transform")
    custom = _make_references(frame, schema, matrix)
    if not custom:
        raise ValueError("Adaptation has no valid positive fit references")
    merged = {row["identity"]: copy.deepcopy(row) for row in base_profile["references"]}
    overlaps = set(merged) & {row["identity"] for row in custom}
    for row in custom:
        existing = merged.get(row["identity"])
        merged[row["identity"]] = _merge_reference(existing, row) if existing else row
    refs = [merged[key] for key in sorted(merged)]
    weights = [1. / len(refs)] * len(refs) if mode == "fixed" else _balanced_weights(refs)
    refs = [{**row, "weight": weight} for row, weight in zip(refs, weights, strict=True)]
    adaptation = {"base_profile_sha256": base_profile["profile_sha256"], "custom_fit_rows": len(frame),
                  "custom_fit_identities": len(custom), "overlapping_fit_identities": len(overlaps),
                  "new_fit_identities": len(custom) - len(overlaps),
                  "transform": "frozen_base", "domain_threshold": "frozen_base"}
    result = seal_profile({**base_profile, "mode": mode, "context": copy.deepcopy(context if context is not None else base_profile["context"]),
        "references": refs, "curves": _profile_curves(refs, schema), "support_settings": settings,
        "training_counts": {**_support(refs), "rows": base_profile["training_counts"]["rows"] + len(frame),
                            "tasks": len({item["task_id"] for row in refs for item in row["contexts"]})},
        "policy": {"status": "uncalibrated", "tau": None, "rescue_enabled": False},
        "provenance": {**base_profile["provenance"], **(provenance or {}), "adaptation": adaptation},
    })
    validate_profile(result)
    return result


def _compatible_pool(refs, context, *, task=False):
    keys = list(_CONTEXT_KEYS if task else ("species", "endpoint"))
    result = []
    for row in refs:
        contexts = [item for item in row["contexts"]
                    if all(not context.get(key) or item.get(key) == str(context[key]) for key in keys)]
        if contexts:
            result.append({**row, "contexts": contexts,
                "document_ids": sorted({value for item in contexts for value in item.get("document_ids", row["document_ids"])}),
                "assay_ids": sorted({value for item in contexts for value in item.get("assay_ids", row["assay_ids"])})})
    return result


def _joint(refs, failed, sides):
    return [row for row in refs if all(
        name in row["failed"] and set(sides[name]).issubset(row["sides"][name]) for name in failed)]


def _effective_n(refs):
    values = np.asarray([row["weight"] for row in refs], dtype=float)
    return float(values.sum() ** 2 / np.dot(values, values)) if len(values) else 0.


def _curves_for(refs, failed):
    return {name: _curve([row["magnitudes"][name] for row in refs], [row["weight"] for row in refs])
            for name in failed} if refs else {}


def _percentile(curve, value):
    index = int(np.searchsorted(curve["x"], value, side="right")) - 1
    return float(curve["cdf"][index]) if index >= 0 else 0.


def _resolve_context(refs, context):
    result = dict(context)
    specific = result.get("task_id") or result.get("target_id")
    if specific:
        matches = _compatible_pool(refs, result, task=True)
        for key in _CONTEXT_KEYS:
            values = {item[key] for row in matches for item in row["contexts"] if item.get(key)}
            if not result.get(key) and len(values) == 1:
                result[key] = next(iter(values))
    return result


def _canonical_context(context):
    return {key: str(context[key]) for key in _CONTEXT_KEYS if context.get(key)}


def _calibration_scope(profile, predicted, reconsidered_criteria, context):
    criteria = _active_criteria(profile["annotation_schema"]) if reconsidered_criteria is None else set(reconsidered_criteria)
    requested = _canonical_context({**profile["context"], **(context or {})})
    unique = {json.dumps(item, sort_keys=True): item for item in predicted.tolerance_applied_context}
    resolved = [_canonical_context(_resolve_context(profile["references"], item)) for item in unique.values()]
    contexts = {json.dumps(item, sort_keys=True): item for item in resolved}
    return {"criteria": sorted(criteria), "requested_context": requested,
            "context_mode": "fixed" if requested else "row_context" if profile["mode"] in {"hierarchical", "contextual"} else "global",
            "contexts": [contexts[key] for key in sorted(contexts)]}


def _scope_matches(scope, criteria, requested_context, applied_context):
    if scope is None or scope.get("criteria") != sorted(criteria):
        return False
    if scope["context_mode"] in {"global", "fixed"}:
        if scope["requested_context"] != _canonical_context(requested_context):
            return False
    return _canonical_context(applied_context) in scope["contexts"]


def _reference_pools(profile, identity, context):
    mode = profile["mode"]
    refs = [ref for ref in profile["references"] if ref["identity"] != identity]
    context = _resolve_context(refs, context)
    parent = _compatible_pool(refs, context)
    if mode != "fixed":
        parent = [{**ref, "weight": weight} for ref, weight in zip(parent, _balanced_weights(parent), strict=True)]
    task = _compatible_pool(parent, context, task=True) if context.get("task_id") or context.get("target_id") else []
    if mode != "fixed":
        task = [{**ref, "weight": weight} for ref, weight in zip(task, _balanced_weights(task), strict=True)]
    candidates = task or parent
    distances = _distance_matrix(candidates, profile["representation"]) if mode == "contextual" and candidates else None
    return context, parent, task, distances


def _pattern_pools(parent, task, failed, sides):
    parent_joint, task_joint = _joint(parent, failed, sides), _joint(task, failed, sides)
    return {
        "parent": parent_joint, "task": task_joint,
        "counts": {"parent": _support(parent_joint), "task": _support(task_joint)},
        "marginal": {name: _support(_joint(parent, [name], {name: sides[name]})) for name in failed},
        "parent_curves": _curves_for(parent_joint, failed), "task_curves": _curves_for(task_joint, failed),
        "parent_n": _effective_n(parent_joint), "task_n": _effective_n(task_joint),
    }


def _score(profile, row, vector, failed, sides, context, pools=None, pattern=None):
    settings, mode = profile["support_settings"], profile["mode"]
    identity = _text(row.get("identity")) or _text(row.get("model_smiles"))
    context, parent, task, distance_matrix = pools or _reference_pools(profile, identity, context)
    pattern = pattern or _pattern_pools(parent, task, failed, sides)
    parent_joint, task_joint = pattern["parent"], pattern["task"]
    local, local_joint, nearest = [], [], None
    if mode == "contextual" and parent:
        candidates = task or parent
        distances = _distances(vector, candidates, profile["representation"], distance_matrix)
        order = np.argsort(distances, kind="stable")[:settings["k"]]
        local = [candidates[index] for index in order if distances[index] <= settings["domain_threshold"]]
        local_joint = _joint(local, failed, sides)
        nearest = float(np.min(distances))
    counts = {**pattern["counts"], "local": _support(local_joint)}
    fallback, applicable = "none", parent_joint
    if mode in {"hierarchical", "contextual"}:
        if mode == "contextual" and _sufficient(counts["local"], settings):
            applicable = local_joint
        elif _sufficient(counts["task"], settings):
            applicable = task_joint
            fallback = "task_parent" if mode == "contextual" else "none"
        elif settings["allow_fallback"]:
            fallback = "general_parent"
        else:
            applicable = []
            fallback = "disabled"
    counts["joint"] = counts["parent"] if applicable is parent_joint else (
        counts["task"] if applicable is task_joint else counts["local"] if applicable is local_joint else _support([]))
    counts["marginal"] = pattern["marginal"]
    applicable_n = pattern["parent_n"] if applicable is parent_joint else (
        pattern["task_n"] if applicable is task_joint else _effective_n(applicable))
    reasons = []
    if not parent:
        reasons.append("incompatible_context")
    if mode in {"hierarchical", "contextual"} and not all(context.get(name) for name in ("species", "endpoint")):
        reasons.append("incomplete_context")
    if not _sufficient(counts["joint"], settings):
        reasons.append("insufficient_joint_support")
    magnitudes = _row_pattern(row, profile["annotation_schema"])[2]
    components, limits = {}, {}
    applicable_curves = pattern["parent_curves"] if applicable is parent_joint else (
        pattern["task_curves"] if applicable is task_joint else _curves_for(applicable, failed))
    for name in failed:
        if not applicable:
            continue
        limits[name] = applicable_curves[name]["maximum"]
        if magnitudes[name] > limits[name]:
            reasons.append("magnitude_outside_support")
        if sides[name] == ["presence"]:
            continue
        value = _percentile(applicable_curves[name], magnitudes[name])
        if applicable is not parent_joint and parent_joint and mode != "fixed":
            n = applicable_n
            lam = n / (n + settings["kappa"]) if n + settings["kappa"] else 1.
            parent_value = _percentile(pattern["parent_curves"][name], magnitudes[name])
            if applicable is local_joint and _sufficient(counts["task"], settings):
                task_n = pattern["task_n"]
                task_lam = task_n / (task_n + settings["kappa"]) if task_n + settings["kappa"] else 1.
                parent_value = task_lam * _percentile(pattern["task_curves"][name], magnitudes[name]) + (1. - task_lam) * parent_value
            value = lam * value + (1. - lam) * parent_value
        if mode == "fixed":
            curve = applicable_curves[name]
            index = int(np.searchsorted(curve["cdf"], settings["quantile"], side="left"))
            cutoff = curve["x"][min(index, len(curve["x"]) - 1)]
            if magnitudes[name] > cutoff:
                reasons.append("outside_frozen_quantile")
            limits[name] = cutoff
        components[name] = float(value)
    if mode == "contextual" and fallback == "none" and nearest is not None:
        threshold = settings["domain_threshold"]
        components["reference_distance"] = min(1., nearest / threshold) if threshold > 0 else 0.
    return {
        "supported": not reasons, "nonconformity": max(components.values(), default=0.),
        "support": counts, "reasons": sorted(set(reasons)), "fallback": fallback,
        "context": context, "components": components, "limits": limits,
        "reference_ids": [ref["identity"] for ref in applicable[:20]],
        "nearest_distance": nearest, "effective_n": applicable_n,
    }


def _boolean_mask(value, count, name):
    result = np.asarray(value)
    if result.shape != (count,) or any(type(item) not in (bool, np.bool_) for item in result):
        raise ValueError(f"Tolerance {name} must contain one boolean per candidate")
    return result.astype(bool)


def _active_criteria(schema):
    return {name for name in schema["criterion_columns"] if name.startswith(("rule:lipinski:", "rule:veber:", "alert:"))}


def predict_tolerance(profile, records, *, context=None, baseline_eligible=None,
                      reconsidered_criteria=None, cache_path=None) -> pd.DataFrame:
    validate_profile(profile)
    options = {"cache_path": cache_path} if cache_path is not None else {}
    frame, schema = annotate_tolerance(records, schema=profile["annotation_schema"], **options)
    matrix, _ = representation_matrix(frame, schema, recipe=profile["recipe"])
    active = _active_criteria(schema) if reconsidered_criteria is None else set(reconsidered_criteria)
    reconsidered = active
    if not reconsidered.issubset(schema["criterion_columns"]):
        raise ValueError("Unknown tolerance reconsidered criterion")
    if baseline_eligible is None and "baseline_eligible" in frame:
        baseline_eligible = frame.baseline_eligible.to_numpy()
    supplied_baseline = None if baseline_eligible is None else _boolean_mask(baseline_eligible, len(frame), "baseline eligibility")
    context = {**profile["context"], **(context or {})}
    known_identities = {row["identity"] for row in profile["references"]}

    @lru_cache(maxsize=8)
    def pools_for(identity, context_json):
        return _reference_pools(profile, identity, json.loads(context_json))

    @lru_cache(maxsize=128)
    def patterns_for(identity, context_json, pattern_key):
        _, parent, task, _ = pools_for(identity, context_json)
        failures = [name for name, _ in pattern_key]
        return _pattern_pools(parent, task, failures, {name: list(sides) for name, sides in pattern_key})

    outputs = []
    row_frame = pd.DataFrame(frame, copy=False)
    for index, (_, row) in enumerate(row_frame.iterrows()):
        failed_all, unknown_all, _, sides = _row_pattern(row, schema)
        failed, unknown = [name for name in failed_all if name in active], [name for name in unknown_all if name in active]
        base = bool(supplied_baseline[index]) if supplied_baseline is not None else not failed and not unknown
        hard = any(bool(row.get(name, False)) for name in ("hard_excluded", "manual_excluded", "locked_smarts_excluded")
                   if pd.notna(row.get(name, False)))
        invalid = not bool(row.get("tolerance_structure_valid", row.get("valid", True))) or not _text(row.get("model_smiles"))
        applied_context = context
        if profile["mode"] in {"hierarchical", "contextual"}:
            source_context = {key: _text(row.get(key)) for key in _CONTEXT_KEYS if _text(row.get(key))}
            if context.get("task_id") or context.get("target_id"):
                source_context = {key: value for key, value in source_context.items() if key not in {"task_id", "target_id"}}
            applied_context = {**source_context, **context}
        scored = {
            "supported": False, "nonconformity": 0., "support": {}, "reasons": [], "fallback": "none",
            "context": applied_context, "components": {}, "limits": {}, "reference_ids": [], "nearest_distance": None,
            "effective_n": 0.,
        }
        if failed and not (base or hard or invalid or unknown):
            identity = _text(row.get("identity")) or _text(row.get("model_smiles"))
            identity_key = identity if identity in known_identities else ""
            context_json = json.dumps(applied_context, sort_keys=True, allow_nan=False)
            pattern_key = tuple((name, tuple(sides[name])) for name in failed)
            pools = pools_for(identity_key, context_json)
            pattern = patterns_for(identity_key, context_json, pattern_key)
            scored = _score(profile, row, matrix[index], failed, sides, applied_context, pools, pattern)
        reasons = list(scored["reasons"])
        if hard or invalid:
            state, supported = "excluded", False
            reasons.append("hard_exclusion" if hard else "invalid_structure")
        elif base:
            state, supported = "base_eligible", False
        elif unknown:
            state, supported = "review", False
            reasons.append("unknown_criteria")
        elif set(failed) - reconsidered or not failed:
            state, supported = "excluded", False
            reasons.append("criterion_not_reconsidered" if failed else "baseline_exclusion_unexplained")
        elif not scored["supported"]:
            state, supported = "review", False
        else:
            supported = True
            policy = profile["policy"]
            if not policy["rescue_enabled"]:
                state = "review"
                reasons.append(f"policy_{policy['status']}")
            elif not _scope_matches(policy.get("scope"), reconsidered, context, scored["context"]):
                state = "review"
                reasons.append("calibration_scope_mismatch")
            elif scored["nonconformity"] <= policy["tau"]:
                state = "supported_relaxation"
            else:
                state = "excluded"
                reasons.append("above_calibrated_tau")
        evidence = {"failed_criteria": failed, "unknown_criteria": unknown,
                    "reconsidered_criteria": sorted(set(failed) & reconsidered),
                    "reasons": sorted(set(reasons)), "components": scored["components"],
                    "supported_limits": scored["limits"], "reference_ids": scored["reference_ids"],
                    "nearest_distance": scored["nearest_distance"], "effective_n": scored["effective_n"]}
        eligible = state in {"base_eligible", "supported_relaxation"}
        outputs.append({"record_id": int(row.record_id), "identity": _text(row.get("identity")) or _text(row.model_smiles),
            "state": state, "eligible": eligible, "nonconformity": float(scored["nonconformity"]),
            "baseline_eligible": base and not hard and not invalid, "tolerance_state": state,
            "tolerance_eligible": eligible, "tolerance_nonconformity": float(scored["nonconformity"]),
            "tolerance_supported": supported, "tolerance_support": copy.deepcopy(scored["support"]),
            "tolerance_evidence": evidence, "tolerance_applied_context": copy.deepcopy(scored["context"]),
            "tolerance_fallback": scored["fallback"], "tolerance_profile_sha256": profile["profile_sha256"]})
    columns = ["record_id", "identity", "state", "eligible", "nonconformity", "baseline_eligible",
               "tolerance_state", "tolerance_eligible", "tolerance_nonconformity", "tolerance_supported",
               "tolerance_support", "tolerance_evidence", "tolerance_applied_context", "tolerance_fallback",
               "tolerance_profile_sha256"]
    return pd.DataFrame(outputs, columns=columns)


def _calibration_cohort(profile, frame, *, positive, reconsidered_criteria=None, context=None):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Tolerance calibration cohorts must be nonempty")
    if "split" in frame and not frame.split.eq("policy_cal").all():
        raise ValueError("Tolerance calibration accepts only policy_cal rows")
    if positive:
        name = next((key for key in ("pactivity", "pActivity") if key in frame), None)
        exact = frame.relation.eq("=") if "relation" in frame else pd.Series(True, index=frame.index)
        if name is None or not (pd.to_numeric(frame[name], errors="raise").ge(6.5) & exact).all():
            raise ValueError("Positive policy_cal cohort needs exact pactivity >= 6.5")
    scored = predict_tolerance(profile, frame, reconsidered_criteria=reconsidered_criteria, context=context)
    identities = set(scored.identity)
    if identities & {row["identity"] for row in profile["references"]}:
        raise ValueError("Tolerance fit/calibration identity overlap")
    if profile["mode"] in {"hierarchical", "contextual"}:
        resolved, context_memo = [], {}
        for item in scored.tolerance_applied_context:
            key = json.dumps(item, sort_keys=True, allow_nan=False)
            if key not in context_memo:
                context_memo[key] = _canonical_context(_resolve_context(profile["references"], item))
            resolved.append(copy.deepcopy(context_memo[key]))
        keys = [(identity, json.dumps(item, sort_keys=True)) for identity, item in zip(scored.identity, resolved)]
        if len(set(keys)) != len(keys):
            raise ValueError("Tolerance calibration identity-context pairs must be unique")
        scored = scored.assign(tolerance_applied_context=resolved)
    else:
        signatures = {}
        for row in scored.to_dict("records"):
            signature = json.dumps({key: row[key] for key in
                                    ("baseline_eligible", "tolerance_supported", "nonconformity", "tolerance_evidence")},
                                   sort_keys=True, allow_nan=False)
            if row["identity"] in signatures and signatures[row["identity"]] != signature:
                raise ValueError("Repeated calibration identity has inconsistent global decisions")
            signatures[row["identity"]] = signature
        scored = scored.drop_duplicates("identity").reset_index(drop=True)
    return scored


def _calibration_audit(profile, positive, background, positive_rows, background_rows):
    paired = profile["mode"] in {"hierarchical", "contextual"}
    return {"calibration_count_basis": "identity_context" if paired else "identity",
            "positive_unique_identities": int(positive.identity.nunique()),
            "background_unique_identities": int(background.identity.nunique()) if background is not None else 0,
            "positive_input_rows": int(positive_rows), "background_input_rows": int(background_rows),
            "retention_interpretation": (
                "Pooled empirical identity-context retention with one global tau; pairs sharing a molecule are "
                "nonindependent and tasks are not equally weighted. No per-task retention, confidence or coverage guarantee."
                if paired else "Empirical retention of unique chemical identities; no confidence or coverage guarantee.")}


def calibrate_tolerance(profile, positive_cal, background_cal, *, target_retention=.95,
                        max_expansion=.1, reconsidered_criteria=None, context=None) -> dict:
    """Fit one empirical threshold using identity or identity-context counting units."""
    validate_profile(profile)
    if not 0 < target_retention <= 1 or not 0 <= max_expansion <= 1:
        raise ValueError("Invalid tolerance retention or expansion target")
    positive = _calibration_cohort(profile, positive_cal, positive=True,
                                    reconsidered_criteria=reconsidered_criteria, context=context)
    scope = _calibration_scope(profile, positive, reconsidered_criteria, context)
    if background_cal is None or isinstance(background_cal, pd.DataFrame) and background_cal.empty:
        result = seal_profile({**profile, "policy": {
            "status": "uncalibrated", "tau": None, "rescue_enabled": False,
            "target_retention": float(target_retention), "max_expansion": float(max_expansion),
            "positive_count": len(positive), "background_count": 0,
            "expansion": None, "reason": "independent_background_missing",
            "scope": scope, **_calibration_audit(profile, positive, None, len(positive_cal), 0),
        }})
        validate_profile(result)
        return result
    background = _calibration_cohort(profile, background_cal, positive=False,
                                      reconsidered_criteria=reconsidered_criteria, context=context)
    if set(positive.identity) & set(background.identity):
        raise ValueError("Tolerance positive/background calibration identity overlap")
    baseline = positive.baseline_eligible.to_numpy(bool)
    supported = positive.tolerance_supported.to_numpy(bool)
    background_r = ~background.baseline_eligible.to_numpy(bool)
    required = math.ceil(target_retention * len(positive) - 1e-12) - int(baseline.sum())
    status, tau, expansion = "baseline_sufficient", None, 0.
    achieved = float(baseline.mean())
    if required > 0:
        scores = sorted(positive.loc[supported & ~baseline, "nonconformity"].tolist())
        status = "infeasible"
        if len(scores) >= required:
            candidate_tau = float(scores[required - 1])
            rescued = supported & ~baseline & positive.nonconformity.le(candidate_tau).to_numpy()
            achieved = float((baseline | rescued).mean())
            bg_rescue = background.tolerance_supported.to_numpy(bool) & background_r & background.nonconformity.le(candidate_tau).to_numpy()
            expansion = float(bg_rescue.sum() / background_r.sum()) if background_r.any() else 0.
            if expansion <= max_expansion:
                status, tau = "feasible", candidate_tau
        else:
            achieved = float((baseline | supported).mean())
    policy = {"status": status, "tau": tau, "rescue_enabled": status == "feasible",
              "target_retention": float(target_retention), "max_expansion": float(max_expansion),
              "positive_count": len(positive), "supported_positive_count": int(supported.sum()),
              "baseline_positive_count": int(baseline.sum()), "background_count": len(background),
              "background_rescue_universe": int(background_r.sum()), "achieved_retention": achieved,
              "expansion": expansion, "criterion": "minimum_tau_under_retention_and_background_cap"}
    policy = {**policy, "scope": scope,
              **_calibration_audit(profile, positive, background, len(positive_cal), len(background_cal))}
    background_scope = _calibration_scope(profile, background, reconsidered_criteria, context)
    policy = {**policy, "background_contexts": background_scope["contexts"]}
    background_contexts = {json.dumps(item, sort_keys=True) for item in background_scope["contexts"]}
    missing = [item for item in scope["contexts"] if json.dumps(item, sort_keys=True) not in background_contexts]
    if required > 0 and missing:
        policy = {**policy, "status": "uncalibrated", "tau": None, "rescue_enabled": False,
                  "expansion": None, "achieved_retention": float(baseline.mean()),
                  "reason": "missing_context_matched_background", "missing_background_contexts": missing}
    result = seal_profile({**profile, "policy": policy})
    validate_profile(result)
    return result
