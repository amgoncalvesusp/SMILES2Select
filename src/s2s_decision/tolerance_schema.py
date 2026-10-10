"""Inspectable, finite, hash-verified active-only tolerance packages."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .artifacts import file_hash, write_json

SCHEMA = "s2s-tolerance-profile/1"
MAX_PACKAGE_BYTES = 256 * 1024 * 1024
MODES = frozenset({"fixed", "general", "hierarchical", "contextual"})
RECIPES = frozenset("ABCD")


def _finite_tree(value, depth=0):
    if depth > 30:
        raise ValueError("Tolerance package nesting exceeds 30 levels")
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Tolerance package values must be finite")
        return
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Tolerance package keys must be strings")
        for child in value.values():
            _finite_tree(child, depth + 1)
        return
    if isinstance(value, list):
        if value and all(type(child) is float or type(child) is int for child in value):
            if depth == 30:
                raise ValueError("Tolerance package nesting exceeds 30 levels")
            numbers = np.asarray(value)
            # JSON integers are unbounded; NumPy uses object dtype for very large ones.
            finite = (all(type(child) is int or math.isfinite(child) for child in value)
                      if numbers.dtype.kind == "O" else bool(np.isfinite(numbers).all()))
            if not finite:
                raise ValueError("Tolerance package values must be finite")
            return
        for child in value:
            _finite_tree(child, depth + 1)
        return
    raise ValueError("Tolerance package must contain only JSON values")


def profile_hash(profile: Mapping) -> str:
    _finite_tree(profile)
    content = {key: value for key, value in profile.items() if key != "profile_sha256"}
    text = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def seal_profile(profile: Mapping) -> dict:
    """Create a new content-addressed profile without changing caller objects."""
    return {**copy.deepcopy(dict(profile)), "profile_sha256": profile_hash(profile)}


def _positive_int(value, name, minimum=0):
    if type(value) is not int or not minimum <= value <= 500_000:
        raise ValueError(f"Invalid tolerance {name}")


def _string_list(value, name):
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item or len(item) > 4096 for item in value
    ):
        raise ValueError(f"Invalid tolerance {name}")
    if len(set(value)) != len(value):
        raise ValueError(f"Duplicate tolerance {name}")


def _vector(value, width, name):
    if not isinstance(value, list) or len(value) != width:
        raise ValueError(f"Invalid tolerance {name} dimensions")
    if any(type(item) not in (int, float) for item in value):
        raise ValueError(f"Invalid tolerance {name} values")
    return np.asarray(value, dtype=float)


def _validate_calibration_audit(profile):
    policy = profile["policy"]
    if "calibration_count_basis" not in policy:
        return
    expected = "identity_context" if profile["mode"] in {"hierarchical", "contextual"} else "identity"
    if policy["calibration_count_basis"] != expected:
        raise ValueError("Tolerance calibration audit count basis conflicts with mode")
    for cohort in ("positive", "background"):
        names = (f"{cohort}_unique_identities", f"{cohort}_count", f"{cohort}_input_rows")
        values = [policy[name] for name in names]
        if any(type(value) is not int or value < 0 for value in values) or values != sorted(values):
            raise ValueError("Tolerance calibration audit counts are inconsistent")
        if expected == "identity" and values[0] != values[1]:
            raise ValueError("Tolerance calibration audit identity count is inconsistent")
    if not isinstance(policy["retention_interpretation"], str) or not policy["retention_interpretation"]:
        raise ValueError("Tolerance calibration audit needs a retention interpretation")


def _validate_references(profile, width, criteria):
    refs = profile["references"]
    if not isinstance(refs, list) or not 1 <= len(refs) <= 500_000:
        raise ValueError("Tolerance needs bounded training references")
    identities = set()
    for row in refs:
        identity = row["identity"]
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError("Tolerance reference identities must be unique strings")
        identities.add(identity)
        if not isinstance(row["scaffold"], str):
            raise ValueError("Tolerance reference scaffold must be a string")
        for name in ("failed", "unknown", "document_ids", "assay_ids"):
            _string_list(row[name], name)
        if not set(row["failed"] + row["unknown"]).issubset(criteria):
            raise ValueError("Tolerance reference contains unknown criterion")
        if set(row["failed"]) & set(row["unknown"]):
            raise ValueError("Tolerance criterion cannot be failed and unknown")
        if not isinstance(row["contexts"], list) or not row["contexts"]:
            raise ValueError("Tolerance reference contexts are required")
        for context in row["contexts"]:
            if not isinstance(context, dict) or any(
                not isinstance(context.get(name, ""), str)
                for name in ("task_id", "target_id", "species", "endpoint")
            ):
                raise ValueError("Invalid tolerance reference context")
        if set(row["magnitudes"]) != set(row["failed"]):
            raise ValueError("Tolerance reference magnitudes must match failures")
        if any(type(v) not in (int, float) or v < 0 for v in row["magnitudes"].values()):
            raise ValueError("Tolerance magnitudes must be nonnegative")
        if set(row["sides"]) != set(row["failed"]):
            raise ValueError("Tolerance reference sides must match failures")
        for sides in row["sides"].values():
            _string_list(sides, "sides")
            if not sides or not set(sides).issubset({"lower", "upper", "presence"}):
                raise ValueError("Invalid tolerance violation sides")
        _vector(row["vector"], width, "reference vector")
        if type(row["weight"]) not in (int, float) or row["weight"] <= 0:
            raise ValueError("Tolerance reference weight must be positive")
    return identities


def _validate_distance(spec, annotation, recipe):
    actual = spec.get("schema") == "s2s-tolerance-representation/1"
    if not actual and "distance_blocks" not in spec:
        return
    if spec.get("distance_formula") != "weighted_block_sum_v1":
        raise ValueError("Tolerance distance formula is incompatible")
    blocks = spec["distance_blocks"]
    if not isinstance(blocks, list) or not 1 <= len(blocks) <= 3:
        raise ValueError("Tolerance distance blocks are required")
    indices, weight = [], 0.
    for block in blocks:
        if block.get("kind") not in {"robust_rms", "mean_mismatch", "tanimoto"}:
            raise ValueError("Unknown tolerance block distance")
        items = block["indices"]
        if not isinstance(items, list) or not items or any(type(item) is not int or not 0 <= item < spec["width"] for item in items):
            raise ValueError("Invalid tolerance distance block indices")
        if type(block["weight"]) not in (int, float) or block["weight"] <= 0:
            raise ValueError("Tolerance distance weights must be positive")
        indices += items
        weight += block["weight"]
    if sorted(indices) != list(range(spec["width"])) or not np.isclose(weight, 1.):
        raise ValueError("Tolerance distance blocks must partition features and sum to one")
    if not actual:
        return
    if spec["numeric_columns"] != annotation["numeric_columns"] or spec["binary_columns"] != annotation["binary_columns"]:
        raise ValueError("Tolerance distance feature dictionary mismatch")
    numeric = len(annotation["numeric_columns"])
    raw = numeric + len(annotation["binary_columns"])
    size, offset = {"A": 0, "B": 2048, "C": 315, "D": 2048}[recipe], 2 * raw
    if spec["representation_offset"] != offset or spec["representation_size"] != size or spec["width"] != offset + size + bool(size):
        raise ValueError("Tolerance distance representation layout mismatch")
    binary = [*range(numeric, raw), *range(raw, offset), *([offset + size] if size else [])]
    expected = [("numeric", "robust_rms", list(range(numeric))), ("binary_missing", "mean_mismatch", binary)]
    if size:
        expected += [("erg" if recipe == "C" else "fingerprint", "robust_rms" if recipe == "C" else "tanimoto", list(range(offset, offset + size)))]
    expected = [item for item in expected if item[2]]
    if len(blocks) != len(expected) or any(
        (block["name"], block["kind"], block["indices"]) != item or not np.isclose(block["weight"], 1. / len(expected))
        for block, item in zip(blocks, expected, strict=True)
    ):
        raise ValueError("Tolerance distance blocks disagree with the frozen recipe")


def _validate_structure(profile):
    if profile.get("schema") != SCHEMA:
        raise ValueError("Unsupported tolerance profile schema")
    if profile.get("mode") not in MODES or profile.get("recipe") not in RECIPES:
        raise ValueError("Unsupported tolerance mode or recipe")
    if profile["activity_threshold"] != {"pactivity": 6.5, "relation": ">=", "fit_relation": "="}:
        raise ValueError("Tolerance training activity contract is incompatible")
    annotation = profile["annotation_schema"]
    if annotation.get("schema") != "s2s-tolerance-annotations/1":
        raise ValueError("Unsupported tolerance annotation schema")
    digest = annotation.get("chemistry_hash", "")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("Tolerance chemistry hash is invalid")
    criteria = annotation["criterion_columns"]
    if not isinstance(criteria, dict) or not criteria or len(criteria) > 10_000:
        raise ValueError("Tolerance criterion dictionary is invalid")
    if any(not isinstance(meta, dict) or not meta.get("violation_column")
           or not meta.get("magnitude_column") for meta in criteria.values()):
        raise ValueError("Tolerance criterion metadata is incomplete")
    if not isinstance(profile["context"], dict) or not isinstance(profile["provenance"], dict):
        raise ValueError("Tolerance context/provenance must be objects")
    if any(not isinstance(profile["context"].get(key, ""), str)
           for key in ("task_id", "target_id", "species", "endpoint")):
        raise ValueError("Tolerance context fields must be strings")
    settings = profile["support_settings"]
    for key in ("min_identities", "min_scaffolds", "min_documents"):
        _positive_int(settings[key], key, 1 if key == "min_identities" else 0)
    _positive_int(settings["k"], "k", 1)
    if type(settings["allow_fallback"]) is not bool:
        raise ValueError("Tolerance allow_fallback must be boolean")
    for key in ("kappa", "domain_threshold"):
        if type(settings[key]) not in (int, float) or settings[key] < 0:
            raise ValueError(f"Tolerance {key} must be nonnegative")
    if type(settings["quantile"]) not in (int, float) or not 0 < settings["quantile"] <= 1:
        raise ValueError("Tolerance quantile must be in (0,1]")
    spec = profile["representation"]
    width = spec["width"]
    _positive_int(width, "representation width", 1)
    if width > 8192 or spec.get("recipe") != profile["recipe"]:
        raise ValueError("Invalid tolerance representation recipe/width")
    _vector(spec["center"], width, "center")
    if (_vector(spec["scale"], width, "scale") <= 0).any():
        raise ValueError("Tolerance representation scales must be positive")
    fingerprint_width = spec["fingerprint_width"]
    if fingerprint_width not in (0, 2048) or fingerprint_width > width:
        raise ValueError("Invalid tolerance fingerprint dimensions")
    expected = 2048 if profile["recipe"] in {"B", "D"} else 0
    offset = spec.get("representation_offset", width - fingerprint_width)
    if fingerprint_width != expected or type(offset) is not int or not 0 <= offset <= width - fingerprint_width:
        raise ValueError("Tolerance fingerprint block does not match recipe")
    _validate_distance(spec, annotation, profile["recipe"])
    identities = _validate_references(profile, width, criteria)
    for name, count in profile["training_counts"].items():
        _positive_int(count, name)
    if profile["training_counts"]["identities"] != len(identities):
        raise ValueError("Tolerance training identity count mismatch")
    for name, curve in profile["curves"].items():
        if name not in criteria or not isinstance(curve, dict):
            raise ValueError("Invalid tolerance curve criterion")
        x, cdf = np.asarray(curve["x"], dtype=float), np.asarray(curve["cdf"], dtype=float)
        if x.ndim != 1 or len(x) < 1 or x.shape != cdf.shape:
            raise ValueError("Invalid tolerance curve dimensions")
        if (x < 0).any() or (np.diff(x) <= 0).any() or (np.diff(cdf) < 0).any():
            raise ValueError("Tolerance CDF must be monotone")
        if (cdf <= 0).any() or (cdf > 1).any() or not np.isclose(cdf[-1], 1):
            raise ValueError("Tolerance CDF must end at one")
        if curve["maximum"] != x[-1]:
            raise ValueError("Tolerance maximum must match curve")
    policy = profile["policy"]
    _validate_calibration_audit(profile)
    status = policy["status"]
    if status not in {"uncalibrated", "feasible", "infeasible", "baseline_sufficient"}:
        raise ValueError("Unsupported tolerance calibration status")
    enabled = policy["rescue_enabled"]
    if type(enabled) is not bool or enabled != (status == "feasible"):
        raise ValueError("Tolerance rescue state conflicts with calibration")
    if status == "feasible":
        if type(policy["tau"]) not in (int, float) or not 0 <= policy["tau"] <= 1:
            raise ValueError("Tolerance calibration tau must be in [0,1]")
        scope = policy["scope"]
        _string_list(scope["criteria"], "calibrated criteria")
        if not set(scope["criteria"]).issubset(criteria) or scope["criteria"] != sorted(scope["criteria"]):
            raise ValueError("Tolerance calibrated criterion scope is invalid")
        if scope["context_mode"] not in {"global", "fixed", "row_context"}:
            raise ValueError("Tolerance calibrated context mode is invalid")
        if not isinstance(scope["requested_context"], dict) or not isinstance(scope["contexts"], list) or not scope["contexts"]:
            raise ValueError("Tolerance calibrated contexts are required")
        for context in [scope["requested_context"], *scope["contexts"]]:
            if not isinstance(context, dict) or not set(context).issubset({"task_id", "target_id", "species", "endpoint"}):
                raise ValueError("Invalid tolerance calibration context")
            if any(not isinstance(value, str) or not value for value in context.values()):
                raise ValueError("Tolerance calibration context values must be nonempty strings")
    elif policy["tau"] is not None:
        raise ValueError("Inactive tolerance policy must have null tau")


def validate_profile(profile: Mapping) -> None:
    """Validate at the package trust boundary; errors never become fallback models."""
    if not isinstance(profile, Mapping):
        raise ValueError("Tolerance profile must be an object")
    _finite_tree(profile)
    try:
        _validate_structure(profile)
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as exc:
        raise ValueError("Malformed tolerance profile structure") from exc
    if profile.get("profile_sha256") != profile_hash(profile):
        raise ValueError("Tolerance profile checksum mismatch")


def _read_json(path: Path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_PACKAGE_BYTES:
        raise ValueError("Tolerance package file missing, linked, or too large")
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError, OSError, RecursionError) as exc:
        raise ValueError("Tolerance package JSON cannot be read") from exc


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate tolerance JSON field")
        result[key] = value
    return result


def write_tolerance_package(profile: Mapping, path: str | Path) -> Path:
    validate_profile(profile)
    destination = Path(path)
    encoded_size = len(json.dumps(profile, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")) + 1
    if encoded_size > MAX_PACKAGE_BYTES:
        raise ValueError("Tolerance profile exceeds package size limit")
    if destination.suffix.lower() == ".json":
        write_json(destination, dict(profile))
        return destination
    if destination.exists():
        raise ValueError(f"Tolerance output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".s2s-tolerance-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "package"
        staging.mkdir()
        write_json(staging / "profile.json", dict(profile))
        write_json(staging / "manifest.json", {
            "schema": "s2s-tolerance-package/1", "profile_file": "profile.json",
            "profile_file_sha256": file_hash(staging / "profile.json"),
            "profile_sha256": profile["profile_sha256"],
            "support_storage": "typed_inline_json", "reference_count": len(profile["references"]),
        })
        staging.rename(destination)
    return destination


def read_tolerance_package(path: str | Path) -> dict:
    root = Path(path)
    if root.is_symlink():
        raise ValueError("Tolerance package cannot be a symbolic link")
    if root.is_dir():
        if {entry.name for entry in root.iterdir()} != {"manifest.json", "profile.json"}:
            raise ValueError("Tolerance package contains unsupported files")
        manifest = _read_json(root / "manifest.json")
        if not isinstance(manifest, dict) or manifest.get("schema") != "s2s-tolerance-package/1":
            raise ValueError("Unsupported tolerance package manifest")
        if manifest.get("profile_file") != "profile.json":
            raise ValueError("Tolerance manifest profile_file must be profile.json")
        data = root / "profile.json"
        if data.is_symlink() or data.resolve().parent != root.resolve():
            raise ValueError("Tolerance package path escapes its directory")
        profile = _read_json(data)
        if not isinstance(profile, dict):
            raise ValueError("Tolerance package profile must be an object")
        if file_hash(data) != manifest.get("profile_file_sha256"):
            raise ValueError("Tolerance package file checksum mismatch")
        if profile.get("profile_sha256") != manifest.get("profile_sha256"):
            raise ValueError("Tolerance manifest profile checksum mismatch")
        if len(profile.get("references", [])) != manifest.get("reference_count"):
            raise ValueError("Tolerance package reference count mismatch")
    else:
        profile = _read_json(root)
    validate_profile(profile)
    return profile
