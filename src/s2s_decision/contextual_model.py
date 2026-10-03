"""Inspectable B15 logistic activity packages; no pickle or automatic adoption."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys

from smiles2select.alerts.engine import AlertEngine
from smiles2select.profiles.loader import BUILTIN_DIR, load_profile_file

from .features import chemistry_manifest, validate_chemistry_compatibility
from .schema import DESCRIPTOR_NAMES, fingerprint_matrix

SCHEMA = "s2s-contextual-logistic/1"


def annotate_chemistry(frame):
    """Add individual canonical rules and catalog identities without changing rows."""
    if not isinstance(frame, pd.DataFrame) or "model_smiles" not in frame:
        raise ValueError("Chemical annotation requires model_smiles dataframe")
    extra = {}
    for profile_id in ("lipinski", "veber"):
        profile = load_profile_file(BUILTIN_DIR / f"{profile_id}.json")
        for rule in profile.rules:
            lower, upper = rule.bounds
            if rule.operator != "<=" or lower is not None or not upper or upper <= 0:
                raise ValueError("Unsupported rule magnitude contract")
            values = frame[rule.descriptor].to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite chemical rule descriptor")
            extra[f"rule__{rule.id}__violation"] = values > upper
            extra[f"rule__{rule.id}__normalized_excess"] = np.maximum(values - upper, 0) / upper
    engine = AlertEngine(("pains", "brenk"))
    identifiers, details = [], []
    for smiles in frame.model_smiles:
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if mol is None:
            raise ValueError("Invalid standardized model_smiles")
        matches = [{"catalog_id": hit["catalog_id"], "alert_name": hit["alert_name"],
                    "alert_id": f"{hit['catalog_id']}:" + hashlib.sha256(
                        hit["alert_name"].encode()).hexdigest()[:20]}
                   for hit in engine.scan_to_rows(0, mol)]
        ids = sorted({hit["alert_id"] for hit in matches})
        identifiers.append(ids)
        details.append(matches)
    for name in sorted({name for row in identifiers for name in row}):
        extra[f"alert__{name}"] = [name in row for row in identifiers]
    extra["alert_ids"] = identifiers
    extra["alert_details"] = details
    replaced = set(extra) | {name for name in frame if name.startswith(("rule__", "alert__"))}
    return frame.drop(columns=list(replaced & set(frame))).assign(**extra)


def _numeric(frame, columns):
    if not isinstance(frame, pd.DataFrame) or not set(columns) <= set(frame):
        raise ValueError("Model feature columns are missing")
    try:
        values = frame[columns].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Model feature columns must be numeric") from exc
    if np.isinf(values).any():
        raise ValueError("Infinite chemical feature is forbidden")
    return values


def _validate_columns(columns, fingerprint):
    if fingerprint not in ("morgan", "maccs", "none"):
        raise ValueError("Unknown fingerprint representation")
    rules = {f"rule__{rule.id}__{kind}" for profile in ("lipinski", "veber")
             for rule in load_profile_file(BUILTIN_DIR / f"{profile}.json").rules
             for kind in ("violation", "normalized_excess")}
    fixed = {*DESCRIPTOR_NAMES, "lipinski_violations", "veber_violations", "pains_count", "brenk_count", *rules}
    if not isinstance(columns, list) or not columns or any(not isinstance(c, str) for c in columns):
        raise ValueError("Feature columns must be nonempty names")
    if len(set(columns)) != len(columns) or any(name not in fixed and not re.fullmatch(
            r"alert__(?:pains|brenk):[0-9a-f]{20}", name) for name in columns):
        raise ValueError("Only unique chemical descriptors/rules/alerts may be features")


def fit_spec(frame, columns, fingerprint="morgan"):
    """Fit numeric imputation/scaling on the explicitly supplied training frame."""
    _validate_columns(columns, fingerprint)
    values = _numeric(frame, columns)
    if not len(values):
        raise ValueError("Training feature rows cannot be empty")
    median = np.array([np.nanmedian(col) if np.isfinite(col).any() else 0
                       for col in values.T])
    filled = np.where(np.isnan(values), median, values)
    mean, scale = filled.mean(axis=0), filled.std(axis=0)
    return {"fingerprint": fingerprint, "columns": list(columns),
            "median": median.tolist(), "mean": mean.tolist(),
            "scale": np.where(scale > 0, scale, 1).tolist()}


def feature_matrix(frame, spec):
    _validate_columns(spec["columns"], spec["fingerprint"])
    values = _numeric(frame, spec["columns"])
    size = len(spec["columns"])
    arrays = [np.asarray(spec[key], float) for key in ("median", "mean", "scale")]
    if any(a.shape != (size,) or not np.isfinite(a).all() for a in arrays) or (arrays[2] <= 0).any():
        raise ValueError("Invalid feature preprocessing arrays")
    median, mean, scale = arrays
    numeric = (np.where(np.isnan(values), median, values) - mean) / scale
    if spec["fingerprint"] == "morgan":
        fp = fingerprint_matrix(frame)
    elif spec["fingerprint"] == "maccs":
        fps = []
        for smiles in frame.model_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("Invalid standardized molecule for MACCS")
            fps.append(np.array(MACCSkeys.GenMACCSKeys(mol), dtype=float))
        fp = np.asarray(fps).reshape(len(frame), 167)
    elif spec["fingerprint"] == "none":
        fp = np.empty((len(frame), 0))
    else:
        raise ValueError("Unknown fingerprint representation")
    return np.column_stack((fp, numeric, np.isnan(values).astype(float)))


def feature_names(spec):
    bits = {"morgan": 2048, "maccs": 167, "none": 0}[spec["fingerprint"]]
    return [*[f"{spec['fingerprint']}_{i}" for i in range(bits)],
            *spec["columns"], *[f"missing_{name}" for name in spec["columns"]]]


def max_similarity(frame, references):
    if not references:
        raise ValueError("Applicability requires training fingerprints")
    try:
        refs = [DataStructs.CreateFromBinaryText(bytes.fromhex(text)) for text in references]
        if any(fp.GetNumBits() != 2048 for fp in refs):
            raise ValueError("Invalid reference fingerprint size")
        return np.array([max(DataStructs.BulkTanimotoSimilarity(
            DataStructs.CreateFromBinaryText(bytes.fromhex(text)), refs))
            for text in frame.fingerprint_hex])
    except (TypeError, RuntimeError) as exc:
        raise ValueError("Invalid applicability fingerprint") from exc


def prediction_sets(probabilities, quantiles):
    values, limits = np.asarray(probabilities, float), np.asarray(quantiles, float)
    if limits.shape != (2,) or not np.isfinite(limits).all() or ((limits < 0) | (limits > 1)).any():
        raise ValueError("Invalid conformal class quantiles")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Invalid activity probabilities")
    return [[label for label, error in ((0, p), (1, 1 - p))
             if error <= limits[label] + 1e-15] for p in values]


def validate_bundle(bundle):
    try:
        if bundle["schema"] != SCHEMA or bundle["family"] != "logistic":
            raise ValueError("Unsupported contextual model package")
        spec = bundle["spec"]
        _validate_columns(spec["columns"], spec["fingerprint"])
        for key in ("median", "mean", "scale"):
            values = np.asarray(spec[key], float)
            if values.shape != (len(spec["columns"]),) or not np.isfinite(values).all():
                raise ValueError("Invalid loaded feature scaler")
            if key == "scale" and (values <= 0).any():
                raise ValueError("Feature scales must be positive")
        names = feature_names(bundle["spec"])
        weights = np.asarray(bundle["weights"], float)
        if weights.shape != (len(names),) or not np.isfinite(weights).all():
            raise ValueError("Invalid contextual coefficient vector")
        calibration = bundle["calibration"]
        numbers = [bundle["intercept"], calibration["slope"], calibration["intercept"],
                   bundle["domain_threshold"]]
        if not np.isfinite(numbers).all() or calibration["slope"] < 0:
            raise ValueError("Invalid logistic/calibration parameters")
        if not 0 <= bundle["domain_threshold"] <= 1 or not bundle["training_fingerprints"]:
            raise ValueError("Invalid applicability references or threshold")
        counts = [calibration[key] for key in ("n", "positive_n", "negative_n")]
        if any(type(n) is not int or n < 5 for n in counts) or counts[0] != sum(counts[1:]):
            raise ValueError("Invalid probability calibration class support")
        conformal = bundle["conformal"]
        if not 0 < conformal["alpha"] < 1 or len(conformal["counts"]) != 2 or any(
                type(n) is not int or n < 0 for n in conformal["counts"]):
            raise ValueError("Invalid conformal metadata")
        prediction_sets(np.array([]), bundle["conformal"]["quantiles"])
        validate_chemistry_compatibility(bundle["chemistry"], chemistry_manifest(2048))
        context = bundle["context"]
        if context["target"] != bundle["task_id"].split("_")[0] or context["endpoint"] != bundle["task_id"].split("_")[-1]:
            raise ValueError("Context target/endpoint conflicts with model task")
        if context["chemistry_version"] != bundle["chemistry"]["chemistry_hash"]:
            raise ValueError("Context chemistry version conflicts with model recipe")
        if context["species"] != "Homo sapiens" or context["stage"] != "hit_finding":
            raise ValueError("B15 portable models support human hit-finding outcomes only")
        threshold = bundle["activity_threshold"]
        if threshold != {"value": 1000., "unit": "nM", "relation": "<="} or any(
            context[key] != threshold[other] for key, other in (
                ("activity_threshold", "value"), ("activity_unit", "unit"), ("activity_relation", "relation"))):
            raise ValueError("Activity threshold metadata conflicts with B15 training definition")
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("Incomplete contextual model package") from exc


def load_bundle(path):
    path = Path(path)
    if path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError("Contextual JSON package exceeds 128 MiB limit")
    bundle = json.loads(path.read_text(encoding="utf-8"))
    validate_bundle(bundle)
    return bundle


def predict_portable(frame, bundle):
    validate_bundle(bundle)
    x = feature_matrix(frame, bundle["spec"])
    terms = x * np.asarray(bundle["weights"])
    logits = terms.sum(axis=1) + bundle["intercept"]
    cal = bundle["calibration"]
    z = cal["slope"] * logits + cal["intercept"]
    probabilities = np.exp(-np.logaddexp(0., -z))
    similarity = max_similarity(frame, bundle["training_fingerprints"])
    contributions = pd.DataFrame(terms, columns=feature_names(bundle["spec"]))
    contributions.insert(0, "intercept", bundle["intercept"])
    return {"logits": logits, "probabilities": probabilities,
            "prediction_sets": prediction_sets(probabilities, bundle["conformal"]["quantiles"]),
            "max_similarity": similarity, "in_domain": similarity >= bundle["domain_threshold"],
            "contributions": contributions}


def score_frame(frame, bundle, *, batch_size=1024):
    """Score all rows in bounded matrix batches, preserving order and index."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    validate_bundle(bundle)
    annotated = annotate_chemistry(frame)
    missing_alerts = {name: False for name in bundle["spec"]["columns"]
                      if name.startswith("alert__") and name not in annotated}
    annotated = annotated.assign(**missing_alerts)
    if annotated.empty:
        return _score_chunk(annotated, bundle)
    # Bound dense fingerprint/contribution arrays; final evidence stays O(rows).
    return pd.concat([_score_chunk(annotated.iloc[start:start + batch_size], bundle)
                      for start in range(0, len(annotated), batch_size)])


def _score_chunk(frame, bundle):
    result = predict_portable(frame, bundle)
    terms = result["contributions"]
    explicit = [name for name in terms if name.startswith(("rule__", "alert__"))]
    fp = [name for name in terms if name.startswith(("morgan_", "maccs_"))]
    other = [name for name in terms if name not in ["intercept", *explicit, *fp]]
    summary = terms[["intercept", *explicit]].assign(
        fingerprint=terms[fp].sum(axis=1), descriptors_and_missing=terms[other].sum(axis=1))
    return frame.assign(activity_score=result["probabilities"],
        activity_prediction_set=result["prediction_sets"], in_domain=result["in_domain"],
        max_training_similarity=result["max_similarity"], activity_logit=result["logits"],
        activity_contributions=summary.to_dict("records"))
