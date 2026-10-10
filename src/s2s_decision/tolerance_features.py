"""Chemical tolerance recipes, distinct from historical activity features."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, rdReducedGraphs

from smiles2select.alerts.rdkit_catalogs import get_catalog
from smiles2select.app_metadata import rdkit_version
from smiles2select.chemistry.descriptor_registry import default_registry
from smiles2select.chemistry.fingerprints import FingerprintConfig, fingerprint
from smiles2select.chemistry.scaffolds import murcko_scaffold
from smiles2select.profiles.loader import BUILTIN_DIR, load_profile_file
from smiles2select.rules import operators
from smiles2select.rules.engine import Rule

from .schema import DESCRIPTOR_NAMES
from .tolerance_cache import BATCH_SIZE, ToleranceAnnotationCache, content_hash

SCHEMA = "s2s-tolerance-annotations/1"
DEFAULT_PROFILES = ("lipinski", "veber", "lead_like", "ro3_core", "ro3_extended")
_PROFILE_FILES = {"ro3_core": "rule_of_three_core", "ro3_extended": "rule_of_three_extended"}
_RULE_FIELDS = ("violation", "lower_excess", "upper_excess", "lower_normalized",
                "upper_normalized", "magnitude")


def _hash(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=9)
def _catalog_data(catalog_id):
    catalog = get_catalog(catalog_id)
    entries, definitions = [], defaultdict(list)
    for index in range(catalog.GetNumEntries()):
        entry = catalog.GetEntryWithIdx(index)
        digest = hashlib.sha256(entry.Serialize()).hexdigest()
        definitions[digest].append(index)
        entries.append({"entry_index": index, "alert_name": entry.GetDescription(),
                        "alert_description": entry.GetProp("Scope")
                        if "Scope" in entry.GetPropList() else "",
                        "definition_hash": digest})
    payload = {"schema": "s2s-alert-entry-manifest/2", "catalog_id": catalog_id,
               "rdkit_version": rdkit_version(), "entries": entries}
    digest = _hash(payload)
    counts = Counter(entry["alert_name"] for entry in entries)
    labelled = [{**entry, "catalog_id": catalog_id, "manifest_hash": digest,
                 "criterion_id": f"alert:v2:{catalog_id}:{entry['entry_index']}:{digest[:20]}",
                 "legacy_id": f"{catalog_id}:" + hashlib.sha256(entry["alert_name"].encode()).hexdigest()[:20]
                 if counts[entry["alert_name"]] == 1 else None} for entry in entries]
    return {**payload, "entries": labelled, "manifest_hash": digest}, definitions


def catalog_manifest(catalog_id: str) -> dict:
    """Freeze entry identity without equating homonymous entries or loading bytes."""
    return copy.deepcopy(_catalog_data(catalog_id)[0])


def match_entries(mol: Chem.Mol, catalog_id: str) -> list[dict]:
    """Return native entry hits; site multiplicity is explicitly not computed."""
    if mol is None:
        raise ValueError("Native alert matching requires a valid molecule")
    manifest = _catalog_data(catalog_id)[0]
    indices = _matched_indices(Chem.MolToSmiles(mol), catalog_id)
    return [{**manifest["entries"][index], "entry_hit": True, "site_count": None,
             "site_count_status": "not_computed"} for index in indices]


@lru_cache(maxsize=8192)
def _matched_indices(smiles, catalog_id):
    _, definitions = _catalog_data(catalog_id)
    indices = set()
    for entry in get_catalog(catalog_id).GetMatches(_molecule(smiles)):
        digest = hashlib.sha256(entry.Serialize()).hexdigest()
        if digest not in definitions:
            raise ValueError("Native alert definition conflicts with frozen catalog recipe")
        indices.update(definitions[digest])
    return tuple(sorted(indices))


def _scale(bound, explicit):
    if bound is None:
        return None
    if not np.isfinite(bound):
        raise ValueError("Rule bounds must be finite")
    if explicit is None and bound > 0:
        return float(bound)
    try:
        value = float(explicit)
    except (TypeError, ValueError) as exc:
        raise ValueError("A zero/negative bound requires a positive explicit scale") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError("A zero/negative bound requires a positive explicit scale")
    return value


def annotate_rule(values: pd.Series, rule: Rule, *, scale_lower=None, scale_upper=None) -> dict:
    """Keep operator failures separate from numeric, side-specific distances."""
    lower, upper = rule.bounds
    scale_lower, scale_upper = _scale(lower, scale_lower), _scale(upper, scale_upper)
    observed = values.notna()
    numeric = lower is not None or upper is not None
    if numeric:
        try:
            values = pd.to_numeric(values, errors="raise").astype(float)
        except (ValueError, TypeError) as exc:
            raise ValueError("Numeric rule observations must be numeric") from exc
        if np.isinf(values.to_numpy()).any():
            raise ValueError("Infinite rule observations are forbidden")
        observed = np.isfinite(values)
    failed = ~operators.apply(rule.operator, values, rule.threshold)
    violation = failed.astype(float).where(observed, np.nan)
    empty = pd.Series(np.nan, index=values.index)
    # Absent sides are zero only for numeric rules; categories have no distance.
    low = np.maximum(lower - values, 0) if lower is not None else empty if not numeric else empty.fillna(0)
    high = np.maximum(values - upper, 0) if upper is not None else empty if not numeric else empty.fillna(0)
    low = low.where(observed, np.nan)
    high = high.where(observed, np.nan)
    low_norm = low / scale_lower if lower is not None else low
    high_norm = high / scale_upper if upper is not None else high
    magnitude = pd.Series(np.maximum(low_norm, high_norm), index=values.index) if numeric else empty
    return {"violation": violation, "lower_excess": low, "upper_excess": high,
            "lower_normalized": low_norm, "upper_normalized": high_norm, "magnitude": magnitude}


@lru_cache(maxsize=32)
def _load_profile(profile_id):
    if not isinstance(profile_id, str) or profile_id not in (
        *_PROFILE_FILES, *[p.stem for p in BUILTIN_DIR.glob("*.json")]
    ):
        raise ValueError(f"Unknown built-in tolerance profile: {profile_id}")
    profile = load_profile_file(BUILTIN_DIR / f"{_PROFILE_FILES.get(profile_id, profile_id)}.json")
    if profile.id != profile_id:
        raise ValueError("Tolerance profile ID must match its built-in definition")
    return profile


@lru_cache(maxsize=32)
def _annotation_recipe(profiles, catalogs):
    if len(set(profiles)) != len(profiles) or len(set(catalogs)) != len(catalogs):
        raise ValueError("Tolerance profiles and catalogs must be unique")
    if "pains" in catalogs and set(catalogs) & {"pains_a", "pains_b", "pains_c"}:
        raise ValueError("PAINS aggregate and subcatalogs overlap")
    registry = default_registry()
    criteria, numeric, binary, profile_recipes = {}, list(DESCRIPTOR_NAMES), [], []
    for profile_id in profiles:
        profile = _load_profile(profile_id)
        profile_recipes.append(profile.as_dict())
        for rule in profile.rules:
            prefix = f"tolerance_rule__{profile_id}__{rule.id}"
            columns = {f"{name}_column": f"{prefix}__{name}" for name in _RULE_FIELDS}
            lower, upper = rule.bounds
            metadata = {**rule.as_dict(), **columns, "criterion_id": f"rule:{profile_id}:{rule.id}",
                        "kind": "rule", "lower_limit": lower, "upper_limit": upper,
                        "scale_lower": _scale(lower, None), "scale_upper": _scale(upper, None),
                        "unit": registry.get(rule.descriptor).unit}
            criteria[metadata["criterion_id"]] = metadata
            binary.append(columns["violation_column"])
            numeric.extend(columns[f"{name}_column"] for name in _RULE_FIELDS[1:])
    manifests = [catalog_manifest(catalog) for catalog in catalogs]
    for manifest in manifests:
        for entry in manifest["entries"]:
            column = f"tolerance_alert__{entry['criterion_id']}"
            criteria[entry["criterion_id"]] = {
                **entry, "kind": "alert", "violation_column": column, "magnitude_column": column,
                "site_count_status": "not_computed"}
            binary.append(column)
    payload = {"schema": SCHEMA, "rdkit_version": rdkit_version(), "profile_ids": list(profiles),
               "catalog_ids": list(catalogs), "profiles": profile_recipes, "catalogs": manifests,
               "descriptor_columns": list(DESCRIPTOR_NAMES), "descriptors": registry.metadata(DESCRIPTOR_NAMES),
               "numeric_columns": numeric, "binary_columns": binary, "criterion_columns": criteria,
               "unknown_encoding": "NaN; never absent", "numeric_scale": "positive_bound_magnitude",
               "structure_column": "model_smiles"}
    return {**payload, "chemistry_hash": _hash(payload)}


@lru_cache(maxsize=2048)
def _molecule(smiles):
    return Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles else None


@lru_cache(maxsize=2048)
def _descriptors(smiles, columns):
    mol = _molecule(smiles)
    return default_registry().compute(mol, columns) if mol is not None else dict.fromkeys(columns, np.nan)


def _validate_records(records):
    if not isinstance(records, pd.DataFrame) or not {"record_id", "model_smiles"} <= set(records):
        raise ValueError("Tolerance annotation requires record_id and model_smiles dataframe")
    if not records.columns.is_unique:
        raise ValueError("Tolerance record columns must be unique")
    ids = pd.to_numeric(records.record_id, errors="raise").to_numpy(float)
    if not np.isfinite(ids).all() or (ids % 1 != 0).any() or pd.Series(ids).duplicated().any():
        raise ValueError("Tolerance record_id must contain unique finite integers")


def _input_hash(frame):
    columns = ["record_id", "model_smiles", *DESCRIPTOR_NAMES]
    if not set(columns) <= set(frame):
        return None
    chemistry = pd.DataFrame(frame, copy=False)
    values = pd.util.hash_pandas_object(chemistry[columns], index=True).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def _criterion_annotations(source, mols, valid, recipe):
    extra = {"tolerance_structure_valid": valid}
    if "identity" not in source:
        extra["identity"] = [hashlib.sha256(Chem.MolToSmiles(mol).encode()).hexdigest()
                             if mol is not None else None for mol in mols]
    if "murcko_scaffold" not in source:
        extra["murcko_scaffold"] = [murcko_scaffold(mol) if mol is not None else None for mol in mols]
    for profile_id in recipe["profile_ids"]:
        for rule in _load_profile(profile_id).rules:
            column = f"smarts__{rule.id}" if rule.is_substructure else rule.descriptor
            if column not in source:
                raise ValueError(f"Rule observation missing: {column}")
            result = annotate_rule(source[column].where(valid, np.nan), rule)
            metadata = recipe["criterion_columns"][f"rule:{profile_id}:{rule.id}"]
            extra.update({metadata[f"{name}_column"]: values for name, values in result.items()})
    hit_rows = [[] for _ in range(len(source))]
    for catalog in recipe["catalog_ids"]:
        hits = [match_entries(mol, catalog) if mol is not None else [] for mol in mols]
        for row, matches in zip(hit_rows, hits):
            row.extend(matches)
    return _finalize_annotations(extra, hit_rows, valid, recipe, source.index)


def _finalize_annotations(extra, hit_rows, valid, recipe, index):
    hit_ids = [{hit["criterion_id"] for hit in row} for row in hit_rows]
    for key, metadata in recipe["criterion_columns"].items():
        if metadata["kind"] == "alert":
            values = np.array([float(key in hits) if ok else np.nan for hits, ok in zip(hit_ids, valid)])
            extra[metadata["violation_column"]] = pd.arrays.SparseArray(values, fill_value=0.)
    annotation = pd.DataFrame(extra, index=index)
    failures, unknown = [[] for _ in range(len(index))], [[] for _ in range(len(index))]
    for criterion, metadata in recipe["criterion_columns"].items():
        values = annotation[metadata["violation_column"]].to_numpy(float)
        for pos in np.flatnonzero(values == 1):
            failures[pos].append(criterion)
        for pos in np.flatnonzero(np.isnan(values)):
            unknown[pos].append(criterion)
    annotation["tolerance_failed_criteria"] = failures
    annotation["tolerance_unknown_criteria"] = unknown
    annotation["tolerance_alert_ids"] = [sorted(ids) for ids in hit_ids]
    annotation["tolerance_alert_details"] = hit_rows
    return annotation


def _cache_scalar(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (float, int, np.number)) and not isinstance(value, (bool, np.bool_)):
        number = float(value)
        if np.isnan(number):
            return None
        if not np.isfinite(number):
            raise ValueError("Infinite tolerance annotation cache observations are forbidden")
        return number
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value
    raise ValueError("Unsupported tolerance annotation cache observation type")


def _cache_contract(recipe):
    rules = [metadata for metadata in recipe["criterion_columns"].values() if metadata["kind"] == "rule"]
    columns = [metadata[f"{field}_column"] for metadata in rules for field in _RULE_FIELDS]
    expected = {"tolerance_structure_valid", "identity", "murcko_scaffold", "tolerance_alert_details", *columns}
    alerts = {entry["criterion_id"]: {**entry, "entry_hit": True, "site_count": None,
              "site_count_status": "not_computed"} for catalog in recipe["catalogs"] for entry in catalog["entries"]}
    alert_order = {key: position for position, key in enumerate(alerts)}

    def validate(annotation):
        if not isinstance(annotation, dict) or set(annotation) != expected:
            raise ValueError("Annotation cache chemical fields differ from frozen recipe")
        valid = annotation["tolerance_structure_valid"]
        if type(valid) is not bool or any(annotation[name] is not None and not isinstance(annotation[name], str)
                                          for name in ("identity", "murcko_scaffold")):
            raise ValueError("Invalid annotation cache structure metadata")
        for column in columns:
            value = annotation[column]
            if value is not None and (type(value) not in (int, float) or not np.isfinite(value) or value < 0):
                raise ValueError("Invalid annotation cache numeric values")
            if column.endswith("__violation") and value not in (None, 0, 1):
                raise ValueError("Invalid annotation cache violation value")
            if not valid and value is not None:
                raise ValueError("Invalid structure cannot have known annotation values")
        hits = annotation["tolerance_alert_details"]
        if not isinstance(hits, list) or (not valid and hits):
            raise ValueError("Invalid annotation cache native alerts")
        seen, last = set(), -1
        for hit in hits:
            key = hit.get("criterion_id") if isinstance(hit, dict) else None
            if (key not in alerts or key in seen or hit != alerts[key] or alert_order[key] <= last
                    or any(type(hit[name]) is not type(value) for name, value in alerts[key].items())):
                raise ValueError("Annotation cache alert differs from frozen catalog")
            seen.add(key)
            last = alert_order[key]

    return columns, validate, alerts


def _cached_annotations(source, mols, valid, recipe, cache_path):
    columns, validate, alerts = _cache_contract(recipe)
    observations = list(dict.fromkeys([*DESCRIPTOR_NAMES, *[
        f"smarts__{rule.id}" if rule.is_substructure else rule.descriptor
        for profile in recipe["profile_ids"] for rule in _load_profile(profile).rules]]))
    if not set(observations) <= set(source):
        raise ValueError("Tolerance annotation cache rule observations are missing")
    # These wrappers avoid copying large frame attrs during every row/slice operation.
    bare = pd.DataFrame(source, copy=False)
    keys, annotations = [], []
    with ToleranceAnnotationCache(cache_path, recipe) as cache:
        for mol, text, values in zip(mols, source.model_smiles, bare[observations].itertuples(index=False, name=None)):
            chemistry = Chem.MolToSmiles(mol) if mol is not None else {"invalid_input": _cache_scalar(text)}
            keys.append(content_hash({"recipe_sha256": cache.recipe_sha256, "chemistry": chemistry,
                                      "observations": dict(zip(observations, map(_cache_scalar, values)))}))
        for start in range(0, len(keys), BATCH_SIZE):
            group = keys[start:start + BATCH_SIZE]
            found = cache.fetch(group, validate)
            missing = {key: start + offset for offset, key in enumerate(group) if key not in found}
            if missing:
                positions = list(missing.values())
                part = bare.iloc[positions].drop(columns=["identity", "murcko_scaffold"], errors="ignore")
                calculated = _criterion_annotations(part, [mols[pos] for pos in positions],
                                                    valid.iloc[positions], recipe)
                chemical = ["tolerance_structure_valid", "identity", "murcko_scaffold", *columns,
                            "tolerance_alert_details"]
                entries = {key: {name: copy.deepcopy(value) if name == "tolerance_alert_details" else _cache_scalar(value)
                                  for name, value in zip(chemical, row)}
                           for key, row in zip(missing, calculated[chemical].itertuples(index=False, name=None))}
                cache.store(entries, validate)
                found.update(entries)
            annotations.extend(copy.deepcopy(found[key]) for key in group)
    extra = {"tolerance_structure_valid": valid}
    for name in ("identity", "murcko_scaffold"):
        if name not in source:
            extra[name] = [item[name] for item in annotations]
    extra.update({name: np.asarray([item[name] for item in annotations], dtype=float) for name in columns})
    hit_rows = [[copy.deepcopy(alerts[hit["criterion_id"]]) for hit in item["tolerance_alert_details"]]
                for item in annotations]
    return _finalize_annotations(extra, hit_rows,
                                 valid, recipe, source.index)


def annotate_tolerance(records: pd.DataFrame, *, profiles=DEFAULT_PROFILES,
                       catalogs=("pains", "brenk"), schema=None, cache_path=None) -> tuple[pd.DataFrame, dict]:
    """Annotate all structures; optional SQLite reuses only frozen chemical annotations."""
    _validate_records(records)
    if schema is not None:
        if not isinstance(schema, dict) or schema.get("schema") != SCHEMA:
            raise ValueError("Unsupported tolerance annotation recipe")
        profiles, catalogs = schema.get("profile_ids", ()), schema.get("catalog_ids", ())
    recipe = copy.deepcopy(_annotation_recipe(tuple(profiles), tuple(catalogs)))
    if schema is not None and schema != recipe:
        raise ValueError("Tolerance annotation recipe conflicts with current chemistry")
    required = [*recipe["numeric_columns"], *recipe["binary_columns"], "tolerance_structure_valid",
                "tolerance_failed_criteria", "tolerance_unknown_criteria", "tolerance_alert_ids",
                "tolerance_alert_details"]
    if (cache_path is None and set(required) <= set(records) and records.attrs.get("tolerance_annotation_schema") == recipe
            and records.attrs.get("tolerance_annotation_input_hash") == _input_hash(records)):
        return records.copy(deep=True), recipe
    source = records.copy(deep=True)
    mols = [_molecule(text) for text in source.model_smiles]
    valid = pd.Series([mol is not None for mol in mols], index=source.index)
    missing = tuple(name for name in DESCRIPTOR_NAMES if name not in source)
    if missing:
        computed = pd.DataFrame([_descriptors(text, missing) for text in source.model_smiles],
                                index=source.index, columns=list(missing))
        source = pd.concat([source, computed], axis=1)
    annotation = (_criterion_annotations(source, mols, valid, recipe) if cache_path is None
                  else _cached_annotations(source, mols, valid, recipe, cache_path))
    replaced = [name for name in source if name.startswith(("tolerance_rule__", "tolerance_alert__"))
                or name in annotation]
    output = pd.concat([source.drop(columns=replaced), annotation], axis=1)
    output.attrs.update(copy.deepcopy(records.attrs))
    output.attrs["tolerance_annotation_schema"] = copy.deepcopy(recipe)
    output.attrs["tolerance_annotation_input_hash"] = _input_hash(output)
    return output, recipe


@lru_cache(maxsize=1)
def _feature_morgan():
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=False,
        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())


@lru_cache(maxsize=4096)
def _representation_vector(smiles, recipe):
    mol = _molecule(smiles)
    size = 315 if recipe == "C" else 2048
    if mol is None:
        return np.zeros(size, dtype=np.float32), True
    if recipe == "C":
        values = np.asarray(rdReducedGraphs.GetErGFingerprint(
            mol, fuzzIncrement=.3, minPath=1, maxPath=15), dtype=np.float32)
    else:
        vector = _feature_morgan().GetFingerprint(mol) if recipe == "B" else fingerprint(mol, FingerprintConfig())
        values = np.zeros(size, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(vector, values)
    values.setflags(write=False)
    return values, False


def representation_matrix(frame: pd.DataFrame, schema: dict, recipe="A") -> tuple[np.ndarray, dict]:
    """Return raw finite blocks and masks; fit normalization only on training rows."""
    if recipe not in ("A", "B", "C", "D"):
        raise ValueError("Tolerance representation must be A, B, C, or D")
    if not isinstance(schema, dict) or schema.get("schema") != SCHEMA:
        raise ValueError("Unsupported tolerance annotation recipe")
    expected = _annotation_recipe(tuple(schema["profile_ids"]), tuple(schema["catalog_ids"]))
    if schema != expected:
        raise ValueError("Tolerance representation recipe conflicts with current chemistry")
    columns = schema["numeric_columns"] + schema["binary_columns"]
    if not set(columns) <= set(frame) or "tolerance_structure_valid" not in frame:
        raise ValueError("Tolerance representation annotations are missing")
    values = frame[columns].to_numpy(dtype=np.float32, copy=True)
    if np.isinf(values).any():
        raise ValueError("Infinite tolerance features are forbidden")
    values[~frame.tolerance_structure_valid.to_numpy(bool)] = np.nan
    matrix = np.column_stack((np.nan_to_num(values, nan=0), np.isnan(values).astype(np.float32)))
    names = [*columns, *[f"missing__{name}" for name in columns]]
    size = {"A": 0, "B": 2048, "C": 315, "D": 2048}[recipe]
    kind = {"A": "none", "B": "feature_morgan", "C": "erg", "D": "morgan"}[recipe]
    parameters = {"fuzzIncrement": .3, "minPath": 1, "maxPath": 15} if recipe == "C" else (
        {"radius": 2, "size": 2048, "use_chirality": False} if size else {})
    if size:
        extra = np.zeros((len(frame), size), dtype=np.float32)
        missing = np.zeros((len(frame), 1), dtype=np.float32)
        for pos, smiles in enumerate(frame.model_smiles):
            extra[pos], is_missing = _representation_vector(smiles, recipe)
            missing[pos] = int(is_missing)
        matrix = np.column_stack((matrix, extra, missing))
        names.extend([*[f"{kind}__{i}" for i in range(size)], f"missing__{kind}"])
    spec = {"schema": "s2s-tolerance-representation/1", "recipe": recipe,
            "chemistry_hash": schema["chemistry_hash"], "feature_names": names,
            "numeric_columns": schema["numeric_columns"], "binary_columns": schema["binary_columns"],
            "representation_kind": kind, "representation_size": size,
            "representation_parameters": parameters, "representation_offset": len(columns) * 2,
            "representation_distance": "euclidean_normalized" if recipe == "C" else "tanimoto" if size else "none",
            "normalization": "unfitted; fit numeric/ErG transforms on training only",
            "missing_policy": "zero encoding with explicit observation masks"}
    return matrix.astype(np.float32, copy=False), spec
