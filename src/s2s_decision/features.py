"""Thin chemistry adapter: all calculations come from SMILES2Select."""

import hashlib
import json
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs

from smiles2select.app_metadata import APP_VERSION, rdkit_version
from smiles2select.chemistry.descriptor_registry import default_registry
from smiles2select.chemistry.fingerprints import FingerprintConfig, fingerprint
from smiles2select.chemistry.scaffolds import murcko_scaffold
from smiles2select.chemistry.standardization import StandardizationConfig
from smiles2select.pipeline.workers import process_record
from smiles2select.profiles.loader import BUILTIN_DIR, load_profile_file
from smiles2select.rules.evaluator import evaluate_profiles

from .artifacts import file_hash
from .schema import DESCRIPTOR_NAMES, PROPERTY_NAMES, SCHEMA_VERSION, FeatureSet

STANDARDIZATION = StandardizationConfig(remove_stereo=True)
CHEMISTRY_CONTRACT_VERSION = 1
CHEMISTRY_IMPLEMENTATION = "s2s-decision-features/1"


@lru_cache(maxsize=1)
def _profiles():
    # Explicit builtins exclude user custom profiles that happen to share these IDs.
    return tuple(load_profile_file(BUILTIN_DIR / f"{name}.json") for name in ("lipinski", "veber"))


def chemistry_manifest(bits: int) -> dict:
    if bits not in (1024, 2048):
        raise ValueError("fingerprint bits must be 1024 or 2048")
    manifest = {
        "feature_schema": SCHEMA_VERSION,
        "smiles2select_version": APP_VERSION,
        "rdkit_version": rdkit_version(),
        "standardization": asdict(STANDARDIZATION),
        "fingerprint": asdict(FingerprintConfig(size=bits)),
        "fingerprint_bits": bits,
        "property_names": list(PROPERTY_NAMES),
        "catalogs": ["pains", "brenk"],
        "descriptors": default_registry().metadata(DESCRIPTOR_NAMES),
        "profiles": [profile.as_dict() for profile in _profiles()],
    }
    from rdkit.Chem import RDConfig

    root = Path(RDConfig.RDContribDir)
    manifest["contrib_resources"] = {
        name: file_hash(root / name) if (root / name).is_file() else "missing"
        for name in ("SA_Score/fpscores.pkl.gz", "NP_Score/publicnp.model.gz")
    }
    legacy_hash = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
    semantic = {key: value for key, value in manifest.items() if key != "smiles2select_version"}
    semantic.update(
        chemistry_contract_version=CHEMISTRY_CONTRACT_VERSION,
        chemistry_implementation=CHEMISTRY_IMPLEMENTATION,
    )
    manifest["chemistry_contract_version"] = CHEMISTRY_CONTRACT_VERSION
    manifest["chemistry_implementation"] = CHEMISTRY_IMPLEMENTATION
    manifest["legacy_chemistry_hash"] = legacy_hash
    manifest["chemistry_hash"] = hashlib.sha256(
        json.dumps(semantic, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return manifest


_LEGACY_KEYS = (
    "feature_schema", "smiles2select_version", "rdkit_version", "standardization",
    "fingerprint", "fingerprint_bits", "property_names", "catalogs", "descriptors",
    "profiles", "contrib_resources",
)
_SEMANTIC_KEYS = tuple(key for key in _LEGACY_KEYS if key != "smiles2select_version") + (
    "chemistry_contract_version", "chemistry_implementation",
)


def _semantic_hash(manifest: dict) -> str:
    if any(key not in manifest for key in _SEMANTIC_KEYS):
        raise ValueError("Semantic chemistry recipe is incomplete")
    recipe = {key: manifest[key] for key in _SEMANTIC_KEYS}
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()


def validate_chemistry_compatibility(task: dict, candidate: dict, references=None) -> None:
    """Match semantic contracts; legacy cross-release requires reference parity."""
    model_hash = task.get("chemistry_hash")
    if not isinstance(model_hash, str) or not model_hash:
        raise ValueError("Model chemistry hash is missing")
    if task.get("chemistry_contract_version") is not None:
        if (
            task.get("chemistry_contract_version") != CHEMISTRY_CONTRACT_VERSION
            or task.get("chemistry_implementation") != CHEMISTRY_IMPLEMENTATION
            or candidate.get("chemistry_contract_version") != CHEMISTRY_CONTRACT_VERSION
            or candidate.get("chemistry_implementation") != CHEMISTRY_IMPLEMENTATION
            or model_hash != _semantic_hash(task)
            or candidate.get("chemistry_hash") != _semantic_hash(candidate)
            or model_hash != candidate.get("chemistry_hash")
        ):
            raise ValueError("Candidate/runtime chemistry conflicts with model")
        return
    if any(key not in task for key in _LEGACY_KEYS):
        raise ValueError("Legacy chemistry recipe is incomplete; compatibility cannot be proven")
    recipe = {key: task[key] for key in _LEGACY_KEYS}
    actual = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    if actual != model_hash:
        raise ValueError("Legacy chemistry recipe checksum mismatch")
    if model_hash == candidate.get("chemistry_hash"):
        if candidate.get("chemistry_contract_version") is not None or any(
            key not in candidate for key in _LEGACY_KEYS
        ):
            raise ValueError("Legacy candidate chemistry recipe is incomplete")
        candidate_recipe = {key: candidate[key] for key in _LEGACY_KEYS}
        candidate_hash = hashlib.sha256(
            json.dumps(candidate_recipe, sort_keys=True).encode()
        ).hexdigest()
        if candidate_hash != model_hash:
            raise ValueError("Legacy candidate chemistry recipe checksum mismatch")
        return  # Frozen legacy features with the same verified full-version recipe.
    current = chemistry_manifest(task["fingerprint_bits"])
    expected = {key: current[key] for key in _LEGACY_KEYS}
    expected["smiles2select_version"] = task["smiles2select_version"]
    if recipe != expected:
        raise ValueError("Candidate/runtime chemistry conflicts with legacy model recipe")
    if references is None:
        raise ValueError("Legacy chemistry requires training-reference feature parity")
    _verify_legacy_reference_parity(references, task["fingerprint_bits"])


def _verify_legacy_reference_parity(references: pd.DataFrame, bits: int) -> None:
    required = ("original_smiles", "model_smiles", "identity", "fingerprint_hex",
                "murcko_scaffold", *PROPERTY_NAMES)
    if any(name not in references for name in required) or references.empty:
        raise ValueError("Legacy references lack fields needed for feature parity")
    positions = sorted({0, len(references) // 2, len(references) - 1})
    sample = references.iloc[positions].copy()
    if "record_id" not in sample:
        raise ValueError("Legacy references lack stable record IDs")
    current = featurize(sample[["record_id", "original_smiles"]], bits).records
    for name in required[1:]:
        old_values = sample[name].reset_index(drop=True)
        new_values = current[name].reset_index(drop=True)
        if name in PROPERTY_NAMES:
            left = pd.to_numeric(old_values, errors="raise").to_numpy(dtype=float)
            right = pd.to_numeric(new_values, errors="raise").to_numpy(dtype=float)
            matches = np.isclose(left, right, rtol=1e-6, atol=1e-7, equal_nan=True)
        else:
            matches = old_values.fillna("").astype(str).eq(new_values.fillna("").astype(str))
        if not bool(np.asarray(matches).all()):
            raise ValueError(f"Legacy reference feature parity failed: {name}")


def _molecule_features(record_id: int, text: str, bits: int) -> dict:
    result = process_record(record_id, text, DESCRIPTOR_NAMES, STANDARDIZATION, ("pains", "brenk"))
    base = {
        "valid": result.valid,
        "invalid_reason": result.invalid_reason,
        "model_smiles": result.standardized_smiles,
        "fingerprint_hex": "",
        "identity": None,
        "murcko_scaffold": None,
        **dict.fromkeys(PROPERTY_NAMES, np.nan),
    }
    if not result.valid:
        return base
    mol = Chem.MolFromSmiles(result.standardized_smiles)
    if mol is None:
        raise ValueError("SMILES2Select returned an invalid standardized structure")
    scaffold = murcko_scaffold(mol)
    if not scaffold and mol.GetRingInfo().NumRings():
        raise ValueError("scaffold calculation failed for a cyclic molecule")
    evaluation = evaluate_profiles(
        pd.DataFrame([result.descriptors], index=[record_id]), _profiles()
    )
    return {
        **base,
        **result.descriptors,
        "murcko_scaffold": scaffold,
        "identity": hashlib.sha256(result.standardized_smiles.encode("utf-8")).hexdigest(),
        "fingerprint_hex": DataStructs.BitVectToBinaryText(
            fingerprint(mol, FingerprintConfig(size=bits))
        ).hex(),
        **{
            f"{name}_violations": int(evaluation.violation_count(name).iloc[0])
            for name in ("lipinski", "veber")
        },
        **{
            f"{name}_count": len(
                {row["alert_name"] for row in result.alert_rows if row["catalog_id"] == name}
            )
            for name in ("pains", "brenk")
        },
    }


def featurize(frame: pd.DataFrame, fingerprint_bits: int = 2048) -> FeatureSet:
    """Preserve upstream evidence; compute one coherent model view with shared chemistry.

    Existing unverified descriptors are kept as source__ columns. Reusing them as
    model inputs across differing structure/toolkit settings would invalidate parity.
    """
    chemistry = chemistry_manifest(fingerprint_bits)
    if "original_smiles" not in frame:
        raise ValueError("original_smiles is required")
    source = frame.copy(deep=True)
    if "record_id" not in source:
        source["record_id"] = np.arange(1, len(source) + 1)
    ids = pd.to_numeric(source.record_id, errors="raise")
    if (
        ids.isna().any()
        or not np.isfinite(ids).all()
        or (ids % 1 != 0).any()
        or ids.duplicated().any()
    ):
        raise ValueError("record_id must contain unique finite integers")
    source["record_id"] = ids.astype("int64")
    if "molecule_id" not in source:
        source["molecule_id"] = source.record_id.map(lambda n: f"REC{n:07d}")
    evidence = {
        f"source__{name}": source[name]
        for name in source
        if not name.startswith("source__") and f"source__{name}" not in source
    }
    source = pd.concat([source, pd.DataFrame(evidence, index=source.index)], axis=1)
    computed = []
    # ponytail: bounded per-chunk cache; persistent cache can follow measured repeated-run costs.
    cache = {}
    for row in source.itertuples():
        text = row.original_smiles if isinstance(row.original_smiles, str) else ""
        if text not in cache:
            if len(cache) >= 2000:
                cache.clear()
            cache[text] = _molecule_features(int(row.record_id), text, fingerprint_bits)
        computed.append(cache[text])
    derived = pd.DataFrame(computed, index=source.index)
    upstream_eligible = source.get("eligible", pd.Series(True, index=source.index))
    if not upstream_eligible.dropna().isin([True, False, 0, 1]).all():
        raise ValueError("eligible must be boolean")
    if derived.empty:
        raise ValueError("no molecules to featurize")
    records = pd.concat(
        [source.drop(columns=list(derived.columns), errors="ignore"), derived], axis=1
    )
    records["eligible"] = upstream_eligible.fillna(False).astype(bool) & records.valid
    if "valid" in source:
        upstream_valid = source.valid
        if not upstream_valid.dropna().isin([True, False, 0, 1]).all():
            raise ValueError("upstream valid must be boolean")
        records["eligible"] &= upstream_valid.fillna(False).astype(bool)
    if "duplicate_of" in source:
        records["eligible"] &= source.duplicate_of.isna()
    records.attrs.update(frame.attrs)
    manifest = {
        **frame.attrs.get("provenance", {}),
        **chemistry,
        "chemistry_policy": "shared_smiles2select_model_view; upstream evidence preserved",
        "valid_count": int(records.valid.sum()),
        "eligible_count": int(records.eligible.sum()),
        "missing_property_counts": {
            name: int(records.loc[records.valid, name].isna().sum()) for name in PROPERTY_NAMES
        },
    }
    records.attrs["provenance"] = manifest
    return FeatureSet(records, manifest)
