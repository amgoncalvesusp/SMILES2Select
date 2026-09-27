"""Read actual SMILES2Select artifacts without mistaking selection for activity."""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import file_hash
from .schema import PROPERTY_NAMES, FeatureSet

ALIASES = {
    "ID": "molecule_id",
    "access_code": "molecule_id",
    "Original_SMILES": "original_smiles",
    "Standardized_SMILES": "standardized_smiles",
    "Canonical_SMILES": "canonical_smiles",
    "SMILES": "original_smiles",
    "smiles": "original_smiles",
    "MW": "mol_wt",
    "WLOGP": "rdkit_wlogp",
    "MR": "mol_mr",
    "HBD": "hbd_lipinski",
    "HBA": "hba_lipinski",
    "hbd": "hbd_lipinski",
    "hba": "hba_lipinski",
    "TPSA": "tpsa",
    "RotB": "rotatable_bonds",
    "RingCount": "ring_count",
    "QED": "qed",
    "SA": "sa_score",
    "NP": "np_score",
    "PAINS_Count": "pains_count",
    "Brenk_Count": "brenk_count",
    "Source_File": "source_file",
    "Source_Sheet": "source_sheet",
    "Source_Row": "source_row",
    "Pinned": "pinned",
    "Cluster_ID": "cluster_id",
}


def _sqlite(path: Path) -> tuple[pd.DataFrame, dict, str]:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        required = {
            "molecule_descriptors",
            "profile_results",
            "structural_alerts",
            "final_decisions",
            "run_config",
        }
        tables = {
            r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if not required.issubset(tables):
            raise ValueError("not a complete SMILES2Select run database")
        config = {
            k: json.loads(v) for k, v in connection.execute("SELECT key,value FROM run_config")
        }
        frame = pd.read_sql_query(
            "SELECT * FROM molecule_descriptors ORDER BY record_id", connection
        )
        decisions = pd.read_sql_query("SELECT * FROM final_decisions", connection)
        frame = frame.merge(decisions, on="record_id", how="left", validate="one_to_one")
        profiles = pd.read_sql_query("SELECT * FROM profile_results", connection)
        alerts = pd.read_sql_query("SELECT * FROM structural_alerts", connection)
        for profile in ("lipinski", "veber"):
            counts = profiles.loc[profiles.profile_id.eq(profile)].set_index("record_id")
            frame[f"{profile}_violations"] = frame.record_id.map(counts.violation_count)
        catalogs = {item.strip() for item in str(config.get("alert_catalogs", "")).split(",")}
        evaluated = frame.record_id.isin(decisions.record_id)
        for catalog in ("pains", "brenk"):
            counts = (
                alerts.loc[alerts.catalog_id.eq(catalog)]
                .groupby("record_id")
                .occurrence_count.sum()
            )
            values = frame.record_id.map(counts).fillna(0) if catalog in catalogs else np.nan
            frame[f"{catalog}_count"] = pd.Series(values, index=frame.index).where(evaluated)
        return frame, config, "screening_pool"
    finally:
        connection.close()


def _workbook(path: Path) -> tuple[pd.DataFrame, dict, str]:
    with pd.ExcelFile(path) as book:
        config = {}
        if "CONFIG" in book.sheet_names:
            raw = pd.read_excel(book, "CONFIG", keep_default_na=False)
            if {"section", "key", "value"}.issubset(raw.columns):
                sections = {
                    str(section): dict(zip(group.key, group.value, strict=True))
                    for section, group in raw.groupby("section")
                }
                config = {**sections.get("run", {}), "sections": sections}
            elif len(raw.columns) >= 2:
                config = dict(zip(raw.iloc[:, 0].astype(str), raw.iloc[:, 1], strict=True))
        if "FINAL_SELECTED" in book.sheet_names:
            return (
                pd.read_excel(book, "FINAL_SELECTED", dtype=object, keep_default_na=False),
                config,
                "final_basket",
            )
        if {"01_SELECTED_FINAL", "02_EXCLUDED_FINAL"}.issubset(book.sheet_names):
            frames = [
                pd.read_excel(book, name, dtype=object, keep_default_na=False)
                for name in ("01_SELECTED_FINAL", "02_EXCLUDED_FINAL")
            ]
            # RESERVE also appears in EXCLUDED_FINAL: do not read it twice.
            return pd.concat(frames, ignore_index=True), config, "screening_pool"
        if len(book.sheet_names) == 1:
            return (
                pd.read_excel(book, book.sheet_names[0], dtype=object, keep_default_na=False),
                config,
                "provided_pool",
            )
        raise ValueError("ambiguous workbook: export the SMILES2Select run or final basket")


def _read(path: Path) -> tuple[pd.DataFrame, dict, str]:
    suffix = path.suffix.lower()
    if suffix in (".sqlite", ".sqlite3", ".db"):
        return _sqlite(path)
    if suffix == ".xlsx":
        return _workbook(path)
    if suffix == ".parquet":
        return pd.read_parquet(path), {}, "provided_pool"
    if suffix in (".csv", ".tsv", ".txt"):
        return (
            pd.read_csv(
                path, sep="\t" if suffix == ".tsv" else ",", dtype=object, keep_default_na=False
            ),
            {},
            "provided_pool",
        )
    raise ValueError("supported inputs: SMILES2Select SQLite, XLSX, Parquet, CSV, TSV")


def _true(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin(("true", "1", "1.0", "yes"))


def _normalize(frame: pd.DataFrame, scope: str) -> tuple[pd.DataFrame, bool]:
    if not frame.columns.is_unique or any(str(c).startswith("source__") for c in frame):
        raise ValueError("duplicate or reserved source__ columns in input")
    original = frame.copy(deep=True)
    columns = {}
    occupied = set(frame.columns)
    for alias, canonical in ALIASES.items():
        if alias in frame and canonical not in occupied:
            columns[alias] = canonical
            occupied.add(canonical)
    result = frame.rename(columns=columns).copy()
    generated_ids = "record_id" not in result
    if generated_ids:
        result["record_id"] = np.arange(1, len(result) + 1)
    ids = pd.to_numeric(result.record_id, errors="raise")
    if (
        ids.isna().any()
        or not np.isfinite(ids).all()
        or (ids % 1 != 0).any()
        or ids.duplicated().any()
    ):
        raise ValueError("record_id must contain unique finite integers")
    result["record_id"] = ids.astype("int64")
    if "molecule_id" not in result:
        result["molecule_id"] = result.record_id.map(lambda n: f"REC{n:07d}")
    if "original_smiles" not in result:
        for name in ("canonical_smiles", "standardized_smiles"):
            if name in result:
                result["original_smiles"] = result[name]
                break
    if "original_smiles" not in result:
        raise ValueError("input does not contain a recognized SMILES column")
    result["valid"] = _true(result.valid) if "valid" in result else result.original_smiles.notna()
    provided_eligible = _true(result.eligible) if "eligible" in result else True
    if "Final_Status" in result:
        result["eligible"] = result.Final_Status.eq("SELECTED")
    elif "selected" in result:
        result["eligible"] = _true(result.selected)
    else:
        result["eligible"] = (
            result.valid
        )  # User-provided pool; no claim of prior chemical filtering.
    result["eligible"] &= result.valid & provided_eligible
    if "duplicate_of" in result:
        result["duplicate_of"] = pd.to_numeric(
            result.duplicate_of.replace("", np.nan), errors="raise"
        )
        result["eligible"] &= result.duplicate_of.isna()
    if "pinned" in result:
        result["pinned"] = _true(result.pinned)
    for name in PROPERTY_NAMES:
        result[name] = (
            pd.to_numeric(result[name].replace("", np.nan), errors="raise")
            if name in result
            else np.nan
        )
    for name in ("source_row", "duplicate_of", "cluster_id"):
        if name in result:
            result[name] = pd.to_numeric(result[name].replace("", np.nan), errors="raise")
    numeric_evidence = set(ALIASES).intersection(original.columns) - {
        "ID",
        "access_code",
        "Original_SMILES",
        "Standardized_SMILES",
        "Canonical_SMILES",
        "SMILES",
        "smiles",
        "Source_File",
        "Source_Sheet",
        "Pinned",
    }
    original = original.assign(
        **{
            name: pd.to_numeric(original[name].replace("", np.nan), errors="raise")
            for name in numeric_evidence
        }
    )
    raw = original.rename(columns=lambda name: f"source__{name}")
    return pd.concat([result, raw], axis=1), generated_ids


def read_smiles2select(path: str | Path) -> FeatureSet:
    source = Path(path).resolve()
    frame, config, scope = _read(source)
    digest = file_hash(source)
    sidecar = source.with_suffix(".docking.json")
    docking = None
    if sidecar.exists():
        docking = json.loads(sidecar.read_text(encoding="utf-8"))
        if docking.get("sha256") != digest:
            raise ValueError("docking artifact checksum does not match sidecar")
        scope = "docking_subset"
    records, generated_ids = _normalize(frame, scope)
    manifest = {
        "input_path": str(source),
        "input_sha256": digest,
        "input_scope": scope,
        "source_config": config,
        "record_ids_generated": generated_ids,
        "identity_namespace": digest,
        "docking_manifest": docking,
        "eligibility_policy": "upstream_selected_only"
        if "selected" in frame or "Final_Status" in frame
        else "provided_pool_unfiltered",
        "missing_properties": [name for name in PROPERTY_NAMES if records[name].isna().all()],
        "warnings": [
            "Selection and rule violations are not experimental activity labels.",
            "Selected-only scope does not recover all chemically eligible upstream leftovers.",
        ],
    }
    records.attrs["provenance"] = manifest
    return FeatureSet(records, manifest)
