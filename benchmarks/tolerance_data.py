"""Frozen Papyrus census and global partitions for chemical tolerance.

Only exact, nonconflicting active observations fit positive profiles. Reference
membership is a sampling event and never a biological inactivity label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem

from s2s_decision.artifacts import file_hash
from s2s_decision.data import _chunks, _normalize
from s2s_decision.features import STANDARDIZATION
from smiles2select.app_metadata import rdkit_version
from smiles2select.chemistry.scaffolds import murcko_scaffold
from smiles2select.chemistry.standardization import standardize

SOURCE_HASH = "8004e0d1027a760f205b45264386f792e7d49658da39f77f52e660a6f19760dd"
PROTEIN_HASH = "832e564fb82daea0e4da79abcb44834d10104229382874e79915a1288d80783c"
THRESHOLD = 6.5
ROLES = ("fit", "tune", "policy_cal", "test")
SEED = 20261009


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def seal(path: Path, value) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Frozen preparation configuration changed: {path}")
    else:
        write_json(path, value)


def _ids(*values) -> list[str]:
    return sorted({item.strip() for value in values for item in str(value or "").split(";")
                   if item.strip() and item.strip().lower() not in {"nan", "none"}})


def normalize_record(row: dict, ordinal: int, metadata: dict) -> dict:
    """Use the shared relation parser; preserve aggregates as unpaired sets."""
    value = _normalize(row, ordinal, THRESHOLD)
    target = value["target_id"]
    info = metadata.get(target, {})
    species = str(info.get("species") or row.get("species") or "unknown")
    docs = _ids(row.get("all_doc_ids", ""), row.get("doc_id", ""))
    assays = _ids(row.get("AID", ""), row.get("assay_id", ""))
    parts = _ids(row.get("pchembl_value", ""))
    constituent_values = []
    for part in parts:
        try:
            number = float(part)
        except ValueError:
            continue
        if math.isfinite(number):
            constituent_values.append(number)
    conflict = bool(constituent_values and min(constituent_values) < THRESHOLD
                    <= max(constituent_values))
    endpoint_ok = bool(value["endpoint"] and not value["mixed_endpoint"])
    exact = value["relation"] == "=" and endpoint_ok
    observed = pd.notna(value["y_active"])
    high = value["quality"] == "high"
    status = ("conflicting_constituents" if conflict else
              "unknown" if not observed else
              "positive" if value["y_active"] == 1 else "below_threshold")
    return {
        "record_id": ordinal, "source_row": ordinal,
        "measurement_id": value["measurement_id"], "molecule_id": value["molecule_id"],
        "original_smiles": value["original_smiles"], "target_id": target,
        "species": species, "endpoint": value["endpoint"],
        "task_id": f"{target}|{species}|{value['endpoint']}",
        "target_family": info.get("target_family", "unknown"),
        "classification": info.get("classification", ""),
        "pActivity": value["pactivity"], "measured_pactivity_bound": value["measured_pactivity"],
        "relation": value["relation"], "relation_scale": value["relation_scale"],
        "quality": value["quality"], "activity_status": status,
        "positive_evidence": bool((observed and value["y_active"] == 1)
                                  or any(v >= THRESHOLD for v in constituent_values)),
        "primary_positive": bool(high and exact and observed and value["y_active"] == 1
                                  and not conflict),
        "censored_positive": bool(high and endpoint_ok and not exact
                                   and observed and value["y_active"] == 1 and not conflict),
        "document_id": docs[0] if len(docs) == 1 else "",
        "assay_id": assays[0] if len(assays) == 1 else "",
        "source_document_ids": json.dumps(docs), "source_assay_ids": json.dumps(assays),
        "provenance_linkage": "single_document_single_assay" if len(docs) == len(assays) == 1
                              else "unpaired_aggregate_sets",
        "source": value["source"], "source_years": row.get("all_years", row.get("Year", "")),
        "year": value["year"], "date": value["date"],
        "temporal_status": "aggregate_unreconstructable" if "pchembl_value_Mean" in row
                           else "direct_measurement",
    }


def assign_roles(frame: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Split globally by scaffold; exposed groups cannot enter confirmation."""
    if type(seed) is not int or seed < 0:
        raise ValueError("Nonnegative integer seed required")
    if not {"identity", "murcko_scaffold"} <= set(frame) or frame.identity.map(
            lambda v: not isinstance(v, str) or not v).any():
        raise ValueError("Complete molecular identity required")
    if frame.murcko_scaffold.map(lambda v: not isinstance(v, str)).any():
        raise ValueError("Known scaffold required; empty means acyclic")
    groups = pd.Series([f"scaffold:{s}" if s else f"identity:{i}"
                        for s, i in zip(frame.murcko_scaffold, frame.identity)], index=frame.index)
    exposure = frame.get("historical_exposed", pd.Series(False, index=frame.index)).astype(bool)
    exposed_groups = set(groups.loc[exposure])
    mapping = {}
    for group in sorted(set(groups)):
        fraction = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:16], 16) / 2**64
        # Previously exposed chemistry remains development-only, preserving the
        # fit:tune:policy_cal ratio rather than relabelling it an untouched test.
        cutoffs = (0.60 / 0.85, 0.75 / 0.85, 1.0) if group in exposed_groups else (0.60, 0.75, 0.85)
        mapping[group] = next((role for role, cutoff in zip(ROLES, cutoffs)
                               if fraction < cutoff), "test")
    result = frame.assign(split=groups.map(mapping), split_group=groups,
                          historical_exposed_group=groups.isin(exposed_groups))
    if result.groupby("identity").split.nunique().gt(1).any():
        raise ValueError("An identity spans inconsistent scaffold groups")
    return result


def collapse_positive_pairs(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Consolidate only compatible identity-task observations, keeping provenance."""
    conflicts = frame.groupby(["identity", "task_id"]).activity_status.agg(
        lambda values: "conflicting_constituents" in set(values)
        or {"positive", "below_threshold"} <= set(values))
    bad = set(conflicts.index[conflicts])
    reject = frame.loc[[pair in bad for pair in zip(frame.identity, frame.task_id)]].copy()
    positive = frame.loc[frame.primary_positive & np.array(
        [pair not in bad for pair in zip(frame.identity, frame.task_id)])].copy()
    rows = []
    for _, group in positive.groupby(["identity", "task_id"], sort=True):
        row = group.sort_values("record_id").iloc[0].to_dict()
        row["pActivity"] = float(group.pActivity.median())
        row["source_rows"] = json.dumps(sorted(group.record_id.astype(int).tolist()))
        for name, scalar in (("source_document_ids", "document_id"), ("source_assay_ids", "assay_id")):
            values = sorted({item for text in group[name] for item in json.loads(text)})
            row[name] = json.dumps(values)
            row[scalar] = values[0] if len(values) == 1 else ""
        rows.append(row)
    return pd.DataFrame(rows, columns=[*positive.columns, "source_rows"]), reject


def select_pilot_tasks(frame: pd.DataFrame, limit: int = 10) -> tuple[list[str], dict]:
    """Support-only, family-balanced selection; never rank tasks by outcomes."""
    support = {}
    for task, task_rows in frame.groupby("task_id", sort=True):
        support[task] = {role: {
            "identities": int(part.identity.nunique()),
            "scaffolds": int(part.loc[part.murcko_scaffold.ne(""), "murcko_scaffold"].nunique()),
            "documents": int(part.loc[part.document_id.ne(""), "document_id"].nunique()),
        } for role in ROLES for part in [task_rows.loc[task_rows.split.eq(role)]]}
        support[task]["target_family"] = str(task_rows.target_family.iloc[0])
    candidates = [task for task, counts in support.items()
                  if counts["fit"]["identities"] >= 200 and counts["fit"]["scaffolds"] >= 20
                  and counts["fit"]["documents"] >= 5]
    families = {}
    for task in candidates:
        families.setdefault(support[task]["target_family"], []).append(task)
    for family in families:
        families[family].sort(key=lambda task: hashlib.sha256(f"{SEED}:pilot:{task}".encode()).hexdigest())
    chosen = []
    while len(chosen) < limit and any(families.values()):
        for family in sorted(families):
            if families[family] and len(chosen) < limit:
                chosen.append(families[family].pop(0))
    return chosen, support


def molecule_identity(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"original_smiles": smiles, "valid": False, "error": "invalid SMILES"}
    result = standardize(mol, STANDARDIZATION)
    if not result.valid:
        return {"original_smiles": smiles, "valid": False, "error": result.error}
    return {"original_smiles": smiles, "valid": True, "error": "",
            "model_smiles": result.standardized_smiles,
            "identity": hashlib.sha256(result.standardized_smiles.encode()).hexdigest(),
            "murcko_scaffold": murcko_scaffold(result.mol)}


def load_target_metadata(path: Path) -> dict:
    if file_hash(path) != PROTEIN_HASH:
        raise ValueError("Papyrus target metadata checksum mismatch")
    table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    result = {}
    for row in table.to_dict("records"):
        species = row["Organism"].split(" (")[0].strip() or "unknown"
        classification = row["Classification"]
        result[row["target_id"]] = {
            "species": species, "classification": classification,
            "target_family": classification.split("->")[0].strip() or "unknown"}
    return result


def historical_exposure(study: Path, output: Path) -> tuple[set, set, set, dict]:
    """Use prior inventories and all preserved molecule bundles, irrespective of role."""
    identities, scaffolds, documents, sources = set(), set(), set(), {}
    json_paths = [
        study / "artifacts/s2-decision-dynamic-filter-v1/exposure-audit/exposure-pool.json",
        study / "artifacts/s2-decision-dynamic-filter-v1/exposure-audit/document-exposure.json",
        study / "artigo_SMILES2Select/03_dados/benchmark_independente_v1/historical_exposure_snapshot.json",
    ]
    for path in json_paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        identities.update(data.get("identities", ()))
        scaffolds.update(value for value in data.get("scaffolds", ()) if value)
        for key, values in data.items():
            if "doc" in key.lower() and isinstance(values, list):
                documents.update(value for value in values if isinstance(value, str))
        sources[str(path.relative_to(study))] = file_hash(path)
    roots = [study / "artifacts", study / "artigo_SMILES2Select/03_dados"]
    for base in roots:
        for path in sorted(base.rglob("*.parquet")):
            if output == path or output in path.parents or "feature-chunks" in path.parts:
                continue
            columns = set(pq.read_schema(path).names)
            keep = sorted(columns & {"identity", "murcko_scaffold", "document_id",
                                     "document_chembl_id", "source_document_ids"})
            if "identity" not in keep:
                continue
            frame = pd.read_parquet(path, columns=keep)
            identities.update(frame.identity.dropna().astype(str))
            if "murcko_scaffold" in frame:
                scaffolds.update(frame.murcko_scaffold.dropna().astype(str).loc[
                    frame.murcko_scaffold.fillna("").ne("")])
            for name in ("document_id", "document_chembl_id"):
                if name in frame:
                    documents.update(frame[name].dropna().astype(str).loc[frame[name].fillna("").ne("")])
            if "source_document_ids" in frame:
                for value in frame.source_document_ids.dropna():
                    if isinstance(value, (np.ndarray, list)):
                        documents.update(str(item) for item in value)
                    elif isinstance(value, str):
                        try:
                            docs = json.loads(value)
                        except json.JSONDecodeError:
                            docs = _ids(value)
                        if isinstance(docs, list):
                            documents.update(str(item) for item in docs)
            sources[str(path.relative_to(study))] = file_hash(path)
    for base in roots:
        for path in sorted(base.rglob("records.jsonl")):
            if output in path.parents:
                continue
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if row.get("identity"):
                        identities.add(row["identity"])
                    if row.get("murcko_scaffold"):
                        scaffolds.add(row["murcko_scaffold"])
            sources[str(path.relative_to(study))] = file_hash(path)
    audit = {"identities": len(identities), "scaffolds": len(scaffolds), "documents": len(documents),
             "source_files": sources, "scope": "Historical evidence through B16 when present, including all preserved roles; confirmation excludes exposed identity/scaffold/document sets",
             "document_limit": "Papyrus document IDs and ChEMBL document IDs are different namespaces; unresolved crosswalk does not establish documentary independence"}
    write_json(output / "exposure" / "audit.json", audit)
    write_json(output / "exposure" / "sets.json", {"identities": sorted(identities),
                                                  "scaffolds": sorted(scaffolds), "documents": sorted(documents)})
    return identities, scaffolds, documents, audit


def prepare(study: Path, *, workers: int = 4) -> dict:
    output = study / "artifacts/s2-decision-tolerance-v1"
    source = study / "data/papyrus/05.7++_combined_set_without_stereochemistry.tsv.xz"
    proteins = output / "protocol/sources/05.7_combined_set_protein_targets.tsv.xz"
    output.mkdir(parents=True, exist_ok=True)
    if file_hash(source) != SOURCE_HASH:
        raise ValueError("Papyrus source checksum mismatch")
    protocol = {
        "schema": "s2s-tolerance-protocol/1", "registry_id": "TOL-01", "threshold": THRESHOLD,
        "primary": "High-quality single-endpoint exact aggregate; quarantine constituent threshold conflicts",
        "source_sha256": SOURCE_HASH, "protein_sha256": PROTEIN_HASH,
        "source_release": "Papyrus++ 05.7 corrected 2024, ChEMBL34", "license": "CC-BY-SA-4.0",
        "split_seed": SEED, "split": "global Murcko scaffold hash 60/15/10/15; exposed groups development-only",
        "confirmation": "exclude exposed identity/scaffold/document sets; unresolved documentary crosswalk means source independence is not claimed",
        "temporal": "strict temporal infeasible for unreconstructable Papyrus aggregates",
        "pilot": {"max_tasks": 10, "minimum_fit_identities": 200, "minimum_fit_scaffolds": 20,
                  "minimum_fit_documents": 5, "family_round_robin": True, "outcome_selection": False,
                  "core_fit_cap_per_task": 512, "source_fit_per_origin": 128,
                  "source_other_role_per_origin": 64, "source_row_order": "SHA256(seed:identity)",
                  "source_test": "diagnostic only; not used to select main profile architecture"},
        "support": {"identities": 20, "scaffolds": 5, "documents": 3},
        "retention_goal": 0.95, "reference_rescue_cap": 0.10, "primary_capacity_cap": 0.05,
        "qualification_gain_pp": 5, "qualification_scaffold_loss_margin_pp": 2,
        "capacity_grid": [0.01, 0.05, 0.10, 0.20], "N_grid": [20, 50, 100, 500, 2000],
        "primary_N": 50, "scaffold_caps": [None, 3], "reference_sample_cap": 20000,
        "reference": "Papyrus molecular universe, excluding every known snapshot positive identity; membership only, no inactivity inference",
        "primary_confirmation_comparison": "tune-selected contextual profile versus tune-selected best simple comparator, equal achieved nonzero rescue capacity",
        "finite_policy_states": ["calibrated", "infeasible", "baseline_sufficient"],
        "rdkit_version": rdkit_version(), "standardization_hash": STANDARDIZATION.fingerprint(),
        "script_sha256": file_hash(Path(__file__)),
    }
    seal(output / "protocol" / "protocol.json", protocol)
    complete = output / "data" / "manifest.json"
    if complete.exists():
        manifest = json.loads(complete.read_text(encoding="utf-8"))
        for name, digest in manifest["files"].items():
            if file_hash(output / name) != digest:
                raise ValueError(f"Prepared data checksum mismatch: {name}")
        return manifest
    metadata = load_target_metadata(proteins)
    raw_path = output / "data/raw-records.parquet"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    counts, task_counts, raw = Counter(), {}, []
    started = time.monotonic()
    ordinal = 0
    for chunk in _chunks(source, 10000):
        for source_row in chunk.to_dict("records"):
            ordinal += 1
            row = normalize_record(source_row, ordinal, metadata)
            counts["source_rows"] += 1
            counts[row["activity_status"]] += 1
            counts["primary_positive_rows"] += int(row["primary_positive"])
            counts["censored_positive_rows"] += int(row["censored_positive"])
            counts["unknown_species_rows"] += int(row["species"] == "unknown")
            task_counts.setdefault(row["task_id"], Counter())[row["activity_status"]] += 1
            raw.append(row)
        if ordinal % 100000 == 0:
            print(f"census_rows={ordinal} elapsed_s={time.monotonic()-started:.1f}", flush=True)
    frame = pd.DataFrame(raw)
    del raw
    frame.to_parquet(raw_path, index=False)
    write_json(output / "census/source.json", {"counts": dict(counts), "tasks": {k: dict(v) for k, v in task_counts.items()}})
    print(f"census_complete={len(frame)} positive_rows={int(frame.primary_positive.sum())}", flush=True)
    exposed_ids, exposed_scaffolds, exposed_docs, exposure = historical_exposure(study, output)
    smiles = sorted(set(frame.original_smiles))
    cached_path = study / "artifacts/s2-decision-multitask-v1/data/features-before-identity-collapse.parquet"
    cached = pd.read_parquet(cached_path, columns=["original_smiles", "model_smiles", "identity", "murcko_scaffold", "valid"])
    recipe = json.loads((cached_path.parent / "chemistry.json").read_text(encoding="utf-8"))
    if recipe["rdkit_version"] != rdkit_version() or recipe["standardization"] != {
            "cleanup": True, "remove_salts": True, "neutralize": False,
            "canonical_tautomer": False, "remove_stereo": True}:
        raise ValueError("Cached identity recipe does not match the runtime")
    cached = cached.drop_duplicates("original_smiles").loc[cached.original_smiles.isin(smiles)].copy()
    missing = sorted(set(smiles) - set(cached.original_smiles))
    print(f"identities_cached={len(cached)} new_chemistry={len(missing)}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fresh = pd.DataFrame(pool.map(molecule_identity, missing, chunksize=100))
    molecular = pd.concat([cached, fresh], ignore_index=True).drop_duplicates("original_smiles")
    molecular.to_parquet(output / "data/molecular-identity-map.parquet", index=False)
    frame = frame.merge(molecular, on="original_smiles", validate="many_to_one")
    invalid = frame.loc[~frame.valid.fillna(False)].copy()
    invalid.to_parquet(output / "data/invalid-records.parquet", index=False)
    frame = frame.loc[frame.valid.fillna(False)].copy()
    frame["historical_exposed"] = frame.identity.isin(exposed_ids) | frame.murcko_scaffold.isin(exposed_scaffolds) | frame.source_document_ids.map(
        lambda text: bool(set(json.loads(text)) & exposed_docs))
    # Assign once over all source molecules, including potential background rows.
    frame = assign_roles(frame)
    positive, conflicts = collapse_positive_pairs(frame)
    if positive.empty:
        raise ValueError("No positive records meet the frozen primary contract")
    positive = positive.assign(y_active=1.0, sample_origin="observed_active")
    known_positive = set(frame.loc[frame.positive_evidence, "identity"])
    reference = frame.loc[~frame.identity.isin(known_positive)].drop_duplicates("identity").copy()
    reference["sampling_hash"] = reference.identity.map(lambda x: hashlib.sha256(f"{SEED}:{x}".encode()).hexdigest())
    reference = reference.sort_values("sampling_hash").head(protocol["reference_sample_cap"])
    reference = reference.assign(y_active=np.nan, sample_origin="reference_background", pActivity=np.nan,
                                 activity_status="unlabeled_reference")
    pilot, support = select_pilot_tasks(positive)
    positive.to_parquet(output / "data/positive-records.parquet", index=False)
    reference.to_parquet(output / "data/reference-records.parquet", index=False)
    conflicts.to_parquet(output / "data/conflicting-pairs.parquet", index=False)
    split_map = frame.drop_duplicates("identity")[["identity", "murcko_scaffold", "split", "split_group", "historical_exposed", "historical_exposed_group"]]
    split_map.to_parquet(output / "data/split-manifest.parquet", index=False)
    write_json(output / "data/tasks.json", {"pilot_task_ids": pilot, "support": support})
    summary = {
        "source_rows": len(frame) + len(invalid), "valid_source_rows": len(frame), "invalid_source_rows": len(invalid),
        "source_unique_smiles": len(smiles), "valid_unique_identities": int(frame.identity.nunique()),
        "positive_identity_task_pairs": len(positive), "positive_unique_identities": int(positive.identity.nunique()),
        "positive_tasks": int(positive.task_id.nunique()), "positive_scaffolds": int(positive.loc[positive.murcko_scaffold.ne(""), "murcko_scaffold"].nunique()),
        "conflicting_identity_task_rows": len(conflicts), "reference_unique_identities": len(reference),
        "positive_role_pairs": positive.split.value_counts().astype(int).to_dict(),
        "reference_role_counts": reference.split.value_counts().astype(int).to_dict(),
        "test_exposed_groups": int(positive.loc[positive.split.eq("test"), "historical_exposed_group"].sum()),
        "pilot_task_ids": pilot, "pilot_family_count": len({support[t]["target_family"] for t in pilot}),
        "pilot_support": {t: support[t] for t in pilot}, "historical_exposure": exposure,
        "confirmation_status": "Retrospective scaffold novelty relative to recorded exposure; not independent source or prospective validation",
        "temporal_status": "infeasible_strict_temporal: aggregate measures cannot be reconstructed at each cutoff",
        "document_support": "Distinct IDs in unpaired source sets are provenance diagnostics; automatic support uses unambiguous document_id only",
        "background_label": "NaN biological label; source membership S=0 only in the source-discriminator branch",
    }
    write_json(output / "census/prepared.json", summary)
    names = ["data/positive-records.parquet", "data/reference-records.parquet", "data/conflicting-pairs.parquet", "data/split-manifest.parquet", "data/molecular-identity-map.parquet", "data/raw-records.parquet", "data/invalid-records.parquet", "data/tasks.json", "census/source.json", "census/prepared.json", "protocol/protocol.json", "protocol/sources/05.7_combined_set_protein_targets.tsv.xz", "exposure/audit.json", "exposure/sets.json"]
    manifest = {"schema": "s2s-tolerance-data/1", "created_utc": datetime.now(UTC).isoformat(),
                "summary": summary, "files": {name: file_hash(output / name) for name in names},
                "protocol_sha256": file_hash(output / "protocol/protocol.json")}
    write_json(complete, manifest)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"historical_exposure", "pilot_support"}}, indent=2), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        parser.error("workers must be between 1 and 6")
    prepare(args.study_root.resolve(), workers=args.workers)


if __name__ == "__main__":
    main()
