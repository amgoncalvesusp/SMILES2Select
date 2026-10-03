"""Acquire and prepare two external LIT-PCBA assays without viewing model scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import pandas as pd
import requests
from rdkit import Chem

from s2s_decision.artifacts import file_hash
from s2s_decision.features import STANDARDIZATION, chemistry_manifest, featurize
from smiles2select.chemistry.scaffolds import murcko_scaffold
from smiles2select.chemistry.standardization import standardize

ARCHIVE_URL = "https://drugdesign.unistra.fr/downloads/datasets/LIT-PCBA_AVE_unbiased.tar.gz"
ASSAYS = {"ESR1_ant": ("P03372_WT_IC50", 743080), "PPARG": ("P37231_WT_EC50", 743094)}
FILES = {"active_T.smi", "active_V.smi", "inactive_T.smi", "inactive_V.smi"}


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def download(url: str, path: Path) -> dict:
    if not path.exists():
        response = requests.get(url, timeout=180)
        response.raise_for_status()
        path.write_bytes(response.content)
    return {"url": url, "path": str(path), "sha256": file_hash(path), "bytes": path.stat().st_size}


def extract_smiles(archive: Path, output: Path) -> list[Path]:
    paths = []
    seen = set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or "\\" in member.name:
                raise ValueError(f"Unsafe archive member: {member.name}")
            if len(name.parts) < 2 or name.parts[-2] not in ASSAYS or name.name not in FILES:
                continue
            if not member.isfile():
                continue
            if member.size > 50_000_000:
                raise ValueError("Oversized SMILES member")
            path = output / name.parts[-2] / name.name
            if path in seen:
                raise ValueError(f"Duplicate archive source: {member.name}")
            seen.add(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            data = tar.extractfile(member).read()
            if path.exists() and path.read_bytes() != data:
                raise ValueError(f"Existing source differs: {path}")
            path.write_bytes(data)
            paths.append(path)
    return sorted(paths)


def parse_smi(text: str, assay: str, filename: str) -> list[dict]:
    if assay not in ASSAYS or filename not in FILES:
        raise ValueError("Unexpected assay or source filename")
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2 or not fields[1].isdigit():
            raise ValueError(f"Malformed SMILES/SID at line {number}: {filename}")
        rows.append({
            "assay": assay, "task_id": ASSAYS[assay][0], "pubchem_aid": ASSAYS[assay][1],
            "original_smiles": fields[0], "source_sid": fields[1],
            "source_partition": filename[-5], "source_file": filename, "source_line": number,
            "y_active": int(filename.startswith("active_")),
        })
    return rows


def collapse_records(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = frame.loc[frame.eligible].copy()
    counts = valid.groupby(["assay", "identity"]).y_active.transform("nunique")
    conflicts = valid.loc[counts.gt(1)].copy()
    good = valid.loc[counts.eq(1)]
    keys = ["assay", "identity"]
    evidence = good.groupby(keys, sort=True).agg(
        source_sids=("source_sid", lambda s: json.dumps(sorted(set(s)))),
        source_partitions=("source_partition", lambda s: json.dumps(sorted(set(s)))),
        source_smiles=("original_smiles", lambda s: json.dumps(sorted(set(s)))),
        source_record_count=("source_sid", "size"),
    ).reset_index()
    clean = good.sort_values(keys + ["source_sid"]).drop_duplicates(keys).merge(
        evidence, on=keys, validate="one_to_one"
    )
    return clean.reset_index(drop=True), conflicts


def cohort_flags(frame, exposed_identities, exposed_scaffolds, fitted_identities):
    novel = ~frame.identity.isin(exposed_identities)
    return frame.assign(
        overlap_b09_fitted=frame.identity.isin(fitted_identities),
        overlap_any_exposure=~novel,
        empty_scaffold=frame.murcko_scaffold.eq(""),
        primary_identity_novel=novel,
        secondary_scaffold_novel=novel & ~frame.murcko_scaffold.isin(exposed_scaffolds),
    )


def exposure_pool(study: Path) -> tuple[set, set, set, dict]:
    root = study / "artifacts/s2-decision-multitask-v1/data"
    pre_path, fit_path = root / "features-before-identity-collapse.parquet", root / "molecules.parquet"
    pre = pd.read_parquet(pre_path, columns=["identity", "murcko_scaffold"])
    fitted = set(pd.read_parquet(fit_path, columns=["identity"]).identity)
    scaffold_map = dict(pre.dropna(subset=["identity"]).itertuples(index=False, name=None))
    historical_path = study / "artifacts/data-capacity-v1/excluded-previous-identities.json"
    historical = set(json.loads(historical_path.read_text())["identities"])
    sources = {str(p): file_hash(p) for p in [pre_path, fit_path, historical_path]}
    paths = sorted((study / "artifacts/data-capacity-v1/datasets").rglob("records.jsonl"))
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            historical.add(row["identity"])
            scaffold_map[row["identity"]] = row["murcko_scaffold"]
        sources[str(path)] = file_hash(path)
    for path in sorted((study / "artifacts/data-capacity-v1").rglob("*.csv")):
        if "identity" in pd.read_csv(path, nrows=0).columns:
            historical.update(pd.read_csv(path, usecols=["identity"]).identity.dropna())
            sources[str(path)] = file_hash(path)
    # Recover historical scaffold evidence without changing the fixed exposure identity set.
    candidates = sorted((study / "artifacts").rglob("records.jsonl"))
    candidates += sorted((study / "artigo_SMILES2Select/03_dados").rglob("records.jsonl"))
    for path in candidates:
        missing = historical - scaffold_map.keys()
        if not missing:
            break
        changed = False
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("identity") in missing and row.get("murcko_scaffold") is not None:
                scaffold_map[row["identity"]] = row["murcko_scaffold"]
                changed = True
        if changed:
            sources[str(path)] = file_hash(path)
    missing = historical - scaffold_map.keys()
    if missing:
        snapshot = study / "artigo_SMILES2Select/03_dados/benchmark_independente_v1/historical_exposure_snapshot.json"
        source_paths = [Path(s["path"]) for s in json.loads(snapshot.read_text())["sources"]
                        if s["kind"] == "blind_csv"]
        source_paths.append(study / "artigo_SMILES2Select/03_dados/benchmark_independente_v1/prepared_001/source_annotations.csv")
        sources[str(snapshot)] = file_hash(snapshot)
        for path in source_paths:
            source = pd.read_csv(path)
            column = "original_smiles" if "original_smiles" in source else "smiles"
            for text in source[column].dropna().unique():
                mol = Chem.MolFromSmiles(text)
                if mol is None:
                    continue
                result = standardize(mol, STANDARDIZATION)
                if result.valid:
                    identity = hashlib.sha256(result.standardized_smiles.encode()).hexdigest()
                    if identity in missing:
                        scaffold_map[identity] = murcko_scaffold(result.mol)
            sources[str(path)] = file_hash(path)
        missing = historical - scaffold_map.keys()
    if missing:
        raise ValueError(f"Historical scaffolds unresolved for {len(missing)} identities")
    identities = set(pre.identity.dropna()) | historical
    scaffolds = {scaffold_map[i] for i in identities}
    return identities, scaffolds, fitted, {
        "b09_pre_identity_unique": int(pre.identity.nunique()), "b09_fitted_unique": len(fitted),
        "historical_unique": len(historical), "all_exposed_unique": len(identities),
        "exposed_scaffolds": len(scaffolds), "source_files": sources,
    }


def feature_chunk(frame):
    return featurize(frame, 2048).records


def prepare(study: Path, workers: int) -> None:
    output = study / "artifacts/s2-decision-external-v1/data"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "preparation-manifest.json").exists():
        raise ValueError("Completed preparation exists; preserve it")
    raw_root = study / "data/lit-pcba"
    raw_root.mkdir(parents=True, exist_ok=True)
    archive = raw_root / "LIT-PCBA_AVE_unbiased.tar.gz"
    acquisition = [download(ARCHIVE_URL, archive)]
    for assay, (_, aid) in ASSAYS.items():
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/description/JSON"
        acquisition.append(download(url, raw_root / f"{assay}_AID{aid}_description.json"))
    paths = extract_smiles(archive, raw_root / "smiles")
    expected = {(assay, filename) for assay in ASSAYS for filename in FILES}
    if {(p.parent.name, p.name) for p in paths} != expected:
        raise ValueError("Incomplete assay source file set")
    rows = [row for path in paths for row in parse_smi(path.read_text(), path.parent.name, path.name)]
    raw = pd.DataFrame(rows)
    raw.to_parquet(output / "raw-records.parquet", index=False)
    write_json(output / "source-manifest.json", {
        "acquired_utc": datetime.now(UTC).isoformat(), "downloads": acquisition,
        "smiles_files": {str(p): file_hash(p) for p in paths},
        "raw_counts": raw.groupby(["assay", "y_active"]).size().to_string(),
    })
    unique = raw[["original_smiles"]].drop_duplicates().reset_index(drop=True).assign(
        record_id=lambda f: range(1, len(f) + 1)
    )
    cache = output / "features.parquet"
    cache_manifest = output / "features-manifest.json"
    cache_recipe = {
        "source_smiles_sha256": hashlib.sha256(
            json.dumps(sorted(unique.original_smiles)).encode()
        ).hexdigest(),
        "chemistry_hash": chemistry_manifest(2048)["chemistry_hash"],
        "code_sha256": file_hash(Path(__file__)),
    }
    if cache.exists():
        if not cache_manifest.exists():
            raise ValueError("Unverified feature cache: missing manifest")
        manifest = json.loads(cache_manifest.read_text())
        if manifest != {**cache_recipe, "features_sha256": file_hash(cache)}:
            raise ValueError("Feature cache chemistry, source, code, or content mismatch")
        features = pd.read_parquet(cache)
        if set(features.original_smiles) != set(unique.original_smiles):
            raise ValueError("Feature cache source mismatch")
    else:
        chunks = [unique.iloc[i:i + 500].copy() for i in range(0, len(unique), 500)]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            completed = []
            for chunk in pool.map(feature_chunk, chunks):
                completed.append(chunk)
                print(f"features {sum(map(len, completed))}/{len(unique)}", flush=True)
        features = pd.concat(completed, ignore_index=True)
        features.to_parquet(cache, index=False)
        write_json(cache_manifest, {**cache_recipe, "features_sha256": file_hash(cache)})
    joined = raw.merge(features, on="original_smiles", validate="many_to_one")
    clean, conflicts = collapse_records(joined)
    identities, scaffolds, fitted, exposure = exposure_pool(study)
    dataset = cohort_flags(clean, identities, scaffolds, fitted).assign(
        record_id=lambda f: range(1, len(f) + 1)
    )
    dataset.to_parquet(output / "dataset.parquet", index=False)
    conflicts.to_parquet(output / "conflicts.parquet", index=False)
    joined.loc[~joined.eligible].to_parquet(output / "invalid-records.parquet", index=False)
    write_json(output / "chemistry.json", chemistry_manifest(2048))
    counts = {}
    for assay in ASSAYS:
        records = dataset.loc[dataset.assay.eq(assay)]
        counts[assay] = {
            "raw": raw.loc[raw.assay.eq(assay)].y_active.value_counts().to_dict(),
            "deduplicated": records.y_active.value_counts().to_dict(),
            **{flag: records.loc[records[flag]].y_active.value_counts().to_dict() for flag in [
                "primary_identity_novel", "secondary_scaffold_novel", "overlap_b09_fitted",
                "overlap_any_exposure", "empty_scaffold",
            ]},
        }
    write_json(output / "preparation-manifest.json", {
        "schema": "s2-decision-external-preparation/1", "assays": ASSAYS,
        "label_semantics": "Published LIT-PCBA binary activity; not pActivity >= 6",
        "source_partitions": "T and V combined for external evaluation only; no external training",
        "counts": counts, "unique_source_smiles": len(unique), "exposure": exposure,
        "invalid_records": int((~joined.eligible).sum()), "conflict_records": len(conflicts),
        "code_sha256": file_hash(Path(__file__)),
        "dataset_sha256": file_hash(output / "dataset.parquet"),
        "chemistry_sha256": file_hash(output / "chemistry.json"),
        "source_manifest_sha256": file_hash(output / "source-manifest.json"),
    })
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    prepare(args.study.resolve(), args.workers)
