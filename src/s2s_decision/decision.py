"""Inspect model packages, preview a basket, and explicitly adopt its frozen result."""

import json
import shutil
import tempfile
from collections.abc import Sequence
from concurrent.futures import CancelledError
from numbers import Integral
from pathlib import Path

import numpy as np

from .artifacts import file_hash, read_bundle, spreadsheet_safe, write_bundle, write_json
from .inference import validate_model_manifest
from .schema import CONTEXT_NAMES, SCHEMA_VERSION, FeatureSet
from .selection import select_candidates
from .workflows import prepare_candidates, score_candidates


def _json(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    return payload


def _inspect_model(path, candidates=None, target=None, endpoint=None):
    result = {
        "path": str(path.resolve()),
        "name": path.name,
        "estimator": None,
        "input_layout": None,
        "target_id": None,
        "endpoint": None,
        "threshold": None,
        "compatible": False,
        "reason": None,
        "calibration_status": None,
        "train_reference_count": 0,
        "source_sha256": None,
        "quality_policy": None,
        "split_method": None,
    }
    try:
        model = _json(path / "manifest.json")
        task = model.get("provenance", {})
        if not isinstance(task, dict) or not isinstance(model.get("calibrator"), dict):
            raise ValueError("Invalid task or calibrator metadata")
        result.update(
            estimator=model.get("estimator", "tiny"),
            input_layout=model.get("input_layout", "tiny_branches"),
            target_id=task.get("target_id"),
            endpoint=task.get("endpoint"),
            threshold=task.get("threshold"),
            calibration_status=model.get("calibrator", {}).get("status"),
            source_sha256=task.get("full_source_sha256") or task.get("source_sha256"),
            quality_policy=task.get("quality_policy"),
            split_method=task.get("split_method"),
        )
        validate_model_manifest(model)
        if not model.get("runtime_ready"):
            raise ValueError("Model reference package is incomplete")
        if not task.get("target_id") or not task.get("endpoint"):
            raise ValueError("Model target and endpoint are required")
        if not isinstance(task.get("threshold"), (int, float)) or not np.isfinite(
            task["threshold"]
        ):
            raise ValueError("Model threshold must be finite")
        if file_hash(path / "model.onnx") != model.get("onnx_sha256"):
            raise ValueError("Model checksum mismatch")
        refs = _json(path / "references" / "manifest.json")
        if refs.get("feature_schema") != SCHEMA_VERSION or refs.get("bundle_version") != 1:
            raise ValueError("Incompatible reference schema")
        digest = file_hash(path / "references" / "records.jsonl")
        if digest != refs.get("records_sha256") or digest != model.get("reference_records_sha256"):
            raise ValueError("Reference checksum mismatch")
        result["train_reference_count"] = refs.get("row_count", 0)
        if refs.get("chemistry_hash") != task.get("chemistry_hash"):
            raise ValueError("Reference chemistry conflicts with model")
        if candidates is None:
            from .features import chemistry_manifest

            chemistry = chemistry_manifest(model["fingerprint_bits"])
        else:
            chemistry = candidates.manifest
        from .features import validate_chemistry_compatibility

        if (
            task.get("chemistry_contract_version") is None
            and chemistry.get("chemistry_hash") != task.get("chemistry_hash")
        ):
            reference_rows = read_bundle(path / "references").records
        else:
            reference_rows = None
        validate_chemistry_compatibility(task, chemistry, reference_rows)
        if candidates is not None:
            from .session_adapter import reject_model_identity_collisions

            reject_model_identity_collisions(candidates)
        for key, expected in (("target_id", target), ("endpoint", endpoint)):
            if expected and expected != task[key]:
                raise ValueError(f"Requested {key} conflicts with model")
            if candidates is not None and key in candidates.records:
                values = candidates.records[key].replace(r"^\s*$", None, regex=True).dropna()
                if not values.eq(task[key]).all():
                    raise ValueError(f"Candidate {key} conflicts with model")
        result["compatible"] = True
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        result["reason"] = str(exc)
    return result


def list_models(
    model_root: str | Path,
    candidates_dir: str | Path | None = None,
    target: str | None = None,
    endpoint: str | None = None,
) -> list[dict]:
    """Inspect root and immediate packages only; retain invalid packages with reasons."""
    root = Path(model_root)
    if not root.is_dir():
        raise ValueError("model root must be an existing directory")
    candidates = read_bundle(candidates_dir) if candidates_dir is not None else None
    # A package's references/ directory is data, not another model candidate.
    paths = (
        [root]
        if (root / "manifest.json").exists() or (root / "model.onnx").exists()
        else sorted(child for child in root.iterdir() if child.is_dir())
    )
    return [
        _inspect_model(path, candidates, target, endpoint)
        for path in paths
        if (path / "manifest.json").exists() or (path / "model.onnx").exists()
    ]


def _original_selection(features):
    frame = features.records
    if features.manifest.get("decision_status") == "adopted" and "is_final" in frame:
        return frame.loc[frame.is_final, "record_id"].astype(int).tolist()
    for name in ("source__is_final", "source__selected", "source__Final_Status"):
        if name in frame:
            values = frame[name].astype(str).str.strip().str.lower()
            expected = (
                ("selected",) if name.endswith("Final_Status") else ("true", "1", "1.0", "yes")
            )
            return frame.record_id.loc[values.isin(expected)].astype(int).tolist()
    if features.manifest.get("input_scope") in ("final_basket", "docking_subset"):
        return frame.record_id.astype(int).tolist()
    return None


def _chemical_score(features):
    # The shared BALANCED selector falls back to descending QED when no Pareto data exists.
    records = features.records.drop(
        columns=[
            "activity_probability",
            "predicted_pactivity",
            "priority_percentile",
            "reference_similarity_max",
            "reference_count",
            "reference_distance_status",
            *CONTEXT_NAMES,
        ],
        errors="ignore",
    )
    scored = records.assign(
        priority_score=records.qed,
        calibration_status="not_applicable_chemical_ranking",
        score_status="chemical_qed",
    )
    return FeatureSet(
        scored,
        {
            **{
                key: value
                for key, value in features.manifest.items()
                if key
                not in {
                    "model_sha256",
                    "model_manifest_sha256",
                    "reference_records_sha256",
                    "task",
                    "scored_count",
                    "estimator",
                    "input_layout",
                }
            },
            "calibration_status": "not_applicable_chemical_ranking",
            "ranking_mode": "chemical_qed",
            "meaning": "QED chemical desirability; not activity probability",
        },
    )


def _comparison(proposed, original_ids):
    frame = proposed.records
    known = original_ids is not None
    original = frame.record_id.isin(original_ids or [])
    changes = (
        np.select(
            [original & frame.is_final, original & ~frame.is_final, ~original & frame.is_final],
            ["retained", "removed", "added"],
            default="unselected",
        )
        if known
        else "unknown_original"
    )
    columns = [
        name
        for name in frame
        if name
        in (
            "record_id",
            "molecule_id",
            "original_smiles",
            "priority_score",
            "calibration_status",
            "similarity_active_max",
            "similarity_inactive_max",
            "reference_similarity_max",
            "reference_count",
            "reference_distance_status",
        )
        or name.startswith("source__")
    ]
    table = frame[columns].assign(
        original_selected=original if known else None,
        proposed_selected=frame.is_final,
        change=changes,
    )
    proximity = [
        name for name in ("similarity_active_max", "similarity_inactive_max") if name in frame
    ]
    if "reference_similarity_max" not in table:
        table["reference_similarity_max"] = frame[proximity].max(axis=1) if proximity else np.nan
    return table


def _freeze_files(root):
    paths = [Path("comparison.json"), Path("comparison.csv")]
    paths += [
        Path(part) / name
        for part in ("prepared", "scored", "proposed")
        for name in ("manifest.json", "records.jsonl")
    ]
    return {path.as_posix(): file_hash(root / path) for path in paths}


def preview_decision(
    input_path: str | Path,
    output_dir: str | Path,
    model_dir: str | Path | None = None,
    n: int = 50,
    max_per_scaffold: int | None = None,
    min_scaffolds: int | None = None,
    pins: Sequence[int] = (),
    exclude: Sequence[int] = (),
    max_per_cluster: int | None = None,
) -> dict:
    """Write an immutable preview; no adoption/export occurs here."""
    return _preview_decision(
        input_path, output_dir, model_dir, n, max_per_scaffold,
        min_scaffolds, max_per_cluster, pins, exclude,
    )


def preview_session_decision(
    snapshot, output_dir: str | Path, model_dir: str | Path | None = None, *, cancelled=None,
) -> dict:
    """Preview a frozen workspace snapshot without changing the live basket."""
    if model_dir is None:
        raise ValueError("Choose a compatible model before previewing a model decision")
    constraints = snapshot.constraints
    if constraints.target_count is None:
        raise ValueError("Session preview requires a target count")
    return _preview_decision(
        None, output_dir, model_dir, constraints.target_count,
        constraints.max_per_scaffold, constraints.min_scaffolds,
        constraints.max_per_cluster, snapshot.pinned_ids, snapshot.excluded_ids,
        snapshot=snapshot, cancelled=cancelled,
    )


def _preview_decision(
    input_path, output_dir, model_dir, n, max_per_scaffold, min_scaffolds,
    max_per_cluster, pins, exclude, *, snapshot=None, cancelled=None,
):
    def check_cancelled():
        if cancelled is not None and cancelled():
            raise CancelledError("Model preview cancelled")

    destination = Path(output_dir)
    check_cancelled()
    if destination.exists():
        raise ValueError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".s2s-preview-", dir=destination.parent) as temp:
        root = Path(temp) / "preview"
        root.mkdir()
        bits = 2048
        if model_dir is not None:
            metadata = _inspect_model(Path(model_dir))
            if not metadata["compatible"]:
                raise ValueError(metadata["reason"])
            bits = _json(Path(model_dir) / "manifest.json")["fingerprint_bits"]
        if snapshot is not None:
            write_bundle(snapshot.prepared(bits), root / "prepared")
        else:
            source = Path(input_path)
            if source.is_dir():
                write_bundle(read_bundle(source), root / "prepared")
            else:
                prepare_candidates(source, root / "prepared", bits=bits)
        check_cancelled()
        features = read_bundle(root / "prepared")
        model = None
        if model_dir is None:
            write_bundle(_chemical_score(features), root / "scored")
        else:
            from .session_adapter import reject_model_identity_collisions

            reject_model_identity_collisions(features)
            model = _inspect_model(Path(model_dir), features)
            if not model["compatible"]:
                raise ValueError(model["reason"])
            score_candidates(root / "prepared", model_dir, root / "scored")
        check_cancelled()
        scored = read_bundle(root / "scored")
        proposed = select_candidates(
            scored,
            n,
            max_per_scaffold=max_per_scaffold,
            min_scaffolds=min_scaffolds,
            max_per_cluster=max_per_cluster,
            pinned_ids=pins,
            excluded_ids=exclude,
            selection_scaffold_col=(
                "session_scaffold" if snapshot is not None else "murcko_scaffold"
            ),
        )
        write_bundle(proposed, root / "proposed")
        check_cancelled()
        original = _original_selection(features)
        final = proposed.manifest["final_ids"]
        summary = {
            "preview_version": 1,
            "decision_status": "preview_not_adopted",
            "output": str(destination.resolve()),
            "original_ids": original,
            "proposed_ids": final,
            "original_count": len(original) if original is not None else None,
            "proposed_count": len(final),
            "eligible_count": int(features.records.eligible.sum()),
            "total_count": len(features.records),
            "ranking_mode": "experimental_model" if model else "chemical_qed",
            "model": model,
            "calibration_status": scored.manifest.get(
                "calibration_status", "not_applicable_chemical_ranking"
            ),
            "source": features.manifest,
            "constraints": proposed.manifest["constraints"],
            "pinned_ids": proposed.manifest["pinned_ids"],
            "excluded_ids": proposed.manifest["excluded_ids"],
            "warnings": proposed.manifest["warnings"] + features.manifest.get("warnings", []),
            **{
                name: sorted(ids) if original is not None else None
                for name, ids in {
                    "added_ids": set(final) - set(original or []),
                    "removed_ids": set(original or []) - set(final),
                    "retained_ids": set(final) & set(original or []),
                }.items()
            },
        }
        spreadsheet_safe(_comparison(proposed, original)).to_csv(
            root / "comparison.csv", index=False
        )
        write_json(root / "comparison.json", summary)
        write_json(root / "preview.json", {"preview_version": 1, "files": _freeze_files(root)})
        check_cancelled()
        root.rename(destination)
    return summary


def export_decision(preview_dir: str | Path, destination: str | Path) -> dict:
    """Adopt exactly a previously previewed basket after verifying its frozen files."""
    root, output = Path(preview_dir), Path(destination)
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    summary = verify_preview(root)
    proposed = read_bundle(root / "proposed")
    manifest = {
        **proposed.manifest["input_provenance"],
        **proposed.manifest,
        "decision_status": "adopted",
        "preview_sha256": file_hash(root / "preview.json"),
        "preview": summary,
        "selection_records_sha256": proposed.manifest["records_sha256"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".s2s-export-", dir=output.parent) as temp:
        staging = Path(temp) / "adopted"
        adopted = proposed.records.assign(
            pinned=proposed.records.record_id.isin(proposed.manifest["pinned_ids"]),
            eligible=proposed.records.eligible
            & ~proposed.records.record_id.isin(proposed.manifest["excluded_ids"]),
        )
        final = adopted.loc[adopted.is_final].copy()
        manifest = {**manifest, "eligible_count": int(adopted.eligible.sum())}
        write_bundle(FeatureSet(adopted, manifest), staging)
        for name in ("comparison.json", "comparison.csv", "preview.json"):
            shutil.copyfile(root / name, staging / name)
        spreadsheet_safe(final).to_csv(staging / "final.csv", index=False)
        write_json(
            staging / "selection.json",
            {**read_bundle(staging).manifest, "final_csv_sha256": file_hash(staging / "final.csv")},
        )
        staging.rename(output)
    return {
        "output": str(output.resolve()),
        "decision_status": "adopted",
        "final_count": len(final),
        "preview_sha256": manifest["preview_sha256"],
    }


def verify_preview(preview_dir: str | Path) -> dict:
    """Read a sealed proposal for GUI adoption, without modifying any file."""
    root = Path(preview_dir)
    try:
        seal = _json(root / "preview.json")
        if seal.get("preview_version") != 1 or seal.get("files") != _freeze_files(root):
            raise ValueError("Preview changed: frozen files checksum mismatch")
        summary = _json(root / "comparison.json")
        prepared = read_bundle(root / "prepared")
        scored = read_bundle(root / "scored")
        proposed = read_bundle(root / "proposed")
    except (OSError, KeyError) as exc:
        raise ValueError("Preview is incomplete or changed") from exc
    ids = proposed.records.record_id
    if (
        ids.isna().any()
        or ids.duplicated().any()
        or set(ids) != set(prepared.records.record_id)
        or set(ids) != set(scored.records.record_id)
        or proposed.records.is_final.isna().any()
        or not proposed.records.is_final.isin([True, False]).all()
    ):
        raise ValueError("Preview record identities changed")
    actual = set(proposed.records.loc[proposed.records.is_final, "record_id"])
    declared = summary.get("proposed_ids")
    if (
        not isinstance(declared, list)
        or any(isinstance(value, bool) or not isinstance(value, Integral) for value in declared)
        or len(declared) != len(actual)
        or set(declared) != actual
        or proposed.manifest.get("final_ids") != declared
        or proposed.manifest.get("constraints") != summary.get("constraints")
        or proposed.manifest.get("pinned_ids") != summary.get("pinned_ids")
        or proposed.manifest.get("excluded_ids") != summary.get("excluded_ids")
    ):
        raise ValueError("Preview basket or constraints changed")
    return summary
