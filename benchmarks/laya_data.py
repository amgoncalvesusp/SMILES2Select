"""Frozen, matched numeric/text evidence for the exploratory Laya pilot.

This is a limited pre-docking ablation. It does not create new measurements,
change the B15 partitions, or turn historically observed tests into new holdouts.
"""

import argparse
import hashlib
import io
import json
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
from contextual_policy_fit import fit_logistic, predict_estimator

from s2s_decision.artifacts import file_hash, write_json
from s2s_decision.context import build_context
from s2s_decision.contextual_model import feature_matrix, fit_spec
from s2s_decision.schema import PROPERTY_NAMES, fingerprint_matrix

DEFAULT_TASK = "P22303_HUMAN_IC50"
TASK_TEXT = {DEFAULT_TASK: "Human AChE IC50 <=1000 nM.",
             "P00918_HUMAN_Ki": "Human CA2 Ki <=1000 nM."}
ALIASES = ("MW", "LogP", "MR", "TPSA", "HBD", "HBA", "RB", "heavy_atoms",
           "heteroatoms", "rings", "aromatic_rings", "charge", "Csp3", "QED",
           "SA", "NP", "Lipinski", "Veber", "PAINS", "Brenk")
CONTEXT = ("similarity_active_max", "similarity_inactive_max")
NUMERIC_COLUMNS = (*PROPERTY_NAMES, *[f"missing_{name}" for name in PROPERTY_NAMES],
                   *CONTEXT, "auxiliary_molecular_logit")
EXTRA_ALIASES = ("ActiveSim", "InactiveSim", "LRLogit")
RAW_LOGIT = "raw_auxiliary_molecular_logit"
STATE_SCHEMA = "s2s-laya-matched-state/1"
SOURCE_METADATA = ("molecule_id", "original_smiles", "valid", "eligible", "source_rows", "measurement_ids",
                   "source_document_ids", "source_assay_ids", "source_document_count",
                   "source_assay_count", "assay_context", "year_min", "year_max",
                   "species", "endpoint", "target_id", "evaluation_status")


def _groups(frame):
    return pd.Series([f"scaffold:{scaffold}" if scaffold else f"identity:{identity}"
                      for scaffold, identity in zip(frame.murcko_scaffold, frame.identity)],
                     index=frame.index)


def _validate(frame, task_id, folds, seed):
    required = {"task_id", "record_id", "identity", "murcko_scaffold", "split",
                "y_active", "fingerprint_hex", "model_smiles", *PROPERTY_NAMES}
    if task_id not in TASK_TEXT or type(folds) is not int or folds < 2 or type(seed) is not int or seed < 0:
        raise ValueError("Unsupported task or invalid fold/seed configuration")
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique or not required <= set(frame):
        raise ValueError("Complete unique panel columns required")
    selected = frame.loc[frame.task_id.eq(task_id)].copy()
    if selected.empty or selected.identity.duplicated().any() or any(
            not isinstance(value, str) or not value.strip() for value in selected.identity):
        raise ValueError("Nonempty unique molecular identity required")
    supplied_ids = selected.record_id.tolist()
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
           or not 0 <= value <= np.iinfo(np.int64).max for value in supplied_ids):
        raise ValueError("Unique nonnegative integer record IDs required without coercion")
    ids = np.asarray(supplied_ids, dtype=np.int64)
    if len(set(ids)) != len(ids):
        raise ValueError("Unique nonnegative integer record IDs required")
    if not selected.y_active.isin([0, 1]).all():
        raise ValueError("Observed binary labels required; unknown is not inactive")
    if not set(selected.split) <= {"train", "validation", "calibration", "test", "recent"}:
        raise ValueError("Unknown split; preserve the source partitions")
    if any(not isinstance(value, str) for value in selected.murcko_scaffold):
        raise ValueError("Scaffold must be known; empty string denotes acyclic")
    internal = selected.loc[selected.split.ne("recent") & selected.murcko_scaffold.ne("")]
    if internal.groupby("murcko_scaffold").split.nunique().gt(1).any():
        raise ValueError("Scaffold overlap across development splits")
    try:
        values = selected[list(PROPERTY_NAMES)].to_numpy(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Property values must be numeric") from exc
    if np.isinf(values).any():
        raise ValueError("Properties must be finite or explicitly missing")
    fingerprint_matrix(selected)
    selected = selected.assign(record_id=ids.astype(np.int64)).sort_values("record_id").reset_index(drop=True)
    train = selected.loc[selected.split.eq("train")]
    if set(train.y_active) != {0, 1}:
        raise ValueError("Both training classes required")
    if _groups(train).nunique() < folds:
        raise ValueError("Too few independent training groups for requested folds")
    return selected


def fold_assignments(train, folds=5, seed=42):
    """Assign label-blind scaffold groups; acyclic structures group by identity."""
    groups = _groups(train)
    order = sorted(groups.unique())
    np.random.default_rng(seed).shuffle(order)
    fold_for = {group: index % folds for index, group in enumerate(order)}
    return groups.map(fold_for)


def _reference_hash(frame):
    if "y_active" not in frame or not frame.y_active.isin([0, 1]).all():
        raise ValueError("Measured reference labels required")
    rows = [[row.identity, row.murcko_scaffold, row.fingerprint_hex, int(row.y_active)]
            for row in frame.sort_values("identity").itertuples()]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _fit_predict(references, query):
    if set(references.y_active) != {0, 1}:
        raise ValueError("Both classes required in every fold reference pool; no reseeding")
    if set(references.identity) & set(query.identity):
        raise ValueError("Auxiliary fit/query identity overlap")
    spec = fit_spec(references, list(PROPERTY_NAMES))
    estimator = fit_logistic(feature_matrix(references, spec), references.y_active.to_numpy(), c=1.)
    logits = predict_estimator(feature_matrix(query, spec), estimator)
    context = build_context(query, references)[list(CONTEXT)].to_numpy(float)
    if not np.isfinite(logits).all() or not np.isfinite(context).all():
        raise ValueError("Auxiliary predictions and reference similarities must be finite")
    saved = {"spec": spec, "estimator": estimator,
             "fit_identities": references.identity.tolist(), "query_identities": query.identity.tolist(),
             "fit_groups": sorted(_groups(references).unique()),
             "query_groups": sorted(_groups(query).unique()),
             "reference_sha256": _reference_hash(references),
             "reference_pool": "identical to fit_identities; labels from this pool only"}
    return logits, context, saved


def render_state(values, task_id=DEFAULT_TASK):
    values = np.asarray(values, dtype=float)
    if task_id not in TASK_TEXT or values.shape != (43,) or not np.isfinite(values).all():
        raise ValueError("Finite 43-column state and supported task required")
    if not np.isin(values[20:40], [0, 1]).all():
        raise ValueError("Missingness flags must be binary")
    def number(value):
        return f"{value:.3f}".rstrip("0").rstrip(".") if value else "0"
    properties = ";".join(f"{key}={number(value)}" for key, value in zip(ALIASES, values[:20]))
    missing = "".join(str(int(value)) for value in values[20:40])
    extra = ";".join(f"{key}={number(value)}" for key, value in zip(EXTRA_ALIASES, values[40:]))
    return f"{TASK_TEXT[task_id]}\n{properties}\nMissing(1=missing,property order)={missing}\n{extra}"


def parse_state(text, task_id=DEFAULT_TASK):
    """Verify exact recoverability; reject extra or reordered predictive evidence."""
    try:
        lines = text.splitlines()
        props = [float(item.split("=")[1]) for item in lines[1].split(";")]
        flags = [int(value) for value in lines[2].rsplit("=", 1)[1]]
        extra = [float(item.split("=")[1]) for item in lines[3].split(";")]
        values = np.asarray([*props, *flags, *extra], dtype=float)
        if render_state(values, task_id) != text:
            raise ValueError("State differs from canonical representation")
        return values
    except (IndexError, TypeError, AttributeError) as exc:
        raise ValueError("Malformed state") from exc


def _state_frame(frame, raw_logits, contexts, task_id):
    properties = frame[list(PROPERTY_NAMES)].to_numpy(float)
    numeric = np.column_stack((np.nan_to_num(properties, nan=0.), np.isnan(properties), contexts, raw_logits))
    numeric = np.round(numeric, 3)
    numeric[numeric == 0] = 0.
    keep = ("record_id", "identity", "task_id", "split", "y_active", "murcko_scaffold",
            "model_smiles", "fingerprint_hex", *SOURCE_METADATA)
    metadata = frame[[name for name in keep if name in frame]].reset_index(drop=True).copy()
    result = pd.concat([metadata, pd.DataFrame(numeric, columns=NUMERIC_COLUMNS)], axis=1)
    return result.assign(row_id=result.record_id, **{RAW_LOGIT: raw_logits},
                         state_text=[render_state(row, task_id) for row in numeric])


def validate_prepared(frame, manifest):
    if manifest.get("numeric_columns") != list(NUMERIC_COLUMNS) or manifest.get("state_schema") != STATE_SCHEMA:
        raise ValueError("Prepared state schema mismatch")
    required = {"row_id", "identity", "state_text", *NUMERIC_COLUMNS}
    if (not required <= set(frame) or frame.empty or not frame.columns.is_unique
            or frame.row_id.duplicated().any() or frame.identity.duplicated().any()):
        raise ValueError("Prepared rows require unique identities and record IDs")
    numeric = frame[list(NUMERIC_COLUMNS)].to_numpy(float)
    for index, text in enumerate(frame.state_text):
        if not np.array_equal(parse_state(text, manifest["task_id"]), numeric[index]):
            raise ValueError("Numeric/text evidence mismatch")
    return {"rows": len(frame), "numeric_columns": len(NUMERIC_COLUMNS), "status": "exact parity"}


def inference_states(raw_frame, fulltrain_reference_frame, auxiliary_models, task_id=DEFAULT_TASK):
    """Deployment transform without fitting or using query labels; raw cached properties required."""
    model = auxiliary_models["full_train"]
    references = fulltrain_reference_frame
    if (set(references.identity) != set(model["fit_identities"])
            or _reference_hash(references) != model["reference_sha256"]):
        raise ValueError("Frozen training reference evidence mismatch")
    query = raw_frame.sort_values("record_id").reset_index(drop=True).copy()
    if query.empty or query.identity.duplicated().any() or query.record_id.duplicated().any():
        raise ValueError("Inference requires unique nonempty query records")
    logits = predict_estimator(feature_matrix(query, model["spec"]), model["estimator"])
    contexts = build_context(query, references)[list(CONTEXT)].to_numpy(float)
    return _state_frame(query, logits, contexts, task_id)


def prepare_panel(frame, task_id=DEFAULT_TASK, folds=5, seed=42):
    frame = _validate(frame, task_id, folds, seed)
    train = frame.loc[frame.split.eq("train")]
    assignments = fold_assignments(train, folds, seed)
    raw_logits, contexts = np.full(len(frame), np.nan), np.full((len(frame), 2), np.nan)
    models = {}
    for fold in range(folds):
        query = train.loc[assignments.eq(fold)]
        references = train.loc[assignments.ne(fold)]
        logits, context, saved = _fit_predict(references, query)
        raw_logits[query.index], contexts[query.index] = logits, context
        models[f"fold_{fold}"] = saved
    held = frame.loc[~frame.split.eq("train")]
    if held.empty:
        raise ValueError("Preserved held-out partitions required")
    logits, context, models["full_train"] = _fit_predict(train, held)
    raw_logits[held.index], contexts[held.index] = logits, context
    result = _state_frame(frame, raw_logits, contexts, task_id)
    manifest = {"schema": "s2s-laya-pilot-data/1", "state_schema": STATE_SCHEMA,
        "task_id": task_id, "task_text": TASK_TEXT[task_id], "numeric_columns": list(NUMERIC_COLUMNS),
        "raw_auxiliary_logit_column": RAW_LOGIT, "rounding_decimals": 3,
        "missing_values": "fixed zero sentinel plus explicit indicator; no imputation",
        "text_property_aliases": dict(zip(ALIASES, PROPERTY_NAMES)), "empty_scaffold_grouping": "identity",
        "source_grouping": "not enforced; source/document overlap may remain",
        "row_order": "ascending original integer record_id", "folds": folds, "seed": seed,
        "fold_assignments": dict(zip(train.identity, assignments.astype(int))),
        "auxiliary_model": "Morgan2048 + 20 properties/missingness; fixed C1 LR; fold-local preprocessing",
        "reference_context": "maximum active/inactive Tanimoto; fold-train or full-train references only",
        "calibration_use": "source calibration partition reserved for probability calibration; no conformal",
        "evaluation_status": "historically observed; exploratory reuse, not a fresh holdout",
        "omitted_features": ["individual alerts", "rule magnitudes", "assay text", "raw Morgan bits in meta state",
                             "docking", "trajectory/stage outcomes", "other five reference-context values"],
        "token_budget_status": "not measured; real tokenizer must reject any truncation before use",
        "splits": {split: {"n": len(rows), "positive_n": int(rows.y_active.sum())}
                   for split, rows in frame.groupby("split")}}
    validate_prepared(result, manifest)
    return result, manifest, models


def build_pilot_data(panel_path, output_dir, task_id=DEFAULT_TASK, folds=5, seed=42):
    panel_path, output_dir = Path(panel_path), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace an existing data directory: {output_dir}")
    source = panel_path.read_bytes()
    source_hash = hashlib.sha256(source).hexdigest()
    frame, manifest, models = prepare_panel(pd.read_parquet(io.BytesIO(source)), task_id, folds, seed)
    if file_hash(panel_path) != source_hash:
        raise ValueError("Panel source changed during preparation")
    output_dir.mkdir(parents=True)
    frame.to_parquet(output_dir / "records.parquet", index=False)
    states = [{"row_id": int(row.row_id), "state_text": row.state_text} for row in frame.itertuples()]
    (output_dir / "states.jsonl").write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in states),
                                            encoding="utf-8")
    write_json(output_dir / "auxiliary_models.json", models)
    manifest = {**manifest, "input_path": str(panel_path.resolve()), "input_sha256": source_hash,
        "preparation_source_sha256": file_hash(Path(__file__)),
        "output_sha256": {name: file_hash(output_dir / name)
                          for name in ("records.parquet", "states.jsonl", "auxiliary_models.json")}}
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task", default=DEFAULT_TASK, choices=tuple(TASK_TEXT))
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args(argv)
    manifest = build_pilot_data(args.panel, args.output, args.task, args.folds)
    print(json.dumps({"output": str(args.output), "splits": manifest["splits"]}, indent=2))


if __name__ == "__main__":
    main()
