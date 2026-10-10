"""Local supervised source-origin pilot; no output is a bioactivity probability.

All families receive the same six-decimal numeric information. Text encoders
consume sparse molecular annotation cards, with absent fields explicitly zero.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

EVENT = "active_sample_origin_vs_reference_origin"
ORIGINS = {"observed_active": 1, "active_sample": 1, "reference_background": 0}


def source_arrays(frame):
    if not {"identity", "split", "sample_origin"} <= set(frame):
        raise ValueError("Source pilot requires explicit sample origin, identity and split")
    if not set(frame.sample_origin) <= set(ORIGINS):
        raise ValueError("Unexpected sample origin; biological labels cannot replace origin")
    if frame.identity.duplicated().any():
        raise ValueError("Source pilot requires unique chemical identity across all tasks/roles")
    if not set(frame.split) <= {"fit", "tune", "policy_cal", "test"}:
        raise ValueError("Unexpected source pilot partition")
    y = frame.sample_origin.map(ORIGINS).to_numpy(dtype=np.int64)
    roles = frame.split.to_numpy()
    for split in ("fit", "tune"):
        if len(np.unique(y[roles == split])) != 2:
            raise ValueError(f"Source {split} requires both observed-active and reference origins")
    return y, roles


def common_card_matrix(x, spec):
    """Frozen common CPU context recipe; no outcome- or variance-based feature choice."""
    from s2s_decision.schema import DESCRIPTOR_NAMES

    def allowed(name):
        base = name.removeprefix("missing__")
        return (base in DESCRIPTOR_NAMES or base.startswith("tolerance_alert__")
                or (base.startswith("tolerance_rule__") and base.endswith("__violation")))

    names = spec["feature_names"]
    indices = [index for index, name in enumerate(names) if allowed(name)]
    if np.asarray(x).ndim != 2 or np.asarray(x).shape[1] != len(names) or not indices:
        raise ValueError("Shared context recipe requires aligned annotated feature dictionary")
    return np.asarray(x)[:, indices].copy(), {
        "recipe": "A-common-context-v1", "feature_names": [names[index] for index in indices],
        "source_indices": indices, "source_recipe": spec,
        "omitted": "redundant numeric rule magnitude/value/excess columns; all models omit identically",
        "selection": "fixed descriptor/violation/alert/missing-mask whitelist; no fit or test variance selection"}


def molecular_cards(x, *, compact=False):
    values = np.asarray(x, dtype=float)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Molecular cards require a finite nonempty numeric matrix")
    rounded = np.round(values, 6)
    if compact:
        cards = [json.dumps({"one": np.flatnonzero(row == 1.).tolist(),
            "value": {str(i): format(float(value), ".6f").rstrip("0").rstrip(".")
                      for i, value in enumerate(row) if value != 0. and value != 1.},
            "zero_default": True}, separators=(",", ":"), allow_nan=False) for row in rounded]
        return cards, rounded
    cards = [json.dumps({"features": {str(i): float(value) for i, value in enumerate(row)
               if value != 0.}, "implicit_zero": True}, separators=(",", ":"), allow_nan=False)
             for row in rounded]
    return cards, rounded


def _source_question():
    from benchmarks.laya_backend import BinaryQuestion

    return BinaryQuestion("Choose molecular sample origin. Listed one indices equal 1; value indices "
        "have numeric values; all other fields equal 0. This is not an activity probability.",
        "Observed experimental active sample", "Unlabeled reference library sample")


def verify_card_budgets(cards, laya_manifest, von_snapshot):
    """Check every row before any family is trained; no truncation or dropped molecule."""
    import importlib.util

    from transformers import AutoTokenizer

    from benchmarks.laya_backend import verify_assets

    assets = verify_assets(Path(laya_manifest))
    root, von_root = Path(assets["root"]), Path(von_snapshot)
    module_spec = importlib.util.spec_from_file_location("_source_budget_laya_common", root/"upstream/laya/common.py")
    common = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(common)
    laya_tok = AutoTokenizer.from_pretrained(root/"model/tokenizer", local_files_only=True, trust_remote_code=False)
    question = _source_question()
    lengths = []
    for card in cards:
        ids, _, _, stats = common.build_sequence(laya_tok, card, question.internal(),
            max_len=512, head_max_len=192, return_stats=True, return_truncation_stats=True)
        if stats["state_tokens_dropped"]:
            raise ValueError("Common card recipe still exceeds untruncated Laya budget")
        lengths.append(len(ids))
    von_tok = AutoTokenizer.from_pretrained(von_root, local_files_only=True, trust_remote_code=False)
    digit_split = json.loads((von_root/"marker_calibration.json").read_text("utf-8")).get("digit_split", False)
    von_cards = [re.sub(r"\d+", lambda m: " ".join(m.group(0)), card) for card in cards] if digit_split else cards
    von_lengths = [len(ids) for ids in von_tok(von_cards, truncation=False)["input_ids"]]
    if max(von_lengths) > 512:
        raise ValueError("Common card recipe still exceeds untruncated Von budget")
    return {"rows_checked": len(cards), "laya_max_total_tokens": max(lengths),
            "von_max_total_tokens": max(von_lengths), "von_digit_split": digit_split,
            "truncated_rows": 0, "dropped_rows": 0}


def _checks(x, y, split):
    x, y, split = np.asarray(x, dtype=float), np.asarray(y), np.asarray(split)
    if (x.ndim != 2 or y.shape != (len(x),) or split.shape != y.shape
            or not np.isfinite(x).all() or not np.isin(y, (0, 1)).all()):
        raise ValueError("Aligned finite source features, binary origins and partitions required")
    fit, tune = np.flatnonzero(split == "fit"), np.flatnonzero(split == "tune")
    if any(len(np.unique(y[rows])) != 2 for rows in (fit, tune)):
        raise ValueError("Both origins required in fit and tune")
    prior = float(y[fit].mean())
    return x, y.astype(int), fit, tune, prior


def _source_diagnostics(y, logits):
    from scipy.special import expit
    from sklearn.metrics import average_precision_score, roc_auc_score

    return {"origin_ap": float(average_precision_score(y, expit(logits))),
            "origin_auc": float(roc_auc_score(y, logits)),
            "scope": "sample-origin discrimination, not biological precision or AUC"}


def _result(logits, prior, **metadata):
    return {"scores": np.asarray(logits)-np.log(prior/(1-prior)), "sampling_prior": prior,
            "event": EVENT, "interpretation": "sampling-prior-corrected log density ratio",
            "biological_probability": False, **metadata}


def train_numeric(x, y, split, *, families=("logistic", "boosting", "tiny"), seed=42,
                  output=None):
    """Fit-only transforms; tune chooses fixed two-candidate grids, never test labels."""
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits

    from benchmarks.contextual_policy_fit import fit_boosting, predict_estimator

    raw, y, fit, tune, prior = _checks(x, y, split)
    split = np.asarray(split)
    mean, scale = raw[fit].mean(axis=0), raw[fit].std(axis=0)
    scale[scale < 1e-8] = 1.
    values = ((raw-mean)/scale).astype(np.float32)
    result = {}
    if output is not None:
        Path(output).mkdir(parents=True, exist_ok=True)
        np.savez(Path(output)/"fit-only-transform.npz", mean=mean, scale=scale)
    for family in families:
        started, candidates, events = time.perf_counter(), [], []
        if family == "logistic":
            for c in (.1, 1.):
                with threadpool_limits(limits=4):
                    estimator = LogisticRegression(C=c, max_iter=1000,
                        random_state=seed).fit(values[fit], y[fit])
                logits = estimator.decision_function(values)
                model = {"family": "logistic", "weights": estimator.coef_[0].tolist(),
                         "intercept": float(estimator.intercept_[0]), "C": c}
                candidates.append((logits, model))
                events.append({"C": c, **_source_diagnostics(y[tune], logits[tune])})
        elif family == "boosting":
            for depth in (2, 3):
                _, model = fit_boosting(values[fit], y[fit], depth, seed)
                logits = predict_estimator(values, model)
                candidates.append((logits, model))
                events.append({"depth": depth, **_source_diagnostics(y[tune], logits[tune])})
        elif family == "tiny":
            candidates, events = _tiny_candidates(values, y, fit, tune, seed, output)
        else:
            raise ValueError(f"Unknown numeric source model: {family}")
        chosen = max(range(len(events)), key=lambda i: (events[i]["origin_ap"], -i))
        logits, model = candidates[chosen]
        model_path = None
        if output is not None and family != "tiny":
            model_path = Path(output)/f"{family}.json"
            model_path.write_text(json.dumps(model, allow_nan=False)+"\n", encoding="utf-8")
        result[family] = _result(logits, prior, fit_count=len(fit), tune_count=len(tune),
            elapsed_seconds=time.perf_counter()-started, selected_candidate=chosen,
            grid=events, recipe="shared annotation card information A, fit-only scaling",
            model=model, model_file=str(model_path) if model_path else None,
            source_diagnostics={part: _source_diagnostics(y[split == part], logits[split == part])
                for part in ("tune", "policy_cal")
                if len(np.unique(y[split == part])) == 2})
    return result


def _tiny_candidates(values, y, fit, tune, seed, output):
    import torch
    from safetensors.torch import save_file

    torch.set_num_threads(4)
    candidates, events = [], []
    for epochs in (10, 20):
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            model = torch.nn.Sequential(torch.nn.Linear(values.shape[1], 128), torch.nn.ReLU(),
                torch.nn.Linear(128, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
            vx, truth = torch.from_numpy(values), torch.tensor(y, dtype=torch.float32)
            losses = []
            for epoch in range(epochs):
                order = fit[torch.randperm(len(fit)).numpy()]
                loss_sum = 0.
                for start in range(0, len(order), 32):
                    ids = order[start:start+32]
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        model(vx[ids]).squeeze(1), truth[ids])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    loss_sum += float(loss.detach())*len(ids)
                losses.append(loss_sum/len(fit))
            with torch.no_grad():
                logits = model(vx).squeeze(1).numpy()
            if output is not None:
                save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()},
                          str(Path(output)/f"tiny-epochs{epochs}.safetensors"))
                import onnxruntime as ort

                onnx_path = Path(output)/f"tiny-epochs{epochs}.onnx"
                torch.onnx.export(model.eval(), vx[:2], str(onnx_path), dynamo=False, opset_version=17,
                    input_names=["features"], output_names=["logit"],
                    dynamic_axes={"features": {0: "batch"}, "logit": {0: "batch"}})
                session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
                exported = session.run(None, {"features": values})[0].ravel()
                if not np.allclose(exported, logits, rtol=1e-5, atol=1e-5):
                    raise ValueError("New source Tiny ONNX differs from trained BCE head")
        candidates.append((logits, {"family": "tiny", "architecture": [values.shape[1], 128, 32, 1],
            "activation": "ReLU", "loss": "BCEWithLogitsLoss", "optimizer": "AdamW",
            "epochs": epochs, "seed": seed, "learning_rate": .001, "weight_decay": .01}))
        events.append({"epochs": epochs, "loss_history": losses,
                       "onnx_validated": output is not None,
                       **_source_diagnostics(y[tune], logits[tune])})
    return candidates, events


def train_laya(cards, y, split, manifest, output, *, seed=42):
    import torch

    from benchmarks.laya_backend import LayaBackend

    _, y, fit, tune, prior = _checks(np.zeros((len(cards), 1)), y, split)
    torch.set_num_threads(4)
    started = time.perf_counter()
    backend = LayaBackend.load(Path(manifest))
    question = _source_question()
    cache = backend.encode_cache(cards, question, Path(output)/"cache",
                                 row_ids=[f"source-{i}" for i in range(len(cards))], batch_size=4)
    candidates, events = [], []
    for epochs in (1, 2):
        history = backend.train_head(cache, y, indices=fit, epochs=epochs, seed=seed, reset=True,
                                     learning_rate=1e-4, batch_size=4)
        paired = backend.predict_cached(cache)
        logits = paired[:, 0]-paired[:, 1]
        checkpoint = Path(output)/f"laya-epochs{epochs}.safetensors"
        receipt = backend.save_head(checkpoint)
        candidates.append((logits, receipt))
        events.append({"epochs": epochs, "loss_history": history,
                       **_source_diagnostics(y[tune], logits[tune])})
    chosen = max(range(len(events)), key=lambda i: (events[i]["origin_ap"], -i))
    return _result(candidates[chosen][0], prior, elapsed_seconds=time.perf_counter()-started,
        selected_candidate=chosen, grid=events, checkpoint=candidates[chosen][1],
        adaptation="Pinned Laya frozen encoder, original decision head, supervised origin CE",
        assets=backend.assets, fit_count=len(fit), tune_count=len(tune))


def train_von(cards, y, split, snapshot, manifest, output, *, seed=42):
    """Pinned Von encoder + new GELU molecular-origin head; no shipped score calibration."""
    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    from benchmarks.von_assay import verify_snapshot

    _, y, fit, tune, prior = _checks(np.zeros((len(cards), 1)), y, split)
    root, output = Path(snapshot), Path(output)
    assets = json.loads(Path(manifest).read_text("utf-8"))
    verify_snapshot(root, assets)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    config = AutoConfig.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    config.reference_compile = False
    encoder = AutoModel.from_config(config, trust_remote_code=False, attn_implementation="sdpa")
    decision_weights = torch.load(root/"option_marker.pt", map_location="cpu", weights_only=True)
    encoder.load_state_dict({key.removeprefix("encoder."): value for key, value in decision_weights.items()
                             if key.startswith("encoder.")}, strict=True)
    del decision_weights
    encoder.requires_grad_(False).eval()
    if encoder.config.model_type != "modernbert":
        raise ValueError("Pinned Von encoder must be ModernBERT")
    limit = min(512, encoder.config.max_position_embeddings)
    calibration = json.loads((root/"marker_calibration.json").read_text("utf-8"))
    digit_split = calibration.get("digit_split", False)
    encoder_cards = [re.sub(r"\d+", lambda m: " ".join(m.group(0)), card) for card in cards] if digit_split else cards
    tokens = tokenizer(encoder_cards, truncation=False, padding=True, return_tensors="pt")
    if tokens.input_ids.shape[1] > limit:
        raise ValueError("Shared molecular cards exceed untruncated Von context budget")
    hidden = []
    with torch.no_grad():
        for start in range(0, len(cards), 4):
            ids, mask = tokens.input_ids[start:start+4], tokens.attention_mask[start:start+4]
            states = encoder(input_ids=ids, attention_mask=mask).last_hidden_state
            pooled = (states*mask[:, :, None]).sum(1)/mask.sum(1)[:, None]
            if not torch.isfinite(pooled).all():
                raise ValueError("Von encoder returned nonfinite molecular representations")
            hidden.append(pooled)
    vectors = torch.cat(hidden)
    np.save(output/"frozen-mean-pooled.npy", vectors.numpy(), allow_pickle=False)
    del encoder
    events, candidates = [], []
    for epochs in (10, 20):
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            width = vectors.shape[1]
            head = torch.nn.Sequential(torch.nn.LayerNorm(width), torch.nn.Linear(width, width//2),
                torch.nn.GELU(), torch.nn.LayerNorm(width//2), torch.nn.Linear(width//2, 1))
            optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=.01)
            truth = torch.tensor(y, dtype=torch.float32)
            history = []
            for epoch in range(epochs):
                order = fit[torch.randperm(len(fit)).numpy()]
                total = 0.
                for start in range(0, len(order), 32):
                    ids = order[start:start+32]
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        head(vectors[ids]).squeeze(1), truth[ids])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    total += float(loss.detach())*len(ids)
                history.append(total/len(fit))
            head.eval()
            with torch.no_grad():
                logits = head(vectors).squeeze(1).numpy()
            checkpoint = output/f"von-molecular-epochs{epochs}.safetensors"
            save_file({k: v.detach().contiguous() for k, v in head.state_dict().items()}, str(checkpoint))
        candidates.append((logits, str(checkpoint)))
        events.append({"epochs": epochs, "loss_history": history,
                       **_source_diagnostics(y[tune], logits[tune])})
    chosen = max(range(len(events)), key=lambda i: (events[i]["origin_ap"], -i))
    return _result(candidates[chosen][0], prior, elapsed_seconds=time.perf_counter()-started,
        selected_candidate=chosen, grid=events, checkpoint=candidates[chosen][1], assets=assets,
        fit_count=len(fit), tune_count=len(tune),
        adaptation="Pinned Von ModernBERT frozen encoder, mean pooling, new GELU/BCE molecular-origin head",
        option_marker_encoder_weights_used=True, shipped_option_head_used=False,
        digit_split=digit_split, shipped_score_calibration_used=False)


def resume_qex_model(path, properties):
    """Reuse an actual fit only when the exact training matrix and RDKit match."""
    from rdkit import rdBase

    from s2s_decision.qex import QEXModel

    path = Path(path)
    receipt = json.loads(path.read_text('utf-8'))
    if receipt.get('status') != 'fit':
        raise ValueError('QEX resume requires a successful fitted artifact')
    model = QEXModel.from_dict(receipt['model'])
    values = np.asarray(properties, dtype='<f8')
    digest = hashlib.sha256(values.tobytes()).hexdigest()
    if (model.training_count != len(values) or model.training_sha256 != digest
            or model.rdkit_version != rdBase.rdkitVersion):
        raise ValueError('QEX resume training chemistry or RDKit mismatch')
    return model, {**receipt, 'resume_source': str(path), 'resume_sha256': file_digest(path)}


def file_digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def export_bundled_candidates(data, fit_snapshot, output):
    """Compact experimental profiles with pooled, explicitly declared query contexts."""
    import pandas as pd

    from benchmarks.tolerance_study import _context, _load_data, _write_json
    from s2s_decision.tolerance_profile import calibrate_tolerance
    from s2s_decision.tolerance_schema import read_tolerance_package, validate_profile

    data, fit_snapshot, output = Path(data), Path(fit_snapshot), Path(output)
    positive, reference, _, tasks = _load_data(data)
    prior = json.loads((fit_snapshot.parent/"execution-freeze.json").read_text("utf-8"))
    if prior["data_manifest_sha256"] != file_digest(data/"manifest.json"):
        raise ValueError("Candidate packages require unchanged frozen calibration data")
    profiles = json.loads(fit_snapshot.read_text("utf-8"))["profiles"]
    output.mkdir(parents=True, exist_ok=False)
    positive_cal = positive.loc[positive.split.eq("policy_cal")].copy().reset_index(drop=True)
    background_cal = reference.loc[reference.split.eq("policy_cal")].copy()
    contexts = [_context(positive.loc[positive.task_id.eq(task)]) for task in tasks]
    queries = []
    for context in contexts:
        saved = {f"reference_source_{key}": background_cal[key] for key in context if key in background_cal}
        queries.append(background_cal.assign(**saved, **context,
            context_assignment_origin="declared_inference_query", query_context_id=context["task_id"]))
    contextual_background = pd.concat(queries, ignore_index=True)
    contextual_background = contextual_background.assign(record_id=range(len(contextual_background)))
    contextual_background.to_parquet(output/"declared-background-queries.parquet", index=False)
    receipts = {}
    for name in ("S1", "S2", "S3-A"):
        profile = profiles[name]
        criteria = [key for key in profile["annotation_schema"]["criterion_columns"]
                    if key.startswith(("rule:lipinski:", "rule:veber:"))]
        positives = positive_cal.drop_duplicates("identity") if name == "S1" else positive_cal
        positives = positives.reset_index(drop=True).assign(record_id=lambda x: range(len(x)))
        background = background_cal if name == "S1" else contextual_background
        background = background.reset_index(drop=True).assign(record_id=lambda x: range(len(x)))
        print(f"Calibrating bundled candidate {name}: {len(positives)} positive rows, {len(background)} reference queries", flush=True)
        calibrated = calibrate_tolerance(profile, positives, background, context={}, reconsidered_criteria=criteria)
        validate_profile(calibrated)
        destination = output/f"{name}.json"
        destination.write_text(json.dumps(calibrated, separators=(",", ":"), allow_nan=False)+"\n", encoding="utf-8")
        verified = read_tolerance_package(destination)
        receipts[name] = {"profile_sha256": verified["profile_sha256"], "file_sha256": file_digest(destination),
                          "file_bytes": destination.stat().st_size, "policy": verified["policy"]}
        _write_json(output/"package-receipts.json", receipts)
    _write_json(output/"completion.json", {"status": "experimental candidate profiles; no method promoted",
        "fit_snapshot_sha256": file_digest(fit_snapshot), "data_manifest_sha256": file_digest(data/"manifest.json"),
        "criteria": "Lipinski/Veber only", "declared_query_contexts": contexts,
        "background_interpretation": "Same unlabeled reference identities queried once under each frozen task context. Assigned contexts are inference queries, not biological measurements.",
        "reference_query_sha256": file_digest(output/"declared-background-queries.parquet"), "packages": receipts})
    return output


def _matched_reference(reference, active_fit, count):
    """Frozen numeric property matching against fit-active centroid, no activity label."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

    def properties(frame):
        values = []
        for smiles in frame.model_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("Matching requires valid standardized molecular structures")
            values.append([Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
                           rdMolDescriptors.CalcTPSA(mol), Lipinski.NumRotatableBonds(mol)])
        return np.asarray(values, dtype=float)

    fit, candidates = properties(active_fit), properties(reference)
    center = np.median(fit, axis=0)
    scale = np.percentile(fit, 75, axis=0)-np.percentile(fit, 25, axis=0)
    scale[scale < 1e-8] = 1.
    distance = np.square((candidates-center)/scale).mean(axis=1)
    ranked = reference.assign(_match_distance=distance, _tie=reference.identity.map(
        lambda identity: hashlib.sha256(identity.encode()).hexdigest())).sort_values(
            ["_match_distance", "_tie"], kind="stable")
    return ranked.head(count).drop(columns=["_match_distance", "_tie"])


def _reuse_encoder_result(name, source):
    """Use verified completed supervised weights/cache, without changing tune choices."""
    import torch
    from safetensors.torch import load_file

    source = Path(source)
    receipt = json.loads((source/f"{name}-training.json").read_text("utf-8"))
    torch.set_num_threads(4)
    if name == "laya":
        from benchmarks.laya_backend import LayaBackend, RepresentationCache

        cache_root = source/"laya/cache"
        info = json.loads((cache_root/"cache.json").read_text("utf-8"))
        cache = RepresentationCache(cache_root, tuple(info["row_ids"]), info["signature"])
        backend = LayaBackend.load(Path(receipt["assets"]["root"])/"assets.json")
        checkpoint = source/"laya"/receipt["checkpoint"]["file"]
        backend.load_head(checkpoint)
        paired = backend.predict_cached(cache)
        logits = paired[:, 0]-paired[:, 1]
        cache_hash = file_digest(cache_root/"cache.json")
    elif name == "von":
        checkpoint = Path(receipt["checkpoint"])
        vectors = torch.from_numpy(np.load(source/"von/frozen-mean-pooled.npy", allow_pickle=False))
        width = vectors.shape[1]
        head = torch.nn.Sequential(torch.nn.LayerNorm(width), torch.nn.Linear(width, width//2),
            torch.nn.GELU(), torch.nn.LayerNorm(width//2), torch.nn.Linear(width//2, 1))
        head.load_state_dict(load_file(str(checkpoint)), strict=True)
        head.eval()
        with torch.no_grad():
            logits = head(vectors).squeeze(1).numpy()
        cache_hash = file_digest(source/"von/frozen-mean-pooled.npy")
    else:
        raise ValueError("Only actual pinned Laya/Von supervised caches can be reused")
    return {**receipt, **_result(logits, receipt["sampling_prior"]),
        "reuse": {"source_training_sha256": file_digest(source/f"{name}-training.json"),
                 "checkpoint_sha256": file_digest(checkpoint), "cache_sha256": cache_hash,
                 "scope": "same supervised checkpoint and frozen tune choice; no retraining or encoder rerun"}}


def run_source_pilot(data, output, study_root, *, heavy=True, common=True, matched_background=False,
                     reuse_heavy=None):
    """Preregistered first supported task, deterministic origin-balanced small CPU pilot."""
    import pandas as pd

    from benchmarks.tolerance_study import (
        _constant_policy,
        _context,
        _hash_sample,
        _load_data,
        _profile_arm,
        _simple_arms,
        _simple_choice,
        _write_json,
        compare_capacity,
        evaluate_arms,
        freeze_threshold,
    )
    from s2s_decision.qex import QEXFitError, fit_qex, predict_qex, qed_properties
    from s2s_decision.tolerance_features import annotate_tolerance, representation_matrix
    from s2s_decision.tolerance_profile import fit_tolerance

    data, output, study_root = Path(data), Path(output), Path(study_root)
    protocol_path = data.parent/"protocol/protocol.json"
    protocol = json.loads(protocol_path.read_text("utf-8"))
    positive, reference, _, tasks = _load_data(data)
    task = tasks[0]
    selected = []
    active_fit = _hash_sample(positive.loc[positive.task_id.eq(task) & positive.split.eq("fit")],
                              protocol["pilot"]["source_fit_per_origin"])
    for role in ("fit", "tune", "policy_cal", "test"):
        count = protocol["pilot"]["source_fit_per_origin" if role == "fit" else "source_other_role_per_origin"]
        background = reference.loc[reference.split.eq(role)]
        sampled_background = _matched_reference(background, active_fit, count) if matched_background else _hash_sample(background, count)
        selected.extend([_hash_sample(positive.loc[positive.task_id.eq(task) & positive.split.eq(role)], count),
                         sampled_background])
    frame = pd.concat(selected, ignore_index=True).drop_duplicates("identity").sort_values(
        "identity", kind="stable").assign(record_id=lambda x: range(len(x)))
    y, roles = source_arrays(frame)
    output.mkdir(parents=True, exist_ok=False)
    freeze = {"schema": "s2s-tolerance-origin-pilot/1", "task": task,
        "selection": "first training-support-qualified protocol task, SHA256(seed:identity) rows",
        "event": EVENT, "recipe": "A-common-context-v1" if common else "full A",
        "engineering_amendment": "Common fixed feature whitelist after recorded full-A context failures; no molecule or outcome selection" if common else None,
        "background": "numeric matched sensitivity: MW/logP/TPSA/RotB; fit-active median/IQR only" if matched_background else "broad deterministic reference background",
        "test_support": "first frozen task retained; only 15 fresh test positives, restricted exploratory evaluation",
        "protocol_sha256": file_digest(protocol_path), "source_sha256": file_digest(__file__),
        "counts": frame.groupby(["split", "sample_origin"]).size().to_dict(),
        "numeric_grids": {"logistic_C": [.1, 1.], "boosting_depth": [2, 3], "tiny_epochs": [10, 20]},
        "text_grids": {"laya_epochs": [1, 2], "von_epochs": [10, 20]},
        "seed": 42, "threads": 4, "biological_probability": False,
        "origin_ap_selection": "diagnostic choice within pilot only; no qualification by AP",
        "laya": {"adaptation": "frozen pinned encoder/original trainable decision head CE", "lr": 1e-4},
        "von": {"adaptation": "frozen pinned encoder/new molecular GELU BCE head", "lr": 1e-4}}
    freeze["counts"] = {f"{role}:{origin}": int(count) for (role, origin), count in freeze["counts"].items()}
    _write_json(output/"execution-freeze.json", freeze)
    frame.to_parquet(output/"pilot-records.parquet", index=False)
    annotated, schema = annotate_tolerance(frame)
    matrix, spec = representation_matrix(annotated, schema, recipe="A")
    if common:
        matrix, spec = common_card_matrix(matrix, spec)
    cards, matrix = molecular_cards(matrix, compact=common)
    if reuse_heavy is not None:
        previous = Path(reuse_heavy)
        prior_records = pd.read_parquet(previous/"pilot-records.parquet")
        if (prior_records.identity.tolist() != frame.identity.tolist()
                or prior_records.split.tolist() != frame.split.tolist()
                or not np.array_equal(np.load(previous/"shared-matrix.npy", allow_pickle=False), matrix)):
            raise ValueError("Encoder reuse requires identical ordered molecules, roles and molecular information")
    np.save(output/"shared-matrix.npy", matrix, allow_pickle=False)
    _write_json(output/"feature-dictionary.json", {"schema": schema, "matrix_spec": spec,
        "precision": "six decimal places, identical numeric and text information",
        "rows": frame.identity.tolist(), "cards": cards})
    von_snapshot = Path(json.loads((study_root/"artifacts/s2-decision-von-pilot-v1/run/inputs.json").read_text("utf-8"))["snapshot"])
    if heavy:
        _write_json(output/"token-budget-check.json", verify_card_budgets(cards,
            study_root/"artifacts/s2s-laya-pilot-v1/runtime/assets.json", von_snapshot))
    models = train_numeric(matrix, y, roles, output=output/"numeric")
    failures = {}
    text_runs = {
        "laya": lambda: train_laya(cards, y, roles,
            study_root/"artifacts/s2s-laya-pilot-v1/runtime/assets.json", output/"laya"),
        "von": lambda: train_von(cards, y, roles,
            von_snapshot,
            study_root/"artifacts/s2-decision-von-pilot-v1/sources/expected-model-manifest.json", output/"von")}
    _write_json(output/"numeric-training.json", {name: {k: v for k, v in model.items() if k != "scores"}
                                                for name, model in models.items()})
    for name, execute in text_runs.items():
        if not heavy:
            failures[name] = {"status": "not_run", "reason": "heavy encoders explicitly disabled in this execution"}
            continue
        print(f"Source-origin molecular training: {name}, {len(frame)} real distinct identities", flush=True)
        try:
            models[name] = _reuse_encoder_result(name, reuse_heavy) if reuse_heavy is not None else execute()
            _write_json(output/f"{name}-training.json", {k: v for k, v in models[name].items() if k != "scores"})
        except (ImportError, ValueError, RuntimeError, OSError) as error:
            import traceback

            failures[name] = {"status": "failed", "reason": str(error), "type": type(error).__name__,
                              "traceback": traceback.format_exc()}
            print(f"Source {name} failed: {error}", flush=True)
    _write_json(output/"failures.json", failures)
    context = _context(positive.loc[positive.task_id.eq(task)])
    fit_active = frame.loc[frame.split.eq("fit") & frame.sample_origin.eq("observed_active")]
    support_profile = fit_tolerance(fit_active,
                                    mode="general", context=context)
    comparators = _simple_arms(annotated)
    comparators["S1"] = _profile_arm(annotated, support_profile, context)
    for name, mode in (("S0", "fixed"), ("S2", "hierarchical"), ("S3-A", "contextual")):
        profile = fit_tolerance(fit_active, mode=mode, recipe="A", context=context)
        comparators[name] = _profile_arm(annotated, profile, context)
    try:
        qex = fit_qex(qed_properties(fit_active.model_smiles.tolist()))
        comparators["QEX"] = comparators["accept_all"].assign(
            rank_score=1-predict_qex(qex, qed_properties(annotated.model_smiles.tolist())))
        _write_json(output/"qex-training.json", {"model": qex.to_dict(), "fit_count": len(fit_active)})
    except QEXFitError as error:
        _write_json(output/"qex-training.json", {"status": "unavailable", "reason": str(error)})
    base = comparators["rigid"]
    support = _profile_arm(annotated, support_profile, context).supported.to_numpy()
    arms = {**comparators, **{name: base.assign(rank_score=-model["scores"], supported=support)
            for name, model in models.items()}}
    best_simple = _simple_choice({name: ranking.loc[ranking.split.eq("tune")].reset_index(drop=True)
                                 for name, ranking in comparators.items()})
    policies = {"rigid": _constant_policy(0.), "simple_1": _constant_policy(1.),
                "simple_2": _constant_policy(2.), "accept_all": _constant_policy(1e300)}
    for name, ranking in arms.items():
        if name in policies:
            continue
        positive_cal = ranking.loc[ranking.split.eq("policy_cal") & ranking.y_active.eq(1)]
        unknown_cal = ranking.loc[ranking.split.eq("policy_cal") & ranking.y_active.isna()]
        policies[name] = freeze_threshold(positive_cal, unknown_cal) if len(positive_cal) and len(unknown_cal) else {
            "status": "unavailable", "tau": None, "reason": "missing calibration origins"}
    _write_json(output/"frozen-policies.json", policies)
    _write_json(output/"configuration-freeze.json", {"best_simple": best_simple,
        "selected_candidates": {name: model["selected_candidate"] for name, model in models.items()},
        "test_outcomes_accessed_for_selection": False, "policies": policies})
    filter_rows, final_rows, paired = [], [], []
    for role in ("tune", "policy_cal", "test"):
        universe = base.loc[base.split.eq(role)].reset_index(drop=True)
        role_arms = {name: rank.loc[rank.split.eq(role)].reset_index(drop=True) for name, rank in arms.items()}
        if not len(universe):
            continue
        metrics, final = evaluate_arms(universe, role_arms, policies,
            training_scaffolds=frame.loc[frame.split.eq("fit"), "murcko_scaffold"].unique())
        filter_rows.extend({"task": task, "split": role, **row} for row in metrics)
        final_rows.extend({"task": task, "split": role, **row} for row in final)
        for name in role_arms:
            if name != "logistic":
                paired.append({"task": task, "split": role, "family": name,
                               **compare_capacity(role_arms[name], role_arms["logistic"], .05)})
            if name in models:
                paired.append({"task": task, "split": role, "family": name,
                    "comparator": best_simple, **compare_capacity(role_arms[name], role_arms[best_simple], .05)})
    pd.DataFrame(filter_rows).to_csv(output/"filter-metrics.csv", index=False)
    pd.DataFrame(final_rows).to_json(output/"final-metrics.jsonl", orient="records", lines=True)
    _write_json(output/"paired-capacity.json", paired)
    scores = frame[["record_id", "identity", "split", "sample_origin"]].copy()
    for name, model in models.items():
        scores[f"{name}_log_density_ratio"] = model["scores"]
    scores.to_parquet(output/"scores.parquet", index=False)
    test = roles == "test"
    _write_json(output/"test-origin-diagnostics.json", {name: _source_diagnostics(y[test], model["scores"][test])
        for name, model in models.items() if len(np.unique(y[test])) == 2})
    _write_json(output/"completion.json", {"families_completed": list(models), "failures": failures,
        "status": "small exploratory origin pilot; no model promoted",
        "main_limit": "One task and small chemistry-disjoint test; origin discrimination is not bioactivity validation",
        "files": {p.relative_to(output).as_posix(): file_digest(p) for p in sorted(output.rglob("*"))
                  if p.is_file() and p.name != "completion.json"}})
    return output
