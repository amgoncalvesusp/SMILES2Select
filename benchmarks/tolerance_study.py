"""TOL-01 evaluator: separate frozen policies from label-blind rescue capacity.

Unknown reference compounds never become biological negatives. All new rankings
use the same final QED-priority balanced selector and disclose unfilled budgets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

CAPACITIES = (.01, .05, .10, .20)
BUDGETS = (20, 50, 100, 500, 2000)
QUOTAS = (None, 3)


def _bool(values, size, name):
    raw = np.asarray(values)
    if raw.shape != (size,) or any(not isinstance(x, (bool, np.bool_)) for x in raw):
        raise ValueError(f"{name} requires aligned boolean values")
    return raw.astype(bool)


def _validate_rank(frame):
    required = {"record_id", "identity", "baseline_eligible", "rank_score", "supported"}
    if not required <= set(frame) or frame.record_id.duplicated().any():
        raise ValueError("Ranked candidates require unique record_id and ranking fields")
    baseline = _bool(frame.baseline_eligible, len(frame), "baseline_eligible")
    supported = _bool(frame.supported, len(frame), "supported")
    scores = pd.to_numeric(frame.rank_score, errors="coerce").to_numpy(float)
    if not np.isfinite(scores[supported & ~baseline]).all():
        raise ValueError("Supported rescue scores must be finite")
    return baseline, supported, scores


def _rescue_order(frame):
    baseline, supported, scores = _validate_rank(frame)
    ties = [hashlib.sha256(str(x).encode()).hexdigest() for x in frame.identity]
    ordered = pd.DataFrame({"row": np.arange(len(frame)), "score": scores, "tie": ties})
    return ordered.loc[supported & ~baseline].sort_values(["score", "tie", "row"]).row.to_numpy()


def capacity_eligibility(frame, fraction, *, rescue_count=None):
    """Lower rank_score is better; labels and experimental values are not read."""
    if (isinstance(fraction, (bool, np.bool_)) or not isinstance(fraction, (int, float))
            or not np.isfinite(fraction) or not 0 <= fraction <= 1):
        raise ValueError("Rescue fraction must be finite and inside [0,1]")
    baseline, _, _ = _validate_rank(frame)
    r_count = int((~baseline).sum())
    budget = math.floor(fraction * r_count) if rescue_count is None else rescue_count
    if type(budget) is not int or not 0 <= budget <= r_count:
        raise ValueError("Rescue count must be an integer within R")
    order = _rescue_order(frame)
    chosen = order[:budget]
    eligibility = baseline.copy()
    eligibility[chosen] = True
    return eligibility, {"R_count": r_count, "rescue_budget": budget,
        "rescue_count": len(chosen), "rescue_deficit": max(0, budget-len(chosen)),
        "rescue_fraction_achieved": len(chosen)/r_count if r_count else 0.,
        "status": "nondiscriminant" if not r_count else "complete"}


def filter_metrics(frame, eligible, *, training_scaffolds=()):
    """Only measured known positives enter biological denominators."""
    baseline = _bool(frame.baseline_eligible, len(frame), "baseline_eligible")
    accepted = _bool(eligible, len(frame), "eligible")
    if (baseline & ~accepted).any():
        raise ValueError("Baseline eligibility cannot be removed")
    positive = frame.y_active.eq(1).to_numpy()
    known = frame.y_active.notna().to_numpy()
    rescue = accepted & ~baseline
    pos_r = positive & ~baseline
    has_scaffold = frame.murcko_scaffold.ne("").to_numpy()
    active_scaffolds = set(frame.loc[positive & has_scaffold, "murcko_scaffold"])
    selected_scaffolds = set(frame.loc[positive & accepted & has_scaffold, "murcko_scaffold"])
    rescued_scaffolds = set(frame.loc[positive & rescue & has_scaffold, "murcko_scaffold"])
    novel = active_scaffolds - set(training_scaffolds)
    return {"C_count": len(frame), "B_count": int(baseline.sum()),
        "R_count": int((~baseline).sum()), "eligible_count": int(accepted.sum()),
        "rescue_count": int(rescue.sum()), "active_count": int(positive.sum()),
        "active_R_count": int(pos_r.sum()), "active_retained": int((positive & accepted).sum()),
        "active_rescue_count": int((positive & rescue).sum()),
        "active_retention": float((positive & accepted).sum()/positive.sum()) if positive.any() else None,
        "active_rescue_retention": float((positive & rescue).sum()/pos_r.sum()) if pos_r.any() else None,
        "rescue_fraction": float(rescue.sum()/(~baseline).sum()) if (~baseline).any() else 0.,
        "unknown_count": int((~known).sum()), "unknown_rescue_count": int((rescue & ~known).sum()),
        "active_scaffold_count": len(active_scaffolds), "active_scaffolds_retained": len(selected_scaffolds),
        "active_scaffolds_rescued": len(rescued_scaffolds),
        "active_scaffold_retention": len(selected_scaffolds)/len(active_scaffolds) if active_scaffolds else None,
        "novel_active_scaffold_count": len(novel), "novel_active_scaffolds_retained": len(novel & selected_scaffolds),
        "reduction_fraction": 1-float(accepted.sum()/len(frame)) if len(frame) else None}


def compare_capacity(left, right, fraction):
    """Paired comparison has identical nonzero rescue count; unsupported cases stay out."""
    if (left.record_id.tolist() != right.record_id.tolist()
            or left.identity.tolist() != right.identity.tolist()
            or left.baseline_eligible.tolist() != right.baseline_eligible.tolist()):
        raise ValueError("Capacity comparison requires the same ordered universe and baseline")
    _, lmeta = capacity_eligibility(left, fraction)
    _, rmeta = capacity_eligibility(right, fraction)
    common = min(lmeta["rescue_count"], rmeta["rescue_count"])
    if not common:
        return {"status": "not_estimable", "reason": "no common nonzero rescue capacity",
                "common_rescue_count": 0, "left_requested": lmeta, "right_requested": rmeta}
    lelig, _ = capacity_eligibility(left, fraction, rescue_count=common)
    relig, _ = capacity_eligibility(right, fraction, rescue_count=common)
    lm, rm = filter_metrics(left, lelig), filter_metrics(right, relig)
    final = {"left_final": final_metrics(left, lelig, 50, 3),
             "right_final": final_metrics(right, relig, 50, 3)} if "qed" in left and "qed" in right else {}
    return {"status": "estimable", "common_rescue_count": common,
        "left_requested": lmeta, "right_requested": rmeta, "left": lm, "right": rm,
        "active_rescue_retention_difference": None if lm["active_rescue_retention"] is None else
            lm["active_rescue_retention"]-rm["active_rescue_retention"], **final}


def paired_bootstrap(rows, *, repeats=1000, seed=20261009):
    """Exploratory paired macro CI; shared scaffold groups keep a common multiplier.

    This conditional resampling does not establish document/source independence.
    It conditions on frozen rankings and capacity, not repeated model training.
    """
    required = {"task", "group", "left_rescued", "right_rescued"}
    if not required <= set(rows) or rows.empty or type(repeats) is not int or repeats < 2:
        raise ValueError("Bootstrap requires positive-R paired outcomes and grouped tasks")
    tasks, groups = sorted(rows.task.unique()), sorted(rows.group.unique())
    if len(tasks) < 2 or len(groups) < 2:
        return {"status": "not_estimable", "reason": "fewer than two tasks or groups"}
    task_index = {name: i for i, name in enumerate(tasks)}
    group_index = {name: i for i, name in enumerate(groups)}
    denominator = np.zeros((len(tasks), len(groups)))
    difference = np.zeros_like(denominator)
    for row in rows.itertuples(index=False):
        t, g = task_index[row.task], group_index[row.group]
        denominator[t, g] += 1
        difference[t, g] += int(row.left_rescued)-int(row.right_rescued)
    rng, samples = np.random.default_rng(seed), []
    for _ in range(repeats):
        group_weights = np.bincount(rng.integers(len(groups), size=len(groups)), minlength=len(groups))
        task_weights = np.bincount(rng.integers(len(tasks), size=len(tasks)), minlength=len(tasks))
        total = denominator @ group_weights
        valid = (total > 0) & (task_weights > 0)
        if valid.any():
            scores = (difference @ group_weights)[valid]/total[valid]
            samples.append(float(np.average(scores, weights=task_weights[valid])))
    point = float(np.mean(difference.sum(axis=1)/denominator.sum(axis=1)))
    return {"status": "exploratory", "macro_difference": point,
        "ci95": np.quantile(samples, [.025, .975]).tolist(), "repeats": repeats,
        "tasks": len(tasks), "global_scaffold_groups": len(groups), "seed": seed,
        "resampling": "paired task draws plus global shared-scaffold multiplicities",
        "limitations": "conditional on frozen rankings; documentary/source independence unresolved; not a qualification claim"}


def freeze_threshold(positive_cal, background_cal, *, target_retention=.95, max_expansion=.1):
    """Minimum expansion feasible calibration threshold, counting unsupported positives."""
    if not 0 < target_retention <= 1 or not 0 <= max_expansion <= 1 or positive_cal.empty:
        raise ValueError("Calibration requires positives and valid retention/expansion goals")
    if not positive_cal.y_active.eq(1).all() or background_cal.y_active.notna().any():
        raise ValueError("Calibration requires known positives and unknown reference compounds")
    pb, ps, px = _validate_rank(positive_cal)
    rb, rs, rx = _validate_rank(background_cal)
    thresholds = [-math.inf, *sorted(set(px[ps & ~pb]))]
    maximum = float((pb | ps).mean())
    choices = []
    for tau in thresholds:
        retained = float((pb | (ps & (px <= tau))).mean())
        expansion = float((rs & ~rb & (rx <= tau)).sum()/(~rb).sum()) if (~rb).any() else 0.
        if retained >= target_retention and expansion <= max_expansion:
            choices.append((expansion, tau, retained))
    if not choices:
        return {"status": "infeasible", "tau": None, "maximum_supported_retention": maximum,
                "target_retention": target_retention, "max_expansion": max_expansion}
    expansion, tau, retained = min(choices)
    return {"status": "feasible", "tau": None if tau == -math.inf else float(tau),
            "baseline_only": tau == -math.inf, "calibration_retention": retained,
            "calibration_expansion": expansion, "maximum_supported_retention": maximum,
            "target_retention": target_retention, "max_expansion": max_expansion}


def validate_partitions(frame, *, scaffold_disjoint=True):
    if not {"identity", "murcko_scaffold", "split"} <= set(frame):
        raise ValueError("Dataset requires identity/scaffold/split fields")
    if not set(frame.split) <= {"fit", "tune", "policy_cal", "test"}:
        raise ValueError("Unexpected partition role")
    if (frame.groupby("identity").split.nunique() > 1).any():
        raise ValueError("Chemical identity appears across partitions")
    cyclic = frame.loc[frame.murcko_scaffold.ne("")]
    if scaffold_disjoint and (cyclic.groupby("murcko_scaffold").split.nunique() > 1).any():
        raise ValueError("Chemical scaffold appears across partitions")


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def _hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def final_metrics(frame, eligible, n, quota):
    from s2s_decision.selection import select_candidates

    if "qed" not in frame:
        raise ValueError("Final selector requires canonical annotated qed descriptor")
    candidates = frame[["record_id", "identity", "murcko_scaffold"]].assign(
        eligible=eligible, valid=True, priority_score=frame.qed)
    candidates.attrs = {}
    outcome = select_candidates(candidates, n, max_per_scaffold=quota)
    final = outcome.records.is_final.to_numpy()
    active = frame.y_active.eq(1).to_numpy()
    return {"requested_n": n, "quota": quota, "selected_n": int(final.sum()),
        "shortfall": max(0, n-int(final.sum())), "active_selected": int((final & active).sum()),
        "active_scaffolds_selected": frame.loc[final & active & frame.murcko_scaffold.ne(""), "murcko_scaffold"].nunique(),
        "nondiscriminant": n >= int(np.asarray(eligible).sum()) and final.sum() == np.asarray(eligible).sum(),
        "selected_ids": frame.loc[final, "record_id"].tolist()}


def _policy_eligibility(frame, policy):
    baseline, support, scores = _validate_rank(frame)
    tau = policy.get("tau")
    if policy["status"] != "feasible" or policy.get("baseline_only") or tau is None:
        return baseline
    return baseline | (support & (scores <= tau))


def evaluate_arms(frame, arms, policies, *, training_scaffolds=(), budgets=BUDGETS):
    """Scores/support are supplied frozen; this stage never fits a model."""
    view = frame.copy(deep=False)
    view.attrs = {}
    rows, baskets = [], []
    for arm, ranking in arms.items():
        if ranking.record_id.tolist() != frame.record_id.tolist():
            raise ValueError("Arm row order mismatch")
        ranking = ranking.copy(deep=False)
        ranking.attrs = {}
        for kind, fraction in [("frozen_policy", None), *(("capacity", q) for q in CAPACITIES)]:
            eligible, capacity = (_policy_eligibility(ranking, policies[arm]), {}) if fraction is None else \
                capacity_eligibility(ranking, fraction)
            metrics = filter_metrics(ranking, eligible, training_scaffolds=training_scaffolds)
            rows.append({"arm": arm, "evaluation": kind, "capacity": fraction, **capacity, **metrics})
            for n in budgets:
                for quota in QUOTAS:
                    baskets.append({"arm": arm, "evaluation": kind, "capacity": fraction,
                                    **final_metrics(view, eligible, n, quota)})
    return rows, baskets


def _simple_arms(frame, *, rules_only=False):
    failures = frame.tolerance_failed_criteria.map(lambda ids: [x for x in ids if
        x.startswith("rule:lipinski:") or x.startswith("rule:veber:") or
        (not rules_only and x.startswith("alert:"))])
    counts = failures.map(len).to_numpy()
    alerts = failures.map(lambda ids: any(x.startswith("alert:") for x in ids)).to_numpy()
    base = counts == 0
    if "hard_excluded" in frame:
        if frame.hard_excluded.any():
            raise ValueError("Benchmark C must omit mandatory exclusions")
    ranked = frame.assign(baseline_eligible=base, rank_score=counts.astype(float))
    return {"rigid": ranked.assign(supported=base),
        "simple_1": ranked.assign(supported=(counts <= 1) & ~alerts),
        "simple_2": ranked.assign(supported=(counts <= 2) & ~alerts),
        "accept_all": ranked.assign(supported=True)}


def _hash_sample(frame, count, seed=20261009):
    if count >= len(frame):
        return frame.copy()
    hashes = frame.identity.map(lambda identity: hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest())
    return frame.loc[hashes.sort_values(kind="stable").index[:count]].copy()


def _profile_arm(frame, profile, context, *, rules_only=False):
    from s2s_decision.tolerance_profile import predict_tolerance

    base = _simple_arms(frame, rules_only=rules_only)["rigid"].baseline_eligible.to_numpy()
    criteria = [name for name in profile["annotation_schema"]["criterion_columns"]
                if name.startswith(("rule:lipinski:", "rule:veber:"))] if rules_only else None
    evidence = predict_tolerance(profile, frame, context=context, baseline_eligible=base,
                                 reconsidered_criteria=criteria)
    return frame.assign(baseline_eligible=base, rank_score=evidence.nonconformity.to_numpy(),
        supported=evidence.tolerance_supported.to_numpy(),
        evidence_json=evidence.tolerance_evidence.map(lambda x: json.dumps(x, sort_keys=True)),
        fallback=evidence.tolerance_fallback.to_numpy())


def _constant_policy(tau):
    return {"status": "feasible", "tau": tau, "scope": "predefined comparator, no calibration"}


def _simple_choice(arms):
    scores, available = [], []
    for name in ("simple_1", "simple_2", "S0", "QEX"):
        if name not in arms:
            continue
        selected, meta = capacity_eligibility(arms[name], .05)
        result = filter_metrics(arms[name], selected)
        score = result["active_rescue_retention"]
        candidate = (score if score is not None else -1., -meta["rescue_count"], name)
        scores.append(candidate)
        if meta["rescue_count"] > 0:
            available.append(candidate)
    return max(available or scores)[2]


def _context(frame):
    row = frame.iloc[0]
    return {key: str(row[key]) for key in ("task_id", "target_id", "species", "endpoint")
            if key in frame and pd.notna(row[key])}


def _load_data(root):
    manifest = json.loads((root/"manifest.json").read_text("utf-8"))
    inventory = manifest.get("files")
    consumed = {"data/positive-records.parquet", "data/reference-records.parquet", "data/tasks.json", "protocol/protocol.json"}
    if manifest.get("schema") != "s2s-tolerance-data/1" or not isinstance(inventory, dict) or not consumed <= set(inventory):
        raise ValueError("Frozen tolerance data manifest lacks the required consumed inventory")
    parent = root.parent.resolve()
    for name, digest in inventory.items():
        path = (parent/name).resolve()
        if (not isinstance(name, str) or "\\" in name or ":" in name or Path(name).is_absolute()
                or ".." in Path(name).parts or not path.is_relative_to(parent)
                or not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
                or path.is_symlink() or _hash(path) != digest):
            raise ValueError(f"Frozen tolerance data checksum or path mismatch: {name}")
    positive = pd.read_parquet(root/"positive-records.parquet")
    reference = pd.read_parquet(root/"reference-records.parquet")
    tasks_payload = json.loads((root/"tasks.json").read_text("utf-8"))
    if isinstance(tasks_payload, list):
        tasks = [value["task_id"] if isinstance(value, dict) else value for value in tasks_payload]
    else:
        tasks = tasks_payload.get("pilot_task_ids", tasks_payload.get("pilot_tasks", tasks_payload.get("tasks", [])))
        tasks = [value["task_id"] if isinstance(value, dict) else value for value in tasks]
    if not tasks:
        tasks = manifest.get("pilot_task_ids", manifest.get("summary", {}).get("pilot_task_ids", manifest.get("pilot_tasks", [])))
    if not tasks:
        raise ValueError("P0 requires tasks frozen by training support before fitting")
    positive = positive.loc[positive.task_id.isin(tasks)].copy()
    positive = positive.assign(y_active=1, sample_origin="observed_active")
    reference = reference.assign(y_active=np.nan, sample_origin="reference_background")
    validate_partitions(pd.concat([positive, reference], ignore_index=True))
    return positive, reference, manifest, tasks


def run_study(data, output, *, rules_only=False, resume_profiles=None, resume_qex=None):
    """Execute registered core pilot, preserve every arm and freeze choices before test."""
    from s2s_decision.qex import QEXFitError, fit_qex, predict_qex, qed_properties
    from s2s_decision.tolerance_features import annotate_tolerance
    from s2s_decision.tolerance_profile import calibrate_tolerance, fit_tolerance
    from s2s_decision.tolerance_schema import seal_profile, write_tolerance_package

    data, output = Path(data), Path(output)
    protocol_path = data.parent/"protocol/protocol.json"
    protocol = json.loads(protocol_path.read_text("utf-8"))
    positive, reference, manifest, tasks = _load_data(data)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output/"execution-freeze.json", {"created_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": _hash(protocol_path), "data_manifest_sha256": _hash(data/"manifest.json"),
        "source_sha256": _hash(__file__), "tasks": tasks, "protocol": protocol,
        "status": "computational pilot; qualification requires independent supported confirmation",
        "biological_labels_used_for_ranking": False, "final_priority": "QED fixed; balanced selector",
        "capacity_tiebreak": "SHA256(identity), row index", "core_fit_cap": 512,
        "representations": ["A", "B", "C", "D"], "model_options": ["S0", "S1", "S2", "specific", "S3-A/B/C/D"],
        "criterion_scope": "Lipinski/Veber only" if rules_only else "Lipinski/Veber + PAINS/Brenk entry IDs"})
    scenario = "rules_only" if rules_only else "rules_plus_alerts"
    fit_parts = [_hash_sample(positive.loc[positive.task_id.eq(task) & positive.split.eq("fit")], 512)
                 for task in tasks]
    training = pd.concat(fit_parts, ignore_index=True).drop_duplicates(["identity", "task_id"])
    _write_json(output/"training-counts.json", {"full_fit_rows": int(positive.split.eq("fit").sum()),
        "pilot_fit_rows": len(training), "pilot_fit_identities": training.identity.nunique(),
        "by_task": training.groupby("task_id").identity.nunique().to_dict()})
    training.to_parquet(output/"training-records.parquet", index=False)
    started = time.perf_counter()
    profiles = {}
    if resume_profiles is not None:
        source = Path(resume_profiles)
        prior = json.loads((source.parent/"execution-freeze.json").read_text("utf-8"))
        if prior["protocol_sha256"] != _hash(protocol_path) or prior["data_manifest_sha256"] != _hash(data/"manifest.json"):
            raise ValueError("Resume requires unchanged frozen protocol and data")
        profiles = json.loads(source.read_text("utf-8"))["profiles"]
        _write_json(output/"resume.json", {"source": str(source), "sha256": _hash(source),
            "reason": "reuse unchanged fit-only profiles after evaluator metadata fix; no model/selection retuning"})
    else:
        for name, mode, recipe in [("S0", "fixed", "A"), ("S1", "general", "A"),
                ("S2", "hierarchical", "A"), *((f"S3-{r}", "contextual", r) for r in "ABCD")]:
            print(f"Fitting tolerance {name}: {len(training)} records", flush=True)
            profiles[name] = fit_tolerance(training, mode=mode, recipe=recipe,
                provenance={"study": "TOL-01", "purpose": "capped computational pilot",
                            "protocol_sha256": _hash(protocol_path)})
    if resume_profiles is None:
        _write_json(output/"profile-training.json", {"elapsed_seconds": time.perf_counter()-started, "profiles": profiles})
    else:
        _write_json(output/"profile-training-reference.json", {"source": str(resume_profiles),
            "sha256": _hash(resume_profiles), "loaded_profile_count": len(profiles),
            "load_seconds": time.perf_counter()-started})
    qex, qex_status = None, {"status": "not_fit"}
    try:
        properties = qed_properties(training.drop_duplicates("identity").model_smiles.tolist())
        if resume_qex is not None:
            from benchmarks.tolerance_source_models import resume_qex_model
            qex, qex_status = resume_qex_model(resume_qex, properties)
        else:
            qex = fit_qex(properties)
            qex_status = {"status": "fit", "training_identities": training.identity.nunique(), "model": qex.to_dict()}
    except QEXFitError as error:
        qex_status = {"status": "unavailable", "reason": str(error)}
    _write_json(output/"qex.json", qex_status)
    all_filter, all_final, all_pair, frozen = [], [], [], {}
    timings, score_tables, runtimes, bootstrap_rows = [], [], [], []
    for task in tasks:
        print(f"Freezing development configuration {len(frozen)+1}/{len(tasks)}: {task}", flush=True)
        task_rows = positive.loc[positive.task_id.eq(task)]
        context = _context(task_rows)
        task_profile = fit_tolerance(training.loc[training.task_id.eq(task)], mode="general", context=context)
        task_profiles = {**profiles, "specific": task_profile}
        role_arms = {}
        for role in ("tune", "policy_cal"):
            part = pd.concat([task_rows.loc[task_rows.split.eq(role)],
                              reference.loc[reference.split.eq(role)]], ignore_index=True)
            part = part.drop_duplicates("identity").sort_values("identity", kind="stable").assign(record_id=lambda x: range(len(x)))
            if part.empty:
                continue
            annotated, _ = annotate_tolerance(part)
            arms = _simple_arms(annotated, rules_only=rules_only)
            for name, profile in task_profiles.items():
                begin = time.perf_counter()
                arms[name] = _profile_arm(annotated, profile, context if name not in ("S0", "S1") else {}, rules_only=rules_only)
                timings.append({"task": task, "role": role, "arm": name,
                                "rows": len(part), "seconds": time.perf_counter()-begin})
            if qex is not None:
                values = predict_qex(qex, qed_properties(annotated.model_smiles.tolist()))
                arms["QEX"] = arms["accept_all"].assign(rank_score=1-values)
            role_arms[role] = arms
        if not role_arms.get("tune") or not role_arms.get("policy_cal"):
            frozen[task] = {"status": "unavailable", "reason": "missing development partition"}
            continue
        best_simple = _simple_choice(role_arms["tune"])
        contextual_choices = [(filter_metrics(rank, capacity_eligibility(rank, .05)[0])["active_rescue_retention"], name)
            for name, rank in role_arms["tune"].items() if name.startswith("S3-")]
        best_context = max(((score if score is not None else -1., name) for score, name in contextual_choices))[1]
        policies = {"rigid": _constant_policy(0.), "simple_1": _constant_policy(1.),
                    "simple_2": _constant_policy(2.), "accept_all": _constant_policy(1e300)}
        for name, ranked in role_arms["policy_cal"].items():
            if name not in policies:
                positives = ranked.loc[ranked.y_active.eq(1)]
                unknown = ranked.loc[ranked.y_active.isna()]
                policies[name] = freeze_threshold(positives, unknown) if len(positives) and len(unknown) else {
                    "status": "unavailable", "tau": None, "reason": "missing calibration origins"}
        calibrated_profiles = {}
        for name in {"S0", "S1", "S2", "S3-A", best_context}:
            ranked = role_arms["policy_cal"][name]
            positives, unknown = ranked.loc[ranked.y_active.eq(1)], ranked.loc[ranked.y_active.isna()]
            if positives.empty or unknown.empty:
                continue
            scope = [key for key in task_profiles[name]["annotation_schema"]["criterion_columns"]
                     if key.startswith(("rule:lipinski:", "rule:veber:"))] if rules_only else None
            applied = context if name not in ("S0", "S1") else {}
            calibrated = calibrate_tolerance(task_profiles[name], positives, unknown, reconsidered_criteria=scope, context=applied)
            native_policy = calibrated["policy"]
            benchmark_policy = policies[name]
            if native_policy["status"] == "feasible" and (benchmark_policy["status"] != "feasible"
                    or not np.isclose(native_policy["tau"], benchmark_policy["tau"], rtol=0, atol=1e-12)):
                raise ValueError("Native calibrated package and independent evaluator threshold disagree")
            if name not in ("S0", "S1"):
                calibrated = seal_profile({**calibrated, "context": context})
            calibrated_profiles[name] = calibrated
        frozen[task] = {"status": "frozen", "best_simple": best_simple, "best_context": best_context,
                        "policies": policies, "context": context,
                        "native_policies": {name: profile["policy"] for name, profile in calibrated_profiles.items()}}
        _write_json(output/"frozen-selection.json", frozen)
        runtimes.append((task, task_rows, context, task_profiles, role_arms, policies,
                         best_simple, best_context, calibrated_profiles))
    _write_json(output/"configuration-freeze.json", {"created_utc": datetime.now(UTC).isoformat(),
        "all_task_configurations": frozen, "test_accessed_for_selection": False,
        "primary_comparison": "task-specific tune choices, frozen before any test inference"})
    global_cal = positive.loc[positive.split.eq("policy_cal")].drop_duplicates("identity")
    global_bg = reference.loc[reference.split.eq("policy_cal")]
    if len(global_cal) and len(global_bg):
        criteria = [key for key in profiles["S1"]["annotation_schema"]["criterion_columns"]
                    if key.startswith(("rule:lipinski:", "rule:veber:"))] if rules_only else None
        global_profile = calibrate_tolerance(profiles["S1"], global_cal, global_bg,
                                            reconsidered_criteria=criteria, context={})
        write_tolerance_package(global_profile, output/"packages/general-S1")
    for task, task_rows, context, task_profiles, role_arms, policies, best_simple, best_context, calibrated_profiles in runtimes:
        for name in {"S2", "S3-A"} if task == tasks[0] else set():
            if name in calibrated_profiles:
                write_tolerance_package(calibrated_profiles[name],
                    output/"packages"/f"{hashlib.sha256(task.encode()).hexdigest()[:10]}-{name}")
        for role in ("tune", "policy_cal", "test"):
            if role == "test":
                part = pd.concat([task_rows.loc[task_rows.split.eq(role)], reference.loc[reference.split.eq(role)]], ignore_index=True)
                part = part.drop_duplicates("identity").sort_values("identity", kind="stable").assign(record_id=lambda x: range(len(x)))
                if part.empty or not part.y_active.eq(1).any():
                    _write_json(output/f"test-unavailable-{hashlib.sha256(task.encode()).hexdigest()[:10]}.json",
                        {"task": task, "status": "not_estimable", "reason": "no untouched test positives"})
                    continue
                annotated, _ = annotate_tolerance(part)
                arms = _simple_arms(annotated, rules_only=rules_only)
                for name, profile in task_profiles.items():
                    begin = time.perf_counter()
                    arms[name] = _profile_arm(annotated, profile, context if name not in ("S0", "S1") else {}, rules_only=rules_only)
                    timings.append({"task": task, "role": role, "arm": name,
                                    "rows": len(part), "seconds": time.perf_counter()-begin})
                if qex is not None:
                    arms["QEX"] = arms["accept_all"].assign(rank_score=1-predict_qex(qex, qed_properties(annotated.model_smiles.tolist())))
            else:
                arms = role_arms[role]
                annotated = next(iter(arms.values()))
            metrics, baskets = evaluate_arms(annotated, arms, policies,
                training_scaffolds=training.murcko_scaffold.unique())
            all_filter.extend({"task": task, "split": role, "scenario": scenario, **row} for row in metrics)
            all_final.extend({"task": task, "split": role, "scenario": scenario, **row} for row in baskets)
            for q in CAPACITIES:
                comparison = compare_capacity(arms[best_context], arms[best_simple], q)
                all_pair.append({"task": task, "split": role, "capacity": q,
                    "contextual": best_context, "simple": best_simple, **comparison})
                if q == .05 and comparison["status"] == "estimable":
                    common = comparison["common_rescue_count"]
                    left = capacity_eligibility(arms[best_context], q, rescue_count=common)[0]
                    right = capacity_eligibility(arms[best_simple], q, rescue_count=common)[0]
                    mask = annotated.y_active.eq(1).to_numpy() & ~arms[best_context].baseline_eligible.to_numpy()
                    for index in np.flatnonzero(mask):
                        row = annotated.iloc[index]
                        bootstrap_rows.append({"task": task, "split": role,
                            "group": row.murcko_scaffold or row.identity,
                            "left_rescued": bool(left[index]), "right_rescued": bool(right[index])})
            for name, ranking in arms.items():
                cols = [key for key in ("record_id", "identity", "murcko_scaffold", "y_active", "baseline_eligible",
                    "rank_score", "supported", "evidence_json", "fallback") if key in ranking]
                score_tables.append(ranking[cols].assign(task=task, split=role, arm=name))
        pd.DataFrame(all_filter).to_csv(output/"filter-metrics.csv", index=False)
        pd.DataFrame(all_final).to_json(output/"final-metrics.jsonl", orient="records", lines=True)
        _write_json(output/"paired-capacity.json", all_pair)
        pd.DataFrame(timings).to_csv(output/"timings.csv", index=False)
        pd.concat(score_tables, ignore_index=True).to_parquet(output/"scores.parquet", index=False)
    if bootstrap_rows:
        paired_rows = pd.DataFrame(bootstrap_rows)
        paired_rows.to_csv(output/"primary-paired-observations.csv", index=False)
        _write_json(output/"primary-bootstrap.json", {role: paired_bootstrap(part)
            for role, part in paired_rows.groupby("split")})
    _write_json(output/"completion.json", {"created_utc": datetime.now(UTC).isoformat(),
        "status": "completed exploratory computational pilot", "tasks": tasks,
        "qualification": "not promoted; capped pilot and documentary independence unresolved",
        "files": {p.relative_to(output).as_posix(): _hash(p) for p in sorted(output.rglob("*"))
                  if p.is_file() and p.name != "completion.json"}})
    return output


def _chemistry_resource_batch(records):
    from s2s_decision.tolerance_features import annotate_tolerance, representation_matrix

    begin = time.perf_counter()
    annotated, schema = annotate_tolerance(records.reset_index(drop=True))
    elapsed = time.perf_counter()-begin
    batch = {"start": int(records.record_id.min()), "rows": len(annotated), "annotation_seconds": elapsed,
             "valid_count": int(annotated.tolerance_structure_valid.sum())}
    for recipe in "ABCD":
        begin = time.perf_counter()
        matrix, _ = representation_matrix(annotated, schema, recipe=recipe)
        batch[f"{recipe}_seconds"] = time.perf_counter()-begin
        del matrix
    return batch


def measure_unique_chemistry(identity_map, output, *, size=100000, batch_size=1000, workers=8):
    """Measure real distinct source chemistry; cache-warm timings are reported separately."""
    import platform
    import threading
    from concurrent.futures import ProcessPoolExecutor

    import psutil

    identity_map, output = Path(identity_map), Path(output)
    records = pd.read_parquet(identity_map)
    records = records.loc[records.model_smiles.notna()].drop_duplicates("identity").drop_duplicates("model_smiles")
    if len(records) < size:
        raise ValueError("Not enough distinct chemical identities for the declared performance workload")
    records = _hash_sample(records, size).reset_index(drop=True).assign(record_id=range(size))
    output.mkdir(parents=True, exist_ok=False)
    records[["record_id", "identity", "model_smiles"]].to_parquet(output/"workload.parquet", index=False)
    process, samples, stop = psutil.Process(), [], threading.Event()

    def monitor():
        while not stop.wait(.2):
            aggregate = process.memory_info().rss
            for child in process.children(recursive=True):
                try:
                    aggregate += child.memory_info().rss
                except psutil.NoSuchProcess:
                    continue
            samples.append(aggregate)

    baseline = process.memory_info().rss
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    recipe_times = dict.fromkeys("ABCD", 0.)
    annotation_seconds, valid_count, batches = 0., 0, []
    wall_started = time.perf_counter()
    try:
        chunks = (records.iloc[start:start+batch_size].copy() for start in range(0, size, batch_size))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for batch in pool.map(_chemistry_resource_batch, chunks, chunksize=1):
                annotation_seconds += batch["annotation_seconds"]
                valid_count += batch["valid_count"]
                for recipe in "ABCD":
                    recipe_times[recipe] += batch[f"{recipe}_seconds"]
                batches.append(batch)
                completed = sum(item["rows"] for item in batches)
                if len(batches) % 10 == 1:
                    print(f"Unique chemistry resource workload: {completed}/{size}", flush=True)
                    _write_json(output/"progress.json", {"completed_rows": completed,
                        "worker_annotation_seconds_sum": annotation_seconds,
                        "representation_worker_seconds_sum": recipe_times,
                        "wall_seconds": time.perf_counter()-wall_started})
    finally:
        stop.set()
        thread.join(timeout=1.)
    metadata = {"status": "completed", "unique_identity_count": records.identity.nunique(),
        "unique_smiles_count": records.model_smiles.nunique(), "valid_count": valid_count,
        "source_sha256": _hash(identity_map), "workload_sha256": _hash(output/"workload.parquet"),
        "cpu": platform.processor(), "logical_cpus": psutil.cpu_count(), "python": platform.python_version(),
        "batch_size": batch_size, "annotation_seconds": annotation_seconds,
        "workers": workers, "wall_seconds": time.perf_counter()-wall_started,
        "worker_time_interpretation": "annotation/representation sums overlap across processes; use wall_seconds for elapsed workload",
        "representation_seconds": recipe_times, "initial_rss_bytes": baseline,
        "peak_rss_bytes": max(samples, default=baseline),
        "additional_peak_rss_bytes": max(samples, default=baseline)-baseline,
        "scope": "pre-standardized distinct chemistry annotation/representations; excludes model fitting, statistical inference and selection",
        "cache": "cold distinct structures; annotation/representation reuse within each batch",
        "thread_configuration": f"{workers} Python worker processes; native library settings recorded by execution environment"}
    _write_json(output/"resource-results.json", metadata)
    pd.DataFrame(batches).to_csv(output/"batches.csv", index=False)
    return metadata


def export_global_general(data, trained, output, *, rules_only=True):
    """Calibration/package branch independent of task tuning; enables resource measurements."""
    from s2s_decision.tolerance_profile import calibrate_tolerance
    from s2s_decision.tolerance_schema import write_tolerance_package

    data, trained, output = Path(data), Path(trained), Path(output)
    positive, reference, _, _ = _load_data(data)
    prior = json.loads((trained.parent/"execution-freeze.json").read_text("utf-8"))
    if prior["data_manifest_sha256"] != _hash(data/"manifest.json"):
        raise ValueError("General package requires unchanged frozen data")
    profile = json.loads(trained.read_text("utf-8"))["profiles"]["S1"]
    positives = positive.loc[positive.split.eq("policy_cal")].drop_duplicates("identity")
    background = reference.loc[reference.split.eq("policy_cal")]
    criteria = [key for key in profile["annotation_schema"]["criterion_columns"]
                if key.startswith(("rule:lipinski:", "rule:veber:"))] if rules_only else None
    calibrated = calibrate_tolerance(profile, positives, background, context={}, reconsidered_criteria=criteria)
    write_tolerance_package(calibrated, output)
    _write_json(output.parent/f"{output.name}-calibration-source.json", {"fit_snapshot_sha256": _hash(trained),
        "data_manifest_sha256": _hash(data/"manifest.json"), "policy": calibrated["policy"],
        "scope": "same frozen general fit and all pilot positive calibration identities; no tune/test used"})
    return output


def _resource_records(path):
    records = pd.read_parquet(path)
    needed = {"record_id", "identity", "model_smiles"}
    if not needed <= set(records) or records.empty or any(records[key].duplicated().any() for key in needed):
        raise ValueError("Resource workload requires actual distinct IDs, identities and SMILES")
    return records


def _resource_annotate(payload):
    from s2s_decision.tolerance_features import annotate_tolerance

    records, destination = payload
    begin = time.perf_counter()
    annotated, _ = annotate_tolerance(records.reset_index(drop=True))
    dense = pd.DataFrame(annotated, copy=False)
    sparse = {name: dense[name].sparse.to_dense() for name, dtype in dense.dtypes.items()
              if isinstance(dtype, pd.SparseDtype)}
    dense = dense.assign(**sparse)
    dense.attrs = dict(annotated.attrs)
    dense.to_parquet(destination, index=False)
    return {"path": str(destination), "rows": len(annotated), "seconds": time.perf_counter()-begin}


def _resource_initialize(package):
    from s2s_decision.tolerance_schema import read_tolerance_package

    global _RESOURCE_PROFILE
    _RESOURCE_PROFILE = read_tolerance_package(package)


def _resource_infer(path):
    from s2s_decision.tolerance_profile import predict_tolerance

    annotated = pd.read_parquet(path)
    begin = time.perf_counter()
    scope = _RESOURCE_PROFILE.get("policy", {}).get("scope", {}).get("criteria")
    scored = predict_tolerance(_RESOURCE_PROFILE, annotated, context={}, reconsidered_criteria=scope)
    slim = annotated[["record_id", "identity", "murcko_scaffold", "qed"]].assign(
        eligible=scored.eligible.to_numpy(), valid=annotated.tolerance_structure_valid.to_numpy(),
        state=scored.state.to_numpy(), nonconformity=scored.nonconformity.to_numpy())
    slim.attrs = {}
    return slim, time.perf_counter()-begin


def measure_statistical_resource(workload, package, output, *, workers=8, batch_size=2500, n=2000):
    """Real annotation, frozen statistical inference and native final selector, timed separately."""
    import platform
    import threading
    from concurrent.futures import ProcessPoolExecutor

    import psutil

    from s2s_decision.selection import select_candidates
    from s2s_decision.tolerance_schema import read_tolerance_package

    workload, package, output = Path(workload), Path(package), Path(output)
    records = _resource_records(workload)
    profile = read_tolerance_package(package)
    output.mkdir(parents=True, exist_ok=False)
    (output/"annotations").mkdir()
    _write_json(output/"execution-freeze.json", {"source_sha256": _hash(__file__),
        "profile_hash": profile["profile_sha256"], "workload_sha256": _hash(workload),
        "unique_identities": len(records), "workers": workers, "batch_size": batch_size,
        "requested_n": n, "scaffold_cap": 3, "priority": "fixed QED balanced native selector"})
    process, stop, samples = psutil.Process(), threading.Event(), []

    def monitor():
        while not stop.wait(.2):
            rss = process.memory_info().rss
            for child in process.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except psutil.NoSuchProcess:
                    continue
            samples.append(rss)

    initial = process.memory_info().rss
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    wall_started = time.perf_counter()
    batches = [(records.iloc[start:start+batch_size].copy(), output/"annotations"/f"{start:06d}.parquet")
               for start in range(0, len(records), batch_size)]
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            annotated = list(pool.map(_resource_annotate, batches, chunksize=1))
        annotation_wall = time.perf_counter()-wall_started
        print(f"Statistical resource annotation complete: {len(records)} distinct identities", flush=True)
        begin = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers, initializer=_resource_initialize,
                                 initargs=(package,)) as pool:
            predictions = list(pool.map(_resource_infer, [item["path"] for item in annotated], chunksize=1))
        inference_wall = time.perf_counter()-begin
        print("Statistical resource frozen inference complete", flush=True)
        candidates = pd.concat([item[0] for item in predictions], ignore_index=True).sort_values("record_id", kind="stable")
        begin = time.perf_counter()
        selected = select_candidates(candidates.assign(priority_score=candidates.qed), n, max_per_scaffold=3)
        selection_wall = time.perf_counter()-begin
    finally:
        stop.set()
        thread.join(timeout=1.)
    candidates.assign(is_final=selected.records.is_final.to_numpy()).to_parquet(output/"decisions.parquet", index=False)
    metadata = {"status": "completed", "profile_hash": profile["profile_sha256"],
        "profile_mode": profile["mode"], "profile_policy": profile["policy"],
        "unique_identity_count": len(records), "unique_smiles_count": records.model_smiles.nunique(),
        "workers": workers, "batch_size": batch_size, "cpu": platform.processor(),
        "logical_cpus": psutil.cpu_count(), "python": platform.python_version(),
        "annotation_wall_seconds": annotation_wall, "inference_wall_seconds": inference_wall,
        "selection_wall_seconds": selection_wall, "total_wall_seconds": time.perf_counter()-wall_started,
        "worker_inference_seconds_sum": sum(item[1] for item in predictions),
        "initial_rss_bytes": initial, "peak_aggregate_rss_bytes": max(samples, default=initial),
        "additional_peak_rss_bytes": max(samples, default=initial)-initial,
        "state_counts": candidates.state.value_counts().to_dict(), "selection": selected.manifest,
        "scope": "pre-standardized unique molecules; fresh annotation then package loading/inference and QED selection; no fitting/standardization",
        "implementation_gate_under_5_minutes": inference_wall+selection_wall <= 300.,
        "implementation_gate_under_2_gib_additional_rss": max(samples, default=initial)-initial <= 2*1024**3}
    _write_json(output/"resource-results.json", metadata)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rules-only", action="store_true")
    parser.add_argument("--resume-profiles", type=Path)
    parser.add_argument("--resume-qex", type=Path)
    args = parser.parse_args(argv)
    run_study(args.data, args.output, rules_only=args.rules_only,
              resume_profiles=args.resume_profiles, resume_qex=args.resume_qex)


if __name__ == "__main__":
    main()
