"""Run orchestration.

Import -> map -> validate -> standardize -> canonicalize -> plan descriptors ->
parallel calculation -> vectorized rule evaluation -> alerts -> scores ->
decision -> SQLite.

Each descriptor is computed exactly once per record, and only the descriptors
the selected profiles, scores and alerts actually need are computed at all.
"""

from __future__ import annotations

import functools
import hashlib
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from smiles2select.alerts import engine as alert_engine
from smiles2select.chemistry.descriptor_planner import DescriptorPlan, DescriptorPlanner
from smiles2select.chemistry.descriptor_registry import default_registry
from smiles2select.chemistry.duplicates import find_duplicates
from smiles2select.chemistry.scaffolds import scaffolds_from_smiles
from smiles2select.chemistry.standardization import StandardizationConfig
from smiles2select.decision.engine import DecisionEngine, DecisionResult
from smiles2select.io.importer import load_records
from smiles2select.pipeline.checkpoint_store import ChunkCheckpointStore
from smiles2select.pipeline.chunking import MoleculeChunk
from smiles2select.pipeline.config import RunConfig
from smiles2select.pipeline.resource_estimation import estimate_worker_count
from smiles2select.pipeline.streaming import run_streaming
from smiles2select.pipeline.workers import RecordResult, alerts_to_tuples, process_chunk
from smiles2select.profiles.loader import builtin_registry
from smiles2select.profiles.registry import Profile
from smiles2select.profiles.validator import validate_all
from smiles2select.rules.evaluator import ProfileEvaluation, evaluate_profiles
from smiles2select.scores.consensus import score_table
from smiles2select.selection.diversity import DiversityReport, pick_by_scaffold, pick_diverse
from smiles2select.storage.descriptor_cache import DescriptorCache, rebind
from smiles2select.storage.sqlite_store import SqliteStore

ProgressCallback = Callable[[int, int, str], None]

# Descriptor id -> column name in molecule_descriptors, where they differ.
_STORAGE_ALIASES = {"hbd_lipinski": "hbd", "hba_lipinski": "hba"}

_RESERVED_COLUMNS = frozenset(
    {
        "input_order",
        "molecule_id",
        "original_smiles",
        "standardized_smiles",
        "canonical_smiles",
        "source_file",
        "source_sheet",
        "source_row",
        "valid",
        "invalid_reason",
        "duplicate_of",
        "evaluable",
    }
)


@dataclass(frozen=True)
class RunResult:
    """Everything one run produced, in memory and on disk."""

    records: pd.DataFrame
    descriptors: pd.DataFrame
    evaluation: ProfileEvaluation
    alerts: pd.DataFrame
    scores: pd.DataFrame
    decision: DecisionResult
    plan: DescriptorPlan
    profiles: tuple[Profile, ...]
    config: RunConfig
    database_path: Path | None
    #: ``(hits, misses)`` when a persistent cache was used, otherwise None.
    cache_stats: tuple[int, int] | None = None
    #: Present when a MaxMin diversity pick narrowed the selection.
    diversity: DiversityReport | None = None

    @property
    def total_records(self) -> int:
        return len(self.records)

    @property
    def invalid_count(self) -> int:
        return int((~self.descriptors["valid"].astype(bool)).sum())

    @property
    def duplicate_count(self) -> int:
        return int(self.descriptors["duplicate_of"].notna().sum())

    @property
    def evaluated_count(self) -> int:
        return len(self.evaluation.status)

    def profile_summary(self) -> pd.DataFrame:
        """Approved / rejected / percentage per profile, for the summary sheet."""
        total = self.evaluated_count
        rows = []
        for profile in self.profiles:
            passed = int(self.evaluation.passed(profile.id).sum())
            rows.append(
                {
                    "profile_id": profile.id,
                    "profile": profile.name,
                    "approved": passed,
                    "rejected": total - passed,
                    "percentage": round(100.0 * passed / total, 2) if total else 0.0,
                }
            )
        return pd.DataFrame(rows)


def run(config: RunConfig, progress: ProgressCallback | None = None) -> RunResult:
    """Execute a full run and persist it to SQLite."""
    report = progress or (lambda done, total, stage: None)

    report(0, 1, "Importando")
    records = load_records(config.sources)
    if config.max_records is not None:
        records = records.head(config.max_records)
    if records.empty:
        raise ValueError("no records were imported from the selected sources")

    descriptor_registry = default_registry()
    profile_registry = builtin_registry(config.profile_directories)
    profiles = profile_registry.select(config.profile_ids)
    validate_all(profiles, descriptor_registry)

    plan = DescriptorPlanner(descriptor_registry).resolve(
        profiles=profiles, scores=config.score_ids(), alerts=config.alert_catalogs
    )
    substructure_rules = tuple(
        (rule.id, rule.smarts)
        for profile in profiles
        for rule in profile.rules
        if rule.is_substructure and rule.smarts
    )

    results, cache_stats = _calculate(config, records, plan, substructure_rules, report)
    descriptors = _descriptor_frame(records, results, plan)
    alerts = alert_engine.alerts_frame(row for result in results for row in result.alert_rows)
    alerts = _apply_alert_actions(alerts, config)

    evaluable = descriptors[descriptors["evaluable"].astype(bool)]
    if evaluable.empty:
        raise ValueError("every record was invalid or duplicated; nothing to evaluate")

    report(0, 1, "Avaliando regras")
    evaluation = evaluate_profiles(evaluable[_rule_columns(evaluable)], profiles)

    qed_values = evaluable["qed"] if "qed" in evaluable.columns else None
    counts = alert_engine.alert_counts(alerts, evaluation.status.index)
    scores = score_table(
        evaluation.status, [profile.id for profile in profiles], qed_values, counts
    )

    report(0, 1, "Aplicando política de seleção")
    decision = DecisionEngine(config.policy).decide(evaluation.status, scores, alerts)

    diversity_report = None
    if config.needs_scaffolds:
        report(0, 1, "Selecionando subconjunto diverso")
        descriptors["murcko_scaffold"] = pd.Series(
            scaffolds_from_smiles(descriptors["canonical_smiles"].fillna("").tolist()),
            index=descriptors.index,
        )
        decision, diversity_report = _apply_diversity(decision, descriptors, config)

    database_path = _persist(config, records, descriptors, evaluation, alerts, scores, decision)
    report(1, 1, "Concluído")

    return RunResult(
        records=records,
        descriptors=descriptors,
        evaluation=evaluation,
        alerts=alerts,
        scores=scores,
        decision=decision,
        plan=plan,
        profiles=profiles,
        config=config,
        database_path=database_path,
        cache_stats=cache_stats,
        diversity=diversity_report,
    )


def _apply_diversity(
    decision: DecisionResult, descriptors: pd.DataFrame, config: RunConfig
) -> tuple[DecisionResult, DiversityReport | None]:
    """Narrow the selected set to a diverse subset.

    Runs last, on molecules that already passed every rule: diversity picking
    can only remove, never rescue an excluded molecule. Each drop is recorded
    with its reason, like any other exclusion.
    """
    decisions = decision.decisions.copy()
    selected = decisions.index[decisions["selected"].astype(bool)]
    if selected.empty:
        return decision, None

    smiles = descriptors.loc[selected, "canonical_smiles"].fillna("").tolist()
    keep = set(selected)
    report: DiversityReport | None = None

    if config.per_scaffold_limit is not None:
        positions = pick_by_scaffold(smiles, config.per_scaffold_limit)
        keep &= {selected[position] for position in positions}
        _drop(decisions, selected, keep, f"limite de {config.per_scaffold_limit} por scaffold")
        selected = decisions.index[decisions["selected"].astype(bool)]
        smiles = descriptors.loc[selected, "canonical_smiles"].fillna("").tolist()
        keep = set(selected)

    if config.diversity_pick is not None and len(selected) > config.diversity_pick:
        report = pick_diverse(smiles, config.diversity_pick, config.fingerprint_config)
        keep = {selected[position] for position in report.picked_index}
        _drop(
            decisions,
            selected,
            keep,
            f"fora do subconjunto diverso (MaxMin, n={config.diversity_pick})",
        )

    return DecisionResult(decisions=decisions), report


def _drop(decisions: pd.DataFrame, candidates: pd.Index, keep: set, reason: str) -> None:
    """Deselect the candidates that were not kept, recording why."""
    dropped = [record_id for record_id in candidates if record_id not in keep]
    if not dropped:
        return
    decisions.loc[dropped, "selected"] = False
    existing = decisions.loc[dropped, "exclusion_reasons"]
    decisions.loc[dropped, "exclusion_reasons"] = [
        f"{text}; {reason}" if text else reason for text in existing
    ]


def _calculate(
    config: RunConfig,
    records: pd.DataFrame,
    plan: DescriptorPlan,
    substructure_rules: tuple[tuple[str, str], ...],
    report: ProgressCallback,
) -> tuple[list, tuple[int, int] | None]:
    """Run the chemistry workers over every record, in chunks.

    With a cache configured, molecules already described under the same
    standardization, RDKit version and alert catalogues are read back instead
    of recomputed; only the remainder reaches the workers.
    """
    pairs = list(zip(records.index.tolist(), records["original_smiles"].tolist(), strict=True))
    smiles_by_record = dict(pairs)
    custom_alerts = alerts_to_tuples(config.custom_alerts)
    input_hash = _hash_records(pairs)

    cache = _open_cache(config)
    cached = cache.fetch((smiles for _, smiles in pairs), plan.descriptor_ids) if cache else {}
    pending = [(record_id, smiles) for record_id, smiles in pairs if smiles not in cached]
    reused = [rebind(cached[smiles], record_id) for record_id, smiles in pairs if smiles in cached]

    report(len(reused), len(pairs), "Calculando descritores")
    computed = _run_workers(
        config,
        pending,
        plan,
        substructure_rules,
        custom_alerts,
        report,
        len(reused),
        len(pairs),
        input_hash,
    )

    stats: tuple[int, int] | None = None
    if cache is not None:
        cache.store(smiles_by_record, computed)
        stats = (cache.hits, cache.misses)
        cache.close()

    return reused + computed, stats


def _open_cache(config: RunConfig) -> DescriptorCache | None:
    """Open the persistent cache, or return None when caching is off."""
    if config.cache_path is None:
        return None
    return DescriptorCache(
        config.cache_path,
        config.standardization,
        config.alert_catalogs,
        tuple(alert.smarts for alert in config.custom_alerts),
    )


def _run_workers(
    config: RunConfig,
    pending: Sequence[tuple[int, str]],
    plan: DescriptorPlan,
    substructure_rules: tuple[tuple[str, str], ...],
    custom_alerts: tuple[tuple[str, str, str, str], ...],
    report: ProgressCallback,
    already_done: int,
    total: int,
    input_hash: str,
) -> list:
    """Compute the records the cache could not supply.

    Streams chunk-by-chunk instead of collecting every worker result before
    returning: each finished chunk is committed to a checkpoint immediately,
    so a crash here loses at most one in-flight window of chunks, not the
    whole run, and a second attempt with the same input and config resumes
    instead of recomputing everything.
    """
    if not pending:
        return []

    run_chunk = functools.partial(
        _run_chunk,
        descriptor_ids=plan.descriptor_ids,
        standardization=config.standardization,
        catalog_ids=config.alert_catalogs,
        custom_alerts=custom_alerts,
        substructure_rules=substructure_rules,
    )

    diagnostics = config.diagnostics
    n_jobs = diagnostics.resolved_n_jobs(_resolve_n_jobs(config))
    chunk_size = diagnostics.resolved_chunk_size(config.chunk_size)
    log_directory = resolve_log_directory(config)
    checkpoint_path = resolve_checkpoint_path(config)
    keep_checkpoint = config.checkpoint_path is not None

    checkpoint = ChunkCheckpointStore(
        checkpoint_path, input_hash=input_hash, config_hash=config.fingerprint()
    )
    try:
        streaming_report = run_streaming(
            pending,
            run_chunk,
            checkpoint=checkpoint,
            n_jobs=n_jobs,
            initial_chunk_size=chunk_size,
            log_directory=log_directory,
            log_memory=diagnostics.log_memory,
            report=report,
            already_done=already_done,
            total=total,
        )
    finally:
        checkpoint.close()

    if not keep_checkpoint:
        Path(checkpoint_path).unlink(missing_ok=True)

    return streaming_report.results


def _run_chunk(
    chunk: MoleculeChunk,
    *,
    descriptor_ids: tuple[str, ...],
    standardization: StandardizationConfig,
    catalog_ids: tuple[str, ...],
    custom_alerts: tuple[tuple[str, str, str, str], ...],
    substructure_rules: tuple[tuple[str, str], ...],
) -> list[RecordResult]:
    """Adapts ``process_chunk`` (a plain sequence of pairs) to ``MoleculeChunk``.

    Must stay a module-level function, never a closure or lambda: it is
    pickled and sent to worker processes, including on Windows where that
    requires the target to be importable by name.
    """
    return process_chunk(
        chunk.records,
        descriptor_ids,
        standardization,
        catalog_ids,
        custom_alerts,
        substructure_rules,
    )


def _resolve_n_jobs(config: RunConfig) -> int:
    """An explicit positive n_jobs is honored exactly; -1/0/None auto-sizes
    from available memory and CPU count instead of handing joblib every core.
    """
    if config.n_jobs is not None and config.n_jobs > 0:
        return config.n_jobs
    return estimate_worker_count(None)


def resolve_log_directory(config: RunConfig) -> str:
    if config.diagnostics.log_directory is not None:
        return str(config.diagnostics.log_directory)
    if config.database_path is not None:
        return str(config.database_path.parent / f"{config.database_path.stem}_logs")
    return str(Path(tempfile.gettempdir()) / "smiles2select_logs")


def resolve_checkpoint_path(config: RunConfig) -> Path:
    if config.checkpoint_path is not None:
        return config.checkpoint_path
    if config.database_path is not None:
        return config.database_path.with_suffix(".checkpoint.sqlite")
    return Path(tempfile.gettempdir()) / f"smiles2select_{config.fingerprint()}.checkpoint.sqlite"


def _hash_records(pairs: Sequence[tuple[int, str]]) -> str:
    """Identity of the input for checkpoint resume: order-sensitive, so a
    reordered or edited source file is treated as different input, not a
    partially-completed one."""
    digest = hashlib.sha256()
    for record_id, smiles in pairs:
        digest.update(f"{record_id}\x00{smiles}\n".encode())
    return digest.hexdigest()


def _descriptor_frame(
    records: pd.DataFrame, results: Sequence, plan: DescriptorPlan
) -> pd.DataFrame:
    """Assemble the per-record descriptor table, flagging invalids and duplicates."""
    rows = []
    for result in results:
        row: dict[str, object] = {
            "record_id": result.record_id,
            "valid": result.valid,
            "invalid_reason": result.invalid_reason,
            "standardized_smiles": result.standardized_smiles,
            "canonical_smiles": result.canonical_smiles,
        }
        row.update(result.descriptors)
        row.update(result.substructure_flags)
        rows.append(row)

    computed = pd.DataFrame(rows).set_index("record_id").sort_index()
    frame = records.join(computed, how="left")
    frame["valid"] = frame["valid"].fillna(False).astype(bool)

    duplicates = find_duplicates(frame["canonical_smiles"].tolist())
    duplicate_of = pd.Series(
        {
            frame.index[position]: frame.index[first]
            for position, first in duplicates.duplicate_of.items()
        },
        dtype="Int64",
    )
    frame["duplicate_of"] = duplicate_of.reindex(frame.index)
    frame["evaluable"] = frame["valid"] & frame["duplicate_of"].isna()

    for descriptor_id in plan.descriptor_ids:
        if descriptor_id not in frame.columns:
            frame[descriptor_id] = pd.NA
    return frame


def _rule_columns(frame: pd.DataFrame) -> list[str]:
    """Columns a rule may read: descriptors plus substructure flags."""
    return [column for column in frame.columns if column not in _RESERVED_COLUMNS]


def _apply_alert_actions(alerts: pd.DataFrame, config: RunConfig) -> pd.DataFrame:
    """Stamp each alert row with the action configured for its catalogue."""
    if alerts.empty:
        return alerts
    updated = alerts.copy()
    updated["action"] = updated["catalog_id"].map(config.policy.alert_policy.action_for)
    return updated


def _persist(
    config: RunConfig,
    records: pd.DataFrame,
    descriptors: pd.DataFrame,
    evaluation: ProfileEvaluation,
    alerts: pd.DataFrame,
    scores: pd.DataFrame,
    decision: DecisionResult,
) -> Path | None:
    """Write every table to the run database."""
    if config.database_path is None:
        return None

    storage = descriptors.rename(columns=_STORAGE_ALIASES)
    storage = storage.assign(valid=storage["valid"].astype(int))

    # The run database holds exactly one run; re-running the same path replaces it.
    with SqliteStore(config.database_path, overwrite=True) as store:
        store.write_frame(storage, "molecule_descriptors", index_label="record_id")
        store.write_frame(_profile_results(evaluation), "profile_results")
        store.write_frame(evaluation.failures, "rule_failures")
        store.write_frame(alerts, "structural_alerts")
        store.write_frame(
            decision.decisions.assign(selected=decision.decisions["selected"].astype(int)),
            "final_decisions",
            index_label="record_id",
        )
        store.write_config(
            {
                **dict(config.summary_rows()),
                "score_columns": list(scores.columns),
                "total_records": len(records),
            }
        )
    return config.database_path


def _profile_results(evaluation: ProfileEvaluation) -> pd.DataFrame:
    """Long form of the per-profile verdicts, one row per record x profile."""
    frames = []
    for profile_id in evaluation.profile_ids():
        frames.append(
            pd.DataFrame(
                {
                    "record_id": evaluation.status.index,
                    "profile_id": profile_id,
                    "passed": evaluation.passed(profile_id).astype(int).to_numpy(),
                    "violation_count": evaluation.violation_count(profile_id).to_numpy(),
                    "failure_mask": evaluation.failure_mask(profile_id).to_numpy(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
