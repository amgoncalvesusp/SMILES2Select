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
from smiles2select.chemical_space.zones import zone_from_expression
from smiles2select.chemistry.descriptor_planner import DescriptorPlan, DescriptorPlanner
from smiles2select.chemistry.descriptor_registry import default_registry
from smiles2select.chemistry.duplicates import find_duplicates
from smiles2select.chemistry.scaffolds import scaffolds_from_smiles
from smiles2select.chemistry.standardization import StandardizationConfig
from smiles2select.decision.engine import DecisionEngine, DecisionResult
from smiles2select.io.importer import load_records
from smiles2select.pipeline.cancellation import CancellationToken
from smiles2select.pipeline.checkpoint_store import ChunkCheckpointStore
from smiles2select.pipeline.chunking import MoleculeChunk
from smiles2select.pipeline.config import RunConfig
from smiles2select.pipeline.resource_estimation import estimate_worker_count
from smiles2select.pipeline.streaming import run_streaming
from smiles2select.pipeline.workers import RecordResult, alerts_to_tuples, process_chunk
from smiles2select.profiles.loader import builtin_registry
from smiles2select.profiles.registry import Profile
from smiles2select.profiles.validator import validate_all
from smiles2select.reference.duplicates import ExactDuplicateReport, compare_exact_duplicates
from smiles2select.reference.libraries import LibraryRole, ReferenceLibrary, prepare_library_frame
from smiles2select.reference.similarity import (
    ReferenceSimilarityResult,
    compute_reference_similarity,
)
from smiles2select.rules.evaluator import ProfileEvaluation, evaluate_profiles
from smiles2select.scores.consensus import score_table
from smiles2select.selection.diversity import DiversityReport, pick_by_scaffold, pick_diverse
from smiles2select.selection.reference_diversity import (
    reference_neighborhood,
    reference_seeded_maxmin,
)
from smiles2select.selection.zone_allocator import ZoneAllocationResult, allocate_zones
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
    #: Present when one or more reference libraries were configured.
    reference_libraries: tuple[ReferenceLibrary, ...] = ()
    background_libraries: tuple[ReferenceLibrary, ...] = ()
    reference_duplicates: ExactDuplicateReport | None = None
    reference_similarity: ReferenceSimilarityResult | None = None
    reserve_ids: tuple[int, ...] = ()
    zone_allocation: ZoneAllocationResult | None = None

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


def run(
    config: RunConfig,
    progress: ProgressCallback | None = None,
    cancellation: CancellationToken | None = None,
) -> RunResult:
    """Execute a full run and persist it to SQLite."""
    token = cancellation or CancellationToken()

    def report(done: int, total: int, stage: str) -> None:
        token.raise_if_cancelled()
        if progress is not None:
            progress(done, total, stage)

    report(0, 1, "Importing")
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
    reference_libraries, background_libraries, reference_duplicates, reference_similarity = _reference_analysis(
        config, descriptors, report
    )
    alerts = alert_engine.alerts_frame(row for result in results for row in result.alert_rows)
    alerts = _apply_alert_actions(alerts, config)

    evaluable = descriptors[descriptors["evaluable"].astype(bool)]
    if evaluable.empty:
        raise ValueError("every record was invalid or duplicated; nothing to evaluate")

    report(0, 1, "Evaluating rules")
    evaluation = evaluate_profiles(evaluable[_rule_columns(evaluable)], profiles)

    qed_values = evaluable["qed"] if "qed" in evaluable.columns else None
    counts = alert_engine.alert_counts(alerts, evaluation.status.index)
    scores = score_table(
        evaluation.status, [profile.id for profile in profiles], qed_values, counts
    )

    report(0, 1, "Applying selection policy")
    decision = DecisionEngine(config.policy).decide(evaluation.status, scores, alerts)
    if reference_duplicates is not None and config.exclude_reference_duplicates:
        decision = _exclude_reference_duplicates(decision, reference_duplicates)

    diversity_report = None
    if config.needs_scaffolds:
        report(0, 1, "Selecting diverse subset")
        descriptors["murcko_scaffold"] = pd.Series(
            scaffolds_from_smiles(descriptors["canonical_smiles"].fillna("").tolist()),
            index=descriptors.index,
        )
        decision, diversity_report = _apply_diversity(decision, descriptors, config)

    reserve_ids: tuple[int, ...] = ()
    if config.selection_strategy != "traditional" or config.final_count is not None or config.reserve_count:
        decision, reserve_ids, zone_allocation = _apply_selection_strategy(
            decision, descriptors, reference_libraries, config
        )
    else:
        zone_allocation = None

    database_path = _persist(
        config,
        records,
        descriptors,
        evaluation,
        alerts,
        scores,
        decision,
        reference_libraries=reference_libraries,
        background_libraries=background_libraries,
        reference_duplicates=reference_duplicates,
        reference_similarity=reference_similarity,
        reserve_ids=reserve_ids,
        zone_allocation=zone_allocation,
    )
    report(1, 1, "Completed")

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
        reference_libraries=reference_libraries,
        background_libraries=background_libraries,
        reference_duplicates=reference_duplicates,
        reference_similarity=reference_similarity,
        reserve_ids=reserve_ids,
        zone_allocation=zone_allocation,
    )


def _reference_analysis(
    config: RunConfig,
    descriptors: pd.DataFrame,
    report: ProgressCallback,
) -> tuple[
    tuple[ReferenceLibrary, ...],
    tuple[ReferenceLibrary, ...],
    ExactDuplicateReport | None,
    ReferenceSimilarityResult | None,
]:
    """Prepare configured references and attach exact overlap/similarity columns."""

    if not config.reference_sources and not config.background_sources:
        return (), (), None, None
    sources = [(source, LibraryRole.REFERENCE) for source in config.reference_sources]
    sources.extend((source, LibraryRole.BACKGROUND) for source in config.background_sources)
    report(0, len(sources), "Analyzing reference and background libraries")
    libraries: list[ReferenceLibrary] = []
    for index, (source, role) in enumerate(sources, start=1):
        reference_records = load_records((source,))
        library_id = Path(source.path).stem or f"reference_{index}"
        if any(existing.library_id == library_id for existing in libraries):
            library_id = f"{library_id}_{index}"
        libraries.append(
            prepare_library_frame(
                reference_records,
                library_id,
                smiles_column="original_smiles",
                id_column="molecule_id",
                role=role,
                source=source.label,
            )
        )
        report(index, len(sources), "Analyzing reference and background libraries")

    candidate_frame = descriptors[["molecule_id", "canonical_smiles"]].copy()
    candidate_frame["canonical_smiles"] = candidate_frame["canonical_smiles"].where(
        candidate_frame["canonical_smiles"].notna(), ""
    )
    candidate_library = prepare_library_frame(
        candidate_frame,
        "candidates",
        smiles_column="canonical_smiles",
        id_column="molecule_id",
    )
    duplicate_report = compare_exact_duplicates(
        candidate_library.molecules,
        libraries,
        candidate_smiles_column="canonical_smiles",
        candidate_id_column="reference_id",
    )
    similarity_result = compute_reference_similarity(
        candidate_library.molecules,
        libraries,
        candidate_smiles_column="canonical_smiles",
        candidate_id_column="reference_id",
        fingerprint_config=config.fingerprint_config,
        search=config.reference_search,
    )
    descriptors[duplicate_report.annotations.columns] = duplicate_report.annotations.reindex(
        descriptors.index
    )
    descriptors[similarity_result.annotations.columns] = similarity_result.annotations.reindex(
        descriptors.index
    )
    return (
        tuple(library for library in libraries if library.spec.role is LibraryRole.REFERENCE),
        tuple(library for library in libraries if library.spec.role is LibraryRole.BACKGROUND),
        duplicate_report,
        similarity_result,
    )


def _exclude_reference_duplicates(
    decision: DecisionResult, duplicate_report: ExactDuplicateReport
) -> DecisionResult:
    decisions = decision.decisions.copy()
    duplicate_ids = duplicate_report.annotations.index[
        duplicate_report.annotations["is_reference_duplicate"].astype(bool)
    ]
    duplicate_ids = decisions.index.intersection(duplicate_ids)
    if duplicate_ids.empty:
        return decision
    decisions.loc[duplicate_ids, "selected"] = False
    current = decisions.loc[duplicate_ids, "exclusion_reasons"]
    decisions.loc[duplicate_ids, "exclusion_reasons"] = [
        f"{reason}; exact reference duplicate" if reason else "exact reference duplicate"
        for reason in current
    ]
    return DecisionResult(decisions=decisions)


def _apply_selection_strategy(
    decision: DecisionResult,
    descriptors: pd.DataFrame,
    reference_libraries: tuple[ReferenceLibrary, ...],
    config: RunConfig,
) -> tuple[DecisionResult, tuple[int, ...], ZoneAllocationResult | None]:
    """Apply the optional Hub strategy layer and split final versus reserve IDs."""

    decisions = decision.decisions.copy()
    reasons = decisions["exclusion_reasons"].fillna("").astype(str)
    eligible = decisions.index[reasons.eq("")]
    strategy = config.selection_strategy
    ordered = list(eligible)

    if config.zones:
        zone_frame = descriptors.loc[eligible]
        zones = [
            zone_from_expression(
                zone_frame,
                str(item.get("expression", "")),
                zone_id=str(item["zone_id"]),
                name=str(item.get("name", item["zone_id"])),
                priority=int(item.get("priority", 0)),
                quota=(int(item["quota"]) if item.get("quota") is not None else None),
            )
            for item in config.zones
            if str(item.get("expression", "")).strip()
        ]
        ranking_column = (
            "reference_novelty" if "reference_novelty" in zone_frame.columns else "qed"
        )
        ranking = zone_frame[ranking_column].fillna(float("-inf")).to_dict()
        allocation = allocate_zones(
            zone_frame,
            zones,
            final_count=config.final_count if config.final_count is not None else len(eligible),
            reserve_count=config.reserve_count or 0,
            ranking=ranking,
        )
        final_ids = tuple(int(record_id) for record_id in allocation.final_ids)
        reserve_ids = tuple(int(record_id) for record_id in allocation.reserve_ids)
        _drop(
            decisions,
            eligible,
            set(final_ids),
            "not selected by zone quotas or final quota",
        )
        decisions["selection_status"] = "EXCLUDED"
        decisions.loc[list(final_ids), "selection_status"] = "FINAL_SELECTED"
        if reserve_ids:
            decisions.loc[list(reserve_ids), "selection_status"] = "RESERVE"
        return DecisionResult(decisions=decisions), reserve_ids, allocation

    if strategy in {"reference_novelty", "reference_aware_diversity"}:
        if "reference_novelty" in descriptors.columns:
            pool = eligible[descriptors.loc[eligible, "reference_novelty"].notna()]
            if config.exclude_reference_duplicates and "is_reference_duplicate" in descriptors:
                pool = pool[~descriptors.loc[pool, "is_reference_duplicate"].fillna(False).astype(bool)]
        else:
            pool = pd.Index([], dtype=eligible.dtype)
        if strategy == "reference_novelty":
            ordered = descriptors.loc[pool].sort_values(
                "reference_novelty", ascending=False, kind="mergesort"
            ).index.tolist()
        else:
            reference_smiles = [
                str(value)
                for library in reference_libraries
                for value in library.valid["canonical_smiles"].tolist()
            ]
            candidate_smiles = descriptors.loc[pool, "canonical_smiles"].fillna("").tolist()
            requested = (config.final_count if config.final_count is not None else len(pool)) + (
                config.reserve_count or 0
            )
            picked = reference_seeded_maxmin(
                candidate_smiles,
                reference_smiles,
                max(1, requested),
                fingerprint_config=config.fingerprint_config,
                seed=config.selection_seed,
            )
            ordered = [pool[position] for position in picked.picked_indices]
    elif strategy == "reference_neighborhood":
        if "max_reference_similarity" in descriptors.columns:
            pool = reference_neighborhood(
                descriptors.loc[eligible],
                minimum_similarity=config.reference_min_similarity,
                maximum_similarity=config.reference_max_similarity,
            )
            ordered = list(pool)
        else:
            ordered = []
    elif strategy == "diversity_first":
        candidate_smiles = descriptors.loc[eligible, "canonical_smiles"].fillna("").tolist()
        requested = (config.final_count if config.final_count is not None else len(eligible)) + (
            config.reserve_count or 0
        )
        picked = reference_seeded_maxmin(
            candidate_smiles,
            (),
            max(1, requested),
            fingerprint_config=config.fingerprint_config,
            seed=config.selection_seed,
        )
        ordered = [eligible[position] for position in picked.picked_indices]

    target = config.final_count if config.final_count is not None else len(ordered)
    target = min(max(target, 0), len(ordered))
    final_ids = tuple(int(record_id) for record_id in ordered[:target])
    reserve_start = target
    reserve_stop = reserve_start + (config.reserve_count or 0)
    reserve_ids = tuple(int(record_id) for record_id in ordered[reserve_start:reserve_stop])

    _drop(
        decisions,
        eligible,
        set(final_ids),
        f"not selected by {strategy.replace('_', ' ')} strategy or final quota",
    )
    decisions["selection_status"] = "EXCLUDED"
    decisions.loc[list(final_ids), "selection_status"] = "FINAL_SELECTED"
    if reserve_ids:
        decisions.loc[list(reserve_ids), "selection_status"] = "RESERVE"
    return DecisionResult(decisions=decisions), reserve_ids, None


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
        _drop(decisions, selected, keep, f"limit of {config.per_scaffold_limit} per scaffold")
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
            f"outside the diverse subset (MaxMin, n={config.diversity_pick})",
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

    report(len(reused), len(pairs), "Computing descriptors")
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
    *,
    reference_libraries: tuple[ReferenceLibrary, ...] = (),
    background_libraries: tuple[ReferenceLibrary, ...] = (),
    reference_duplicates: ExactDuplicateReport | None = None,
    reference_similarity: ReferenceSimilarityResult | None = None,
    reserve_ids: tuple[int, ...] = (),
    zone_allocation: ZoneAllocationResult | None = None,
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
        if reference_libraries or background_libraries:
            store.write_frame(
                pd.DataFrame(
                    [library.spec.as_dict() for library in (*reference_libraries, *background_libraries)]
                ),
                "reference_libraries",
            )
        if reference_duplicates is not None:
            store.write_frame(reference_duplicates.overlaps, "reference_overlap")
        if reference_similarity is not None:
            store.write_frame(reference_similarity.pairs, "reference_similarity")
        if reserve_ids:
            store.write_frame(
                pd.DataFrame({"record_id": list(reserve_ids)}),
                "reserve_selection",
            )
        if zone_allocation is not None:
            store.write_frame(zone_allocation.memberships, "zone_memberships")
            store.write_frame(zone_allocation.allocation_table, "zone_allocation")
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
