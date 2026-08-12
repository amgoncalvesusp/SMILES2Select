"""Command-line interface.

Example::

    smiles2select library.xlsx --smiles-column SMILES --id-column ID \\
        --profiles lipinski,veber,ghose,egan,muegge \\
        --mandatory lipinski,veber --excel results.xlsx
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

from smiles2select.alerts.custom_smarts import SmartsAlert
from smiles2select.alerts.policies import AlertPolicy
from smiles2select.app_metadata import APP_NAME, APP_VERSION, DISCLAIMER
from smiles2select.chemistry.fingerprints import FingerprintConfig
from smiles2select.chemistry.standardization import StandardizationConfig
from smiles2select.decision.explanations import policy_sentence, restrictiveness_warning
from smiles2select.decision.policies import DecisionPolicy, ProfileRole
from smiles2select.export import docking, excel, parquet
from smiles2select.io.importer import ColumnMapping, SourceFile, guess_mapping, preview_columns
from smiles2select.pipeline import presets
from smiles2select.pipeline.config import RunConfig
from smiles2select.pipeline.diagnostics import ParallelDiagnosticsConfig, safe_mode_config
from smiles2select.pipeline.runner import RunResult, run
from smiles2select.profiles.loader import builtin_registry
from smiles2select.scores.qed import QedSelection
from smiles2select.selection_intelligence import recipes

DEFAULT_PROFILES = ("lipinski", "veber", "ghose", "egan", "muegge")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smiles2select",
        description=f"{APP_NAME} {APP_VERSION} — multi-rule drug-likeness selector",
        epilog=DISCLAIMER,
    )
    parser.add_argument(
        "inputs", nargs="*", type=Path, help="input files (.csv, .tsv, .smi, .xlsx)"
    )
    parser.add_argument("--sheet", default=None, help="spreadsheet sheet name")
    parser.add_argument("--smiles-column", default=None, help="column holding the SMILES")
    parser.add_argument("--id-column", default=None, help="column holding the molecule id")
    parser.add_argument(
        "--reference",
        action="append",
        type=Path,
        default=[],
        metavar="FILE",
        help="reference library file (repeatable; references are never selected)",
    )
    parser.add_argument(
        "--background",
        action="append",
        type=Path,
        default=[],
        metavar="FILE",
        help="context/background library file (repeatable; never selected)",
    )
    parser.add_argument(
        "--reference-smiles-column",
        default=None,
        help="SMILES column in reference libraries",
    )
    parser.add_argument(
        "--reference-id-column",
        default=None,
        help="identifier column in reference libraries",
    )
    parser.add_argument("--background-smiles-column", default=None)
    parser.add_argument("--background-id-column", default=None)
    parser.add_argument(
        "--reference-search",
        choices=("exact", "fast"),
        default="exact",
        help="reference nearest-neighbor discovery mode",
    )
    parser.add_argument(
        "--exclude-reference-duplicates",
        action="store_true",
        help="exclude exact candidate/reference duplicates from the final selection",
    )
    parser.add_argument(
        "--reference-min-similarity", type=float, default=0.0,
        help="lower bound for reference-neighborhood selection",
    )
    parser.add_argument(
        "--reference-max-similarity", type=float, default=1.0,
        help="upper bound for reference-neighborhood selection",
    )
    parser.add_argument(
        "--selection-strategy",
        choices=(
            "traditional", "balanced", "diversity_first", "reference_novelty",
            "reference_neighborhood", "reference_aware_diversity", "stratified", "manual_assisted",
        ),
        default="traditional",
        help="optional strategy layer applied after chemical eligibility",
    )
    parser.add_argument("--final-count", type=int, default=None)
    parser.add_argument("--reserve-count", type=int, default=None)
    parser.add_argument("--selection-seed", type=int, default=0xF00D)
    parser.add_argument(
        "--profiles",
        default=",".join(DEFAULT_PROFILES),
        help="comma-separated profile ids to evaluate",
    )
    parser.add_argument(
        "--mandatory",
        default="lipinski,veber",
        help="profiles that must pass for a molecule to be selected",
    )
    parser.add_argument(
        "--consensus",
        default=None,
        metavar="N:PROFILES",
        help="approve on N of M profiles, e.g. 3:lipinski,veber,ghose,egan,muegge",
    )
    parser.add_argument("--expression", default=None, help="custom selection expression")
    parser.add_argument(
        "--qed-mode",
        choices=("compute", "rank", "threshold", "top_percentile"),
        default="rank",
    )
    parser.add_argument("--qed-threshold", type=float, default=None)
    parser.add_argument("--qed-percentile", type=float, default=None)
    parser.add_argument(
        "--alerts", default="pains,brenk", help="comma-separated alert catalogs ('none' disables)"
    )
    parser.add_argument(
        "--alert-action",
        action="append",
        default=[],
        metavar="CATALOG=ACTION",
        help="inform|warn|penalize|exclude per catalog (repeatable)",
    )
    parser.add_argument("--custom-smarts", action="append", default=[], metavar="NAME=SMARTS")
    parser.add_argument("--no-qed", action="store_true", help="do not compute QED")
    parser.add_argument(
        "--sa-score", action="store_true", help="compute synthetic accessibility (1 easy - 10 hard)"
    )
    parser.add_argument(
        "--np-score",
        action="store_true",
        help="compute natural-product likeness (about -5 synthetic to +5 natural-like)",
    )
    parser.add_argument(
        "--diverse",
        type=int,
        default=None,
        metavar="N",
        help="after selection, keep a diverse subset of N molecules (MaxMin)",
    )
    parser.add_argument(
        "--per-scaffold",
        type=int,
        default=None,
        metavar="N",
        help="after selection, keep at most N molecules per Murcko scaffold",
    )
    parser.add_argument(
        "--docking",
        type=Path,
        default=None,
        help="write the selection as a SMILES2Docking input sheet",
    )
    parser.add_argument(
        "--docking-all",
        action="store_true",
        help="hand every valid molecule to docking, not only the selected ones",
    )
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--database", type=Path, default=None, help="SQLite output path")
    parser.add_argument("--excel", type=Path, default=None, help="Excel report path")
    parser.add_argument("--parquet", type=Path, default=None, help="Parquet output path")
    parser.add_argument(
        "--parquet-selected", action="store_true", help="write only selected molecules to Parquet"
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="persistent descriptor cache reused across runs",
    )
    parser.add_argument(
        "--compact", action="store_true", help="single RULE_FAILURES sheet instead of one per rule"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="worker processes (-1 = auto, sized from available memory and CPU count)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=2000, help="initial chunk size (adapts during the run)"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="checkpoint file; rerunning the same command resumes from the last completed chunk",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="safe-diagnostics mode: 1 worker, small chunks, full fault/memory logging",
    )
    parser.add_argument(
        "--preset", type=Path, default=None, help="load a saved preset (profiles, roles, alerts)"
    )
    parser.add_argument(
        "--save-preset", type=Path, default=None, help="write the resolved settings as a preset"
    )
    parser.add_argument(
        "--selection-plan",
        type=Path,
        default=None,
        help="replay strategy, fingerprint, final/reserve counts and seed from a selection recipe",
    )
    parser.add_argument(
        "--save-selection-plan",
        type=Path,
        default=None,
        help="save the resolved run as a machine-readable selection recipe",
    )
    parser.add_argument(
        "--profile-dir",
        action="append",
        type=Path,
        default=[],
        metavar="DIR",
        help="extra directory with custom profile JSON files (repeatable)",
    )
    parser.add_argument("--list-profiles", action="store_true", help="list profiles and exit")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_profiles:
        _print_profiles()
        return 0
    if not args.inputs:
        parser.error("at least one input file is required")

    try:
        config = _build_config(args)
    except (ValueError, KeyError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    if args.save_preset is not None:
        presets.save(_preset_from_config(config, args.save_preset.stem), args.save_preset)

    if not args.quiet:
        print(policy_sentence(config.policy))
        warning = restrictiveness_warning(config.policy)
        if warning:
            print(f"WARNING: {warning}")

    progress = None if args.quiet else _print_progress
    try:
        result = run(config, progress)
    except Exception as exc:
        print(f"run failed: {exc}", file=sys.stderr)
        return 1

    if args.save_selection_plan is not None:
        recipes.save(_recipe_from_result(result), args.save_selection_plan)

    if config.excel_path is not None:
        excel.export(result, config.excel_path, excel.ExportOptions(detailed=not args.compact))

    if config.parquet_path is not None:
        try:
            parquet.export(result, config.parquet_path, selected_only=args.parquet_selected)
        except parquet.ParquetUnavailableError as exc:
            print(f"parquet export skipped: {exc}", file=sys.stderr)
            return 1

    if args.docking is not None:
        try:
            docking.export(
                result,
                args.docking,
                docking.DockingExportOptions(selected_only=not args.docking_all),
            )
        except ValueError as exc:
            print(f"docking export failed: {exc}", file=sys.stderr)
            return 1

    _print_report(result, quiet=args.quiet)
    return 0


def _build_config(args: argparse.Namespace) -> RunConfig:
    """Resolve the command line, and the preset it may point at, into a config.

    An explicit flag always wins over the preset: the preset supplies the
    settings the user did not state on this command line.
    """
    if args.selection_plan is not None:
        return _config_from_selection_plan(args, recipes.load(args.selection_plan))

    preset = presets.load(args.preset) if args.preset else None
    if preset is not None:
        available = builtin_registry(tuple(args.profile_dir)).ids()
        problems = preset.validate(available)
        if problems:
            raise ValueError("\n".join(problems))
        return _config_from_preset(args, preset)

    profile_ids = tuple(_split(args.profiles))
    return RunConfig(
        sources=tuple(_build_sources(args)),
        profile_ids=profile_ids,
        policy=_build_policy(args, profile_ids),
        alert_catalogs=tuple(_split(args.alerts)) if args.alerts != "none" else (),
        custom_alerts=tuple(_build_custom_alerts(args.custom_smarts)),
        compute_qed=not args.no_qed,
        compute_sa=args.sa_score,
        compute_np=args.np_score,
        diversity_pick=args.diverse,
        per_scaffold_limit=args.per_scaffold,
        drop_duplicates=not args.keep_duplicates,
        profile_directories=tuple(args.profile_dir),
        n_jobs=args.jobs,
        chunk_size=args.chunk_size,
        database_path=args.database,
        excel_path=args.excel,
        parquet_path=args.parquet,
        cache_path=args.cache,
        detailed_export=not args.compact,
        checkpoint_path=args.checkpoint,
        diagnostics=_build_diagnostics(args),
        reference_sources=tuple(_build_reference_sources(args)),
        background_sources=tuple(_build_background_sources(args)),
        reference_search=args.reference_search,
        exclude_reference_duplicates=args.exclude_reference_duplicates,
        reference_min_similarity=args.reference_min_similarity,
        reference_max_similarity=args.reference_max_similarity,
        selection_strategy=args.selection_strategy,
        final_count=args.final_count,
        reserve_count=args.reserve_count,
        selection_seed=args.selection_seed,
    )


def _config_from_selection_plan(args: argparse.Namespace, recipe: recipes.SelectionRecipe) -> RunConfig:
    """Replay the deterministic Hub layer while letting the caller choose new inputs."""
    profile_ids = tuple(_split(args.profiles))
    standardization_fields = set(StandardizationConfig.__dataclass_fields__)
    standardization_payload = {
        key: value for key, value in recipe.standardization.items() if key in standardization_fields
    }
    fingerprint_payload = recipe.fingerprint
    fingerprint = FingerprintConfig(
        radius=int(fingerprint_payload.get("radius", 2)),
        size=int(fingerprint_payload.get("bits", fingerprint_payload.get("size", 2048))),
        use_chirality=bool(fingerprint_payload.get("use_chirality", False)),
    )
    reference_paths = list(_build_reference_sources(args))
    if not reference_paths:
        for item in recipe.reference_libraries:
            source = Path(str(item.get("source", "")))
            if source.exists():
                reference_paths.append(
                SourceFile(path=source, mapping=guess_mapping(preview_columns(source)))
                )
    background_paths = list(_build_background_sources(args))
    if not background_paths:
        for item in recipe.background_libraries:
            source = Path(str(item.get("source", "")))
            if source.exists():
                background_paths.append(
                    SourceFile(path=source, mapping=guess_mapping(preview_columns(source)))
                )
    return RunConfig(
        sources=tuple(_build_sources(args)),
        profile_ids=profile_ids,
        policy=_build_policy(args, profile_ids),
        standardization=StandardizationConfig(**standardization_payload),
        alert_catalogs=tuple(_split(args.alerts)) if args.alerts != "none" else (),
        custom_alerts=tuple(_build_custom_alerts(args.custom_smarts)),
        compute_qed=not args.no_qed,
        compute_sa=args.sa_score,
        compute_np=args.np_score,
        diversity_pick=args.diverse,
        per_scaffold_limit=args.per_scaffold,
        fingerprint_config=fingerprint,
        drop_duplicates=not args.keep_duplicates,
        profile_directories=tuple(args.profile_dir),
        n_jobs=args.jobs,
        chunk_size=args.chunk_size,
        database_path=args.database,
        excel_path=args.excel,
        parquet_path=args.parquet,
        cache_path=args.cache,
        detailed_export=not args.compact,
        checkpoint_path=args.checkpoint,
        diagnostics=_build_diagnostics(args),
        reference_sources=tuple(reference_paths),
        background_sources=tuple(background_paths),
        reference_search=("fast" if "HNSW" in str(fingerprint_payload.get("search", "")) else args.reference_search),
        exclude_reference_duplicates=args.exclude_reference_duplicates,
        reference_min_similarity=args.reference_min_similarity,
        reference_max_similarity=args.reference_max_similarity,
        selection_strategy=recipe.strategy,
        final_count=recipe.target_count,
        reserve_count=recipe.reserve_count,
        selection_seed=recipe.seed if recipe.seed is not None else args.selection_seed,
        zones=tuple(recipe.zones),
    )


def _recipe_from_result(result: RunResult) -> recipes.SelectionRecipe:
    """Capture the run-level settings needed to replay its Hub decision."""
    return recipes.SelectionRecipe(
        name="cli-selection",
        input_hash=result.config.fingerprint(),
        required_filters=tuple(result.config.profile_ids),
        target_count=result.config.final_count,
        strategy=result.config.selection_strategy,
        reference_libraries=tuple(
            library.spec.as_dict() for library in result.reference_libraries
        ),
        background_libraries=tuple(
            library.spec.as_dict() for library in result.background_libraries
        ),
        standardization=asdict(result.config.standardization),
        fingerprint={
            "radius": result.config.fingerprint_config.radius,
            "bits": result.config.fingerprint_config.size,
            "use_chirality": result.config.fingerprint_config.use_chirality,
            "search": result.config.reference_search,
        },
        reserve_count=result.config.reserve_count,
        seed=result.config.selection_seed,
        zones=tuple(result.config.zones),
    )


def _build_diagnostics(args: argparse.Namespace) -> ParallelDiagnosticsConfig:
    return safe_mode_config() if args.diagnose else ParallelDiagnosticsConfig()


def _config_from_preset(args: argparse.Namespace, preset: presets.RunPreset) -> RunConfig:
    """Build the run configuration from a preset plus this command's paths."""
    return RunConfig(
        sources=tuple(_build_sources(args)),
        profile_ids=preset.profile_ids(),
        policy=preset.build_policy(),
        standardization=preset.standardization,
        alert_catalogs=preset.active_catalogs,
        custom_alerts=preset.custom_alerts,
        compute_qed=preset.compute_qed,
        compute_sa=args.sa_score,
        compute_np=args.np_score,
        diversity_pick=args.diverse,
        per_scaffold_limit=args.per_scaffold,
        drop_duplicates=preset.drop_duplicates,
        profile_directories=tuple(args.profile_dir),
        n_jobs=args.jobs,
        chunk_size=args.chunk_size,
        database_path=args.database,
        excel_path=args.excel,
        parquet_path=args.parquet,
        cache_path=args.cache,
        detailed_export=preset.detailed_export and not args.compact,
        checkpoint_path=args.checkpoint,
        diagnostics=_build_diagnostics(args),
        reference_sources=tuple(_build_reference_sources(args)),
        background_sources=tuple(_build_background_sources(args)),
        reference_search=args.reference_search,
        exclude_reference_duplicates=args.exclude_reference_duplicates,
        reference_min_similarity=args.reference_min_similarity,
        reference_max_similarity=args.reference_max_similarity,
        selection_strategy=args.selection_strategy,
        final_count=args.final_count,
        reserve_count=args.reserve_count,
        selection_seed=args.selection_seed,
    )


def _build_sources(args: argparse.Namespace) -> list[SourceFile]:
    sources: list[SourceFile] = []
    for path in args.inputs:
        if args.smiles_column:
            mapping = ColumnMapping(smiles=args.smiles_column, molecule_id=args.id_column)
        else:
            mapping = guess_mapping(preview_columns(path, args.sheet))
            if args.id_column:
                mapping = ColumnMapping(smiles=mapping.smiles, molecule_id=args.id_column)
        sources.append(SourceFile(path=Path(path), mapping=mapping, sheet=args.sheet))
    return sources


def _build_reference_sources(args: argparse.Namespace) -> list[SourceFile]:
    sources: list[SourceFile] = []
    for path in args.reference:
        if args.reference_smiles_column:
            mapping = ColumnMapping(
                smiles=args.reference_smiles_column,
                molecule_id=args.reference_id_column,
            )
        else:
            mapping = guess_mapping(preview_columns(path, args.sheet))
            if args.reference_id_column:
                mapping = ColumnMapping(smiles=mapping.smiles, molecule_id=args.reference_id_column)
        sources.append(SourceFile(path=Path(path), mapping=mapping, sheet=args.sheet))
    return sources


def _build_background_sources(args: argparse.Namespace) -> list[SourceFile]:
    sources: list[SourceFile] = []
    for path in args.background:
        if args.background_smiles_column:
            mapping = ColumnMapping(
                smiles=args.background_smiles_column,
                molecule_id=args.background_id_column,
            )
        else:
            mapping = guess_mapping(preview_columns(path, args.sheet))
            if args.background_id_column:
                mapping = ColumnMapping(
                    smiles=mapping.smiles,
                    molecule_id=args.background_id_column,
                )
        sources.append(SourceFile(path=Path(path), mapping=mapping, sheet=args.sheet))
    return sources


def _build_policy(args: argparse.Namespace, profile_ids: tuple[str, ...]) -> DecisionPolicy:
    roles: dict[str, ProfileRole] = dict.fromkeys(profile_ids, "informative")
    consensus_min: int | None = None

    for profile_id in _split(args.mandatory):
        if profile_id not in profile_ids:
            raise ValueError(f"mandatory profile '{profile_id}' is not in --profiles")
        roles[profile_id] = "mandatory"

    if args.consensus:
        count, _, listed = args.consensus.partition(":")
        consensus_min = int(count)
        for profile_id in _split(listed):
            if profile_id not in profile_ids:
                raise ValueError(f"consensus profile '{profile_id}' is not in --profiles")
            roles[profile_id] = "consensus"

    alert_policy = AlertPolicy()
    for item in args.alert_action:
        catalog_id, _, action = item.partition("=")
        alert_policy = alert_policy.with_action(catalog_id.strip(), action.strip())

    return DecisionPolicy(
        id="cli",
        roles=roles,
        consensus_min_pass=consensus_min,
        expression=args.expression,
        qed=QedSelection(
            mode=args.qed_mode, threshold=args.qed_threshold, percentile=args.qed_percentile
        ),
        alert_policy=alert_policy,
    )


def _preset_from_config(config: RunConfig, name: str) -> presets.RunPreset:
    """Capture a resolved configuration as a reusable preset."""
    return presets.RunPreset(
        name=name,
        roles=dict(config.policy.roles),
        consensus_min_pass=config.policy.consensus_min_pass,
        expression=config.policy.expression,
        qed=config.policy.qed,
        active_catalogs=config.alert_catalogs,
        alert_actions=dict(config.policy.alert_policy.actions),
        custom_alerts=config.custom_alerts,
        standardization=config.standardization,
        compute_qed=config.compute_qed,
        drop_duplicates=config.drop_duplicates,
        detailed_export=config.detailed_export,
    )


def _build_custom_alerts(items: list[str]) -> list[SmartsAlert]:
    alerts: list[SmartsAlert] = []
    for index, item in enumerate(items, start=1):
        name, _, smarts = item.partition("=")
        alert = SmartsAlert(id=f"custom_{index}", name=name.strip(), smarts=smarts.strip())
        alert.compile_pattern()  # fail before the run, not during it
        alerts.append(alert)
    return alerts


def _split(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _print_profiles() -> None:
    for profile in builtin_registry().all():
        print(
            f"{profile.id:<16} {profile.name:<45} {profile.category:<20} {profile.policy_label()}"
        )
        if profile.notes:
            print(f"{'':<16} {profile.notes}")


def _print_progress(done: int, total: int, stage: str) -> None:
    if total <= 0:
        print(f"  {stage}...")
        return
    print(f"  {stage}: {done}/{total}", end="\r", flush=True)
    if done >= total:
        print()


def _print_report(result: RunResult, *, quiet: bool) -> None:
    if quiet:
        print(result.decision.selected_count)
        return
    print()
    print(f"Policy:              {result.config.policy.id}")
    print(f"Records processed:   {result.total_records}")
    print(f"Invalid:             {result.invalid_count}")
    print(f"Duplicates:          {result.duplicate_count}")
    print(f"Evaluated:           {result.evaluated_count}")
    print(f"Final selected:      {result.decision.selected_count}")
    print(f"Final excluded:      {result.decision.excluded_count}")
    print()
    print(result.profile_summary().to_string(index=False))
    if result.diversity is not None:
        print()
        for label, value in result.diversity.summary_rows():
            print(f"  {label}: {value}")
    if result.cache_stats is not None:
        hits, misses = result.cache_stats
        print(f"\nCache: {hits} reused, {misses} computed")
    if result.database_path:
        print(f"\nDatabase: {result.database_path}")
    if result.config.excel_path:
        print(f"Excel: {result.config.excel_path}")
    if result.config.parquet_path:
        print(f"Parquet: {result.config.parquet_path}")


if __name__ == "__main__":
    # See gui/app.py for why this is required before anything touches
    # multiprocessing: a frozen build re-executes this module per worker.
    from multiprocessing import freeze_support

    freeze_support()
    raise SystemExit(main())
