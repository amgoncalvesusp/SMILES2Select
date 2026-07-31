"""Command-line interface.

Example::

    smiles2select library.xlsx --smiles-column SMILES --id-column ID \\
        --profiles lipinski,veber,ghose,egan,muegge \\
        --mandatory lipinski,veber --excel results.xlsx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from smiles2select.alerts.custom_smarts import SmartsAlert
from smiles2select.alerts.policies import AlertPolicy
from smiles2select.app_metadata import APP_NAME, APP_VERSION, DISCLAIMER
from smiles2select.decision.explanations import policy_sentence, restrictiveness_warning
from smiles2select.decision.policies import DecisionPolicy, ProfileRole
from smiles2select.export import docking, excel, parquet
from smiles2select.io.importer import ColumnMapping, SourceFile, guess_mapping, preview_columns
from smiles2select.pipeline import presets
from smiles2select.pipeline.config import RunConfig
from smiles2select.pipeline.runner import RunResult, run
from smiles2select.profiles.loader import builtin_registry
from smiles2select.scores.qed import QedSelection

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
    parser.add_argument("--jobs", type=int, default=-1, help="worker processes (-1 = all cores)")
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument(
        "--preset", type=Path, default=None, help="load a saved preset (profiles, roles, alerts)"
    )
    parser.add_argument(
        "--save-preset", type=Path, default=None, help="write the resolved settings as a preset"
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
            print(f"AVISO: {warning}")

    progress = None if args.quiet else _print_progress
    try:
        result = run(config, progress)
    except Exception as exc:
        print(f"run failed: {exc}", file=sys.stderr)
        return 1

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
    )


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
    print(f"Política:            {result.config.policy.id}")
    print(f"Total processado:    {result.total_records}")
    print(f"Inválidos:           {result.invalid_count}")
    print(f"Duplicatas:          {result.duplicate_count}")
    print(f"Avaliados:           {result.evaluated_count}")
    print(f"Selecionados finais: {result.decision.selected_count}")
    print(f"Excluídos finais:    {result.decision.excluded_count}")
    print()
    print(result.profile_summary().to_string(index=False))
    if result.diversity is not None:
        print()
        for label, value in result.diversity.summary_rows():
            print(f"  {label}: {value}")
    if result.cache_stats is not None:
        hits, misses = result.cache_stats
        print(f"\nCache: {hits} reaproveitadas, {misses} calculadas")
    if result.database_path:
        print(f"\nBanco: {result.database_path}")
    if result.config.excel_path:
        print(f"Excel: {result.config.excel_path}")
    if result.config.parquet_path:
        print(f"Parquet: {result.config.parquet_path}")


if __name__ == "__main__":
    raise SystemExit(main())
