"""Local commands; optional training dependencies are imported only when requested."""

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .artifacts import file_hash, read_bundle, spreadsheet_safe, write_bundle, write_json


def parser():
    root = argparse.ArgumentParser(
        description="S2S-Decision Tiny: experimental local prioritization"
    )
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in (
        "audit",
        "prepare",
        "dataset",
        "train",
        "benchmark",
        "evaluate",
        "predict",
        "select",
        "train-baseline",
        "models",
        "preview",
        "adopt",
    ):
        sub = commands.add_parser(name)
        sub.add_argument("--input", required=True, type=Path)
        sub.add_argument("--output", required=True, type=Path)
        if name in ("audit", "dataset"):
            sub.add_argument("--threshold", type=float, default=6.0)
        if name in ("prepare", "dataset"):
            sub.add_argument("--fingerprint-bits", type=int, choices=(1024, 2048), default=2048)
        if name == "dataset":
            sub.add_argument("--target", required=True)
            sub.add_argument("--endpoint", required=True, choices=("Ki", "Kd", "IC50", "EC50"))
            sub.add_argument(
                "--split", choices=("scaffold", "temporal", "random"), default="scaffold"
            )
            sub.add_argument("--max-rows", type=int, default=250000)
        if name in ("dataset", "train", "train-baseline"):
            sub.add_argument("--seed", type=int, default=42)
        if name == "train-baseline":
            sub.add_argument(
                "--estimator", choices=("logistic", "gradient_boosting"), required=True
            )
            sub.add_argument("--input-layout", choices=("scalar", "scalar_fingerprint"))
            sub.add_argument("--threads", type=int, default=4)
        if name == "models":
            sub.add_argument("--candidates", type=Path)
            sub.add_argument("--target")
            sub.add_argument("--endpoint", choices=("Ki", "Kd", "IC50", "EC50"))
        if name == "preview":
            sub.add_argument("--model", type=Path)
        if name == "train":
            sub.add_argument("--epochs", type=int, default=200)
            sub.add_argument("--patience", type=int, default=20)
            sub.add_argument("--learning-rate", type=float, default=1e-3)
            sub.add_argument("--weight-decay", type=float, default=1e-4)
            sub.add_argument("--width-multiplier", type=int, choices=(1, 2), default=1)
            sub.add_argument("--threads", type=int, default=4)
            sub.add_argument("--resume", action="store_true")
        if name in ("train", "predict"):
            sub.add_argument("--batch-size", type=int, default=1024)
        if name in ("evaluate", "predict"):
            sub.add_argument("--model", type=Path, required=True)
        if name in ("benchmark", "evaluate", "select", "preview"):
            sub.add_argument("--n", type=int, required=True)
        if name in ("select", "preview"):
            sub.add_argument("--max-per-scaffold", type=int)
            sub.add_argument("--min-scaffolds", type=int)
            sub.add_argument("--pins", nargs="*", type=int, default=[])
            sub.add_argument("--exclude", nargs="*", type=int, default=[])
        if name == "select":
            sub.add_argument("--max-per-cluster", type=int)
    return root


def _run(args):
    from . import workflows

    name = args.command
    if args.output.exists() and not (name == "train" and args.resume):
        raise ValueError("output already exists; choose a new path (train supports --resume)")
    if name == "train-baseline":
        return workflows.train_baseline_dataset(
            args.input,
            args.output,
            estimator=args.estimator,
            seed=args.seed,
            threads=args.threads,
            input_layout=args.input_layout,
        )
    if name in ("models", "preview", "adopt"):
        from .decision import export_decision, list_models, preview_decision

        if name == "models":
            report = {
                "models": list_models(
                    args.input,
                    candidates_dir=args.candidates,
                    target=args.target,
                    endpoint=args.endpoint,
                )
            }
            write_json(args.output, report)
            return report
        if name == "adopt":
            return export_decision(args.input, args.output)
        return preview_decision(
            args.input,
            args.output,
            model_dir=args.model,
            n=args.n,
            max_per_scaffold=args.max_per_scaffold,
            min_scaffolds=args.min_scaffolds,
            pins=tuple(args.pins),
            exclude=tuple(args.exclude),
        )
    if name == "audit":
        from .data import audit_source

        result = audit_source(args.input, args.threshold)
        write_json(args.output, result)
        return result
    if name == "prepare":
        return workflows.prepare_candidates(args.input, args.output, args.fingerprint_bits)
    if name == "dataset":
        return workflows.prepare_dataset(
            args.input,
            args.output,
            args.target,
            args.endpoint,
            args.threshold,
            args.split,
            args.seed,
            args.fingerprint_bits,
            args.max_rows,
        )
    if name == "train":
        from .training import TrainingConfig

        config = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            seed=args.seed,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            resume=args.resume,
            width_multiplier=args.width_multiplier,
        )
        return workflows.train_dataset(args.input, args.output, config, args.threads)
    if name == "predict":
        return workflows.score_candidates(args.input, args.model, args.output, args.batch_size)
    if name == "evaluate":
        report = workflows.evaluate_dataset(args.input, args.model, args.n)
        write_json(args.output, report)
        return report
    if name == "benchmark":
        from .training import benchmark_baselines

        dataset = read_bundle(args.input)
        report = benchmark_baselines(
            dataset.records, n=args.n, fingerprint_bits=dataset.manifest["fingerprint_bits"]
        )
        write_json(
            args.output, {"dataset_sha256": dataset.manifest["records_sha256"], "baselines": report}
        )
        return report
    return _select(args)


def _select(args):
    from .selection import select_candidates

    source = read_bundle(args.input)
    selected = select_candidates(
        source,
        args.n,
        args.max_per_scaffold,
        args.min_scaffolds,
        args.max_per_cluster,
        tuple(args.pins),
        tuple(args.exclude),
    )
    write_bundle(selected, args.output)
    final = selected.records.loc[selected.records.is_final]
    final_path = args.output / "final.csv"
    spreadsheet_safe(final).to_csv(final_path, index=False)
    # Keep full molecular provenance; this table does not overwrite upstream baskets.
    write_json(
        args.output / "selection.json",
        {
            **selected.manifest,
            "final_csv_sha256": file_hash(final_path),
            "csv_text_escaping": "spreadsheet formula prefixes escaped with apostrophe; raw text retained in records.jsonl",
        },
    )
    return {
        "output": str(args.output),
        "requested": args.n,
        "final": len(final),
        "warnings": selected.manifest.get("warnings", []),
    }


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        report = _run(args)
    except (ValueError, OSError, ImportError, KeyError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))
    return 0
