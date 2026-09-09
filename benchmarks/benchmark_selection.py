"""Reproducible count-constrained selection on cached descriptors, without chemistry work."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from smiles2select import APP_VERSION
from smiles2select.selection_intelligence.constrained_selection import (
    SelectionConstraints,
    Strategy,
    select,
)


@dataclass(frozen=True)
class BenchmarkConfig:
    count: int = 1000
    seed: int = 42
    repeats: int = 3
    thresholds: tuple[float, ...] = (0.0, 0.5, 0.7)
    max_per_scaffold: int | None = None

    def __post_init__(self):
        SelectionConstraints(target_count=self.count, max_per_scaffold=self.max_per_scaffold)
        if self.repeats < 1 or self.seed < 0:
            raise ValueError("repeats must be positive and seed nonnegative")
        if not self.thresholds or any(not 0 <= t <= 1 for t in self.thresholds):
            raise ValueError("QED thresholds must be finite numbers between 0 and 1")
        if len(set(self.thresholds)) != len(self.thresholds):
            raise ValueError("QED thresholds must be unique")


def validate_frame(frame: pd.DataFrame, activity_column: str | None = None) -> pd.DataFrame:
    """Only allow known selection features; activity lives in a separate nullable column."""
    required = ["record_id", "approved", "qed", "murcko_scaffold"]
    if activity_column in required:
        raise ValueError("activity must be separate from selection features")
    columns = required + ([activity_column] if activity_column else [])
    if any(column not in frame for column in columns):
        raise ValueError(f"Required columns: {columns}")
    if frame.empty:
        raise ValueError("Input must contain at least one molecule")
    data = frame.loc[:, columns].copy()
    ids = pd.to_numeric(data["record_id"], errors="raise")
    if ids.isna().any() or not np.isfinite(ids).all() or (ids != ids.astype("int64")).any():
        raise ValueError("record_id must contain finite integers")
    if ids.duplicated().any():
        raise ValueError("record_id must be unique")
    approved = pd.to_numeric(data["approved"], errors="raise")
    if not approved.isin([0, 1]).all():
        raise ValueError("approved must be 0 or 1; missing eligibility is not approval")
    qed = pd.to_numeric(data["qed"], errors="raise")
    if not qed.between(0, 1).all():
        raise ValueError("qed must be finite and between 0 and 1")
    data = data.assign(
        record_id=ids.astype("int64"),
        approved=approved.astype(bool),
        qed=qed,
        murcko_scaffold=data["murcko_scaffold"].replace(r"^\s*$", None, regex=True),
    )
    if activity_column:
        labels = pd.to_numeric(data[activity_column], errors="raise")
        if not labels.dropna().isin([0, 1]).all():
            raise ValueError("Activity labels must be 0, 1 or missing")
        data = data.drop(columns=activity_column).assign(activity_label=labels)
    return data.set_index("record_id")


def synthetic_frame(count: int, seed: int) -> pd.DataFrame:
    """Artificial independent descriptors/scaffold identifiers; no activity claims."""
    if count < 1:
        raise ValueError("molecules must be positive")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "approved": rng.random(count) >= 0.1,
            "qed": rng.random(count),
            "murcko_scaffold": pd.Categorical(rng.integers(0, max(1, count // 20), count)),
        },
        index=pd.RangeIndex(1, count + 1, name="record_id"),
    )


def _ids_hash(ids) -> str:
    return hashlib.sha256(np.sort(np.asarray(ids, dtype="<i8")).tobytes()).hexdigest()


def _activity_metrics(pool: pd.DataFrame, selected: pd.DataFrame) -> dict:
    keys = (
        "known_active_pool",
        "known_active_selected",
        "known_inactive_selected",
        "activity_unknown_selected",
        "active_recovery",
    )
    if "activity_label" not in pool:
        return dict.fromkeys(keys)
    labels = selected["activity_label"]
    active_pool = int(pool["activity_label"].eq(1).sum())
    active_selected = int(labels.eq(1).sum())
    return dict(
        zip(
            keys,
            (
                active_pool,
                active_selected,
                int(labels.eq(0).sum()),
                int(labels.isna().sum()),
                active_selected / active_pool if active_pool else None,
            ),
        )
    )


def _one_run(pool, config, strategy, repeat, threshold, universe):
    started = time.perf_counter()
    # ponytail: benchmark existing ranking primitives; no all-pairs chemical distances.
    candidates = pool.loc[:, ["qed", "murcko_scaffold"]]
    if strategy == "seeded_random":
        priorities = np.random.default_rng(config.seed + repeat).random(len(universe))
        candidates = candidates.assign(qed=priorities[universe.get_indexer(pool.index)])
    elif strategy == "rare_scaffolds_first":
        sizes = candidates["murcko_scaffold"].value_counts()
        candidates = candidates.assign(
            scaffold_size=candidates["murcko_scaffold"].map(sizes).astype(float)
        )
    constraints = SelectionConstraints(
        target_count=config.count, max_per_scaffold=config.max_per_scaffold
    )
    selected = select(
        candidates,
        constraints,
        Strategy.SCAFFOLD_COVERAGE if strategy == "rare_scaffolds_first" else Strategy.BALANCED,
        explain_rejections=False,
    )
    seconds = time.perf_counter() - started
    chosen = pool.loc[list(selected.selected_ids)]
    scaffolds_pool = int(pool["murcko_scaffold"].nunique())
    scaffolds_selected = int(chosen["murcko_scaffold"].nunique())
    return {
        "strategy": strategy,
        "repeat": repeat,
        "seed": config.seed + repeat,
        "qed_minimum": threshold,
        "eligible_count": len(pool),
        "eligible_sha256": _ids_hash(pool.index),
        "selected_count": selected.count,
        "shortfall": selected.shortfall(constraints),
        "scaffolds_pool": scaffolds_pool,
        "missing_scaffold_pool": int(pool["murcko_scaffold"].isna().sum()),
        "missing_scaffold_selected": int(chosen["murcko_scaffold"].isna().sum()),
        "scaffolds_selected": scaffolds_selected,
        "scaffold_coverage": scaffolds_selected / scaffolds_pool if scaffolds_pool else None,
        "selection_sha256": _ids_hash(selected.selected_ids),
        "seconds": round(seconds, 6),
        **_activity_metrics(pool, chosen),
    }, frozenset(selected.selected_ids)


def _stability(selections: dict, scenario_count: int) -> list[dict]:
    result = []
    for (strategy, repeat), scenarios in selections.items():
        counts = Counter(record_id for chosen in scenarios for record_id in chosen)
        union = len(counts)
        stable = sum(count == scenario_count for count in counts.values())
        result.append(
            {
                "strategy": strategy,
                "repeat": repeat,
                "scenario_count": scenario_count,
                "union_count": union,
                "stable_core_count": stable,
                "core_to_union": stable / union if union else None,
                "selection_frequency_histogram": dict(sorted(Counter(counts.values()).items())),
            }
        )
    return result


def run_benchmark(
    frame: pd.DataFrame,
    config: BenchmarkConfig,
    *,
    source: str,
    source_hash: str,
    source_kind: str = "cached_csv",
) -> dict:
    """Measure selectors fairly within each threshold's identical eligible pool."""
    runs, overlaps, selections = [], [], {}
    for threshold in config.thresholds:
        pool = frame.loc[frame["approved"] & frame["qed"].ge(threshold)]
        baseline = None
        for strategy in ("qed_ranked", "rare_scaffolds_first", "seeded_random"):
            for repeat in range(config.repeats if strategy == "seeded_random" else 1):
                row, chosen = _one_run(pool, config, strategy, repeat, threshold, frame.index)
                runs.append(row)
                selections.setdefault((strategy, repeat), []).append(chosen)
                if baseline is None:
                    baseline = chosen
                union = len(baseline | chosen)
                overlaps.append(
                    {
                        "qed_minimum": threshold,
                        "strategy": strategy,
                        "repeat": repeat,
                        "reference": "qed_ranked",
                        "jaccard": len(baseline & chosen) / union if union else None,
                    }
                )
    return {
        "schema_version": 1,
        "source": {"path": source, "sha256": source_hash, "kind": source_kind},
        "config": asdict(config),
        "molecule_count": len(frame),
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "smiles2select": APP_VERSION,
            "benchmark_sha256": _file_hash(Path(__file__)),
            "selector_sha256": _file_hash(Path(sys.modules[select.__module__].__file__)),
            **{name: version(name) for name in ("numpy", "pandas")},
        },
        "runs": runs,
        "overlaps": overlaps,
        "stability": _stability(selections, len(config.thresholds)),
        "limitations": [
            "Chemistry parsing, standardization, descriptor calculation and GUI are not measured.",
            "Synthetic mode tests scaling only; synthetic scaffold IDs are not chemical structures.",
            "Activity labels never rank candidates; missing labels are unknown, never inactive.",
            "Missing scaffold identifiers are counted separately; scaffold quotas cannot constrain them.",
            "Known-active recovery denominator is the eligible pool in that scenario, not the library.",
            "Stability frequency is decision sensitivity, not probability of biological activity.",
            "Timings include ordering and quotas, exclude input loading and metrics; compare on same hardware.",
            "RSS sampled for the whole process, not each method; 20 ms sampling can miss transient peaks.",
        ],
    }


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--synthetic", type=int, metavar="MOLECULES")
    parser.add_argument("--activity-column")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.0, 0.5, 0.7])
    parser.add_argument("--max-per-scaffold", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".json":
        parser.error("--output must have a .json extension")
    if args.input and args.input.resolve() in (
        args.output.resolve(),
        args.output.with_suffix(".csv").resolve(),
    ):
        parser.error("Output paths must differ from the input CSV")
    process = psutil.Process()
    samples = [process.memory_info().rss]
    stopped = threading.Event()

    def sample():
        while not stopped.wait(0.02):
            samples[:] = [samples[0], max(samples[-1], process.memory_info().rss)]

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        config = BenchmarkConfig(
            args.count, args.seed, args.repeats, tuple(args.thresholds), args.max_per_scaffold
        )
        if args.input:
            columns = ["record_id", "approved", "qed", "murcko_scaffold"]
            if args.activity_column:
                columns.append(args.activity_column)
            raw = pd.read_csv(args.input, usecols=columns, dtype={"record_id": "string"})
            frame = validate_frame(raw, args.activity_column)
            del raw
            source_hash = _file_hash(args.input)
        else:
            if args.activity_column:
                raise ValueError("Synthetic mode has no activity labels")
            frame = synthetic_frame(args.synthetic, args.seed)
            source_hash = hashlib.sha256(
                pd.util.hash_pandas_object(frame).values.tobytes()
            ).hexdigest()
        report = run_benchmark(
            frame,
            config,
            source=str(args.input or "synthetic"),
            source_hash=source_hash,
            source_kind="cached_csv" if args.input else "synthetic_smoke",
        )
        report["memory"] = {
            "process_rss_baseline_mib": samples[0] / 2**20,
            "process_rss_peak_sampled_mib": max(samples[-1], process.memory_info().rss) / 2**20,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(report, indent=2, allow_nan=False)
        args.output.write_text(payload + "\n", encoding="utf-8")
        pd.DataFrame(report["runs"]).to_csv(args.output.with_suffix(".csv"), index=False)
        print(payload)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    finally:
        stopped.set()
        sampler.join()
    return 0


if __name__ == "__main__":
    sys.exit(main())
