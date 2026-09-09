# Reproducible selection benchmark

`benchmarks/benchmark_selection.py` compares cached-descriptor selection under a
**molecule count** limit. It needs no downloads, additional dependencies, prices,
currency conversion or regional cost assumptions. Run from the source checkout:

```powershell
$env:PYTHONPATH = "src"
python benchmarks/benchmark_selection.py --input cached.csv --count 1000 --max-per-scaffold 5 --activity-column active --output benchmark.json
python benchmarks/benchmark_selection.py --synthetic 1000000 --count 1000 --max-per-scaffold 5 --output smoke.json
```

The tool writes JSON with provenance, comparison and stability metrics, plus a
CSV containing one row per selection. Output must end in `.json`; neither output
may overwrite the input CSV. Retain both output files and the exact input.

## Input and scientific meaning

CSV columns:

| Column | Meaning |
|---|---|
| `record_id` | Unique signed integer identifying the molecule |
| `approved` | Cached eligibility: exactly `0` or `1`; missing is rejected |
| `qed` | Previously computed QED, finite and within `[0, 1]` |
| `murcko_scaffold` | Previously computed scaffold identifier; missing is reported separately |
| Optional activity column | `1` experimentally active, `0` experimentally inactive, blank unknown |

Eligibility and selection are separate. Every method in a scenario receives
exactly the same approved pool, QED threshold, target count and scaffold quota.
Unknown scaffolds are excluded from coverage counts and cannot be constrained
by scaffold quotas; inspect the explicit missing-scaffold counts before drawing
comparisons. Unfilled target counts are reported as shortfalls.

Activity columns are isolated from selector inputs and used **only after
selection**. Unknown activity is never converted to inactivity. Recovery is
known active molecules selected divided by known active molecules in that
scenario's eligible pool; zero known actives produces an unavailable result.
That conditional metric does not measure losses caused by initial eligibility
filters. Compare thresholds with their eligible counts, not recovery alone.

Real labels can come from a curated experimental dataset, including a suitable
Papyrus subset. Prepare them externally for a defined target, assay type,
activity threshold, quality filter and aggregation policy. Preserve those
decisions and the dataset release alongside the input. Do not infer inactivity
from absent records, similarity, target mismatches or unavailable measurements.
This benchmark does not download Papyrus or resolve conflicting assay labels.

## Methods and stability

- `qed_ranked`: highest cached QED first, followed by the same quota selector.
- `rare_scaffolds_first`: existing scaffold-size ordering, favoring rarer
  scaffold groups; this is a greedy heuristic, not exact coverage optimization.
- `seeded_random`: independent seeded priorities with identical quota handling.
  Default: three random repeats. Each repeat keeps each molecule's priority
  fixed across thresholds. Input ordering is therefore part of reproducibility.

Defaults are seed `42` and minimum QED scenarios `0`, `0.5`, `0.7`.
Configure these with `--seed`, `--repeats` and `--thresholds 0 0.4 0.6`.
The thresholds are illustrative sensitivity settings, not endorsed scientific
cutoffs. Methods compare against QED ranking through Jaccard overlap. Each
method/repeat also reports its union, stable core and a histogram of the number
of scenarios selecting each molecule. A stable selection is not evidence of
biological activity. Empty unions produce unavailable overlap fractions.

Input SHA-256, exact settings, package versions, benchmark/selector source
hashes and selection hashes support reruns. Selection/eligible hashes are
SHA-256 over sorted signed 64-bit integer IDs encoded little-endian. Synthetic
source hashes use pandas row hashes including the index; retain package versions
when comparing them. Activity/descriptor changes also change the real CSV hash.

## Scaling and measured scope

Selections reuse the compact selector, with no rejection dictionary for the
entire library and no all-pairs distance matrix. Sorting is O(n log n); arrays
are O(n), and stability storage scales with selected sets. CSV input is loaded
into memory: this is not an out-of-core chemistry pipeline.

Local Windows measurement, 2026-09-08, Python 3.11, independent CLI processes
(AMD64 Family 25 Model 33, approximately 64 GiB installed RAM):
seed 42, target 1,000, maximum 5 per scaffold, three thresholds and three random
repeats (15 selections total). Synthetic cached features, no experimental labels:

| Input rows | QED ranking, threshold 0 | All 15 timed selections | Sampled whole-process peak RSS |
|---|---:|---:|---:|
| 100,000 | 0.020 s | 0.203 s | 95.3 MiB |
| 1,000,000 | 0.162 s | 1.447 s | 212.1 MiB |
| 3,000,000 | 0.493 s | 4.679 s | 460.6 MiB |

These are **synthetic selector smoke measurements**, not end-to-end performance
claims. They exclude SMILES parsing, standardization, descriptor calculation,
GUI rendering, loading and metric computation from selection timings. Whole
process RSS includes input generation and reporting; 20 ms sampling may miss
brief peaks, and RSS cannot be assigned to an individual method. Sequential
methods can benefit from warm caches; repeat in fresh processes on the same
hardware, report dispersion, and include chemistry/IO separately for a paper.

A separate Qt offscreen smoke opened one million cached rows in 29.35 s.
Process RSS was 613.8 MiB before constructing the workspace and 997.6 MiB
after its first events. Exact Pareto was disabled and the map deferred as
intended. This fixture repeats six molecules: it measures GUI bookkeeping,
not one million unique structures or chemistry processing. These RSS snapshots
are not peak measurements. The workspace and selection basket still retain
O(n) state in memory; full-library processing is not a disk-backed pipeline.

The benchmark validates reproducibility and establishes simple baselines. It
does not establish superiority, compare exact Pareto fronts, or replace a
multi-dataset retrospective and prospective scientific study.
