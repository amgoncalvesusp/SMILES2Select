# Large-library performance

Reference comparison uses one candidate fingerprint against packed reference
fingerprints and does not materialize a dense pairwise matrix. Large workflows
should use chunk processing, persistent caches, packed fingerprints, Parquet,
SQLite and optional approximate-neighbour indexes where validated.

Do not run quadratic Butina clustering on a 200,000-molecule library without a
size guard. Benchmark wall time, peak memory, fingerprint throughput,
reference-search throughput, projection time, selection time and export time on
the target workstation.

The repository includes a deterministic smoke benchmark:

```text
python benchmarks/benchmark_hub.py --molecules 10000 --references 2000 --output benchmark-results.json
```

The JSON records the active reference-search method, projection time, density
preparation time, displayed point count and density-tile count. Production
reports should also record the workstation, Python/RDKit versions and peak RSS.
