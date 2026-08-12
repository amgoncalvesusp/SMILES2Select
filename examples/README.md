# SMILES2Select examples

These examples assume the project is installed or that `PYTHONPATH=src` is set.

## Reference-aware final and reserve selection

```bash
python -m smiles2select.cli candidates.csv \
  --reference known-bioactives.csv \
  --reference-smiles-column SMILES --reference-id-column ID \
  --exclude-reference-duplicates \
  --selection-strategy reference_aware_diversity \
  --final-count 3000 --reserve-count 500 \
  --save-selection-plan selection-plan.json \
  --excel selection-report.xlsx --database selection.sqlite
```

## Replay a saved plan

```bash
python -m smiles2select.cli candidates.csv \
  --reference known-bioactives.csv \
  --selection-plan selection-plan.json \
  --excel replayed-report.xlsx
```

## Benchmark reference comparison

```bash
python benchmarks/benchmark_hub.py --molecules 10000 --references 2000 \
  --output benchmark-results.json
```

The benchmark reports the active reference-search method, projection time,
density-rendering time and number of display points. The optional `fast`
reference mode requires `hnswlib`; when unavailable, use exact search.
