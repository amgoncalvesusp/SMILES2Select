# SMILES2Select 3.0

SMILES2Select is a multi-rule drug-likeness and chemical-space selection tool
for large SMILES libraries. Version 3.0 adds the Chemical Space Selection Hub:
explicit candidate, reference and background libraries; exact overlap detection;
fingerprint-based reference similarity and novelty; deterministic diversity and
zone quotas; final/reserve outputs; an English interactive workspace; and
machine-readable selection recipes.

Chemical similarity is computed with Morgan fingerprints and exact Tanimoto
verification. 2D map distances are visualization-only and are never used as a
similarity, novelty or selection threshold.

## Installation

RDKit is easiest to install with conda:

```bash
conda env create -f environment/environment.yml
conda activate smiles2select
pip install -e .
```

Optional capabilities are available as extras:

```bash
pip install -e ".[maps,fastsearch,parquet]"
```

## Quick start

The graphical application has seven English steps:

```bash
smiles2select-gui
```

The CLI can run a reference-aware selection directly:

```bash
smiles2select candidates.csv --smiles-column SMILES --id-column ID \
  --reference ecbd.csv --reference-smiles-column SMILES \
  --reference-id-column ID --exclude-reference-duplicates \
  --selection-strategy reference_aware_diversity \
  --final-count 3000 --reserve-count 500 \
  --excel selection.xlsx --database run.sqlite
```

Background/context libraries are explicit and are never selected:

```bash
smiles2select candidates.csv --smiles-column SMILES \
  --reference known.csv --background public-context.csv \
  --selection-strategy reference_novelty --final-count 500
```

## Selection semantics

The pipeline keeps four kinds of evidence separate:

| Layer | Role |
| --- | --- |
| Rules | pass/fail and failure explanations |
| Scores | continuous ranking information such as QED, SA and NP score |
| Alerts | structural flags with configurable inform/warn/penalize/exclude actions |
| Selection policy | final eligibility, strategy, quotas and reserve allocation |

The default policy makes Lipinski and Veber mandatory, calculates the other
selected profiles as informative outputs, ranks with QED, and warns on PAINS
and Brenk. These are heuristics, not predictions of efficacy, safety or oral
bioavailability.

Available selection strategies include `traditional`, `balanced`,
`diversity_first`, `reference_novelty`, `reference_neighborhood`,
`reference_aware_diversity`, `stratified` and `manual_assisted`.

`--reference-search exact` performs exhaustive reference comparison. The
optional `fast` mode uses HNSW only for neighbour discovery and re-ranks the
discovered pool with exact RDKit Tanimoto. It fails explicitly when `hnswlib`
is not installed rather than silently changing method.

## Reproducible recipes

Save and replay the Hub decision layer with JSON:

```bash
smiles2select library.csv --smiles-column SMILES --profiles lipinski \
  --mandatory lipinski --selection-strategy diversity_first \
  --final-count 100 --reserve-count 25 \
  --save-selection-plan selection.selection.json

smiles2select another-library.csv --smiles-column SMILES \
  --selection-plan selection.selection.json
```

Recipes record schema version, toolkit version, fingerprint settings, seed,
library provenance, zones, projection provenance and final/reserve counts.
Legacy 2.0 recipes remain readable.

## Chemical Space Hub

The workspace provides:

- deterministic property PCA plus optional structural UMAP/TMAP methods;
- separate reference overlays and layers for candidate/reference/background data;
- progressive point rendering and density tiles for large libraries;
- color controls for selection status, Pareto rank and reference similarity;
- lasso selection, Pareto objectives, scaffold/cluster quotas and a live basket;
- method cards and consequence-aware explanations for strategy changes;
- exact duplicate, nearest-reference, novelty and coverage inspectors;
- final and reserve status, undo/redo, audit history and Excel/recipe export.

The overlay is a visual context layer. Fingerprint similarity remains the
scientific metric shown in the inspector and saved in the database.

The optional dockability envelope is available as a built-in operational
profile. It is a preparation-oriented filter, not a biological prediction.
`natural_product_exploration_preset()` keeps it mandatory while treating
classical drug-likeness profiles as informative by default.

## Outputs

Excel exports include summary, final, reserve, excluded, profile, alert,
reference-overlap, reference-similarity, zone-membership, zone-allocation and
configuration sheets when those data exist. SQLite stores descriptors, profile
results, failures, alerts, decisions, reference provenance, exact overlaps,
similarity pairs, reserves, zones and the run configuration. Parquet is
available for large wide tables.

## Performance and cancellation

Descriptor work is chunked, checkpointed and executed in background workers.
The GUI exposes cooperative cancellation; completed chunks remain in the
checkpoint and can be resumed with the same configuration. Large exact Butina
clustering is guarded before quadratic memory allocation. Reference comparison
does not materialize a candidate-by-reference similarity matrix.

Run the deterministic smoke benchmark with:

```bash
python benchmarks/benchmark_hub.py --molecules 10000 --references 2000 \
  --output benchmark-results.json
```

See the detailed methodology in:

- [Chemical Space Hub](docs/CHEMICAL_SPACE_HUB.md)
- [Reference libraries](docs/REFERENCE_LIBRARIES.md)
- [Visualization](docs/CHEMICAL_SPACE_VISUALIZATION.md)
- [Selection strategies](docs/SELECTION_STRATEGIES.md)
- [Selection zones](docs/SELECTION_ZONES.md)
- [Reference-aware selection](docs/REFERENCE_AWARE_SELECTION.md)
- [Natural-product selection](docs/NATURAL_PRODUCT_SELECTION.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Large-library performance](docs/LARGE_LIBRARY_PERFORMANCE.md)
- [Examples](examples/README.md)
- [Changelog](CHANGELOG.md)

## Development

```bash
$env:QT_QPA_PLATFORM = "offscreen"  # PowerShell/CI
$env:PYTHONPATH = "src"
python -m pytest -q
python -m ruff check src tests benchmarks
```

The test suite covers chemistry, profiles, policies, reference libraries,
exact/approximate search contracts, zones, projections, progressive density,
recipes, persistence, GUI interaction, cancellation and CLI integration.
