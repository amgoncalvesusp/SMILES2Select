# Changelog

## 3.2.0 - 2026-09-08

### Added

- Guided workspace controls for researchers: essential actions first, expandable advanced criteria, notebook-friendly scrolling, human-readable strategy labels and on-demand 2D structures.
- Independent A/B scenario previews, threshold overrides, explicit adoption with undo/redo, selection versus eligibility frequencies, complete CSV output and fingerprint-bound study JSON replay.
- Chunked replay of cached chemical criteria, preserving profile roles, violation tolerances, structural alerts and global QED percentile policy. Unsupported zone allocations are reported explicitly.
- Optional local Papyrus evidence indexes scoped by target, endpoint, quality and molecular identity. Compressed input is streamed; indexed evidence is queried on inspection, with source and identity limitations preserved. No download or biological inference is performed automatically.
- Reproducible cached-descriptor selection benchmark with seeded random and ranked baselines, fixed molecule-count limits, optional external activity labels and input/software hashes.
- Adopted-scenario export context and replay sidecars. Original screening status is distinguished from scenario eligibility and the actual final selection.

### Performance

- Large selections can omit per-rejection dictionaries and stop scanning once the target is reached; selection IDs and quota semantics remain unchanged.
- Scenario pools above 2,000 candidates use explicitly labeled weighted objective percentiles instead of quadratic exact Pareto ranking. Large maps and tables bound display work without subsampling the final selector's candidate universe.
- Scenario, map and export work uses background jobs; closing windows while jobs run is guarded. Final-selection Excel data is materialized only for selected records.
- Benchmarks cover synthetic cached libraries up to three million records. They do not claim end-to-end SMILES processing or biological performance; run/basket RAM still grows with input size.

### Fixed

- Automatic workspace selection now excludes chemically rejected candidates unless already selected and pinned with a manual justification. Re-selection replaces the previous set in one undo step and preserves retained manual decisions.
- Selection basket updates validate the whole action before changing state, preventing partial changes when justification is missing.
- Constrained selection prioritizes distinct scaffolds when a minimum is requested and reports pinned molecules that exceed count or quota limits. Coverage remains greedy; unmet minimums are reported.
- Workspace exports follow the current basket after manual changes and undo/redo. The inspector refreshes after decisions, and invalid Pareto objectives clear outdated points.
- Exported strategy reflects the last automatic selection still applied, including after undo; changing the strategy control alone does not rewrite its provenance.
- Pillow now requires version 12.3.0 or later to exclude known vulnerabilities in older versions.

## 3.1.0 - 2026-09-02

### Added

- Docking preparability layer: per-molecule flags for uncommon elements, multiple fragments, macrocycles, peptide character and undefined stereochemistry, driven by declarative engine profiles (Vina, GOLD, Glide) rather than hardcoded rules. Results reach the `preparability_flags` SQLite table, the `PREPARABILITY_FLAGS` Excel sheet and the docking hand-off.
- Docking budget panel on the Results screen, shown only when preparability was computed: selected molecules, estimated 3D structures for the selected set, a flag breakdown by severity, and a log-scale histogram of the per-molecule preparation cost. The panel is included in the chart export.
- New preparability descriptors, computed once in the parallel workers like every other descriptor: undefined and defined stereocenters, fragment count, largest ring size, amide bonds, bridgehead and spiro atoms, and an optional capped tautomer count.
- Progress reporting for the preparability stage, which is single-threaded and previously left the progress bar silent on large libraries.

### Removed

- Removed TMAP support, which has no reliable Windows installation. Projection defaults to deterministic PCA, with UMAP available when installed.
- Removed the duplicate `smiles2select-workspace` entry point, which only called the GUI entry point.

## 3.0.3 - 2026-08-13

### Added

- Added the SMILES2Select chemical-selection artwork as the application, executable, shortcut and installer icon.
- Bundled transparent high-resolution PNG and multi-resolution ICO assets for Windows and Linux-compatible runtime use.
- Windows runtime surfaces now select the multi-resolution ICO; Linux-compatible surfaces retain the transparent PNG.

All notable changes to SMILES2Select are documented here.

## 3.0.2 — 2026-08-13

### Fixed

- Fixed Windows PyInstaller workers failing with `ValueError: not enough values to unpack (expected 2, got 1)` when processing large libraries. Frozen builds now use the standard-library spawn executor instead of the incompatible loky command-line path.

### Added

- Added PNG and SVG export for the four result charts: approval by profile, violations, profile intersection and descriptor distributions.
- Connected the Results page to the interactive Chemical Space Hub, including point inspection, Pareto view, lasso shortlisting, constrained auto-selection, undo/redo and selection export.

## 3.0.1 — 2026-08-13

### Changed

- Added explicit software authorship and Zenodo metadata for Adriano Marques
  Gonçalves, Universidade de Araraquara (UNIARA).
- Registered this patch release as the Zenodo DOI release record.
- Synchronized the package and desktop-bundle version to 3.0.1.

## 3.0.0 — 2026-08-12

### Added

- Chemical Space Selection Hub workflow with candidate, reference and background libraries.
- Exact fingerprint-based reference overlap, nearest-neighbour similarity and novelty analysis.
- Optional HNSW neighbour discovery with exact RDKit Tanimoto verification.
- Reference-aware diversity, novelty, neighbourhood and stratified selection strategies.
- Named selection zones with overlap-aware final/reserve quota allocation and shortfall reporting.
- Candidate/reference overlays, projection and color controls, progressive density rendering and method/consequence panels.
- Cooperative background processing cancellation while preserving completed checkpoints.
- SQLite, Excel, Parquet and machine-readable selection-recipe provenance for final and reserve sets.
- CLI replay/save support and a deterministic large-library benchmark.
- Windows portable bundle and per-user Setup installer containing the complete PyInstaller runtime payload.
- Windows native dependency manifest covering every bundled EXE, DLL and PYD file.

### Changed

- Migrated the normal GUI, CLI, reports, logs, built-in profile names and documentation to English.
- Raised the application version to SMILES2Select 3.0.0.
- Kept legacy 2.0 selection recipes readable and traditional single-library workflows operational.
