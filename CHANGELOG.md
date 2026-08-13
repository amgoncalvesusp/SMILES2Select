# Changelog

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
