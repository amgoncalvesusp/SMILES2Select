# Changelog

## 3.5.0 - 2026-10-03

- Add an offline English Methods tab and menu covering chemical methods, all model families, actual training procedures and experimental limits.
- Separate applied basket criteria from settings for the next selection in the workspace.
- Integrate experimental contextual selection with rule and alert evidence, optional SMARTS, endpoint-specific risk predictions, explicit preview/adoption, review budgets and saved-session provenance.
- Include nine contextual activity packages (three predictors, each with three calibration variants) and one endpoint-specific risk package alongside the existing twelve ONNX models, including Tiny.
- Preserve retrospective study limitations and unfavorable comparisons; no model was refitted for this release.
- Add reproducible contextual, multitask, target-panel and comparator research tooling and regression tests.
- Raise the optional training dependency minimum to PyTorch 2.13 to exclude a known TorchScript vulnerability; desktop inference does not use PyTorch.
- Verify Methods resources and contextual/risk inference in frozen executables and installed Windows/Linux builds.

## 3.4.1 - 2026-09-28

- Restore all twelve trained task-specific ONNX alternatives, including three Tiny packages; preserve the original validation-selected baselines and historical model hashes.
- Add estimator/input labels and per-model explanations and selection tips in the workspace.
- Restore explicit QED-only ranking and minimum-scaffold controls, including shared model constraints, scenarios, recipes and saved-session compatibility.
- Add a Linux x86_64 per-user installer with desktop integration and uninstall, alongside the Windows installer and portable bundles.
- Expand real-model workspace and frozen-bundle tests to cover all estimator families; automate installer checks, wheel upload and release checksums.

## 3.4.0 - 2026-09-26

- Integrated S2S-Decision into the SMILES2Select package and desktop workspace, with target-specific ONNX previews, explicit adoption, undo/redo and model-aware exports.
- Added complete, integrity-checked `.s2s.sqlite` sessions that restore processed results, basket and history without original source or model files for historical export.
- Added validated user model import, optional training runtime, source/frozen worker dispatch and ONNX Runtime CPU desktop packaging.
- Separated model chemistry compatibility from the application version and preserved legacy model verification without changing historical manifests.
- Raised minimum Python version to 3.11; model selection remains experimental and is not a claim of biological activity.

## 3.3.1 - 2026-09-16

- Fixed final selection by molecular cores for libraries above 5,000 evaluable molecules: missing Murcko scaffolds are now calculated on demand in the background. Previously the operation failed and left the original, potentially much larger selection unchanged.
- Molecular-core coverage allocates the first feasible molecule from each core before filling the requested count, including the large-library percentile-ranking path. Ties remain deterministic; quotas and preserved pins can still constrain the achievable count and are disclosed.
- Both GUI exports are disabled for unapplied criteria and running jobs. A count different from the requested value requires explicit acknowledgement. Undo/redo restores the acknowledged controls, and adopted scenarios synchronize supported objectives and quotas.
- Background completion errors are shown in the interface, controls recover, and completed map jobs no longer retain a misleading building message. Large-library selection refreshes the map automatically.
- Added in-app criterion guidance covering strategy use, descriptor directions, quotas, ranking limits and scientific interpretation.
- Scenario-study schema 2 verifies selector version, final-ID identity and required scaffold inputs on replay; legacy schema 1 studies are rejected explicitly rather than silently replayed with a changed algorithm. Cached scaffold availability does not change the underlying candidate identity.

## 3.3.0 - 2026-09-16

### Fixed

- The Chemical Space Hub now opens with the original final selection. Every final molecule is retained in the map and display layers, with gold-square markers above contextual points, explicit counts/legend, a final-only filter and inspection of overlapping records without coordinate jitter.
- Added a direct SMILES2Docking CSV/XLSX export of the current basket, including justified manual changes, with a SHA-256 manifest, exact identities, active history and applied criteria. Formula-like identifiers remain literal text in all XLSX exports.
- Automatic decisions freeze count, quotas, objectives and ranking evidence. Control edits and undo/redo no longer rewrite the criteria associated with an existing selection. Original pipeline decisions are distinguished from subsequent Hub selections.
- Deterministic record-ID tie-breaking and explicit weighted-percentile priority for large libraries. The Hub now applies its objective controls above the exact Pareto limit as well.
- Crossed scaffold/cluster quotas use deterministic augmenting-path repair when ranked greedy filling misses a feasible target. Pins remain fixed; count excesses, shortfalls and unmet heuristic scaffold minima are disclosed. The first chosen cluster member is no longer described as a centroid or structural representative.

### Scientific provenance

- Recipes record ordered-data hashes, complete run/profile configuration, software versions, actual final IDs, manual overrides and map sampling metadata. The displayed applied target is separate from unapplied edits.
- Updated method documentation defines ordering, ranking universes, quotas, map limitations and the distinction between audit records and supported scenario replay. These changes do not assert biological activity or optimal global selection quality.

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
