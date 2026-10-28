# SMILES2Select 3.7.1

**Release date:** 2026-10-28

SMILES2Select 3.7.1 carries the updated Chemical Space Hub from the 3.7.0 source tag and makes final Qt/plot cleanup explicit. Chemical selection methods, trained model packages and scientific limits remain unchanged.

## Shutdown cleanup and the preserved 3.7.0 tag

The [3.7.0 CI run](https://github.com/amgoncalvesusp/SMILES2Select/actions/runs/40780896027) reported 1,947 passing Linux tests and 18 skips, then emitted a shared-QObject warning and exited with a segmentation fault (139) during process shutdown. Builds and publication were skipped; no 3.7.0 installers were published. The source tag, commit `005c2f7` and failure evidence are preserved.

A local Linux GUI reproduction with Qt 6.12 likewise reported 264 passing tests before a shutdown SIGSEGV. Passing assertions alone therefore did not establish successful process completion.

Version 3.7.1 closes pyqtgraph child plots before native widget destruction during final cleanup. The application entry point performs cleanup after the event loop returns, while QApplication remains alive; test-session cleanup also drains remaining Qt graphs. Ordinary window close/reopen behavior remains available. Subprocess regressions require a clean exit without shared-QObject warnings or remaining widgets. Final 3.7.1 tests, tagged CI and installer verification are pending; the earlier passing assertions are not evidence that this follow-up has passed.

## Chemical Space Hub changes included

- **Requested N**, the applied method, final count, count difference and selection/export actions remain visible above compact forms. **Configure** shows **Chemical ranking**, **Model (experimental)**, **Contextual (experimental)** or **Tolerance (experimental)** without applying a method. Use **Create selection** or the chosen method's explicit preview/adopt actions. **Refresh map** and **Display...** sit beside the map.
- **Table** exposes every evaluable Hub candidate, including chemical screening failures. Stable **Record ID** values link sorting, keyboard navigation, maps, Pareto and preview review tables even when external molecule IDs repeat. **Final for export**, **Chemical status** and **Selection status** remain separate; inspection and sorting do not change membership.
- The on-demand 2D inspector separates identity/properties, selection/decision and evidence/provenance, with escaped display values. A green inspection ring is independent of gold final-molecule markers. Records without plotted coordinates remain inspectable with an explicit notice; no coordinates are invented.
- **Compare molecules...** opens a modeless, read-only window. Explicitly choose **Keep current molecule as A**, then inspect another record for B. A is a temporary view reference, not a selection or pin. Both panes refresh current basket states and applied evidence after decisions, adoption, undo and redo. Pending preview evidence stays in the main inspector; no new similarity score is calculated. Empty, unavailable and identical records clear their panes, and busy/closing guards remain in place.

The [3.7.0 feature notes](RELEASE_NOTES_3.7.0.md), [Hub guide](CHEMICAL_SPACE_HUB.md) and [GUI guide](GUI_GUIDE.md) give the full workflow. Historical 3.7.0 notes are retained as source-tag documentation, not evidence of published installers.

## Compatibility, security and scientific scope

The explicit boolean filter mask retains empty-reference context behavior under pandas 3.1 without changing context features, thresholds or mathematics. No model is retrained and no new application dependency is added.

The Pillow runtime minimum remains **13.0.0**, and the setuptools build minimum remains **84.1.0**. The release gate audits the resolved installed runtime, transitive dependencies and build tools with **pip-audit** before PyInstaller packaging, without re-resolving packages or suppressing advisories. The final isolated Windows audit checked 87 installed packages with no known vulnerabilities and no skipped packages; the CI gates independently audit their resolved build environments.

The twelve ONNX alternatives, including Tiny, nine contextual packages and three statistical tolerance profiles remain available. Nine contextual packages represent three biological predictors with three calibration variants each. AI selection and training remain **experimental and need broader independent validation**. The included tolerance profiles' infeasible retention/expansion calibration still disables automatic rescue; chemical compatibility does not establish activity or safety.

Requested N remains a molecule-count budget governed by eligibility, quotas, pins, manual exclusions and locked SMARTS. Chemical ranking and QED priority, explicit proposal adoption, undo/redo and saved-session/export schemas are preserved. Optional [SMARTS patterns](CONTEXTUAL_SELECTION.md#optional-smarts) still use required-pattern AND, excluded-pattern OR and preferred-pattern ranking with local RDKit matching; [SMARTS.plus](https://smarts.plus/) remains optional external help. Map distances remain visualization-only. No new biological-performance evidence is claimed; see the [Methods guide](../src/smiles2select/assets/methods.md) and [chemical-tolerance workflow](CHEMICAL_TOLERANCE.md).

## Verification scope

The final Windows source suite passed 1,952 tests with 19 expected skips, 86.86% combined line/branch coverage and successful process termination. The matching Linux Qt 6.12 reproducer passed the same 264 GUI/workspace tests and exited with code zero, with no surviving widgets; six shutdown regressions passed in a separate Linux process. These are overlapping source-checkout checks, not 270 tests in one process.

Release gates include successful source-test process completion, native-dependency checks and frozen/installed desktop resource verification on Windows and Linux. The `--verify-desktop-resources` diagnostic exercises the Hub table, linked inspection and molecule comparison through its `hub_gui` result. Publication depends on successful tagged CI and asset verification.

Linux release jobs pin Ubuntu 24.04 to retain the tested runner baseline.

Earlier source-checkout coverage, GUI captures and audits retain their original scope. They do not establish final 3.7.1 installer success, clean-machine compatibility, physical-display DPI behavior or global window placement under Wayland.

## Authors

- Adriano Marques Gonçalves — Universidade de Araraquara (UNIARA).
- Daniel Andres Grajales Ruiz — IQ-UNESP.

## Downloads

- `SMILES2Select-Setup-3.7.1.exe`: Windows x86_64 installer.
- `SMILES2Select-Setup-3.7.1-linux-x86_64.run`: Linux x86_64 per-user installer.
- `SMILES2Select-windows.zip` and `SMILES2Select-linux.tar.gz`: portable bundles.
- Python wheel, Windows native-dependency manifest and `SHA256SUMS-v3.7.1.txt`: package and integrity evidence.

After publication, download assets from the [3.7.1 release](https://github.com/amgoncalvesusp/SMILES2Select/releases/tag/v3.7.1) and compare their hashes with the checksum file.
