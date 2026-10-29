# SMILES2Select 3.7.3

**Release date:** 2026-10-28

SMILES2Select 3.7.3 corrects frozen-bundle packaging of RDKit's SA/NP scorer modules. It retains the updated Chemical Space Hub, deterministic final cleanup and scoped Qt compatibility constraint without changing descriptor mathematics, selection algorithms or trained model data.

## Packaging correction and withdrawn 3.7.2 release

The 3.7.2 release was withdrawn to a draft after a clean check of its published Linux bundle failed SA-score legacy-reference parity for all twelve bundled ONNX models. Its source tag, history and [release notes](RELEASE_NOTES_3.7.2.md) are preserved. At withdrawal, 3.6.2 was again the latest public release.

The SA/NP score-data files were present, but their Python scorer modules were missing. Bare imports reached through RDKit's host Contrib path were not collected into the frozen bundle. Earlier harnesses imported host RDKit and passed its `RDBASE` setting to the child, masking that dependency. Passing CI and matching release hashes proved the recorded checks and downloaded bytes; they did not prove the bundle could run independently of host RDKit.

The correction uses qualified imports from `rdkit.Contrib.SA_Score` and `rdkit.Contrib.NP_Score`, allowing PyInstaller to collect the Python modules while retaining their existing data-only score resources. No score formula, model weight, threshold or training data is changed.

Frozen-child regressions remove host `RDBASE`, `PYTHONPATH` and Qt/library-path overrides. Publication requires a fresh local frozen build before tagging, clean-child model/resource verification and successful tagged Windows/Linux release gates. Existing 3.7.2 results are historical evidence, not a claim that 3.7.3 binaries have passed.

## Hub and compatibility features retained

The Hub keeps **Requested N**, applied method, final count, count difference and explicit actions visible above compact **Configure** forms. **Table** exposes evaluable candidates through stable **Record ID** values, linked to structured on-demand 2D inspection and a separate map ring. **Compare molecules...** remains modeless and read-only: explicitly keep A as a temporary view reference while B follows inspection. Current basket states and applied evidence refresh after decisions, adoption, undo and redo; pending preview evidence stays in the main inspector. Inspection, sorting and keeping A do not select or pin molecules. See the [Hub guide](CHEMICAL_SPACE_HUB.md) and [GUI guide](GUI_GUIDE.md).

Final cleanup closes child plots before native widget destruction while QApplication remains alive, preserving ordinary window close/reopen. The existing PySide6 >= 6.5 requirement still excludes 6.12.0 only on Linux with Python below 3.12; with the Python 3.11 project minimum, this guards Linux/Python 3.11. The pandas 3.1 empty-reference fix and Pillow >= 13.0.0/setuptools >= 84.1.0 security minimums remain in place. The [3.7.0](RELEASE_NOTES_3.7.0.md), [3.7.1](RELEASE_NOTES_3.7.1.md) and [3.7.2](RELEASE_NOTES_3.7.2.md) notes retain the original scope of earlier changes and checks.

## Scientific and verification scope

The twelve ONNX alternatives, including Tiny, nine contextual packages and three tolerance profiles retain their existing limits. Nine contextual packages represent three predictors with three calibration variants each. AI selection and training remain **experimental and need broader independent validation**. Chemical compatibility does not establish activity or safety; infeasible tolerance calibration still disables automatic rescue. Training uses a separately configured external Python runtime; the tested training configuration is CPU-only.

Requested N remains a molecule-count budget subject to eligibility, quotas, pins, manual exclusions and locked SMARTS. Chemical ranking, QED priority, explicit adoption, undo/redo and session/export schemas are preserved. [Optional SMARTS matching](CONTEXTUAL_SELECTION.md#optional-smarts) remains local and map distances remain visualization-only. No new biological-performance evidence is claimed; see [Methods](../src/smiles2select/assets/methods.md) and the [chemical-tolerance workflow](CHEMICAL_TOLERANCE.md).

Release gates require successful test-process completion, signal/shutdown regressions, audits of resolved installed runtime/transitive/build packages, native-dependency checks and clean-child frozen/installed desktop verification on Windows and Linux. Audits run before packaging without re-resolving packages or suppressing advisories. The desktop diagnostic covers bundled model resources and linked Hub inspection/comparison through `hub_gui`. Checksums remain integrity evidence alongside these execution checks. Runner tests do not certify every operating system, clean-machine setup, physical-display DPI or global window placement under Wayland.

## Authors

- Adriano Marques Gonçalves — Universidade de Araraquara (UNIARA).
- Daniel Andres Grajales Ruiz — IQ-UNESP.

## Release asset names

- `SMILES2Select-Setup-3.7.3.exe`: Windows x86_64 installer.
- `SMILES2Select-Setup-3.7.3-linux-x86_64.run`: Linux x86_64 per-user installer.
- `SMILES2Select-windows.zip` and `SMILES2Select-linux.tar.gz`: portable bundles.
- Python wheel, Windows native-dependency manifest and `SHA256SUMS-v3.7.3.txt`: package and integrity evidence.

When published, obtain assets from the [3.7.3 release page](https://github.com/amgoncalvesusp/SMILES2Select/releases/tag/v3.7.3) and verify their checksums.
