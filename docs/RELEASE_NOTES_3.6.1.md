# SMILES2Select 3.6.1

SMILES2Select now includes **Chemical tolerance (experimental)** in the graphical workspace. It describes whether selected rule violations match chemistry observed among known training actives. Its score represents chemical compatibility; it does not establish biological activity or safety.

## Graphical workflow

Open **Chemical Space Hub → Prioritize by model → Chemical tolerance (experimental)**. Choose an included profile or load a verified custom package, inspect its training counts and calibration status, and explicitly choose rules or native alert entries that may be reconsidered. General, target-conditioned and custom workflows share the existing molecule-count and scaffold/cluster controls.

Use **Preview chemical tolerance** to inspect eligible, supported-relaxation, review and excluded cases. Preview leaves the basket unchanged. **Adopt tolerance proposal** records the evidence in one undoable action. Mandatory exclusions, manual exclusions and locked SMARTS remain binding. Saved sessions and exports retain the applied criteria, model identity and frozen evidence.

**Validate data / train custom profile** opens the existing cancelable Decision Studio. The CLI also provides `tolerance-prepare`, `tolerance-train`, `tolerance-predict` and `tolerance-inspect`. Custom fitting and adaptation require a compatible external Python training runtime when using a frozen installer. Included profiles run locally without that training environment.

## Included models and scientific limits

Three recipe A statistical profiles cover pooled, hierarchical and contextual chemical compatibility. They use 3,829 fitted molecular identities across ten tasks, with recorded-active eligibility at exact pActivity >= 6.5. Their retention/expansion constraints were infeasible, so automatic rescue is disabled. Descriptive evidence and custom adaptation remain available; review cases are not silently accepted.

The twelve historical ONNX alternatives, including Tiny, and nine contextual packages remain accessible through the same workspace. The nine contextual packages are three biological predictors with three calibration variants each. Research-only broad multitask, Von/Jev and source-origin classifiers are not presented as additional validated activity rankers.

Controlled tolerance comparisons did not demonstrate an advantage for contextual models: primary recovery differences were -1.89 and -4.47 percentage points, with both uncertainty intervals including zero. Full training-bank expansion, the complete research parameter grid and independent-source confirmation remain future work. Offline **Methods** and the [chemical-tolerance guide](https://github.com/amgoncalvesusp/SMILES2Select/blob/v3.6.1/docs/CHEMICAL_TOLERANCE.md) explain each method and its limits.

## Reliability and performance

Version 3.6.1 fixes Qt cleanup in the offscreen resource diagnostic and GUI tests. The first 3.6.0 tag passed 1,884 Linux tests but failed during Qt process shutdown (segmentation fault, exit 139); it did not publish installers. Diagnostic windows are now disposed while their QApplication remains alive. The original failed tag and CI evidence are retained.

An optional durable cache stores verified chemistry features with bounded payloads, recipe/toolkit identity and corruption checks. Shared entries exclude biological context. The GUI reports bounded preparation estimates rather than promising peak RAM or duration.

A 100,000-molecule statistical-profile test selected 2,000 molecules in 222.53 seconds for inference plus selection, using 1.31 GiB additional RAM. Including chemical annotation, the run took 888.02 seconds. These scoped measurements are not universal performance guarantees.

Windows packaging now prevents Poppler's versioned ICU library from shadowing Qt's incompatible unversioned dependency. Release checks exercise Methods resources, historical ONNX inference, contextual/risk resources and all three tolerance modes in frozen and installed builds on Windows and Linux.

## Authors

The **About SMILES2Select** menu displays both authors, their affiliations and the application version.

- Adriano Marques Gonçalves — Universidade de Araraquara (UNIARA).
- Daniel Andres Grajales Ruiz — IQ-UNESP.

## Downloads

- `SMILES2Select-Setup-3.6.1.exe`: Windows x86_64 installer.
- `SMILES2Select-Setup-3.6.1-linux-x86_64.run`: Linux x86_64 per-user installer.
- `SMILES2Select-windows.zip` and `SMILES2Select-linux.tar.gz`: portable bundles.
- Python wheel, Windows native-dependency manifest and `SHA256SUMS-v3.6.1.txt`: package and integrity evidence.

AI selection and training remain experimental and need broader independent validation. Installer CI verifies its runner environments; it does not certify every operating-system configuration.
