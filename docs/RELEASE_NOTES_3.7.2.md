# SMILES2Select 3.7.2

**Release date:** 2026-10-28

SMILES2Select 3.7.2 includes the updated Chemical Space Hub and explicit final Qt/plot cleanup, with a narrowly scoped dependency fix for Linux/Python 3.11. Selection mathematics, trained models and scientific limits are preserved.

## Linux/Python dependency fix

The existing **PySide6 >= 6.5** requirement now additionally excludes **6.12.0** when `sys_platform == 'linux' and python_version < '3.12'`. The project's Python minimum is 3.11, so this currently covers Linux/Python 3.11 installations. The constraint does not impose a global PySide6 downgrade.

An isolated probe using only QtCore, without SMILES2Select application code, emitted a signal 100 times under Linux/Python 3.11.17. With the official PySide6 6.12.0 wheel, the `True` reference count fell from 604 to 504 and normal interpreter finalization aborted. The same probe with official 6.11.3 kept the count at 601 and exited normally. The tested Windows 6.12.0 wheel showed no reference-count change.

The affected signal return uses Boolean-return macros in [PySide's 6.12.0 source](https://raw.githubusercontent.com/pyside/pyside-setup/6.12.0/sources/pyside6/libpyside/pysidesignal.cpp). [CPython's limited-API fix](https://github.com/python/cpython/pull/135165) explains the required ownership handling for Boolean singletons on Python 3.11 and older; [PySide's build documentation correction](https://github.com/pyside/pyside-setup/commit/d8298afaa86b60cac814ed6d63e967bef4c2ec1c) specifies newer Python headers for compatible stable-API wheels. Together with the isolated wheel comparison, this supports the scoped dependency exclusion.

## Preserved source tags and included features

Linux CI for 3.7.1 reported 1,953 passing tests and 18 skips, then aborted during Boolean finalization (`bool_dealloc`, exit 134); no 3.7.1 packages were published. The earlier 3.7.0 run reported 1,947 passing tests and 18 skips before a shutdown segmentation fault (139), and published no installers. Both source tags and their original [3.7.0 notes](RELEASE_NOTES_3.7.0.md) and [3.7.1 notes](RELEASE_NOTES_3.7.1.md) are preserved. Their local checks and failed CI runs retain their original scope.

This release includes persistent **Requested N**, applied-method/count/difference controls and explicit actions; compact **Configure** pages; a virtual candidate **Table** linked by stable **Record ID**; structured on-demand 2D inspection with a separate map ring; and modeless, read-only **Compare molecules...**. Explicitly keeping A is a temporary view reference while B follows inspection. Applied evidence and current basket states refresh after decisions, adoption, undo and redo; pending preview evidence stays in the main inspector. Inspection and sorting do not change final membership. See the [Hub guide](CHEMICAL_SPACE_HUB.md) and [GUI guide](GUI_GUIDE.md).

Final cleanup still closes child plots before native widget destruction while QApplication remains alive, including application exit and test-session cleanup. Ordinary window close/reopen is preserved. The pandas 3.1 empty-reference Boolean-mask fix, Pillow >= 13.0.0 runtime minimum and setuptools >= 84.1.0 build minimum are retained. No model is retrained and no new application dependency is added.

## Scientific and release-verification scope

The twelve ONNX alternatives, including Tiny, nine contextual packages and three tolerance profiles retain their existing limits. Nine contextual packages represent three predictors with three calibration variants each. AI selection and training remain **experimental and need broader independent validation**; chemical compatibility does not establish activity or safety. Infeasible tolerance calibration still disables automatic rescue.

Requested N remains a molecule-count budget subject to eligibility, quotas, pins, manual exclusions and locked SMARTS. Chemical ranking, QED priority, explicit proposal adoption, undo/redo and session/export schemas are preserved. [Optional SMARTS matching](CONTEXTUAL_SELECTION.md#optional-smarts) remains local, and map distances remain visualization-only. No new biological-performance evidence is claimed; see [Methods](../src/smiles2select/assets/methods.md) and the [chemical-tolerance workflow](CHEMICAL_TOLERANCE.md).

Release gates require successful source-test process completion, signal-ownership/shutdown regressions, audits of resolved installed runtime/transitive/build packages, native-dependency checks and frozen/installed Windows/Linux desktop verification. Audits run before packaging without re-resolving packages or suppressing advisories. The desktop diagnostic includes linked Hub inspection and comparison through `hub_gui`; Linux release jobs use Ubuntu 24.04. Publication requires successful tagged CI and asset verification. Earlier local checks and probes do not certify every operating system, clean-machine setup, physical-display DPI or global window placement under Wayland.

## Authors

- Adriano Marques Gonçalves — Universidade de Araraquara (UNIARA).
- Daniel Andres Grajales Ruiz — IQ-UNESP.

## Downloads

- `SMILES2Select-Setup-3.7.2.exe`: Windows x86_64 installer.
- `SMILES2Select-Setup-3.7.2-linux-x86_64.run`: Linux x86_64 per-user installer.
- `SMILES2Select-windows.zip` and `SMILES2Select-linux.tar.gz`: portable bundles.
- Python wheel, Windows native-dependency manifest and `SHA256SUMS-v3.7.2.txt`: package and integrity evidence.

Use the [3.7.2 release page](https://github.com/amgoncalvesusp/SMILES2Select/releases/tag/v3.7.2) and verify downloaded assets against the checksum file.
