# SMILES2Select 3.7.0

**Release date:** 2026-10-28

SMILES2Select 3.7.0 makes the Chemical Space Hub easier to configure and inspect. A linked candidate table and a read-only two-molecule comparison window keep identity, current selection and applied evidence visible without changing the scientific selection methods.

## Compact workspace and explicit actions

The Hub keeps **Requested N**, the applied method, final count, difference from the request and selection/export actions above the scrolling forms and map. **Inspect applied criteria...**, **Undo** and **Redo** remain accessible while exploring the workspace.

Use **Configure** to show **Chemical ranking**, **Model (experimental)**, **Contextual (experimental)** or **Tolerance (experimental)**. Switching the form does not apply that method or adopt a proposal. Use **Create selection** for chemical ranking, or the selected method's explicit preview/adopt actions. Current chemical-tolerance navigation is **Chemical Space Hub → Configure → Tolerance (experimental)**.

Shared controls are grouped under **Quotas / objectives**. **Refresh map** and **Display...** are placed beside the map, with projection, coloring, reference-overlay and local Papyrus evidence settings. Map geometry remains a visualization: projected distances are not similarity, novelty or selection thresholds.

## Candidate table and linked 2D inspection

Choose **Table** after **Chemical space** and **Pareto** in the view selector. The virtual, read-only table exposes every evaluable Hub candidate, including chemical screening failures, without a 5,000-row display cap. These records are not necessarily eligible for the chosen model or proposal. The existing basket still displays decided molecules.

**Record ID** identifies an internal record even when external **Molecule ID** values repeat. **Final for export**, **Chemical status** and **Selection status** are separate columns. Sorting changes display order only; numeric values sort numerically, ties are stable and missing values stay last. Descriptor cells retain stored precision and missing values appear blank.

Mouse or keyboard inspection follows the same Record ID across Table, Chemical space, Pareto and the contextual/tolerance review tables. The inspector separates **Identity / properties**, **Selection / decision** and **Evidence / provenance**, generates 2D structures on demand and escapes displayed text. Inspection and sorting do not select molecules.

A green outline ring marks the inspected record at its existing map coordinates, independently of the gold final-molecule squares. **Show final molecules only** hides that ring for a nonfinal record. Records absent from a projection or sampled context remain available in the table and inspector, with an explicit map notice; no coordinates are invented.

## Compare two molecules

Open **Compare molecules...** in the persistent action row. Inspect a record and choose **Keep current molecule as A**, then inspect another record for B through Table, Chemical space, Pareto or a preview review table. The modeless window stays open while the Hub remains usable. A is a temporary view reference: keeping it does not select, shortlist, pin or adopt anything.

Both columns show 2D structures, identity, properties, current basket states and applied evidence. They refresh after decisions, proposal adoption, undo and redo. The **Applied method** banner follows the applied ranking even when controls are edited or a proposal is pending. Pending model, contextual and tolerance evidence stays in the main inspector; draft Pareto ranks and map-location notices are omitted from the comparison. No new similarity score is calculated.

Empty or unavailable records clear their panes. When A and B are the same record, B asks for another molecule. A is retained when the window is closed and reopened within the same Hub, but is not saved in sessions. Opening the comparison and keeping A are disabled during computation; **Close** remains available. Closing the Hub also closes the comparison.

## Optional SMARTS help and scientific scope

The existing optional [SMARTS guide](CONTEXTUAL_SELECTION.md#optional-smarts) explains required-pattern AND, excluded-pattern OR, preferred-pattern ranking and local RDKit matching. [SMARTS.plus](https://smarts.plus/) is optional external help for designing and inspecting a pattern; matching remains local in SMILES2Select.

Chemical-selection algorithms, selection constraints, saved-session/export schemas and trained model packages are preserved. No new model was trained and no application dependency was added for this release. The twelve ONNX alternatives, including Tiny, nine contextual packages and three statistical tolerance profiles remain available with their existing scope and limitations. Nine contextual packages are three biological predictors with three calibration variants each, not nine independent biological models.

An explicit boolean filter mask preserves empty-reference context behavior under pandas 3.1 without changing context features, thresholds or mathematics or retraining models.

AI selection and training remain **experimental and need broader independent validation**. Chemical compatibility does not establish activity or safety. The included tolerance profiles' infeasible retention/expansion calibration still disables automatic rescue. This interface update supplies no new biological-performance evidence. See the [Methods guide](../src/smiles2select/assets/methods.md), [chemical-tolerance workflow](CHEMICAL_TOLERANCE.md), [Hub guide](CHEMICAL_SPACE_HUB.md) and [GUI guide](GUI_GUIDE.md).

## Security maintenance

The Pillow runtime minimum is now **13.0.0**, and the setuptools build minimum is **84.1.0**, excluding known vulnerabilities in earlier versions. No new application dependency is added. Release builds run **pip-audit** against the resolved installed runtime, transitive dependencies and build tools before PyInstaller packaging, without re-resolving packages or suppressing advisories; a failed audit blocks bundle creation.

## Verification scope

The final Windows source suite passed 1,946 tests with 19 expected skips and 86.87% combined line/branch coverage across the source tree, using an isolated environment. Earlier targeted checks passed 260 Windows tests and 264 Linux tests, with 84.93% workspace coverage and 97.67% comparison-module coverage, plus 30 GUI captures. These overlapping source-checkout results are not confirmation that the tagged release or its installers have passed CI.

A clean Windows pre-release audit checked 87 installed packages and reported no known vulnerabilities and no skipped packages within the scope above.

The release workflow includes source tests, native-dependency checks and frozen/installed desktop resource verification on Windows and Linux. The fixed `--verify-desktop-resources` diagnostic also exercises the Hub table, linked inspection and molecule comparison, reporting an additive `hub_gui` result. Publication still depends on successful tagged CI and asset verification. Runner checks and captures do not establish clean-machine compatibility, physical-display DPI behavior or global window placement under Wayland.

## Authors

- Adriano Marques Gonçalves — Universidade de Araraquara (UNIARA).
- Daniel Andres Grajales Ruiz — IQ-UNESP.

## Downloads

- `SMILES2Select-Setup-3.7.0.exe`: Windows x86_64 installer.
- `SMILES2Select-Setup-3.7.0-linux-x86_64.run`: Linux x86_64 per-user installer.
- `SMILES2Select-windows.zip` and `SMILES2Select-linux.tar.gz`: portable bundles.
- Python wheel, Windows native-dependency manifest and `SHA256SUMS-v3.7.0.txt`: package and integrity evidence.

After publication, download assets from the [3.7.0 release](https://github.com/amgoncalvesusp/SMILES2Select/releases/tag/v3.7.0) and compare their hashes with the checksum file.
