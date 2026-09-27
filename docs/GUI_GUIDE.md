# Guided selection and scenario comparison

## Model proposals and saved sessions (3.4.0)

After processing, open the Chemical Space Hub and expand **Prioritize by model**.
Choose one of the three bundled task-specific models, or import a compatible
ONNX model folder, then choose its target, endpoint and model.
The panel shows task threshold, estimator, calibration status and training
reference count. Preview scores the chemically eligible pool using the current
count, quotas, pins, exclusions and reference rules. The basket stays unchanged
until **Adopt proposal**. Review retained, added, removed and requested versus
obtained counts before adopting. Undo and redo include the model decision.
**Cancel preview** discards the pending result after the current computation
stage finishes; no partial proposal is adopted.

A model cannot be used for another target or endpoint. A missing or incompatible
model produces a reason; there is no automatic QED fallback. Model scores are
task-specific estimates, not confirmed activity or universal probabilities.
The ordinary chemical strategies stay available. Sessions with zone allocation
currently block model proposals.

Use **Save session** in the Hub to create a `.s2s.sqlite` file. Use **Open
session** in the Hub or **Open saved session** on the main window to restore
the basket, history, criteria and results. Original input and ONNX files may
have moved: historical inspection and export still work; new inference needs
the original compatible model package. A recipe JSON records a decision but
does not replace this complete session file. For very large libraries, session
saving can take time and disk space proportional to the full run.

**Advanced model tools** in the main window hosts S2S-Decision training and
evaluation. Choose an external Python with `smiles2select[train]` installed
when using a frozen desktop build; dependencies are checked before a job starts.
Training does not alter an adopted basket.

The interface is intended for researchers who know their scientific question but
may be new to cheminformatics. Start with a local SMILES library, map its columns,
review the structure preparation and profile roles, and inspect the selection
summary before processing. The seven steps retain your choices when moving back.

## Essential and advanced controls

The wizard pages scroll on smaller screens. The selection policy summary remains
visible while advanced Boolean expressions, consensus and alert controls can be
expanded. Hiding settings never resets them. Drug-likeness profiles are heuristics;
the defaults are a starting configuration, not a biological recommendation.

In the Chemical Space Hub, set the number of molecules and choose a strategy.
Open **How to choose criteria...** beside the strategy for a scrollable guide to
each strategy, descriptor, ranking direction, quota and reproducibility limit.
Each property selector also shows a short explanation of its selected descriptor.
Changing a descriptor preserves the chosen direction, so review the pair together.
Expand advanced controls for objective directions, target intervals, scaffold or
cluster quotas and map settings. Inspect a point to see its structure, properties
and decision. Molecular drawings are generated only for the inspected molecule.

No universal prices, exchange rates or regional purchasing assumptions are used.
The budget in this version is a **molecule count**.

For a first comparison, state your question: property quality, coverage of
molecular cores, a justified property interval or expert fixed choices. Use
**Cover more molecular cores** to take one eligible representative per feasible
Murcko core before filling remaining places. It computes missing cores on demand;
it does not depend on drawing the map. Review acyclic structures, which share the
empty core, before imposing a strict maximum per scaffold. See the
[selection guide](SELECTION_STRATEGIES.md) for meanings, limitations and sources.

## Compare before adopting

1. Open **Compare scenarios A / B** and preview A.
2. Duplicate A into B, change a threshold, count or strategy, then preview B.
3. Compare retained, entered and removed molecules, scaffold coverage and overlap.
4. Export the selection/eligibility frequencies if useful.
5. Adopt a preview explicitly. Undo restores the previous selection.

Previews use the cached descriptors and do not alter the basket. Each scenario
captures its criteria and manual flags. Threshold changes replay the original
profile roles, violation tolerances, alert exclusions, QED policy and supported
reference restrictions. Chemically rejected molecules are not rescued merely by
pinning them. A changed-policy adoption that overrides the original screen needs
an explicit justification in the interface.

**Selection frequency** is the fraction of evaluated scenarios that select a
molecule. **Eligibility frequency** is the fraction where it meets the evaluated
policy. With two scenarios, frequencies can only be 0, 0.5 or 1. These are neither
activity probabilities nor statistical confidence estimates. Fixed and rescued
decisions are identified separately; they should not be interpreted as automatic
evidence of robustness. The existing `robustness_score` describes rule margins and
is a different quantity.

## Reproducible files

Save the study JSON to retain criteria, manual flags, ranking method and a hash
of the exact data and policy. Loading validates this identity before replaying.
The recipe contains no complete million-row molecule table. Keep the original
input and run configuration alongside it.

For an adopted scenario, the exported workbook includes `SCENARIO_CONTEXT`,
`Scenario_Eligible` and `Original_Screen_Status`. Profile columns describe the
original screening; `Final_Status` describes the actual exported selection.
A `.scenarios.json` sidecar records the adopted scenario separately from the
ordinary `.selection.json` recipe. The study replays the automatic snapshot;
subsequent manual changes remain in the workbook's decision history.

## Large libraries on Windows

Descriptor work remains batched and the processing worker count is sized from
available memory and CPU. Scenario evaluation reuses descriptors in blocks;
large selection paths omit per-rejected-molecule explanation dictionaries.
The full run and basket still occupy RAM proportional to the library size.

Exact Pareto ranking is limited to **2,000 eligible candidates** for scenarios.
Larger pools use deterministic weighted objective percentiles, explicitly labeled
in the UI and exported provenance. This is a different ranking method, not an
exact Pareto front and not a structural-diversity guarantee. There is no hidden
subsample of candidates used to make final selections.

The property-based strategies may return the same set above this threshold for
identical objectives and quotas. Core coverage retains its first-per-core policy;
weighted property percentiles order alternatives after scaffold rarity. The
threshold concerns the candidate pool, not the requested final molecule count.

Maps and tables limit display work. Missing scaffolds are computed when needed
for core coverage or scaffold quotas. Missing structural cluster assignments
still prevent cluster quotas; the interface reports their unavailability.
The property PCA cloud is a projection of descriptors, not a set of structural
clusters, and it does not determine export membership.
Scenario replay currently rejects runs with zone allocations. Such analyses
continue to use the existing pipeline, rather than silently losing zone rules.

Time and memory depend on descriptor count, alerts, library size and selected
count. The [benchmark](SELECTION_BENCHMARK.md) measures cached selection only;
it does not establish full SMILES-to-GUI throughput for a million real molecules.

## Optional Papyrus evidence

Build a local, target-specific Papyrus index with the
[documented command](PAPYRUS.md), then open the index in the Hub. Evidence is
queried on inspection, not by running millions of remote or similarity searches.
Check target, endpoint, quality, source and identity level. Connectivity matches
do not establish stereoisomer-specific activity, and absent evidence remains
unknown. Papyrus evidence does not silently replace selection rules.
## Final molecules and docking export (3.3)

The Chemical Space Hub opens with the molecules selected by the completed run.
Gold squares identify the actual final molecules that will be exported. Circles
give candidate context; triangles are references. Pale density bubbles summarize
population and are not selected molecules or cluster centroids. The map displays
the number of finals shown against the total final count.

Use **Show final molecules only** to isolate the final library. Overlapping
molecules keep their true projection coordinates; click the location to choose a
record from the inspection menu. Clicking only inspects. The inspector explicitly
states whether the molecule is included in docking export. Clicking a basket row
also opens that molecule, and final rows are listed first.

Set **Number of molecules**, strategy and optional quotas, then click **Create
selection**. The summary distinguishes the applied count from edited criteria
that have not been applied, and reports count differences. Changing controls alone
does not alter the final set. Undo restores the previous final library and criteria.

Use **Export to SMILES2Docking** for a CSV or XLSX table with `access_code` and
`smiles`; choose those columns in SMILES2Docking. The adjacent `.docking.json`
contains the file hash and decision provenance. **Export scientific report** saves
the full analysis workbook and selection recipe. Both use the current final set.
