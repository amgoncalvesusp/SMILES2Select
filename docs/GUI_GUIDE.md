# Guided selection and scenario comparison

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
Expand advanced controls for objective directions, target intervals, scaffold or
cluster quotas and map settings. Inspect a point to see its structure, properties
and decision. Molecular drawings are generated only for the inspected molecule.

No universal prices, exchange rates or regional purchasing assumptions are used.
The budget in this version is a **molecule count**.

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

Maps and tables limit display work. Missing scaffold/cluster data cannot be used
to enforce corresponding quotas; the interface or scenario evaluator reports
the unavailable analysis instead of treating missing data as measured coverage.
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
