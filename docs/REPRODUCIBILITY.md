# Reproducibility

Selection recipes record candidate and reference libraries, standardization,
fingerprint parameters, projection settings, zones, strategy, final/reserve
counts and random seed. Exact reference results record the search mode and
verified Tanimoto values.

The recipe is machine-readable JSON. The same recipe plus the same inputs
should reproduce the same automatic selection; changed source hashes, toolkit
versions or fingerprint parameters must be treated as a different analysis.

## Applied decisions in version 3.3

Opening the Hub imports the original pipeline's final molecules. Its initial
recipe is marked `original_pipeline`; GUI objectives have not yet been applied.
Create selection replaces this set. Each automatic action freezes its requested
count, quotas, objectives, strategy and available Pareto table. Editing controls
does not rewrite that snapshot. Undo/redo restores the corresponding snapshot.
Manual decisions and the actual final IDs remain explicit.

The input hash covers ordered input records and computed descriptors, including
schemas and row identities. A separate selection-data hash includes candidate
annotations, scores, chemical eligibility and reference data. These are SHA-256
digests of versioned pandas value-hash encodings, not checksums of the original
file bytes. Keep the original files and recorded Python, pandas, NumPy and RDKit
versions. The recipe includes complete run configuration and profile definitions.

Workspace ranking uses the evaluable library as its ranking universe and applies
chemical/reference eligibility before final allocation. Up to 2,000 evaluable
records use exact Pareto ranking; larger libraries use weighted objective
percentiles. Scenario studies rank their eligible pool and record this separately.
The constrained selector is deterministic, with ascending record ID for ties;
it does not randomly sample final molecules. The map's seed and projection affect
visualization only. See [Selection strategies](SELECTION_STRATEGIES.md) for sort
keys, quota repair and limitations.

The SMILES2Docking hand-off contains the current basket, including justified
manual overrides. Its `.docking.json` companion stores the exact exported IDs and
SMILES, count/shortfall, reasons, active action history, recipe and file SHA-256.
Keep these files together. A generic workspace recipe is an audit record; it is
not a claim of an automatic replay loader for arbitrary manual sessions. The
existing scenario-study JSON loader replays supported A/B studies against their
recorded data fingerprint.

Version 3.3.1 calculates missing Murcko scaffolds on demand when the chosen
strategy or quotas require them, including libraries above 5,000 molecules.
Each distinct canonical SMILES is calculated once within the operation. A known
empty scaffold represents an acyclic molecule; a missing value is not treated as
an acyclic core. Invalid structures fail explicitly. The completed scaffold
values are included in the scientific export.

Scenario-study schema 2 binds replay to the selector version and SHA-256 of the
sorted final record IDs. Scaffold-based studies also verify the completed
scaffold inputs. Availability of this derived cache alone does not change the
candidate fingerprint. Legacy schema 1 studies cannot prove parity with the
changed coverage algorithm and are refused explicitly; preserve the original
software and study for those analyses. A new study is a new analysis, not a
replacement for historical evidence.

GUI exports require the displayed criteria to have been applied. If pins, quotas
or manual decisions produce a different final count, the export dialog states
both numbers and requires explicit acknowledgement. A scenario whose options
cannot be represented by the Hub controls can be previewed but cannot be adopted
through those controls. The original library is not silently exported in place
of a requested selection that failed.

Map metadata records the projected candidate IDs, sampling seed and population
counts. Every final molecule is included even if the usual display budget is
exceeded. Recomputing a projection after the selected set changes may change its
axes; do not interpret that movement as a change to molecular properties.
