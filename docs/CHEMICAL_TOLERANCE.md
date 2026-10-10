# Chemical tolerance (experimental)

Chemical tolerance evaluates whether a molecule's rule violations are compatible with chemistry observed among measured actives. It adds a separate workflow to the existing SMILES2Select workspace. Historical activity predictors keep their own thresholds, packages and interpretation.

The workflow has four algorithms:

| Algorithm | Evidence learned from the fit partition | Main restriction |
|---|---|---|
| S0, fixed learned limits | Empirical magnitude limits with equal reference weights | Marginal limits still require joint support for the complete violated pattern. |
| S1, general | Pooled active distributions balanced by task and scaffold | Compatibility applies to the observed training population. |
| S2, hierarchical | Target distributions shrunk toward compatible species/endpoint distributions | Missing target support produces an explicit parent fallback or review. |
| S3, contextual | Joint evidence within a fit-only molecular neighborhood | Unsupported patterns, distant chemistry and excessive magnitudes require review. |

The four representation recipes are A (properties, rules, magnitudes, individual alerts and missing-value masks), B (A plus Feature Morgan, 2,048 bits), C (A plus the continuous 315-value ErG vector) and D (A plus Morgan, 2,048 bits). Numerical and binary evidence are separate distance blocks. Numerical distances use fit-only robust scaling; binary masks use mean mismatch; fingerprints use Tanimoto distance. Blocks have frozen equal weights, rather than allowing a fingerprint block to replace chemical properties.

## Preparing a custom profile

The statistical fit uses valid structures with exact, compatible endpoint measurements at pActivity >= 6.5. A missing value does not become an inactive observation. Concentrations in M, mM, uM, nM and pM are converted to pActivity; Greek micro symbols are accepted. Censoring directions are reversed under the logarithmic conversion. Conflicting exact identity/task measurements are quarantined. Endpoint, target, species, document and assay identifiers are retained.

```bash
python -m s2s_decision tolerance-prepare --input measurements.csv --output prepared
python -m s2s_decision tolerance-train --input prepared --output profile --mode general --recipe A
python -m s2s_decision tolerance-inspect --input profile --output inspection.json
python -m s2s_decision tolerance-predict --input library.csv --model profile --output decisions
```

Preparation assigns identities and scaffolds to frozen fit, tune, policy calibration and test roles. Training accepts the prepared folder or the original CSV, but it verifies prepared-data checksums before reuse. Target, species and endpoint defaults must be specified before preparation if the source does not contain them. Supplying a target name later does not create activity data for that target.

Independent positive and reference calibration files must be supplied together with `--positive-cal` and `--background-cal`. The positive file must contain exact positive measurements in the requested context. The reference file describes an unlabeled sample evaluated in that same query context. Neither biological inactivity nor precision can be inferred from reference membership. Calibration rejects overlap with reserved fit, tune or test identities/scaffolds and removes known positive identities from the reference population.

The policy chooses the smallest threshold that meets the requested positive-retention goal and reference-rescue cap. Unsupported positives remain in the retention denominator. The threshold is bound to exact criterion identities and calibration contexts. An infeasible or uncalibrated policy produces descriptive evidence and cannot enable automatic rescue. These are empirical checks, without a distribution-free guarantee after chemical or assay shift.

For adaptation, add `--base-profile existing_profile`. The base annotation dictionary, representation transforms and domain threshold remain frozen; custom fit references are merged and their weights/distributions are recomputed. The previous policy threshold is discarded. Reserved custom or external calibration groups must not overlap the shared fit bank. Adaptation therefore needs fresh independent policy calibration.

## Using the workspace

1. Analyze the library and set the absolute requested N, quotas and pins using the existing selection controls.
2. Open **Chemical tolerance (experimental)**. Load an included profile or a verified package. Choose General, Target-conditioned or Custom use, and inspect its actual algorithm, training counts, criterion scope and policy status.
3. Check only the rules or native alert entries authorized for reconsideration. Hard exclusions, manual exclusions and locked SMARTS remain binding. A partial reconsideration of a coupled consensus policy is rejected. Selecting an alert entry does not disable its catalog.
4. Generate a preview and inspect evidence. Molecules already eligible under the reference policy remain eligible. Unsupported candidates receive **review** and stay outside automatic selection.
5. Adopt explicitly. Adoption is undoable. Export or save the session to retain the frozen decision and molecular evidence.

The final basket uses the existing balanced selector with QED priority. Eligibility preservation does not imply that every previously selected molecule remains in a basket of fixed N; admitted candidates compete for the same places. Infeasible N or quotas produce a visible deficit. The selector does not fill that deficit with review or excluded candidates.

Changes to the library, rule definitions, package bytes, context, criterion identities, N or quotas invalidate the proposal. Verification replays inference and selection before adoption. Saved sessions retain the evidence without requiring the original package; recalculation requires matching package bytes and hashes.

Custom preparation and training use the existing cancelable worker and Decision Studio. Frozen applications require an external compatible Python environment for training. Statistical inference uses the bundled numerical and chemistry runtime and does not load a transformer at startup.

Before launching custom training, Studio shows a provisional resource estimate. A raw file supplies its byte size; row counts remain unknown until preparation. Prepared metadata supplies row and fit-row bounds and the A–D feature dimension. The displayed float32 matrix size covers one array, not peak RAM: chemistry, tables, copies, reference banks, adaptation and calibration require additional memory. No training-duration estimate is invented. Manifest preview reads are bounded and do not scan the corpus on the interface thread; the worker still validates the dataset before fitting.

When a screening run has a persistent descriptor-cache file configured, tolerance preview and adoption reuse that file through a separate SQLite annotation table. Its key binds canonical chemistry, the versioned annotation recipe and actual descriptor/SMARTS observations. Source row IDs and biological labels are not cached as chemical evidence. Reopening the cache preserves the same annotations and decisions; changing the recipe or supplied observations invalidates reuse. Bounded JSON entries have checksums, and corrupted entries are logged and recomputed. A run without a configured cache follows the ordinary computation path. Prepared custom-data folders also preserve their annotated rows independently of this optional cache.

## Packages and records

The profile schema is `s2s-tolerance-profile/1`. A standalone JSON is supported. Directory packages contain exactly `manifest.json` and `profile.json`; the annotation dictionary, bounded support bank and fitted recipe are validated inside the profile. The maximum individual package-file size is 256 MiB. Model cards belong outside a strict two-file package directory.

Validation rejects unexpected files, symbolic links, duplicate JSON keys, nonfinite values, invalid dimensions, nonmonotone CDFs and checksum mismatches. It does not load pickle or execute package code. Writes are staged and published atomically. Existing output paths are preserved.

Prediction produces `records.jsonl`, `decisions.csv` and `report.json`. JSONL retains the complete input metadata, fingerprint and structured evidence. CSV escapes spreadsheet formulas and omits the long hexadecimal fingerprint to avoid numeric overflow or spreadsheet precision loss. All prediction records remain present, including invalid structures with explicit evidence.

## Evaluation and present status

TOL-01 separates eligibility from final selection and frozen calibration from equal-capacity comparisons. Rescue capacities of 1%, 5%, 10% and 20% of the baseline-rejected universe are research settings; the product continues to accept an absolute N. Every arm uses the same library, hard exclusions, QED priority, selector and quotas.

Logistic regression, gradient boosting, a newly trained single-output Tiny, Laya and Von are a separate sample-origin pilot. They distinguish observed actives from an unlabeled reference sample, using the same chemical information. Their scores are not biological activity probabilities. Training receipts, unsuccessful context-budget attempts, fitted checkpoints, policy transfer and controlled-capacity outcomes are retained in the study directory.

The current source dataset contains aggregates that cannot support a strict temporal reconstruction. Historical exposure and documentary overlap also reduce independent test availability. Broad scientific qualification and default promotion require a sufficiently supported independent cohort. A technically functioning model remains experimental when that requirement is unmet.

The complete implementation study is stored under `S2S-Decision/artifacts/s2-decision-tolerance-v1/`. Its protocol, curation manifests, training receipts, comparison tables, resource measurements and final report are separate from earlier activity-model benchmarks.
