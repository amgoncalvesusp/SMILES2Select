# Included chemical-tolerance profiles

These three recipe A profiles are experimental descriptive models from TOL-01. They evaluate compatibility with observed active chemistry. They are not activity classifiers, safety models or validated target-selection policies.

| File | Algorithm | Policy status | Automatic rescue |
|---|---|---|---|
| General_S1_A_experimental.json | Task/scaffold-balanced pooled distributions | infeasible | Disabled |
| Hierarchical_S2_A_experimental.json | Task/parent effective-support shrinkage | infeasible | Disabled |
| Contextual_S3_A_experimental.json | Fit-only molecular neighborhood with explicit fallback | infeasible | Disabled |

Each fit bank contains 3,830 positive identity/task records, 3,829 chemical identities, 1,732 nonempty scaffolds, 520 documents and 532 assay identifiers across ten tasks. The source is Papyrus++ 05.7, corrected 2024 and derived from ChEMBL34. Exact compatible endpoint measurements with pActivity >= 6.5 were retained; threshold conflicts were quarantined. The fit cap was 512 identities per task. The wider source census is not the number of training molecules in these packages.

Representation A has 1,442 properties, rule/alert and missing-mask features. Fit-only numerical scaling, separate numerical/binary blocks, equal block weights and applicability parameters are frozen. Joint relaxation support requires twenty identities, five nonempty scaffolds and three documents. A candidate's own identity cannot support its relaxation.

S1 calibration uses 996 positive identities and 2,141 unlabeled reference identities. S2/S3 calibration uses 998 positive identity/context pairs representing 996 identities and 21,410 reference query pairs representing 2,141 identities evaluated in ten declared contexts. Repeated query contexts are not new biological experiments or independent molecules.

The criterion scope is the six Lipinski/Veber rule identities recorded in each policy. The positive-retention goal is 95% and the reference-rescue cap is 10%. Reaching about 95% retention would require admitting 78.49% of reference R for S1, 72.28% for S2 and 72.58% for S3-A. Those candidate thresholds violate the cap. Therefore every distributed operational tau is null and rescue is disabled. Candidate retention/expansion fields describe the infeasible search result; they are not outcomes of an enabled policy.

These profiles can explain observed compatibility and preserve already eligible molecules. Review candidates do not enter automatic selection. Do not lower support or reinterpret the calibration fields to claim that these packages qualify a new compound as active. Custom fitting and independent recalibration are available through the existing training worker.

The same-namespace documentary-disjoint restriction leaves only six positives in one of the ten pilot tasks. An unresolved historical Papyrus/ChEMBL document-identifier crosswalk prevents a full publication-independence claim. The larger scaffold test is exploratory; strict temporal reconstruction is unavailable for the aggregate source. No profile has been qualified as the default molecular filter.

The complete study, training receipts, per-molecule evidence, comparison results and resource measurements are stored in `S2S-Decision/artifacts/s2-decision-tolerance-v1/`. Canonical model and file hashes are embedded in the profiles and the study's distributed-profile receipt. These assets are separate from historical activity-model packages and published 3.5.0 installers.
