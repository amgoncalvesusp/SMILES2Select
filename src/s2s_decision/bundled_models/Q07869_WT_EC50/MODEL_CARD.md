# Q07869_WT_EC50 / logistic_scalar_morgan

## Task and selection status

Logistic regression with scalar descriptors, reference context and Morgan fingerprints. Scope: **Q07869_WT / EC50 / pActivity >= 6**.

Validation-selected baseline. This historical choice was frozen among the three baseline candidates before inspecting test predictions. Validation average precision (AP): 0.900646. This number describes one retrospective partition; it is not evidence of universal superiority. No test-set result was used to revise the frozen baseline choice.

## When to choose this model

Use as a compact baseline when you want a linear combination of molecular properties, similarities to training references and structural bits. Compare its selected basket with the nonlinear alternatives using the same eligible pool and scaffold limit. Correlated inputs and unfamiliar scaffolds can still produce misleading scores.

Choose only when the target, endpoint and threshold match your question. Run a preview before adopting the basket. Keep input filters, requested count and per-scaffold cap equal when comparing models; inspect selected structures, applicability warnings and remaining scaffold diversity. A model cannot rank molecules removed by upstream eligibility filters. If no bundled task matches, use a compatible task-specific imported model or the general selection strategies.

## Data and interpretation

Trained on Papyrus++ 05.7, with **345 training reference records**. The references in this package belong only to the training partition. Scaffold-based train, validation, calibration and test partitions were prepared under seed 42. The separate calibration partition fitted the probability transform. Activity probability is conditional on this task and its historical labels; it is not a universal probability of biological activity, experimental confirmation or probability of project advancement. Predictions on unfamiliar chemistry require particular caution.

The downstream internal retrospective evaluation used held-out records from the same data source. It does not establish independent-source or prospective validity. Tiny's optional predicted pActivity is a model estimate; it does not replace an assay.

## Provenance and license

Manifest, ONNX and reference bytes are unchanged from `S2S-Decision/artifacts/product-validation-v1/models/Q07869_WT_EC50/logistic_scalar_morgan`. The historical manifest predates subsequent held-out evaluation and is preserved rather than rewritten. This release card adds task-specific usage guidance and current evaluation context.

Frozen manifest SHA-256: `fc5b7a1aa7382645e0e4fb52abd394b5778ff5cf95193b69d8ac0a1f0ad96912`. ONNX SHA-256: `0267bd1f9b1e71660c91f3d9f527bda526417b9258afc5f58b0a219109870a3d`.

Protocol and frozen choices: `S2S-Decision/artifacts/product-validation-v1/{plan.json,source-extraction.json,validation-choice-freeze.json,REPORT.md}`. Chemical contract: SMILES2Select 3.3.1 / RDKit 2026.03.5, verified by the compatible integrated runtime. Attribution and resource license: [CC BY-SA 4.0](../LICENSE_CC_BY_SA_4.0.md), with [full license](../LICENSE_CC_BY_SA_4.0.txt). Application code has a separate MIT license.
