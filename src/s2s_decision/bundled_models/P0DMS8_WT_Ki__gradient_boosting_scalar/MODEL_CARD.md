# P0DMS8_WT_Ki / gradient_boosting_scalar

## Task and selection status

Gradient boosting with scalar descriptors and reference context. Scope: **P0DMS8_WT / Ki / pActivity >= 6**.

Comparison model. This package was not the historically selected baseline; inclusion in the catalog does not change that frozen choice. Validation average precision (AP): 0.933853. This number describes one retrospective partition; it is not evidence of universal superiority. No test-set result was used to revise the frozen baseline choice.

## When to choose this model

Use to prioritize nonlinear combinations of properties and reference-context features without direct Morgan fingerprint bits in the estimator. Its context still uses Morgan similarity to training references; this is not a structure-free model. Compare with gradient boosting plus Morgan to examine whether explicit structural bits change your basket.

Choose only when the target, endpoint and threshold match your question. Run a preview before adopting the basket. Keep input filters, requested count and per-scaffold cap equal when comparing models; inspect selected structures, applicability warnings and remaining scaffold diversity. A model cannot rank molecules removed by upstream eligibility filters. If no bundled task matches, use a compatible task-specific imported model or the general selection strategies.

## Data and interpretation

Trained on Papyrus++ 05.7, with **1,000 training reference records**. The references in this package belong only to the training partition. Scaffold-based train, validation, calibration and test partitions were prepared under seed 42. The separate calibration partition fitted the probability transform. Activity probability is conditional on this task and its historical labels; it is not a universal probability of biological activity, experimental confirmation or probability of project advancement. Predictions on unfamiliar chemistry require particular caution.

The downstream internal retrospective evaluation used held-out records from the same data source. It does not establish independent-source or prospective validity. Tiny's optional predicted pActivity is a model estimate; it does not replace an assay.

## Provenance and license

Manifest, ONNX and reference bytes are unchanged from `S2S-Decision/artifacts/product-validation-v1/models/P0DMS8_WT_Ki/gradient_boosting_scalar`. The historical manifest predates subsequent held-out evaluation and is preserved rather than rewritten. This release card adds task-specific usage guidance and current evaluation context.

Frozen manifest SHA-256: `bb68b68912d6dc14839a83a89d774fa356ffccd9affa343a25587e7eaf83e148`. ONNX SHA-256: `1dca37a1b9f9e5d7a12d8f769cefee236fd2f55f62adadc67cfce283d5aa12a3`.

Protocol and frozen choices: `S2S-Decision/artifacts/product-validation-v1/{plan.json,source-extraction.json,validation-choice-freeze.json,REPORT.md}`. Chemical contract: SMILES2Select 3.3.1 / RDKit 2026.03.5, verified by the compatible integrated runtime. Attribution and resource license: [CC BY-SA 4.0](../LICENSE_CC_BY_SA_4.0.md), with [full license](../LICENSE_CC_BY_SA_4.0.txt). Application code has a separate MIT license.
