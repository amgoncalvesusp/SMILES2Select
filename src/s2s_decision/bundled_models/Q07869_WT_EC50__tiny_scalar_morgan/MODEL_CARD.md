# Q07869_WT_EC50 / tiny_scalar_morgan

## Task and selection status

Tiny neural network with property, reference-context and Morgan fingerprint branches. Scope: **Q07869_WT / EC50 / pActivity >= 6**.

Comparison model. This package was not the historically selected baseline; inclusion in the catalog does not change that frozen choice. Tiny remains experimental, even where its validation average precision was higher than a baseline. Validation average precision (AP): 0.927677. This number describes one retrospective partition; it is not evidence of universal superiority. No test-set result was used to revise the frozen baseline choice.

## When to choose this model

Use as an experimental comparator to the simpler baselines for the same target and endpoint. Inspect out-of-domain warnings and disagreements between baskets. Tiny learns nonlinear combinations and also has a pActivity regression output, but neither its complexity nor that extra output establishes better prospective performance.

Choose only when the target, endpoint and threshold match your question. Run a preview before adopting the basket. Keep input filters, requested count and per-scaffold cap equal when comparing models; inspect selected structures, applicability warnings and remaining scaffold diversity. A model cannot rank molecules removed by upstream eligibility filters. If no bundled task matches, use a compatible task-specific imported model or the general selection strategies.

## Data and interpretation

Trained on Papyrus++ 05.7, with **345 training reference records**. The references in this package belong only to the training partition. Scaffold-based train, validation, calibration and test partitions were prepared under seed 42. The separate calibration partition fitted the probability transform. Activity probability is conditional on this task and its historical labels; it is not a universal probability of biological activity, experimental confirmation or probability of project advancement. Predictions on unfamiliar chemistry require particular caution.

The downstream internal retrospective evaluation used held-out records from the same data source. It does not establish independent-source or prospective validity. Tiny's optional predicted pActivity is a model estimate; it does not replace an assay.

Historical protocol observation: test feature tensors were processed in evaluation-only forward passes and potentially ONNX parity before the choice freeze. `protocol-observation-02.json` records that chronology; test performance was not used for fitting, early stopping, calibration or model choice. This was not a strict no-access protocol for test features.

## Provenance and license

Manifest, ONNX and reference bytes are unchanged from `S2S-Decision/artifacts/product-validation-v1/models/Q07869_WT_EC50/tiny_scalar_morgan`. The historical manifest predates subsequent held-out evaluation and is preserved rather than rewritten. This release card adds task-specific usage guidance and current evaluation context.

Frozen manifest SHA-256: `4e139abc0b401b727ac9a7c84bb7594d89c5942ae08e2fc36aecc1a7e681e3d6`. ONNX SHA-256: `6b7ec93307a41b2ecaa342647abd0d4dffaab33cb07c593765d3be2d029f0e8f`.

Protocol and frozen choices: `S2S-Decision/artifacts/product-validation-v1/{plan.json,source-extraction.json,validation-choice-freeze.json,REPORT.md}`. Chemical contract: SMILES2Select 3.3.1 / RDKit 2026.03.5, verified by the compatible integrated runtime. Attribution and resource license: [CC BY-SA 4.0](../LICENSE_CC_BY_SA_4.0.md), with [full license](../LICENSE_CC_BY_SA_4.0.txt). Application code has a separate MIT license.
