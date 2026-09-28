# Qualification of bundled models

Updated 28 September 2026 for SMILES2Select 3.4.1. Resource qualification and installer verification are distinct checks.

## Catalog correction

Version 3.4.0 bundled only the three historically validation-selected baselines. Version 3.4.1 restores all 12 qualified packages from `S2S-Decision/artifacts/product-validation-v1/models`: four candidates for each of three tasks. The nine restored comparators include all three Tiny models. No model was retrained. Original manifests, ONNX weights and training-reference files remain byte-identical to the source packages; hashes match `validation-choice-freeze.json`.

| Task | Historically selected baseline | Training references per package |
|---|---|---:|
| Q72547_WT / IC50 / pActivity >= 6 | `gradient_boosting_scalar` | 1,803 |
| P0DMS8_WT / Ki / pActivity >= 6 | `logistic_scalar_morgan` | 1,000 |
| Q07869_WT / EC50 / pActivity >= 6 | `logistic_scalar_morgan` | 345 |

The three baseline choices were frozen from validation performance before held-out performance evaluation. Restoring comparators does not revise those choices. Tiny remains experimental, including where its validation average precision exceeded that of a baseline. Catalog availability does not imply universal suitability or superiority.

The historical Tiny implementation processed test feature tensors in evaluation-only forward passes and potentially during ONNX export parity before the choice freeze. `protocol-observation-02.json` records this chronology. Those operations did not use test performance to fit weights, stop training, calibrate probabilities or select models; they must not be described as strict non-access to test features.

## Package names and selection guidance

The three names already distributed in 3.4.0 remain unchanged: `Q72547_WT_IC50`, `P0DMS8_WT_Ki`, and `Q07869_WT_EC50`. New comparator names use `<task>__<candidate>`, for example `Q72547_WT_IC50__tiny_scalar_morgan`.

| Candidate available for every task | Inputs and intended comparison |
|---|---|
| `logistic_scalar_morgan` | Linear baseline using properties, reference context and Morgan bits. Compare with nonlinear models under identical eligibility filters and scaffold limits. |
| `gradient_boosting_scalar` | Trees using properties and reference context. Context still uses Morgan similarity to training references; this is not a structure-free model. |
| `gradient_boosting_scalar_morgan` | Trees using properties, reference context and explicit Morgan bits. Compare with scalar boosting to inspect changes from adding structural bits. |
| `tiny_scalar_morgan` | Experimental neural network with property, reference-context and Morgan branches. Assess incremental benefit over simpler baselines; greater complexity does not establish better recovery. |

Choose a model only when its target, endpoint and threshold match the question. Inspect applicability warnings, selected structures and scaffold diversity before adopting a preview. Keep filters, requested count and scaffold cap equal across comparisons. Models cannot restore molecules removed by upstream eligibility filters. When no bundled task matches, use a compatible task-specific imported model or the general selection strategies. Each package contains its own `MODEL_CARD.md` with task scope, model family, selection status and guidance.

## Provenance

All 12 packages share historical chemical hash `a2a3bbe752d2485920b99afd4b3338ab1d0654b8c39894898b1fd04725776923`, SMILES2Select 3.3.1 and RDKit 2026.03.5. Papyrus++ 05.7 consolidated source SHA-256 is `8004e0d1027a760f205b45264386f792e7d49658da39f77f52e660a6f19760dd`. `source-extraction.json` links each task subset to its hash. `plan.json` defines exclusions, scaffold splitting, seed 42 and the selection criterion. `validation-choice-freeze.json` records every manifest and ONNX hash. The 3,148 task-specific reference records are repeated across the four models per task; all belong to training, never validation, calibration or test.

`REPORT.md` contains the subsequent retrospective internal evaluation. It does not establish independent-source or prospective validity. Activity probabilities concern the declared task and historical labels; they do not confirm biological activity or estimate project advancement. Tiny's optional predicted pActivity is an estimate, not an assay measurement.

## License

O [registro oficial do Papyrus 05.7 no Zenodo](https://zenodo.org/records/13987985) lista o arquivo `05.7++_combined_set_without_stereochemistry.tsv.xz` com MD5 `9a325b965fb2ab043679d974d1f79549`; MD5 local coincide. Metadados da [API oficial do registro](https://zenodo.org/api/records/13987985) declaram `metadata.license.id = cc-by-sa-4.0`. O `LICENSE.txt` contido em `05.7_additional_files.zip` repete a licença CC BY-SA 4.0; hash MD5 do ZIP coincide com o exibido no registro. [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) permite redistribuição e adaptação sob suas condições de atribuição e CompartilhaIgual.

All 12 model packages have a separate [CC BY-SA attribution notice](../src/s2s_decision/bundled_models/LICENSE_CC_BY_SA_4.0.md) and [full license](../src/s2s_decision/bundled_models/LICENSE_CC_BY_SA_4.0.txt). Application code remains MIT-licensed. Copied resources are `manifest.json`, `model.onnx`, and `references/{manifest.json,records.jsonl}`. Updated release cards explain subsequent evaluation and usage without modifying frozen manifests or data.

## Compatibility and verification

The runtime retains RDKit 2026.03.5 because existing model chemistry contracts require this version. Later versions require compatibility verification; historical hashes must not be rewritten to bypass a mismatch.

Catalog tests lock all 12 manifest and ONNX hashes, check train-only references, discover all candidates and verify resource licenses. Real inference tests check calibrated finite probabilities, record identifiers and regression availability against each manifest. Frozen-worker tests cover all 12 distributed packages, including Tiny, using eligible and ineligible candidates. These checks establish integrity and execution, not prospective efficacy. Installer qualification additionally requires running those tests against the actual Windows and Linux build outputs.

Local evidence: `S2S-Decision/artifacts/product-validation-v1/{plan.json,source-extraction.json,validation-choice-freeze.json,protocol-observation-02.json,REPORT.md,models/}`. Study artifacts remain unchanged.
