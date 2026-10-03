# SMILES2Select 3.5.0 — Methods and contextual model selection

This release explains how each selection method works and brings the completed
contextual-policy research workflow into the unified desktop application.
Included models are ready for local inference; users do not need to train them
before use. Every AI selection and training mode remains experimental and needs
broader independent validation.

## Interface and methods

- **Methods tab and menu:** an offline English guide explains descriptors,
  profiles, alerts, the six chemical selection strategies, quotas, molecular
  cores, maps, SMARTS, each model family and its training and calibration.
- **Visible selection criteria:** applied basket criteria are shown separately
  from settings for the next operation. Preview does not change the basket;
  adoption remains explicit and reversible.
- **Experimental contextual controls:** inspect rule and alert evidence, choose
  policy settings, optionally use known SMARTS substructures, and allocate a
  separate review budget. The requested number is a user-defined budget, not a
  fixed percentage of the library. Constraints can limit the attainable count.

## Models included

The twelve existing ONNX packages remain available: logistic regression with
Morgan fingerprints, scalar gradient boosting, gradient boosting with Morgan
fingerprints, and Tiny for each of three declared Papyrus tasks. Their original
weights and model qualification remain unchanged.

The contextual panel adds nine JSON activity packages covering human CA2/Ki,
AChE/IC50 and BACE1/IC50. These are **three logistic predictors with three
calibration variants each**, not nine independent predictors. Training used
frozen ChEMBL 37 development partitions, with a positive label defined by an
exact measurement at or below 1,000 nM. Choose and load an included model,
inspect its preview, then adopt it if appropriate for the study.

A separate optional model predicts the deposited **SH-SY5Y ATP viability at
48 hours** endpoint from PubChem AID1347400. It does not predict general toxicity
or prove assay interference. Its test set contained 39 molecules and five
positive calls; applicability to new chemistry remains uncertain.

The contextual full-feature model did not consistently outperform its simpler
logistic baseline in retrospective recovery. Stage settings configure policy;
no stage-success outcomes were learned. Broad multitask and Von experiments are
research tooling, not additional shipped GUI rankers. No models were refitted
or selected by favorable test seeds for this release.

## Reproducibility and training

Contextual decisions retain model hashes, context, evidence and explicit
selection/review outcomes in sessions and exports. Source tooling includes the
contextual, risk, multitask, target-panel and comparator studies and their
regression tests. Advanced training still requires a separately configured
Python environment; desktop inference does not require PyTorch or sklearn.

The new desktop-resource diagnostic checks the rendered Methods guide and
performs real inference with every included contextual package and the risk
model. Release CI runs it against frozen executables and installed Windows and
Linux applications, alongside the existing ONNX and installer lifecycle checks.

## Downloads

- **Windows installer:** `SMILES2Select-Setup-3.5.0.exe`.
- **Linux x86_64 installer:** `SMILES2Select-Setup-3.5.0-linux-x86_64.run`.
  Run `bash SMILES2Select-Setup-3.5.0-linux-x86_64.run` as your normal user.
- **Portable bundles:** `SMILES2Select-windows.zip` and
  `SMILES2Select-linux.tar.gz`.
- Python wheel, Windows native-dependency manifest and
  `SHA256SUMS-v3.5.0.txt` accompany the release.

Application code remains MIT-licensed. Historical ONNX resources retain their
CC BY-SA 4.0 notice; new ChEMBL-derived contextual activity resources carry
CC BY-SA 3.0 attribution. The risk package retains its source attribution.

See the [Methods guide](https://github.com/amgoncalvesusp/SMILES2Select/blob/v3.5.0/src/smiles2select/assets/methods.md),
[contextual workflow](https://github.com/amgoncalvesusp/SMILES2Select/blob/v3.5.0/docs/CONTEXTUAL_SELECTION.md)
and [README](https://github.com/amgoncalvesusp/SMILES2Select/blob/v3.5.0/README.md).
