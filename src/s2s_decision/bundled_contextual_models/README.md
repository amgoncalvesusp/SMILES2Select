# Included contextual models

These packages provide local inference in the experimental contextual controls.
Choose a model explicitly, inspect its preview, then adopt the proposed selection
if it fits your study. Loading a package does not replace the current basket.
The application does not train these models again when you use them.

## Activity packages

Nine JSON files contain three full logistic predictors, each with three
calibration variants. Their tasks are human CA2 Ki, AChE IC50 and BACE1 IC50.
For these models, an exact measurement <=1,000 nM is a positive label. Missing
activity is unknown. Seeds 42, 43 and 44 vary calibration partitions; they do
not identify nine independently trained molecular predictors.

The B15 study fitted Morgan fingerprints, continuous descriptors, rule counts,
supported rule magnitudes and individual structural alerts using the frozen
ChEMBL 37/B13 development partitions. Calibration and conformal subsets remain
separate. Each package includes its feature transformations, coefficients,
support records, context, calibration and training-reference fingerprints.

The full model did not consistently beat its matched activity baseline. At
N=50 with at most three compounds per scaffold, external AChE recovery was
12 versus 14 positive calls for fluorescence and 10 versus 13 for the previously
observed Ellman source. Packaging these models makes comparison possible; it
does not establish superior performance or validate new chemical libraries.
Hit, lead and fragment settings configure profile policies. No stage-success
labels were available for training.

The activity models and retained ChEMBL-derived reference data are distributed
under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/), with
attribution to ChEMBL, EMBL-EBI, release 37. SMILES2Select's B13 curation and B15
feature construction, fitting and calibration are the transformations applied.
See the [ChEMBL data-license statement](https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/general-questions).
Application code remains MIT-licensed.

## Separate risk package

`risk/shsy5y_atp_viability_48h.json` predicts the endpoint deposited by NCATS
in [PubChem AID1347400](https://pubchem.ncbi.nlm.nih.gov/bioassay/1347400):
SH-SY5Y ATP viability at 48 hours. Its curated cohort contained 274 identities: 128 for training, 48 for validation,
20 for probability calibration, 16 for conformal calibration, 39 for testing and
23 reserved because of activity-evaluation scaffold overlap. The test contained
five positive calls. It does not predict
general toxicity or confirm interference with the activity assay. Its chemical
applicability remains unknown, and query compounds do not become measured
observations merely because a prediction is shown.

The generated risk-model parameters are supplied with the project under its MIT
license. Attribute the original assay to its depositor and retain its endpoint
identity. PubChem aggregates deposited information; source attribution and any
source-specific data terms remain relevant. See [PubChem downloads and source
licensing](https://pubchem.ncbi.nlm.nih.gov/docs/downloads).

## Integrity and methods

`info/provenance.json` records the source-study seal and SHA-256 for all ten
packages. Activity packages are byte-identical to the completed B15 outputs.
The risk package replaces an archived developer-local source path with the public
assay URL; its numerical parameters and other metadata are unchanged. Provenance
records both risk-file hashes. No model was refitted, relabelled or selected using
a favorable test seed for this release.

Open **Methods** in the application for the full explanation of the basic
chemical strategies, all included model families, training, uncertainty,
selection constraints and evidence limits. Every AI selection or training mode
is experimental and needs broader independent validation.
