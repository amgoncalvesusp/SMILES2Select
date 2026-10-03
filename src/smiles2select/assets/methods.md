# Methods

SMILES2Select reduces a molecular library to the number you request, using chemical criteria, diversity constraints and, optionally, a trained activity model. The requested count is an absolute budget. It is not a fixed percentage of the starting library. Constraints or too few eligible molecules can leave a shortfall.

**All AI-assisted selection, model training and learned contextual policies in this application are experimental. They require more robust validation on independent chemistry, relevant assays and prospective use before their predictions can support reliable scientific decisions.** A model score helps prioritize compounds; it cannot confirm activity, selectivity, safety or success in a discovery project. Availability in the application means that the method can be executed, not that its biological usefulness has been established.

## Choosing a workflow

Start with the question your experiment will answer. For a broad library without a matching activity model, use explicit property objectives and structural coverage. For a supported target and endpoint, compare a model proposal with a chemical baseline under the same filters, requested count and quotas. A known binding motif can become an optional SMARTS requirement or preference; record why you chose it.

The original processing run and the Chemical Space Hub are separate decision stages. The run standardizes structures, computes properties, applies screening and any configured post-selection. The Hub lets you examine that evidence and replace the final basket. A molecule can pass chemical screening but remain outside the final basket because of count or diversity limits. A plotted point, shortlist entry or high score does not by itself mark a molecule for export.

In the Hub, **Create selection** applies the selected chemical strategy. **Preview model selection** creates a separate model-ranked proposal; changing the chemical-strategy dropdown does not change its model scores. Inspect added, retained and removed molecules before adopting a proposal. Contextual selection can explicitly revisit named optional profile filters, while preserving other mandatory restrictions. No workflow can recover structures absent from the saved input universe.

## Structure preparation

SMILES parsing and molecular sanitization establish whether RDKit can interpret a structure. Cleanup, salt removal, neutralization, tautomer handling and stereochemistry settings change the representation used for calculations. Save those settings with the input. Two apparently similar datasets can produce different identities, descriptors and fingerprints if they use different preparation recipes.

The bundled activity packages use cleanup and salt removal, remove stereochemistry, and leave neutralization and canonical tautomer conversion disabled. Their Morgan fingerprints also omit chirality. This matches their historical training representation, but it means stereoisomers can collapse to one model identity. Tautomers and protonation states may remain distinct. The original structure remains evidence; model preparation does not replace an experimental record with a chemically equivalent measurement.

Model inference recomputes features under the package's chemistry contract and checks compatibility. Its manifest records the preparation recipe, descriptor definitions, profile and alert versions, RDKit version and resource hashes. If the contract does not match, use compatible processing settings or a compatible model package before inference. Valid structures can still lack experimental labels: a parsable SMILES does not imply an observed activity result.

## Descriptors and profiles

Descriptors summarize the supplied molecular representation. Their units and calculation methods matter when selecting cutoffs.

| Descriptor | Interpretation and useful choice | Limit |
|---|---|---|
| Molecular weight | Prefer a justified size window, or smaller structures when the experiment calls for them. The displayed g/mol value is numerically equivalent to molecular mass in Da. | The smallest molecule is not necessarily the best ligand. |
| WLOGP | RDKit Wildman-Crippen estimate of octanol/water logP. An interval often expresses a lipophilicity goal better than maximizing it. | It is not measured solubility, pH-dependent logD, XLOGP3 or MLOGP. |
| TPSA | Topological polar surface area in square angstroms. Useful for comparing calculated polarity. | It does not measure permeability, absorption or distribution. |
| H-bond donors and acceptors | The profile uses the recorded RDKit/Lipinski counting definition. | Counts depend on the implementation and chemical representation. |
| Rotatable bonds, rings, heavy atoms, heteroatoms and fraction Csp3 | Describe flexibility and composition; inspect them alongside the structures. | A count does not describe every conformer, stereochemical feature or binding interaction. |
| QED | Continuous drug-likeness desirability from 0 to 1, combining eight property terms. Higher favors its reference property profile. | It does not predict target activity or safety. Combining QED with its component properties can emphasize the same preference twice. |
| SA score | Synthetic-accessibility heuristic, approximately 1 to 10; lower suggests easier synthesis. | It supplies no synthesis route, yield, supplier availability or price. |
| NP score | Natural-product likeness based on molecular fragments; higher favors patterns associated with natural products. | It does not establish natural origin, safety or bioactivity. |

Profiles define a chemical space rather than a universal order of molecular quality. The application ships the following profiles; its visible rules and saved configuration are the exact operational definitions.

| Profile | Purpose and implemented bounds |
|---|---|
| Lipinski | Oral-drug heuristic: molecular weight at most 500, WLOGP at most 5, donors at most 5 and acceptors at most 10. The native policy tolerates one violation; zero allowed violations gives the strict interpretation. |
| Veber | Flexibility/polarity heuristic: at most 10 rotatable bonds and TPSA at most 140. |
| Ghose | Weight 160-480, WLOGP -0.4 to 5.6, molar refractivity 40-130 and total atom count 20-70. Total atoms and heavy atoms are different descriptors. |
| Egan | The application's SwissADME-compatible bounds use WLOGP at most 5.88 and TPSA at most 131.6. |
| Muegge | RDKit adaptation: weight 200-600, WLOGP -2 to 5, TPSA at most 150, rings at most 7, carbon count greater than 4, heteroatom count greater than 1, rotatable bonds at most 15, acceptors at most 10 and donors at most 5. |
| Lead-like | Starting-point space with room for optimization: weight 250-350, WLOGP at most 3.5 and at most 7 rotatable bonds. |
| Rule of Three core | Fragment-oriented bounds: weight at most 300, WLOGP at most 3, donors and acceptors each at most 3. |
| Rule of Three extended | Adds at most 3 rotatable bonds and TPSA at most 60 to the core profile. |
| Beyond Rule of Five | Extended space: weight at most 1,000, WLOGP -2 to 10, donors at most 6, acceptors at most 15, rotatable bonds at most 20 and TPSA at most 250. These permissive bounds do not validate oral exposure. |
| CNS-like | Configured heuristic: weight at most 400, TPSA at most 90, donors at most 3, WLOGP 1-4, rotatable bonds at most 8 and combined nitrogen/oxygen count at most 5. Its native policy tolerates one violation. It does not predict brain penetration. |
| Dockability Envelope | Broad operational window: weight 120-700, TPSA at most 200, donors at most 10, acceptors at most 16, rotatable bonds at most 18 and WLOGP -3 to 8. It does not predict docking success or binding affinity. |

Other listed profiles require every rule by default. Mandatory, consensus, informative and ranking policies have different effects; review the configured action instead of assuming every displayed profile is an exclusion filter. Applying oral, lead and fragment profiles cumulatively can defeat the intended study design. Fragment hits often begin at weaker potency, so changing to a fragment profile does not justify reusing a hit-potency label as a fragment-success criterion.

Rule margins describe distance from a configured threshold. The Hub's robustness score uses the smallest normalized rule margin, with failures scoring zero. It measures sensitivity to the chosen cutoffs, not experimental confidence. A rule count also loses information: exceeding molecular weight by 1 and by 200 produces the same single violation. Contextual evidence therefore retains the rule identity, observed property, threshold and normalized excess.

## Structural alerts

An alert is a substructure match to a named catalogue pattern. PAINS and Brenk warn by default; NIH, ZINC and custom SMARTS are informative by default. A catalogue hit does not automatically exclude a molecule. The four available actions are **inform**, **warn**, **penalize** and **exclude**. Inform records evidence, warn asks for attention, and exclude changes eligibility. A contextual penalty changes the decision priority. Screening-stage penalties contribute to an alert/consensus score, which affects ordering only when the selected strategy uses that score. QED-only ranking does not silently acquire an alert penalty.

Read the pattern and its intended endpoint before using it. Reactivity, aggregation, optical interference and cell toxicity are different phenomena. PAINS matches do not prove false-positive activity, and absence of a match does not establish safety. For a known active compound with an alert, inspect the assay and available counter-assays. Retain both the measured activity and the alert in its evidence record.

In the contextual workflow, named individual PAINS/Brenk patterns can carry separate evidence and actions. Learned advice requires enough training molecules, both activity classes, distinct scaffolds and documents, plus a named source. Unsupported advice remains a warning. The source of an action is visible: a user-configured restriction and a fitted association are not the same form of evidence.

## Pareto and selection strategies

An objective can maximize a property, minimize it, prefer a desired value or prefer an interval. Values inside an interval have equal desirability; outside it, nearer is better. Objectives rank eligible candidates. They do not add hard screening cutoffs.

Under exact Pareto ranking, molecule A dominates B if A is at least as good in every enabled objective and better in at least one. Successive nondominated fronts receive increasing Pareto ranks. Crowding distance measures spread among the selected properties, while distance to the ideal measures closeness to their preferred values. Neither is a fingerprint-distance calculation.

The Hub provides six chemical strategies:

| Strategy | Ordering and when to use it |
|---|---|
| Rank by QED only | Descending QED, with record ID resolving ties. A transparent chemical baseline for comparing trained models. Other property objectives are ignored; pins and quotas still apply. Missing or invalid QED stops selection. |
| Balance properties and representation | Better Pareto front first, then higher crowding distance, higher rule-margin robustness and higher QED. Use when both property trade-offs and property spread matter. |
| Prioritize favorable property trade-offs | Better front first, then smaller distance to the ideal and higher QED. Use when closeness to the chosen objectives matters more than broad property spread. |
| Spread across property values | Higher crowding distance first, then better front and higher robustness. This can favor extremes and does not maximize structural dissimilarity. |
| Cover more molecular cores | Choose one feasible representative per Murcko core first, favoring rarer cores, then fill remaining places. Use to reduce domination by large analogue series. Crossed quotas can limit attainable coverage. |
| Complete my pinned choices | Preserve eligible pins and order the remaining candidates by Pareto front and QED. Other strategies also preserve pins when that option is enabled. |

These keys are compared in order, not averaged. Missing ranking columns are skipped; missing values in a present column sort last. Integer record ID resolves the final tie. Objective weights do not alter exact Pareto dominance.

Above **2,000 candidates in the ranking universe**, the property strategies use a weighted sum of objective percentile ranks instead of exact Pareto sorting. The threshold concerns ranked candidates, not the final requested count. Identical objectives and constraints can then give identical baskets under several strategy names. Core coverage still reserves feasible new cores; QED-only still uses descending QED at every size. The exported ranking method identifies which path ran. This approximation does not optimize fingerprint diversity.

The original pipeline has additional strategies, including traditional screening, stratified allocation, reference novelty, reference neighborhood and reference-aware diversity. Its `diversity_first` and `reference_aware_diversity` use fingerprint MaxMin selection; the latter starts from the reference library. The Hub's similarly named property-spread strategy uses crowding distance. Record the decision stage as well as its name. Zone allocation belongs to the original pipeline; rerun without zones before replacing that automatic selection in the Hub.

## Fingerprints, similarity and scaffolds

Morgan fingerprints encode local molecular environments in a fixed bit vector. The included activity models use radius 2 and 2,048 bits without chirality. Hash collisions and lost stereochemical information limit what a bit match can mean. The same parameters must be used for the query and references.

Tanimoto similarity for binary fingerprints is the number of shared set bits divided by the number set in either molecule. Larger values indicate closer representations under that fingerprint. They do not give a probability of shared activity. Reference novelty is one minus the highest allowed reference similarity, so a novel-looking compound can also be outside the model's reliable domain.

A Murcko scaffold represents the ring systems and connecting linkers of a molecule. A maximum per scaffold limits repeated cores, while a minimum scaffold count seeks broader coverage. Acyclic molecules share the empty scaffold bucket in the selector. A cap of one can therefore retain only one automatic selection from that entire bucket, despite substantial chemical differences among its members.

Structural clustering uses Butina over Morgan/Tanimoto distances. The default distance cutoff is 0.35. Cluster membership depends on the cutoff and representation; it is not a biological class or a claim that every pair in a cluster has identical similarity. Exact clustering constructs a quadratic distance list and is guarded above 10,000 valid molecules. Cluster quotas require real cached assignments; the shape of a map is not a cluster assignment.

The native selector preserves eligible pins, reserves feasible new cores when requested, and fills in rank order under scaffold and cluster maxima. If crossed upper quotas prevent filling the basket, deterministic augmenting paths can exchange automatic choices to increase its size. This repairs count feasibility; it does not globally optimize molecular quality or minimum-core coverage. Pins can create an excess or make constraints infeasible. Review the achieved count and warnings rather than assuming the requested count was met.

## Chemical-space maps

The default property map uses principal component analysis (PCA) on standardized descriptors. Standardization prevents weight, measured in hundreds, from overwhelming a descriptor such as QED, measured from zero to one. The implementation imputes remaining missing descriptor values from the displayed data's means, reports imputation and retains explained variance. Its deterministic axis convention keeps repeated maps from flipping signs.

Optional UMAP arranges local neighborhoods from descriptors or fingerprints. Its appearance depends on neighborhood settings, metric and seed. Neither a PCA cloud nor a UMAP island is an activity prediction. Projection to two dimensions loses information: nearby points can differ in unplotted dimensions, and overlapping points can hide several molecules.

Each point refers to an actual record. The first selected member of a cluster is not automatically a centroid or medoid. Background sampling affects what the map displays, not final membership. Use the complete exported molecular table to verify a basket.

## Reference evidence and SMARTS

Known active and below-threshold references can help you explore a chemical neighborhood or test novelty. Their activity labels must match the named target and endpoint. A local Papyrus annotation is evidence retained for an identity, not a new prediction. A missing annotation means no retained match; it does not mean inactive.

Optional SMARTS constraints describe substructures. In contextual selection, required patterns restrict eligibility, excluded patterns remove matches, and preferred patterns add a configured ranking preference. A preference changes the decision score without changing the model's activity probability. Invalid SMARTS are rejected. Check a query on known matching and nonmatching structures, particularly when aromaticity, charge or stereochemistry matters. A target-associated motif can guide a hypothesis, but it cannot prove a binding mode or rescue an incompatible activity model.

## Experimental model selection

The application includes two package families: 12 historical ONNX activity models and nine contextual activity packages from the B15 study, with a separate optional B15 risk package. They have different training data and inference contracts. They must not be treated as interchangeable checkpoints of one universal model.

Choose a compatible target, species, endpoint, activity threshold and chemical recipe before comparing algorithms. IC50, Ki and EC50 are distinct quantities even when all are expressed in molar concentration. The label pActivity at least 6 corresponds to a concentration at most 1 micromolar for the declared endpoint. A result below the threshold is negative for that criterion, not proof of no biological effect.

Use the simple model as a comparison before assuming that a neural network or more features will help. Keep eligibility, requested count, pins and scaffold/cluster limits fixed. Inspect changes in recovered known positives, false exclusions and series concentration alongside overall ranking metrics. A model can rank well on an active-rich published collection yet perform poorly when screening a much less active library.

Included models run locally. Training is an optional source/Python workflow with additional dependencies; the desktop executable is not a bundled PyTorch training environment. Loading, selecting or changing the stage of a model does not train it on your library. All new training and all AI-assisted selection remain experimental and need more robust validation.

## Bundled model training

The 12 ONNX packages come from the frozen three-task product study using Papyrus++ 05.7 without stereochemistry. The source audit covered 707,461 rows, but each package learned only its own target/endpoint subset. None was trained on all those rows as a universal activity predictor.

The product study retained Papyrus high-quality single-endpoint records, handled activity relations explicitly, and removed invalid structures, unknown labels, repeated-identity groups and previously exposed identities. Exact measurements can supervise quantitative pActivity. A censored measurement supplies a class only when its bound logically determines that class; it never becomes an exact regression target. Absent measurements are not negative labels. The model does not learn the application's final-selection flag.

Training, validation, calibration and test partitions use Murcko scaffold groups with seed 42, approximately 70/10/10/10 of groups. Molecule counts differ because whole groups stay together. Training fits transforms and predictors; validation selects a Tiny checkpoint and the historical simple-model choice; calibration fits the probability transformation; later testing measures performance. Training context uses five scaffold folds so that a training molecule does not see same-fold labeled references. Other partitions and deployed inference use only packaged training references.

| Papyrus task | Accepted molecules | Training | Validation | Calibration | Test |
|---|---:|---:|---:|---:|---:|
| Q72547_WT / IC50 | 2,560 | 1,803 | 240 | 289 | 228 |
| P0DMS8_WT / Ki | 1,406 | 1,000 | 116 | 118 | 172 |
| Q07869_WT / EC50 | 493 | 345 | 44 | 36 | 68 |

Each package stores its training references. The four alternatives for a task therefore share the same reference pool. Seven context values summarize maximum and top-five mean similarity to positive and negative references, novelty, scaffold frequency and local reference density. Query identity is excluded from its own context. These values preserve measured-reference information; they are not uncertainty estimates.

The common property input contains 20 values: molecular weight, WLOGP, molar refractivity, TPSA, donor/acceptor counts, rotatable bonds, heavy atoms, heteroatoms, rings, aromatic rings, charge, fraction Csp3, QED, SA, NP, Lipinski/Veber violation counts and PAINS/Brenk counts. Training-derived transforms apply log1p to selected counts, median imputation and scaling, with missingness indicators. QED, fraction Csp3 and reference-context values keep their unit-interval scale. Candidate libraries reuse these transforms; they do not refit them.

The original simple-model choice was the highest validation average precision among logistic regression, scalar boosting and boosting with Morgan bits. Tiny was a prespecified comparator, outside that choice rule. The choice was frozen before test-performance evaluation. Historical Tiny export did process test features in evaluation/parity passes, without using their labels or performance for fitting or selection; the preserved record does not claim absolute non-access to test features. This same-source retrospective check was not prospective or independent-source biological validation.

## Bundled model catalogue

The accession names refer to [HIV-1 reverse transcriptase/RNaseH, fragment record Q72547](https://www.uniprot.org/uniprotkb/Q72547/entry), [human adenosine receptor A3, P0DMS8](https://www.uniprot.org/uniprotkb/P0DMS8/entry), and [human PPAR-alpha, Q07869](https://www.uniprot.org/uniprotkb/Q07869/entry). `_WT` is the historical Papyrus task identifier. It does not make a package applicable to every strain, variant or assay involving a similarly named protein.

| Package name | Predictor | Historical role |
|---|---|---|
| Q72547_WT_IC50 | Gradient boosting, properties/context | Validation-selected simple baseline |
| Q72547_WT_IC50__logistic_scalar_morgan | Logistic regression with Morgan | Experimental comparator |
| Q72547_WT_IC50__gradient_boosting_scalar_morgan | Gradient boosting with Morgan | Experimental comparator |
| Q72547_WT_IC50__tiny_scalar_morgan | Tiny neural network | Experimental comparator |
| P0DMS8_WT_Ki | Logistic regression with Morgan | Validation-selected simple baseline |
| P0DMS8_WT_Ki__gradient_boosting_scalar | Gradient boosting, properties/context | Experimental comparator |
| P0DMS8_WT_Ki__gradient_boosting_scalar_morgan | Gradient boosting with Morgan | Experimental comparator |
| P0DMS8_WT_Ki__tiny_scalar_morgan | Tiny neural network | Experimental comparator |
| Q07869_WT_EC50 | Logistic regression with Morgan | Validation-selected simple baseline |
| Q07869_WT_EC50__gradient_boosting_scalar | Gradient boosting, properties/context | Experimental comparator |
| Q07869_WT_EC50__gradient_boosting_scalar_morgan | Gradient boosting with Morgan | Experimental comparator |
| Q07869_WT_EC50__tiny_scalar_morgan | Tiny neural network | Experimental comparator |

All four choices per task estimate its pActivity-at-least-6 class. The label "validation-selected" records the historical selection procedure; it does not certify a model as biologically validated. Model cards and manifests describe later evaluations without altering the frozen training record.

## Tiny

Tiny is a small supervised neural network with three input branches. It is not a language model, a pretrained SMILES encoder or a local reasoning assistant. Each included Tiny package was initialized and trained separately for one task.

| Block | Dimensions in the included models |
|---|---|
| Molecular properties and missingness | 40 → 32 → 16 |
| Morgan bits | 2,048 → 128 → 64 |
| Reference context and missingness | 14 → 32 → 16 |
| Concatenation and fusion | 96 → 64 → 32 |
| Output layer | 32 → 2 |

Each branch/fusion block uses Linear, ReLU, dropout 0.1, Linear and ReLU. The two outputs are an activity logit and a standardized pActivity estimate. The included architecture contains 281,730 trainable parameters. Different branches let structural bits, global properties and measured-reference context contribute before fusion; their widths are engineering choices, not a demonstrated optimum.

The training loss combines binary cross-entropy with 0.2 times Huber loss on exact quantitative pActivity. Each term ignores missing labels. The auxiliary regression term retains potency information beyond the binary threshold. Thus a Tiny-versus-classifier comparison includes a difference in supervision, not just architecture.

The product fits used AdamW, learning rate 0.001, weight decay 0.0001, batch size 256, at most 80 epochs and patience 15. Validation average precision selected the restored checkpoint; a plateau scheduler reduced the learning rate when improvement stopped. No test metric chose an epoch or seed. The generic training command has different defaults, so its configuration must be recorded for a new fit.

Use Tiny to test whether nonlinear combinations change useful selections relative to the simpler alternatives. Its predicted pActivity is an auxiliary estimate and does not rank the basket. More parameters do not establish better recovery; the historical comparisons did not show a consistent Tiny advantage.

## Logistic regression and gradient boosting

Logistic regression adds weighted input features and transforms the result through a sigmoid. The included models use molecular properties, reference context and Morgan bits, with L2 regularization, C=1, an LBFGS solver and a 1,000-iteration ceiling. They are useful supervised baselines because the form is simple. Correlated descriptors and hashed bits still make coefficients difficult to interpret causally.

The included boosting models use histogram gradient boosting with binary log loss, learning rate 0.1, at most 100 iterations, at most 31 leaves per tree and at least 20 samples per leaf. Seed 42 and the remaining estimator settings are recorded in each manifest. These historical fixed configurations are different from the B15 research boosting grid.

The scalar version uses 40 property/missingness inputs and 14 context/missingness inputs. The Morgan version adds 2,048 bits. Scalar boosting is therefore not structure-free: its reference context still depends on fingerprint similarity. Compare the two to assess the added value of direct structural bits under the same task and constraints. Trees can capture nonlinear interactions, but small or biased datasets can support unstable splits.

Logistic regression and boosting provide classification outputs only; they have no quantitative pActivity head. Their ONNX classifiers emit raw class probabilities. The runtime clips these to a finite logit and applies the stored calibration sigmoid. Tiny applies its calibration sigmoid directly to its activity logit. Calibration slopes for these historical packages are constrained positive and fitted only on the calibration partition, with at least 20 labels and five of each class. Otherwise the package reports insufficient calibration rather than claiming a reliable probability.

The packaged status `fitted_held_out_not_prospectively_validated` describes a fitted probability scale on a separate historical subset. It does not guarantee accuracy for a new assay, prevalence or chemical domain. ONNX parity tests verify numerical agreement with the trained predictor; they do not add biological evidence.

## Research S2 Decision

S2 Decision also names a broader research programme. Its Papyrus multitask experiment used 170,508 molecules, 286,234 observed labels and 189 target-endpoint tasks across 173 targets. Only about 0.89% of the label matrix was observed; missing task labels stayed unknown. A shared network used 2,088 inputs, hidden layers of 512 and 256 units and 189 outputs, with masked binary cross-entropy. Each model had 1,249,469 parameters.

Those fits used AdamW, learning rate 0.001, weight decay 0.0001, dropout 0.1, batch size 512, at most 40 epochs and patience six. Validation macro average precision selected checkpoints for seeds 42, 43 and 44. Dedicated calibration was supported for 135 tasks; 54 retained flagged raw outputs. Their historical calibrators were not the positive-slope contract of the bundled single-task models: some fitted slopes reversed rankings. Per-task calibration must be inspected before combining outputs.

This multitask predictor remains a research artifact with a different matrix contract. It is not one of the 12 bundled ONNX classifiers or the nine included contextual packages. Its sparse labels and uneven improvements do not establish a general probability that a molecule is bioactive, and it cannot infer an arbitrary new target from a name or sequence.

Later research compared shared and separate networks for human CA2/Ki, AChE/IC50 and BACE1/IC50 using ChEMBL 37. Results varied by target. Von experiments likewise did not produce a generally superior, qualified production model. Von and Jev are not hidden backends of the local molecular selector. Use only the concrete model package and task actually available in the interface.

## Contextual policy and risk

The included B15 contextual activity packages cover human **CA2/Ki (P00918)**, **AChE/IC50 (P22303)** and **BACE1/IC50 (P56817)**. Each task has calibration seeds 42, 43 and 44. These nine packages contain three target-specific full logistic predictors with different calibration/conformal partitions; they are not nine independent architectures. Choosing a seed does not retrain the model, and the study did not select a best test seed.

B15 used frozen ChEMBL 37 data: 7,120 task rows across 7,063 identities, with exact measurements at or below 1,000 nM as positives. Training counts were 2,606 for CA2, 1,010 for AChE and 736 for BACE1. Internal development identities and nonempty scaffolds were globally separated across tasks; recent cohorts were identity-separated but could share development scaffolds. These are different weights, data and tasks from the historical Papyrus ONNX library.

The full logistic inputs combine Morgan bits, continuous properties, counts, rule identities/magnitudes and supported individual alerts. Median imputation and scaling use training data only. Admission of individual rule/alert features required at least 20 flagged training molecules, five labels of each class, five nonempty scaffolds and three documents. Validation average precision selected C from 0.1, 1 and 10, with smaller C resolving exact ties. Other study controls included feature ablations, MACCS, nonlinear boosting, shared models and assay-context interactions. Their study results do not make every comparator an included GUI model.

Activity calibration uses a custom monotone sigmoid with nonnegative slope and slope regularization 0.0001. The original calibration partition is split by scaffold hash into separate probability and class-conditional conformal subsets. Conformal sets use alpha 0.1 and a finite-sample quantile. They are empirical uncertainty evidence under the observed data distribution, not a guarantee of 90% coverage after chemical, temporal or assay shift.

The policy keeps activity prediction, risk evidence and decision actions separate. It records target, species, endpoint, threshold, stage, assay context and versions. Hit-finding uses Lipinski/Veber annotations, lead uses Lead-like and fragment uses Rule of Three core. Profiles are informative by default. Changing stage changes the configured policy; the study had no measured stage-progression outcomes and trained no stage-success predictor. Unsupported stage or assay context produces review evidence. A high activity score cannot cancel an explicit mandatory exclusion.

An optional included risk model predicts **SH-SY5Y ATP-viability loss at 48 hours**, using PubChem AID 1347400. This is a separate Morgan/logistic endpoint model, not an AChE interference or general safety classifier. Of 274 measured identities, 23 were reserved by activity-evaluation scaffold overlap. The remaining fixed scaffold-hash split contained 128 training, 48 validation, 20 probability-calibration, 16 conformal-calibration and 39 test identities; the test had five risk-positive observations.

The risk classifier uses the 2,048 Morgan bits, L2 regularization and an LBFGS solver with a 2,000-iteration ceiling. Validation average precision selected C from 0.1, 1 and 10, with smaller C resolving ties. Its monotone calibration sigmoid uses slope regularization 0.000001; a separate subset supplies class-conditional conformal sets at alpha 0.1. These choices were fixed before test evaluation. The small sample limits interpretation, and its applicability domain remains unknown.

Loading the risk package predicts a value for the query; it does not mean the query was experimentally measured. The default risk ranking weight is zero and hard risk exclusion is off. Three other prepared risk endpoints failed the study's fixed support gates and have no included fitted model. Cell viability, cell-free autofluorescence and redox interference retain their distinct meanings. Missing risk evidence never becomes a safe label.

The B15 study also compared QEX, an independently authored implementation of published target-specific drug-likeness equations. It fits property desirability to known training actives and combines the terms; it is not a target-activity classifier or proof that an alert is harmless. QEX remains a research comparator rather than a standalone included selector. QED is available directly as a chemical strategy.

The primary retrospective comparisons requested 50 molecules with a maximum of three per scaffold. The full contextual activity model recovered 12 versus 14 positive calls for its simple baseline in an AChE fluorescence cohort, and 10 versus 13 in the previously observed Ellman cohort. Neither those results nor the separate viability measurements established consistent superiority. The external calls concerned deposited AC50 activity labels, not the training IC50 threshold; the fluorescence chemistry was already represented in historical exposure. The included packages remain experimental and require more robust independent validation.

## Reading scores and uncertainty

An activity probability refers to the package's declared class. A priority score can also contain configured penalties or SMARTS preferences, so it is a decision quantity rather than a calibrated probability. A percentile describes a molecule's position within the current eligible library; it is not a second probability and can change when the library changes.

For contextual prediction sets, `{1}` retains the positive class, `{0}` the negative class, `{0, 1}` both, and an empty set neither. Ambiguous, empty or missing sets call for review. Even a singleton remains a model result. Read class-specific coverage: a high overall rate in a mostly negative test set can hide poor positive-class coverage. The historical ONNX packages do not acquire conformal guarantees merely because the contextual workflow supports prediction sets.

Similarity to training references and pattern-support counts describe applicability evidence. They are separate from calibration. A high score with low reference similarity, unknown domain or sparse support is a reason to inspect the molecule and assay assumptions, not to suppress those warnings.

Average precision summarizes retrieval over the observed ranking and depends on prevalence. ROC AUC, Brier error and calibration error answer different questions. For a requested screening budget, also inspect recovered positives, precision, recall, enrichment, false exclusions, achieved N, scaffold coverage and property distributions. A held-out label means held out from that fit; historical reuse can still make the overall experiment exploratory. Technical seeds on one split are not independent biological replicates.

The optional information-seeking queue has its own count and diversity constraints. It prioritizes uncertain, unfamiliar or filter-discordant candidates for review without enlarging the main basket automatically. It generates no labels and has not been validated as a prospective active-learning campaign.

## Reproducing a selection

Keep the input files and hashes, record-ID mapping, software/dependency versions, chemistry recipe, references, descriptor and fingerprint settings, profile/alert actions, objective directions and weights, model package hashes, ranking universe, requested count, quotas and manual decisions. Preserve the actual final IDs and per-molecule reasons with the recipe. Reordering an import can change record IDs and therefore tie resolution.

Preview does not alter the basket. Adoption records one undoable decision; a change in model, context, chemistry, criteria or manual state invalidates an old preview. Save the session for later inspection and use the normal exports to retain the selected molecules, model evidence and context. Replaying upstream chemistry also requires its original inputs and environment; a recipe alone cannot recreate missing source files.

For a new training study, fix the endpoint and label rule before inspecting model outcomes. Separate molecular identities and appropriate scaffold groups; retain a genuine independent or temporal evaluation when possible. Fit transformations and support thresholds on training data, use validation for choices, reserve calibration for probability/uncertainty estimates, and evaluate all prespecified models on the same candidate universe and budget. Keep unfavorable results and failed support gates. Laboratory-free retrospective testing can assess recovery of recorded observations, but it cannot supply new experimental confirmation.

## Sources and packaged records

Each included package carries its training record and source provenance. The model guide displays the chosen package's task, calibration and reference information. Historical ONNX records live under `s2s_decision/bundled_models`; contextual/risk packages and their `info/provenance.json` live under `s2s_decision/bundled_contextual_models`. The source repository also contains `docs/SELECTION_STRATEGIES.md`, `docs/MODEL_CATALOG_QUALIFICATION.md` and `docs/CONTEXTUAL_SELECTION.md`. Application and installer tests check software behavior; the study records describe biological evidence and its limits.

Primary method references include [Lipinski's original discussion of solubility and permeability](https://doi.org/10.1016/S0169-409X(96)00423-1), [Veber's molecular-property study](https://doi.org/10.1021/jm020017n), [the Rule of Three proposal](https://doi.org/10.1016/S1359-6446(03)02831-9), [QED](https://doi.org/10.1038/nchem.1243), [QEX](https://doi.org/10.1007/s11030-018-9842-3), [Papyrus](https://doi.org/10.1186/s13321-022-00672-x), and [RDKit's descriptor and fingerprint documentation](https://www.rdkit.org/docs/RDKit_Book.html). These references explain methods; they do not validate every cutoff, model or combined policy in a new experiment.
