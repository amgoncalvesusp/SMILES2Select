# Contextual selection and review

The contextual policy combines a declared activity model with visible chemical
rules, structural alerts and optional evidence from a separate risk endpoint.
It generates a proposed basket under the requested molecule count and diversity
constraints. This experimental workflow is included in version 3.5.0.
Included or imported JSON packages must be loaded explicitly. The existing
ONNX choices and default chemical selection remain available. All AI selection
and training modes require broader independent validation.

## Open and configure a proposal

1. Process the SMILES library and open **Chemical Space Hub → Prioritize by
   model**. Expand **Contextual policy (experimental)**.
2. Choose an included contextual model and select **Load included contextual model**, or use
   **Load contextual model JSON...** for a compatible external package.
   The target, species and endpoint are read from that package. The
   displayed model, data and chemistry versions identify the prediction recipe.
3. Review **Stage** and **Assay context**. Set the absolute requested molecule
   count and scaffold/cluster constraints through the workspace controls.
4. Review **Revisit these profile filters**. Only the listed upstream profile
   filters may be reconsidered in this proposal. The field initially contains
   `lipinski` and `veber` when those profiles were used in the processed run.
   Other mandatory profiles, manual exclusions, custom expressions and reference
   restrictions remain binding.
5. Configure optional alert actions, SMARTS conditions, risk evidence and review
   budget. Select **Preview contextual policy**.
6. Inspect the requested, eligible and proposed counts, retained/added/removed
   records, molecular evidence and review messages. **Adopt contextual proposal**
   applies the proposal as one undoable workspace action after confirmation.

The contextual preview starts from the valid processed structures while
preserving restrictions that were not explicitly selected for reconsideration.
It cannot recover molecules that were never loaded or were removed before the
processed result was saved. Preview and cancellation leave the current basket
unchanged. A change to context, constraints, model package or manual decisions
invalidates the proposal and requires another preview.

## Context and stages

The decision context records target, species, endpoint, activity threshold,
relation and unit, discovery stage, assay context, source version, model version
and chemistry version. For example, `IC50 <= 1000 nM` identifies a different
classification task from `Ki <= 1000 nM` or `IC50 <= 10 nM`. Incompatible activity
tasks or chemistry contracts are rejected before selection.

| Stage | Default contextual annotations | Interpretation |
| --- | --- | --- |
| `hit_finding` | Lipinski and Veber | Property annotations for examining candidate hits. |
| `lead` | Lead-like | A configured chemical-space view for starting-point optimization. |
| `fragment` | Rule of Three core | A configured fragment-space view. |

These stage profiles are informative by default. Switching stage changes the
profile policy; it does not retrain the activity model or create fragment- or
lead-specific activity probabilities. If the imported model was fitted for
`hit_finding`, another stage remains outside its declared activity context and
is flagged for review. Assay-context changes are also recorded and flagged.
Changing an editable assay description does not supply new experimental labels.

Learned rules that belong to an inactive stage profile remain in the audit
manifest as `inactive_rule_actions`. Their support is preserved, but their
actions are not applied to the new profile. Explicit user rules must belong to
the selected profiles; an arbitrary or misspelled rule identifier is rejected.

## Rules, magnitudes and structural alerts

For each active profile, the evidence records the rule identity, descriptor,
observed value, applicable bounds, violation status and normalized excess. A
molecular weight of 600 against an upper bound of 500 gives an excess of
`(600 - 500) / 500 = 0.2`. Missing descriptors remain missing. They do not become
measured violations or passing values.

Profile tolerances reuse the native definitions unless an explicit value is
chosen. In the action table, **Allowed violations** accepts **Native tolerance**
or an integer from 0 to 99 for profile rows: 0 is strict, while 1 or 2 permits
that many violations. Classical Lipinski uses one violation under its native
policy. The field is disabled for individual rule and alert rows. Individual
PAINS and Brenk patterns retain their catalogue and pattern identity, rather
than being reduced to a single count.

| Action | Effect |
| --- | --- |
| `inform` | Records the matched signal without changing rank or eligibility. |
| `warn` | Adds a review message without changing rank or eligibility. |
| `penalize` | Reduces selection priority by the configured amount. |
| `exclude` | Removes the candidate from that proposed basket. |

Use **Add explicit action** in **Advanced: explicit actions override learned
advice** to configure a profile, individual rule or named PAINS/Brenk pattern.
Choose **Kind**, **Feature / pattern**, **User action**, **Penalty** and, for a
profile, **Allowed violations**. The penalty is used only for `penalize`.
**Remove selected action** removes the
override. Repeated entries for the same feature are rejected. **Unspecified
alert action** supplies the default for patterns without a more specific action.
To apply a user rule outside the stage's default profiles, also add its profile
explicitly; a rule from an inactive profile cannot silently become a constraint.

The source of an action is recorded as user-configured or learned. For learned
individual actions, the operational support gate requires at least 20 molecules,
five positive and five negative labels, five scaffolds, three documents and a
named source. Assay counts are recorded separately. This gate limits unsupported
actions; those counts alone do not establish statistical reliability or absence
of confounding.

Unsupported learned actions become warnings. Learned actions also require a
compatible calibrated evidence context and applicable chemical domain. An
association with activity cannot establish an exclusion justified as toxicity
or interference. A high activity score does not cancel an explicit exclusion.
When an explicit mandatory profile, rule or alert constraint cannot be evaluated
because its input is missing, the candidate remains outside that constrained
basket with an unknown-input explanation.

## Optional SMARTS

Leave all three SMARTS fields blank to disable substructure preferences. Enter
one SMARTS pattern per line when a known substructure is relevant to the target
or to a project-specific constraint.

| Field | Effect |
| --- | --- |
| **Required SMARTS (optional)** | Every entered pattern must match. |
| **Excluded SMARTS (optional)** | Any entered pattern match excludes the candidate. |
| **Preferred SMARTS (optional)** | A match adds the configured ranking bonus; the core default is 0.1. |

Invalid SMARTS produce an error before adoption. A preference changes selection
priority only; it does not alter the model's activity score or provide evidence
of activity. Structural similarity, a privileged scaffold and a target-associated
substructure also do not establish selectivity.

## Activity, risk and uncertainty

`activity_score` retains the imported model output for its declared endpoint.
The separate `priority_score` combines that output with the selected actions;
penalties and preferences make it a decision score, not a calibrated probability.
Neither quantity is a probability of clinical success or a campaign-specific
probability of advancement.

Activity prediction sets are reported as subsets of `{0, 1}`. A singleton set
records the retained model class; `{0, 1}`, an empty set or missing set requires
review. A singleton does not turn the prediction into an experimental result.
Calibration method, calibration sample counts and class support travel with the
package. Applicability is reported separately from calibration. Domain and
coverage estimates remain tied to the chemical and assay distributions used in
their evaluation.

**Load measured-risk model JSON (optional)...** loads a separate model trained
against a measured endpoint. The resulting `risk_score` is a predicted
probability for the package's endpoint class; it is not a measurement of the
query molecule. Its biological system, source, chemistry recipe and package hash
remain distinct from the activity task. A cell-viability endpoint, for example,
is not relabelled as enzyme inhibition merely because both predictions are
shown for one molecule. The current risk adapter reports its applicability
domain as unknown, so risk predictions continue to require review.

**Risk ranking penalty weight** defaults to zero. **Exclude at chosen
measured-risk score (explicit constraint)** is off by default. Enabling either
requires separate risk evidence and source provenance. **Clear risk model**
removes the package and resets these options. With no risk package, the risk
score remains unavailable. Loading a package supplies a prediction, not an
experimental observation; neither a prediction, structural alert nor missing
measurement establishes that the query molecule is toxic, interfering or safe.

Unknown activity remains unknown in the evidence table and receives the lowest
internal ranking priority. Unknown risk, uncalibrated predictions, ambiguous
prediction sets, missing descriptors and out-of-domain molecules remain visible
through `review_required` and `policy_warnings`.

## Quantity, diversity and the separate review budget

The requested quantity is an absolute number of molecules. The native selector
applies scaffold and cluster limits, minimum scaffold coverage, pins and explicit
exclusions. If the eligible pool or constraints cannot supply the requested
number, the manifest reports the achieved count and shortfall. Pins can exceed
the requested count or diversity limits; the resulting conflict is reported.
An ineligible pinned candidate must be resolved or unpinned before selection.

**Additional review budget (informative candidates)** builds a separate subset
for examining uncertain, unfamiliar or filter-discordant candidates. Zero
disables this subset. The subset uses its own absolute count and scaffold cap;
it does not increase the main basket automatically. The review table displays
up to 200 rows, while the stored evidence contains the complete result.

Its priority is a descriptive combination of unknown activity, unavailable or
non-singleton activity prediction sets, unknown/out-of-domain status, and an
activity score of at least 0.5 accompanied by a rule violation. The 0.5 criterion
is a queue heuristic, not an experimentally established active label or a
validated estimate of expected information gain.

The core API can explicitly include optional policy rejects in this separate
queue with `include_policy_exclusions=True`. The GUI review subset uses the
policy-eligible population. Original hard exclusions, invalid structures and
manual excluded IDs remain outside both queue variants. Queue membership never
generates activity labels or experimental confirmation.

## Inspection, session history and export

Inspect a candidate to view its context, model score, prediction set, domain,
rule violations and magnitudes, matched alert identities, support, learned score
contributions and policy messages. Contributions explain the fitted score within
that model; they are associations and do not establish a causal mechanism.

Adoption retains the original processed results and stores the proposed
selection through the existing workspace history. Undo restores the previous
basket. Save the session to retain the applied context and evidence, and export
the selected basket through the existing workspace export controls. The
`MODEL_SCORES` sheet contains model and contextual evidence while the adopted
model proposal is active; the saved recipe records context, constraints, model
hashes and review-queue metadata. A fresh preview requires the compatible model
package even when historical inspection remains available from a saved session.

The implementation separates software verification from scientific evaluation.
Passing tests establish the recorded behavior of the policy, preview and
selection flow. Retrospective benchmarks assess recovery under specified data
and endpoints; they do not establish universal bioactivity prediction, absence
of assay interference, general safety or prospective experimental success.

See [selection strategies](SELECTION_STRATEGIES.md), [the GUI guide](GUI_GUIDE.md)
and [model qualification](MODEL_CATALOG_QUALIFICATION.md) for the existing
chemical and ONNX workflows.
