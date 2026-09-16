# Selection strategies and reproducibility

## Original pipeline and Selection Hub are separate decision stages

The original run applies chemical screening and its configured post-selection
rules. Pipeline strategy identifiers include `traditional`, `balanced`,
`diversity_first`, `reference_novelty`, `reference_neighborhood`,
`reference_aware_diversity`, `stratified` and `manual_assisted`. Reference novelty
uses molecular fingerprints. The pipeline's `diversity_first` and
`reference_aware_diversity` use fingerprint MaxMin selection; the latter starts
from the reference library. Their `selection_seed` belongs to that algorithm.
Zone allocation, reserve counts and reference restrictions belong to this stage.

The Selection Hub starts with the original run's final molecules. Applying a new
Hub selection replaces that final set using the current Hub strategy and
constraints. A molecule can pass chemical screening yet be absent from the
original final set because of allocation or count limits. Chemical approval and
final membership are distinct. Reference eligibility restrictions remain in
force; a run with zone allocation must be rerun without zones before changing
its automatic selection in the Hub.

The Hub's `diversity_first` uses Pareto crowding distance as listed below. It is
not the pipeline's fingerprint MaxMin algorithm. The name alone is insufficient
to reproduce a result: record the decision stage and ranking method.

## What each plotted molecule means

A molecular point identifies one actual `record_id`, not a cluster centroid.
The first selected member of a cluster is an actual exported molecule; the
selector does not compute a medoid or centroid when assigning that explanation.
Several molecules from the same cluster can be final molecules when the quota
allows it. Selection status determines export membership. A shortlist or a map
highlight alone is not final membership.

Map projection and background sampling serve visualization. They do not choose
the final molecules. Retain the complete final table when auditing the selected
set, including molecules whose projected points overlap.

## Hub ranking

For exact Pareto ranking, dominance uses the enabled objectives after converting
them to a common desirability direction. Higher desirability is better. Maximize,
minimize, target value and target interval objectives are supported. Missing
objective values are given a score below the observed minimum for that
objective; missing values are not invented measurements.

The constrained selector compares the following keys **lexicographically**, in
the stated order. It does not average these keys into a single score. A lower
priority key is used only when all earlier available keys tie.

| Hub strategy | Ordered keys |
| --- | --- |
| `balanced` | `pareto_rank` ascending; `crowding_distance` descending; `robustness_score` descending; `qed` descending |
| `pareto_first` | `pareto_rank` ascending; `distance_to_ideal` ascending; `qed` descending |
| `diversity_first` | `crowding_distance` descending; `pareto_rank` ascending; `robustness_score` descending |
| `scaffold_coverage` | `scaffold_size` ascending; `pareto_rank` ascending; `robustness_score` descending |
| `manual_assisted` | `pareto_rank` ascending; `qed` descending |

Unavailable columns are skipped, missing values within a present ranking column
sort last, and ascending integer `record_id` resolves the final tie. For
`scaffold_coverage`, scaffold size is the number of candidates sharing that
scaffold in the applicable selection pool. Crowding distance measures separation
in objective space, not pairwise fingerprint diversity. Objective weights do
not change Pareto dominance and are not an additional key in this exact Hub
ordering.

### Libraries above the exact Pareto threshold

The exact ranking threshold is **2,000 molecules in the ranking universe**, not
2,000 final molecules. At or below that threshold, enabled objectives use exact
Pareto ranking. Above it, ranking uses the weighted sum of objective percentile
ranks:

`score_i = sum(weight_j * percentile_rank(desirability_ij))`

Percentiles use average ranks for ties. Higher scores are better. The internal
`selection_priority` is `-score`, so smaller priority is better; ascending
`record_id` resolves equal priorities. This explicit priority replaces the
strategy keys in the table above. Scaffold and cluster constraints still apply.
It is a deterministic weighted-percentile approximation, not an exact Pareto
front, fingerprint diversity optimization or random sample. Objective weights
therefore have a different role in this mode than in exact Pareto ordering.

The interactive workspace ranks all evaluable candidates before filtering its
eligible selection pool. Scenario replay reevaluates screening first and ranks
its eligible/rescued pool. These ranking universes can yield different ranks or
percentiles even with identical objective names. Preserve the recorded universe
and method when comparing or replaying results. With no enabled objectives, the
scenario selector uses available strategy columns without fabricating Pareto
ranks.

## How the requested final count is filled

The Hub selector uses this sequence:

1. Remove explicitly excluded IDs from the candidate pool.
2. When preservation is enabled, add eligible fixed molecules in ascending ID
   order. Fixed molecules have precedence over automatic count and upper quotas.
3. When a minimum scaffold count is requested, reserve candidates from new
   scaffolds greedily in ranking order, respecting the available count and upper
   quotas.
4. Fill remaining places in ranking order, skipping candidates that would exceed
   the maximum per scaffold or cluster.
5. If crossed scaffold and cluster upper quotas leave a count shortfall, repair
   the selection with deterministic augmenting paths. These paths exchange
   automatic choices until the requested count is reached or no count-increasing
   path remains. Fixed molecules are never displaced and scaffold coverage
   already obtained is preserved.

For example, suppose the ranking is molecule 1 `(scaffold A, cluster 1)`, molecule
2 `(A, 2)`, molecule 3 `(B, 1)`. With requested count 2 and both maxima equal to 1,
a greedy pass picks only molecule 1. The repair replaces it with molecules 2 and
3, reaching the requested count without violating either maximum. Reasons and
quota usage are recomputed for the repaired final set.

This repair maximizes cardinality under the two upper quotas with fixed pins;
it does not globally maximize a quality score. It operates only when the count
is short. The minimum scaffold reservation remains a heuristic: if its minimum
is missed, the warning does not prove that no selection could meet it. For
example, with the same candidates, count 2, minimum 2 scaffolds and only a
maximum of 1 per cluster, the greedy result may contain molecules 1 and 2.
Molecules 2 and 3 would satisfy the minimum, but the count is already filled and
count repair does not perform that separate coverage optimization.

The requested count is a target, not permission to manufacture molecules or
silently relax quotas. A pool or fixed decision can make it unattainable. If
molecule 1 in the crossed-quota example is fixed, the two remaining candidates
cannot be added under the quotas. Conversely, two fixed molecules with target 1
remain selected and generate an over-count warning; their presence can also
exceed scaffold or cluster maxima. Read the final count and warnings before
export. Manual edits after automatic selection can change these outcomes.

## Reproducing and reviewing a selection

The constrained Hub selector performs no random draw and uses no random seed.
Repeating its candidate IDs, ranking values, constraints and manual flags yields
the same ordered final IDs, even if the table rows were rearranged. A seed in the
original run, clustering or map configuration belongs to that upstream
calculation; it cannot substitute for the actual data and ranking parameters.
Reimporting files in a different order can assign different record IDs, so retain
the mapping from IDs to the input molecules.

Keep the exact input files and their hashes, software and dependency versions,
standardization and screening configuration, reference libraries and fingerprint
parameters, cached descriptors and cluster/scaffold assignments, enabled
objectives/directions/weights, ranking universe and method, all count constraints,
and fixed/excluded/rescued/manual decisions. Export the final IDs, per-molecule
reasons, actual count, quota usage and warnings together with the applied
selection recipe. A recipe records the decision; reproducing upstream chemistry
also requires the corresponding inputs and computational environment.

A requested count, successful export or populated cluster map does not establish
biological activity, docking success or global optimality. Review selection
overlap, scaffold and cluster coverage, property distributions, alerts and
reference similarities alongside the complete final molecular table.
