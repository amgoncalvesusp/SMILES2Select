# Selection strategies and reproducibility

## Choosing criteria for a new study

The Hub's **How to choose criteria...** button opens the same scientific guidance
next to the selection controls. Define a study objective before choosing a
strategy. No ranking listed here predicts experimental activity or docking success.

| Research intention | Control and interpretation | Limitation to check |
| --- | --- | --- |
| Favor property trade-offs while retaining spread | Balance properties and representation | Property spread is not fingerprint diversity |
| Favor the best available trade-offs | Prioritize favorable property trade-offs | A Pareto front can contain extreme values; specify appropriate directions |
| Explore a broad range of property values | Spread across property values | May favor property extremes; does not maximize structural dissimilarity |
| Represent different molecular cores | Cover more molecular cores | One representative per feasible core first, then fill; coverage under crossed quotas is heuristic |
| Preserve expert choices and complete a library | Complete my pinned choices | Pins require justification and are also preserved by other strategies |

The initial QED/maximize and molecular-weight/minimize controls are defaults, not
a universal medicinal chemistry recommendation. Choose a different property pair
or interval when the scientific question requires it. The software retains your
direction when changing a descriptor: review both together.

| Property | Meaning and direction to consider | What it does not establish |
| --- | --- | --- |
| `qed` | QED combines property desirabilities into a drug-likeness estimate from 0 to 1; maximize to prefer that profile | Target activity or safety; its component properties already include weight and lipophilicity |
| `mol_wt` | Molecular weight in g/mol; minimizing favors smaller structures; an interval expresses a size window | That the smallest molecule is the best ligand |
| `rdkit_wlogp` | Calculated Wildman–Crippen logP; an interval can express the desired lipophilicity window | Measured solubility, or pH-dependent logD |
| `tpsa` | Topological polar surface area in Å²; an interval can express a polarity window | Experimental permeability or absorption |
| `sa_score` | Synthetic accessibility heuristic, 1–10; lower suggests easier synthesis | A synthesis route, supplier availability, yield or price |
| `np_score` | Natural-product likeness; higher favors fragment patterns associated with natural products | Natural origin, safety or activity |

Descriptor definitions follow [RDKit QED](https://www.rdkit.org/docs/source/rdkit.Chem.QED.html),
the [RDKit Book](https://rdkit.org/docs/RDKit_Book.html), and
the [RDKit SA/NP implementation notes and original papers](https://greglandrum.github.io/rdkit-blog/posts/2023-12-01-using_sascore_and_npscore.html).
These definitions explain the descriptors; the study-specific choice of objectives
and bounds remains a scientific decision that should be recorded.

**Higher/lower is better** continually prefers one direction. **Desired value**
prefers the smallest absolute deviation. **Desired interval** gives every value
inside the interval equal desirability, and penalizes distance outside it. These
are ranking preferences, not new hard eligibility filters. To require an absolute
property cutoff, configure screening or a supported scenario policy accordingly.

**Maximum per scaffold** caps analogues sharing a Murcko core. Missing cores are
computed on demand when selecting. Acyclic structures share the empty Murcko
scaffold, so a limit of 1 can retain only one automatic choice from that group.
**Maximum per cluster** requires actual cached structural cluster assignments;
the shape of a PCA cloud does not define those assignments. Tight quotas can make
the requested count unattainable. Compare scenarios and inspect the count,
property distributions and core coverage before adopting one.

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
| `scaffold_coverage` | one feasible representative per core first; `scaffold_size` ascending; `pareto_rank` ascending; `robustness_score` descending; then fill remaining places |
| `manual_assisted` | `pareto_rank` ascending; `qed` descending |

Unavailable columns are skipped, missing values within a present ranking column
sort last, and ascending integer `record_id` resolves the final tie. For
`scaffold_coverage`, scaffold size is the number of candidates sharing that
scaffold in the eligible selection pool after manual exclusions. The first-per-core
pass respects upper quotas and already represented pinned cores. Missing Murcko
cores are calculated on demand before this policy is applied; this is independent
of whether the map has been built. Crowding distance measures separation
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
property strategy keys in the table above. Consequently, balanced, Pareto-first,
property-spread and manual-assisted strategies can return the same set when
objectives, quotas and pins are identical. For `scaffold_coverage`, the first-per-core
pass and scaffold rarity remain active; property percentiles order alternatives
after scaffold size. Scaffold and cluster constraints still apply.
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
3. For `scaffold_coverage`, reserve one candidate from each feasible new core
   before filling remaining places, preferring rarer cores. Otherwise, when a
   minimum scaffold count is requested, reserve candidates from new
   scaffolds greedily in ranking order, respecting the available count and upper
   quotas. Fixed molecules already represent their cores. This coverage pass
   does not establish a globally optimal coverage under crossed quotas.
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
