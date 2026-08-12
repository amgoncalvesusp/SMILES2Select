# Selection strategies

Available strategy identifiers include `traditional`, `balanced`,
`diversity_first`, `reference_novelty`, `reference_neighborhood`,
`reference_aware_diversity`, `stratified` and `manual_assisted`.

Reference novelty is fingerprint-based. Reference-aware diversity initializes
each candidate's distance from the reference library, then updates distance
from selected candidates. A fixed seed and the same inputs/configuration make
automatic selection reproducible where the algorithm permits.

No strategy is a universal definition of a good molecule. Compare strategies
using overlap, scaffold coverage, similarity distributions, property
distributions, alert counts and the recorded selection reasons.
