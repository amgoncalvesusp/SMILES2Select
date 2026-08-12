# Reproducibility

Selection recipes record candidate and reference libraries, standardization,
fingerprint parameters, projection settings, zones, strategy, final/reserve
counts and random seed. Exact reference results record the search mode and
verified Tanimoto values.

The recipe is machine-readable JSON. The same recipe plus the same inputs
should reproduce the same automatic selection; changed source hashes, toolkit
versions or fingerprint parameters must be treated as a different analysis.
