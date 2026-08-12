# Reference-aware selection

For each candidate:

```text
reference_novelty = 1 - maximum Tanimoto similarity to reference libraries
```

The default representation is Morgan radius 2, 2048 bits, with chirality
explicitly recorded. Similarity thresholds are user choices, not universal
scientific boundaries. A distant molecule is not automatically better, and a
reference duplicate is not automatically chemically invalid.
