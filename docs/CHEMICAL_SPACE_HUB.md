# Chemical Space Selection Hub

SMILES2Select helps researchers explore, compare and deliberately sample
chemical space. Chemical eligibility, drug-likeness classification, structural
alerts, chemical-space location, reference novelty and final sampling remain
separate layers.

The current 2.1 workflow remains available. The Hub foundations are exposed in
the Python API and CLI through reference libraries, exact overlap detection,
fingerprint novelty, selection strategies, final/reserve counts and English
method cards.

Important methodological rule: PCA and UMAP coordinates are for
visualization. Their two-dimensional distance is never used as the molecular
similarity or novelty metric.

## CLI example

```text
smiles2select candidates.csv --smiles-column SMILES --id-column ID \
  --reference ecbd.csv --reference-smiles-column SMILES \
  --reference-id-column ID --exclude-reference-duplicates \
  --selection-strategy reference_aware_diversity \
  --final-count 3000 --reserve-count 500 \
  --database session.sqlite --excel session.xlsx
```

The report records the reference libraries, exact search mode, Morgan
fingerprint settings, nearest reference, maximum similarity and
`reference_novelty = 1 - max_reference_similarity`.
