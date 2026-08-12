# Reference libraries

Candidates are eligible for selection. Reference libraries describe covered
chemistry and are never silently added to the candidate set. More than one
reference library can be supplied.

Exact overlap uses canonical SMILES and InChIKey when available. The report
retains the candidate record, reference library and all matching reference IDs.
Novelty-oriented workflows may exclude exact duplicates from the final set,
but the overlap remains in the report.

Reference nearest-neighbour search currently exposes an honest `Exact search`
mode. It uses chunk-friendly RDKit Tanimoto comparisons and avoids constructing
a full candidate-by-reference matrix. Approximate HNSW search is an explicit
optional backend and must not be labeled exact.
