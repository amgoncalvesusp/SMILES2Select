# Papyrus: optional local experimental evidence

Papyrus can help answer: **has a molecule with this identity been measured against
the target of interest, using this endpoint?** It does not predict activity for
unmeasured candidates and does not automatically change the selection.

## Prepare an index once

Obtain an official Papyrus bioactivity TSV or a curated subset locally. The
application does not download datasets. CSV, TSV, CSV/TSV.gz and CSV/TSV.xz are
supported directly; do not decompress a large dataset just for this step.

Choose an exact `target_id` from the release's protein table, including its
mutation suffix, one endpoint (`IC50`, `EC50`, `KD` or `Ki`), and the release
version. No target-name guessing or cross-endpoint pooling occurs.

From a source installation on Windows PowerShell:

```powershell
$env:PYTHONPATH = 'src'
python -m smiles2select.reference.papyrus 'D:\Papyrus\bioactivities.tsv.xz' 'D:\Papyrus\target_IC50.sqlite' --target-id 'YOUR_TARGET_ID' --endpoint IC50 --dataset-version '05.7'
```

Replace `YOUR_TARGET_ID`, file paths and version with the actual values. The
command prints progress and a final report. It refuses to overwrite an existing
index; use a new filename when changing the target, endpoint or source release.
The CLI uses only the application's existing Python dependencies. For a packaged
GUI installation, generate the index in a Python source installation first,
then open the resulting SQLite file in the workspace's Papyrus evidence action.

## Interpretation

- Default matching uses **connectivity**, the first 14 InChIKey characters. This
  is explicitly not stereospecific evidence. Papyrus releases without
  stereochemistry must not establish the activity of a particular stereoisomer.
- `--identity-level inchikey` instead requires full InChIKey equality. Use this
  only with an appropriately curated source; equality cannot repair differences
  in protonation, tautomer handling or structural standardization.
- Only `high` quality records are included by the CLI. The Python API accepts an
  explicit quality allowlist; including medium/low quality is a deliberate
  scientific choice, and the retained quality is reported.
- Only exact `=` relations and aggregates containing one endpoint are included.
  Mixed-endpoint aggregates and censored measurements are excluded with separate
  counters. They are not converted into exact values or inactive labels.
- `papyrus_records` counts source rows, **not independent experiments**.
  `papyrus_pchembl_min/max` span retained **record means**, not raw assay values
  or a confidence interval. Larger pChEMBL corresponds to a smaller molar
  concentration within the same endpoint; assay context still matters.
- No match means **no retained matching evidence**, not inactivity. Identity,
  endpoint, quality and source filters can all explain missing evidence.
- All retained Activity_ID values and per-record sources remain in SQLite for
  audit. Metadata records target, endpoint, identity level, quality policy,
  supplied release label, source filename, size and modification time. These
  file attributes are not a cryptographic verification of an official release;
  retain the original file and its official checksum for a publication.

## Large libraries and Windows

The importer reads only required columns in chunks of 10,000 rows, uses an 8 MiB
SQLite page cache and writes only the requested target subset. Initial indexing
is a linear scan of the source and may take time on compressed multi-gigabyte
files. No all-pairs fingerprints, multiprocessing, GPU or model training occurs.

Lookup uses a disk index and queries at most 500 identities at a time. The GUI
should calculate an InChIKey only for the inspected molecule. For bulk processing,
reuse precomputed keys and call the API on candidate chunks; the returned
annotation table itself requires memory proportional to the supplied chunk.
No performance claim for millions of real Papyrus records is implied by the
small synthetic regression fixtures.

```python
import pandas as pd
from smiles2select.reference.papyrus import annotate_candidates, read_index_metadata

index_path = r'D:\Papyrus\target_IC50.sqlite'
print(read_index_metadata(index_path))
for candidates in pd.read_csv('candidates_with_inchikey.csv', chunksize=10_000):
    evidence = annotate_candidates(candidates, index_path, inchikey_column='inchikey')
    # Same index as candidates. Write/use this chunk; do not accumulate all chunks.
    print(evidence['papyrus_records'].gt(0).sum())
```

For benchmarking, keep these reference annotations separate from held-out activity
labels. Check identity/scaffold overlap before claiming prospective performance.

## Supported source schema and references

Required columns: `target_id`, `Quality`, `source`, `Activity_ID`, `relation`,
`pchembl_value_Mean`, `type_IC50`, `type_EC50`, `type_KD`, `type_Ki`, `type_other`,
plus `connectivity` (or `InChIKey` fallback). Full-key matching requires
`InChIKey`. Endpoint flags may contain semicolon-separated binary values.
Malformed relevant records fail explicitly; no partial index is published.

The boundary follows the Papyrus 05.7 TSV conventions documented in the authors'
[Papyrus scripts](https://github.com/OlivierBeq/Papyrus-scripts), including their
[filtering implementation](https://github.com/OlivierBeq/Papyrus-scripts/blob/master/src/papyrus_scripts/preprocess.py).
See the [Papyrus paper](https://doi.org/10.1186/s13321-022-00672-x) for curation and
quality definitions, and the authors' linked
[05.7 dataset record](https://zenodo.org/records/13787633) for release files.
Future schemas require validation before use; this adapter does not claim
universal compatibility with every Papyrus release.
