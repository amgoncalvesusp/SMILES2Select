"""Test-only worker that can be told to kill its own process.

Lives inside the package rather than under ``tests/`` so it is reliably
importable by name when ``multiprocessing`` re-imports it in a freshly
spawned child process - a function defined inside a pytest test module is not
guaranteed to be importable that way, since it depends on pytest's import
mode and whether the test directory is even a package.

Not part of the public API: exists only to give the crash-isolation tests a
worker that can genuinely die (``os._exit`` bypasses Python entirely, so no
``try/except`` anywhere could ever catch it - that is the point).
"""

from __future__ import annotations

import os

from smiles2select.pipeline.chunking import MoleculeChunk
from smiles2select.pipeline.workers import RecordResult

#: Present as a "smiles" value in a chunk, this kills the worker instead of
#: producing a result.
POISON_SMILES = "__POISON__"


def crash_on_poison(chunk: MoleculeChunk) -> list[RecordResult]:
    results = []
    for record_id, smiles in chunk.records:
        if smiles == POISON_SMILES:
            os._exit(137)
        results.append(
            RecordResult(
                record_id=record_id,
                valid=True,
                invalid_reason=None,
                standardized_smiles=smiles,
                canonical_smiles=smiles,
                descriptors={"stub": float(record_id)},
                substructure_flags={},
                alert_rows=[],
            )
        )
    return results
