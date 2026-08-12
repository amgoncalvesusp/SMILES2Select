"""Small reproducible benchmark for Chemical Space Hub primitives.

Usage::

    python benchmarks/benchmark_hub.py --molecules 10000 --references 2000

The output is plain JSON so it can be archived beside a run and compared
across workstations. Synthetic data is a smoke benchmark, not a production
library performance claim.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from smiles2select.chemical_space.layers import progressive_layer
from smiles2select.chemical_space.projection_manager import ProjectionConfig, project
from smiles2select.reference.libraries import prepare_library_frame
from smiles2select.reference.similarity import compute_reference_similarity

FRAGMENTS = ("CC", "CCC", "CO", "CN", "c1ccccc1", "C(=O)O", "N", "O")


def synthetic_smiles(count: int, offset: int = 0) -> list[str]:
    return [
        "".join(FRAGMENTS[(index + offset + step) % len(FRAGMENTS)] for step in range(3))
        for index in range(count)
    ]


def run_benchmark(molecules: int, references: int) -> dict[str, object]:
    candidates = pd.DataFrame({"canonical_smiles": synthetic_smiles(molecules)})
    reference = prepare_library_frame(
        pd.DataFrame({"SMILES": synthetic_smiles(references, offset=3)}),
        "benchmark-reference",
        smiles_column="SMILES",
    )

    started = time.perf_counter()
    similarity = compute_reference_similarity(candidates, [reference], chunk_size=512)
    similarity_seconds = time.perf_counter() - started

    descriptor_frame = pd.DataFrame(
        {
            "mol_wt": [100.0 + index % 200 for index in range(molecules)],
            "tpsa": [20.0 + index % 80 for index in range(molecules)],
        },
        index=pd.RangeIndex(molecules),
    )
    started = time.perf_counter()
    projection = project(
        descriptor_frame,
        ProjectionConfig(method="property_pca", features=("mol_wt", "tpsa")),
    )
    projection_seconds = time.perf_counter() - started

    started = time.perf_counter()
    layer = progressive_layer(projection.projection.coordinates, density_threshold=20_000)
    density_seconds = time.perf_counter() - started

    return {
        "molecules": molecules,
        "references": references,
        "similarity_seconds": round(similarity_seconds, 4),
        "projection_seconds": round(projection_seconds, 4),
        "density_seconds": round(density_seconds, 4),
        "reference_method": similarity.method_description,
        "display_points": len(layer.points),
        "density_tiles": len(layer.density),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecules", type=int, default=10_000)
    parser.add_argument("--references", type=int, default=2_000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.molecules < 1 or args.references < 1:
        parser.error("--molecules and --references must be positive")
    result = run_benchmark(args.molecules, args.references)
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
