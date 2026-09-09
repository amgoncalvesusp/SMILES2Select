"""Bounded projection calculation without access to Qt widgets."""

import pandas as pd
from rdkit import Chem

from smiles2select.chemical_space.projection_manager import (
    ProjectionConfig,
    ProjectionResult,
    project,
)
from smiles2select.chemistry.descriptor_registry import default_registry


def compute_map(candidates, run_result, method, overlay):
    config = ProjectionConfig(method=method, fingerprint=run_result.config.fingerprint_config)
    map_candidates = candidates
    if len(map_candidates) > 5000:
        map_candidates = map_candidates.sample(n=5000, random_state=42).sort_index()
    if overlay and method == "property_pca":
        features = tuple(ProjectionConfig().features)
        combined = map_candidates[[column for column in features if column in map_candidates]].copy()
        combined.index = [f"candidate::{record_id}" for record_id in combined.index]
        reference_frames = []
        registry = default_registry()
        for library in run_result.reference_libraries:
            rows = []
            for row_id, row in library.valid.head(2000).iterrows():
                mol = Chem.MolFromSmiles(str(row["canonical_smiles"]))
                values = registry.compute(mol, features) if mol is not None else {}
                rows.append(values)
            frame = pd.DataFrame(rows, index=library.valid.head(2000).index)
            frame.index = [
                f"reference::{library.library_id}::{row_id}" for row_id in frame.index
            ]
            reference_frames.append(frame)
        combined = pd.concat([combined, *reference_frames], axis=0)
        result = project(combined, config)
        coordinates = result.projection.coordinates
        candidate_coordinates = coordinates.loc[coordinates.index.str.startswith("candidate::")]
        candidate_coordinates.index = pd.Index(
            [int(value.split("::", 1)[1]) for value in candidate_coordinates.index]
        )
        reference_coordinates = coordinates.loc[coordinates.index.str.startswith("reference::")]
        value = (
            ProjectionResult(
                projection=type(result.projection)(
                    coordinates=candidate_coordinates,
                    method=result.projection.method,
                    features=result.projection.features,
                    explained_variance=result.projection.explained_variance,
                    parameters=result.projection.parameters,
                ),
                config=result.config,
                input_hash=result.input_hash,
                software_versions=result.software_versions,
            ),
            reference_coordinates,
        )
    else:
        result = project(map_candidates, config)
        value = (result, pd.DataFrame(columns=["x", "y"]))
    return value
