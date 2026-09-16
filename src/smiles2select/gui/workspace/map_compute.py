"""Bounded projection calculation without access to Qt widgets."""

import pandas as pd
from rdkit import Chem

from smiles2select.chemical_space.projection_manager import (
    ProjectionConfig,
    ProjectionResult,
    project,
)
from smiles2select.chemistry.descriptor_registry import default_registry


def compute_map(candidates, run_result, method, overlay, selected_ids=()):
    """Project every final molecule and a reproducible sample of context.

    The context budget is a rendering policy, never a selection cap. Coordinates
    are fitted to this explicit subset; changing the finals may change the fit.
    """
    ordered = candidates.sort_index()
    selected_mask = ordered.index.isin(selected_ids)
    final = ordered.loc[selected_mask]
    context = ordered.loc[~selected_mask]
    context_budget = max(0, 5000 - len(final))
    if len(context) > context_budget:
        context = context.sample(n=context_budget, random_state=42)
    map_candidates = pd.concat([final, context]).sort_index()
    sampling = {
        "policy": "all_finals_plus_seeded_context",
        "seed": 42,
        "point_budget": 5000,
        "candidate_total": len(candidates),
        "candidate_projected": len(map_candidates),
        "selected_requested": len(set(selected_ids)),
        "selected_projected": len(final),
        "context_projected": len(context),
        "candidate_record_ids": [int(value) for value in map_candidates.index],
        "reference_policy": "first_2000_per_library_for_property_pca_overlay",
    }
    config = ProjectionConfig(
        method=method,
        fingerprint=run_result.config.fingerprint_config,
        parameters={"display_sampling": sampling},
    )
    if overlay and method == "property_pca":
        features = tuple(ProjectionConfig().features)
        combined = map_candidates[
            [column for column in features if column in map_candidates]
        ].copy()
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
            frame.index = [f"reference::{library.library_id}::{row_id}" for row_id in frame.index]
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
