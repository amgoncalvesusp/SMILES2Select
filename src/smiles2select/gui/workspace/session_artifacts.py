"""Build an export from the selection and provenance currently applied in the workspace."""

from __future__ import annotations

from typing import TYPE_CHECKING

from smiles2select.export import selection_export
from smiles2select.selection_intelligence import recipes
from smiles2select.selection_intelligence.action_log import ActionType
from smiles2select.selection_intelligence.constrained_selection import (
    SelectionConstraints,
    SelectionOutcome,
    Strategy,
)

if TYPE_CHECKING:
    from smiles2select.gui.workspace.workspace_window import WorkspaceWindow


def build_artifacts(self: WorkspaceWindow) -> selection_export.SessionArtifacts:
    """Everything the export needs, assembled from the current session."""
    selected_ids = self.basket.final_ids()
    selected_set = set(selected_ids)
    selected = self.candidates.reindex(selected_ids)
    reasons = {
        rid: [self.basket.state(rid).note or "Selected by original screening pipeline"]
        for rid in selected_ids
    }
    rejections = {}
    strategy = Strategy(self.strategy.currentData())
    for index, action in enumerate(self.basket.log.applied):
        automatic = (
            self._selection_outcomes.get(index)
            if action.action_type is ActionType.AUTOMATIC_SELECTION else None
        )
        if automatic is not None:
            strategy = automatic.strategy
            reasons = {**reasons, **{
                rid: why for rid, why in automatic.reasons.items()
                if rid in selected_set and not self.basket.state(rid).pinned
            }}
            rejections = {
                rid: why for rid, why in automatic.rejections.items() if rid not in selected_set
            }
        for record_id, state in action.new_state.items():
            previous = action.previous_state.get(record_id, {})
            if all(state.get(key) == previous.get(key) for key in (
                "selection_status", "selection_origin"
            )):
                continue
            if record_id in selected_set and state.get("selection_status") == "FINAL_SELECTED":
                explanation = [action.reason or state.get("selection_origin") or "selected"]
                if state.get("selection_origin") == "AUTOMATIC" and automatic is not None:
                    explanation = automatic.reasons.get(record_id, explanation)
                reasons[record_id] = explanation
    for record_id in self.basket.excluded_ids():
        rejections[record_id] = ["manually excluded"]
    outcome = SelectionOutcome(
        selected_ids=selected_ids,
        reasons=reasons,
        rejections=rejections,
        scaffold_usage=selected["murcko_scaffold"].dropna().astype(str).value_counts().to_dict(),
        cluster_usage=selected["cluster_id"].dropna().astype(int).value_counts().to_dict(),
        strategy=strategy,
    )
    snapshot = self._active_snapshot()
    applied = self._active_applied_selection()
    selected_constraints = (snapshot.spec.constraints if snapshot else applied.constraints
                            if applied else SelectionConstraints(
                                target_count=self.result.config.final_count or None,
                                max_per_scaffold=self.result.config.per_scaffold_limit,
                            ))
    objective_values = (tuple(obj.as_dict() for obj in snapshot.spec.objectives)
                        if snapshot else applied.objectives if applied else ())
    metadata = self._provenance()
    projection = self._projection_results.get((
        str(self.projection_selector.currentData()),
        bool(self.reference_overlay.isChecked() and self.result.reference_libraries),
    ))
    return selection_export.SessionArtifacts(
        result=self.result,
        basket=self.basket,
        outcome=outcome,
        constraints=selected_constraints,
        recipe=recipes.SelectionRecipe(
            name=f"scenario:{snapshot.spec.name}" if snapshot else "workspace" if applied else "original_pipeline",
            input_hash=snapshot.data_fingerprint if snapshot else metadata["input_hash"],
            objectives=objective_values,
            original_thresholds={rule.id: rule.threshold for profile in self.result.profiles
                                 for rule in profile.rules if not rule.is_substructure},
            applied_thresholds=dict(snapshot.spec.thresholds) if snapshot else {},
            target_count=selected_constraints.target_count,
            strategy=strategy.value if snapshot or applied else self.result.config.selection_strategy,
            max_per_scaffold=selected_constraints.max_per_scaffold,
            max_per_cluster=selected_constraints.max_per_cluster,
            pinned_ids=self.basket.pinned_ids(),
            excluded_ids=self.basket.excluded_ids(),
            manual_overrides=tuple(
                recipes.ManualOverride(
                    record_id=int(rid),
                    previous_status=action.previous_state.get(rid, {}).get("selection_status", "UNDECIDED"),
                    new_status=state.get("selection_status", "UNDECIDED"),
                    action=action.action_type.value,
                    reason=action.reason or state.get("manual_note") or "",
                    timestamp=action.timestamp,
                )
                for action in self.basket.log.applied
                if action.action_type is not ActionType.AUTOMATIC_SELECTION
                for rid, state in action.new_state.items()
            ),
            candidate_library="candidate",
            reference_libraries=tuple(
                library.spec.as_dict() for library in self.result.reference_libraries
            ),
            background_libraries=tuple(
                library.spec.as_dict() for library in self.result.background_libraries
            ),
            standardization={
                "config_hash": self.result.config.standardization.fingerprint(),
            },
            fingerprint=(
                {
                    "radius": self.result.reference_similarity.fingerprint.radius,
                    "bits": self.result.reference_similarity.fingerprint.size,
                    "use_chirality": self.result.reference_similarity.fingerprint.use_chirality,
                    "search": self.result.reference_similarity.search_mode,
                }
                if self.result.reference_similarity is not None
                else {}
            ),
            reserve_count=len(self.result.reserve_ids),
            seed=self.result.config.selection_seed,
            final_selected_ids=selected_ids,
            projection=projection[0].recipe_block() if projection else {},
            provenance=(
                {**metadata, **dict(snapshot.provenance), "source": "scenario"} if snapshot else
                applied.provenance if applied else {**metadata, "source": "original_pipeline"}
            ),
        ),
        pareto=applied.pareto if applied and not snapshot else None,
        scenario_snapshot=snapshot,
        clusters=self.candidates.get("cluster_id"),
        scaffolds=self.candidates.get("murcko_scaffold"),
    )
