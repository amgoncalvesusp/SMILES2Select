"""The Selection Intelligence workspace window.

Layout follows the specification: controls on the left, one dominant
visualisation in the centre, the molecule inspector on the right, the basket
below. Not a grid of equally weighted cards - at any moment there is one
question on screen, and the rest of the interface supports answering it.
"""

from __future__ import annotations

import pandas as pd
from PySide6.QtWidgets import (
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QWidget,
)
from rdkit import Chem

from smiles2select.app_metadata import APP_NAME
from smiles2select.chemical_space.clustering import ClusteringTooLargeError, cluster
from smiles2select.chemical_space.projection_manager import (
    ProjectionResult,
)
from smiles2select.chemistry.scaffolds import scaffolds_from_smiles
from smiles2select.decision.engine import DecisionEngine
from smiles2select.explainability.consequences import explain_change
from smiles2select.explainability.method_cards import get_method_card
from smiles2select.export import selection_export
from smiles2select.gui.workspace.guided_controls import STRATEGY_HELP, build_layout
from smiles2select.gui.workspace.jobs import ComputationJob
from smiles2select.gui.workspace.map_compute import compute_map
from smiles2select.gui.workspace.panels import BasketPanel, InspectorPanel
from smiles2select.gui.workspace.views import ChemicalSpaceView, ParetoView
from smiles2select.pipeline.runner import RunResult
from smiles2select.selection_intelligence import recipes
from smiles2select.selection_intelligence.action_log import ActionType
from smiles2select.selection_intelligence.basket import JustificationRequired, SelectionBasket
from smiles2select.selection_intelligence.constrained_selection import (
    SelectionConstraints,
    SelectionOutcome,
    Strategy,
    select,
)
from smiles2select.selection_intelligence.objectives import ObjectiveSet
from smiles2select.selection_intelligence.pareto_ranking import ParetoRanker
from smiles2select.selection_intelligence.scenarios import EXACT_PARETO_LIMIT
from smiles2select.selection_intelligence.states import (
    MoleculeState,
    SelectionOrigin,
    chemical_status_from_run,
)

VIEW_MAP = "Chemical space"
VIEW_PARETO = "Pareto"
OBJECTIVE_CANDIDATES = ("qed", "mol_wt", "rdkit_wlogp", "tpsa", "sa_score", "np_score")


class WorkspaceWindow(QMainWindow):
    """Turns a finished run into an interactive selection session."""

    def __init__(self, result: RunResult, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} - Chemical Space Hub")
        self.resize(1440, 900)

        self._job = None
        self._papyrus_path = None
        self._map_requested = False
        self._scenarios_dialog = None
        self.result = result
        self.candidates = self._build_candidates(result)
        self.basket = self._build_basket(result)
        self.ranker = ParetoRanker()
        self.pareto = None
        self._selection_outcomes: dict[int, SelectionOutcome] = {}
        self._scenario_snapshots = {}
        self._projection_results: dict[tuple[str, bool], tuple[ProjectionResult, pd.DataFrame]] = {}

        self.map_view = ChemicalSpaceView()
        self.pareto_view = ParetoView()
        self.views = QStackedWidget()
        self.views.addWidget(self.map_view)
        self.views.addWidget(self.pareto_view)

        self.inspector = InspectorPanel()
        self.basket_panel = BasketPanel()
        self.warnings = QLabel("")
        self.warnings.setWordWrap(True)
        self.warnings.setStyleSheet("color: #a33;")

        self.setCentralWidget(self._build_layout())
        self._connect()
        self._recompute_pareto()

    # -- construction -------------------------------------------------------

    def _build_candidates(self, result: RunResult) -> pd.DataFrame:
        """One table with the columns every view and the selector may read."""
        evaluable = result.descriptors[result.descriptors["evaluable"].astype(bool)].copy()
        if "murcko_scaffold" not in evaluable.columns:
            evaluable["murcko_scaffold"] = (
                scaffolds_from_smiles(evaluable["canonical_smiles"].fillna("").tolist())
                if len(evaluable) <= 5000 else pd.NA
            )
        try:
            if len(evaluable) > 2000:
                raise ClusteringTooLargeError("Exact clustering deferred for large library")
            clusters = cluster(evaluable["canonical_smiles"].fillna("").tolist(), evaluable.index)
            evaluable["cluster_id"] = clusters.labels
        except ClusteringTooLargeError:
            # The map remains usable; a large library needs a scalable cluster
            # backend rather than a hidden quadratic allocation.
            evaluable["cluster_id"] = pd.Series(pd.NA, index=evaluable.index, dtype="Int64")
        for column in ("qed", "sa_score"):
            if column not in evaluable.columns:
                evaluable[column] = pd.NA
        return evaluable

    def _build_basket(self, result: RunResult) -> SelectionBasket:
        """Seed the basket with the chemical verdict of every record."""
        # Screening eligibility precedes count/diversity/reference allocation.
        decisions = DecisionEngine(result.config.policy).decide(
            result.evaluation.status, result.scores, result.alerts
        ).decisions
        states = [
            MoleculeState(
                record_id=int(record_id),
                chemical_status=chemical_status_from_run(
                    valid=bool(valid),
                    passed=bool(decisions["selected"].get(record_id, False)),
                ),
            )
            for record_id, valid in zip(result.descriptors.index, result.descriptors["valid"])
        ]
        return SelectionBasket(states)

    def _build_layout(self) -> QWidget:
        return build_layout(self, OBJECTIVE_CANDIDATES)

    def _connect(self) -> None:
        self.view_selector.currentTextChanged.connect(self._switch_view)
        self.first_objective.currentTextChanged.connect(self._recompute_pareto)
        self.second_objective.currentTextChanged.connect(self._recompute_pareto)
        self.projection_selector.currentIndexChanged.connect(self._request_map)
        self.color_selector.currentIndexChanged.connect(lambda _index: self.refresh())
        self.reference_overlay.stateChanged.connect(self._request_map)
        self.strategy.currentTextChanged.connect(self._update_method_card)
        for widget in (self.target_count, self.per_scaffold, self.per_cluster):
            widget.valueChanged.connect(self._update_summary)
        for editor in self.objective_editors:
            editor.direction.currentIndexChanged.connect(self._recompute_pareto)
            editor.low.valueChanged.connect(self._recompute_pareto)
            editor.high.valueChanged.connect(self._recompute_pareto)
        self._update_method_card(self.strategy.currentText())

        for view in (self.map_view, self.pareto_view):
            view.point_clicked.connect(self._show_molecule)
            view.region_selected.connect(self._shortlist_region)

        self.inspector.shortlist_requested.connect(lambda rid: self._decide(rid, "shortlist"))
        self.inspector.select_requested.connect(lambda rid: self._decide(rid, "final"))
        self.inspector.exclude_requested.connect(lambda rid: self._decide(rid, "exclude"))
        self.inspector.pin_requested.connect(lambda rid: self._decide(rid, "pin"))

        self.basket_panel.undo_requested.connect(self._undo)
        self.basket_panel.redo_requested.connect(self._redo)
        self.basket_panel.auto_select_requested.connect(self._auto_select)
        self.basket_panel.export_requested.connect(self._export)

    # -- behaviour ----------------------------------------------------------

    def _update_summary(self) -> None:
        try:
            objectives = "; ".join(
                f"{editor.field.currentText()}: {editor.direction.currentText()}"
                + (f" [{editor.low.value():g}, {editor.high.value():g}]"
                   if editor.direction.currentData() == "target_range" else
                   f" {editor.low.value():g}"
                   if editor.direction.currentData() == "target_value" else "")
                for editor in self.objective_editors
            )
        except (AttributeError, ValueError):
            objectives = f"{self.first_objective.currentText()} / {self.second_objective.currentText()}"
        self.criteria_summary.setText(
            f"Current criteria: {self.target_count.value():,} molecules; "
            f"{self.strategy.currentText()}. {objectives}. "
            f"Scaffold limit: {self.per_scaffold.value() or 'none'}; "
            f"cluster limit: {self.per_cluster.value() or 'none'}. "
            "Chemical screening rules remain in effect."
        )

    def _open_scenarios(self) -> None:
        from smiles2select.gui.workspace.scenario_dialog import ScenarioDialog

        if self._scenarios_dialog is None:
            self._scenarios_dialog = ScenarioDialog(self)
        self._scenarios_dialog.show()
        self._scenarios_dialog.raise_()

    def _request_map(self) -> None:
        if self.is_busy:
            return
        method = str(self.projection_selector.currentData())
        overlay = bool(self.reference_overlay.isChecked() and self.result.reference_libraries)
        key = (method, overlay)
        self.performance_notice.setText(
            f"Map: at most {min(5000, len(self.candidates)):,} sampled candidates "
            f"of {len(self.candidates):,}; fixed seed 42. Reference overlay: at most "
            "2,000 per library. Selection still considers all candidates."
        )
        candidates, result = self.candidates, self.result

        def completed(value):
            self._projection_results[key] = value
            self._map_requested = True
            self.refresh()

        self._start_job(lambda: compute_map(candidates, result, method, overlay), completed,
                        "Building a bounded map in the background...")

    def _attach_papyrus(self) -> None:
        from smiles2select.reference.papyrus import read_index_metadata

        path, _ = QFileDialog.getOpenFileName(
            self, "Choose local Papyrus evidence index", "", "SQLite index (*.sqlite *.db);;All files (*)"
        )
        if not path:
            return
        try:
            metadata = read_index_metadata(path)
        except Exception as exc:
            QMessageBox.warning(self, "Papyrus index unavailable", str(exc))
            return
        self._papyrus_path = path
        self.papyrus_button.setToolTip(str(metadata))
        self.warnings.setText(
            "Papyrus evidence attached. Inspect a molecule to query measured evidence. "
            "Missing evidence is unknown; connectivity matches do not establish stereospecific activity."
        )
        if self.inspector.record_id is not None:
            self._show_molecule(self.inspector.record_id)

    def _start_job(self, function, completed, message) -> None:
        if self._job is not None:
            return
        self.warnings.setText(message)
        self.job_progress.show()
        self.controls_scroll.setEnabled(False)
        self.basket_panel.setEnabled(False)
        self.inspector.setEnabled(False)
        self.views.setEnabled(False)
        self._job = ComputationJob(function, self)
        self._job.completed.connect(completed)
        self._job.failed.connect(self.warnings.setText)
        self._job.finished.connect(self._finish_job)
        self._job.start()

    def _finish_job(self) -> None:
        job, self._job = self._job, None
        self.job_progress.hide()
        for widget in (self.controls_scroll, self.basket_panel, self.inspector, self.views):
            widget.setEnabled(True)
        if job is not None:
            job.deleteLater()

    @property
    def is_busy(self) -> bool:
        return self._job is not None or bool(
            self._scenarios_dialog is not None and self._scenarios_dialog.is_busy
        )

    def closeEvent(self, event) -> None:
        if self._job is not None and self._job.isRunning():
            self.warnings.setText("Computation is finishing. Close the workspace when it completes.")
            event.ignore()
            return
        if self._scenarios_dialog is not None and self._scenarios_dialog.is_busy:
            self.warnings.setText("Scenario computation is running; wait for completion before closing.")
            event.ignore()
            return
        super().closeEvent(event)

    def _switch_view(self, name: str) -> None:
        self.views.setCurrentIndex(0 if name == VIEW_MAP else 1)
        self.refresh()

    def _update_method_card(self, strategy: str) -> None:
        """Keep method bias and consequence visible while changing controls."""
        strategy = self.strategy.currentData()
        try:
            card = get_method_card(strategy)
        except KeyError:
            card = get_method_card("balanced")
        self.strategy_help.setText(STRATEGY_HELP.get(strategy, strategy))
        self._update_summary()
        self.method_card.setText(
            f"<b>{card.title}</b><br>{card.what_it_does}<br>"
            f"<i>Favors:</i> {card.what_it_favors}<br>"
            f"<i>May underrepresent:</i> {card.may_underrepresent}"
        )
        self.consequence.setText(
            explain_change("selection_strategy", "previous", strategy).message
        )

    def objectives(self) -> ObjectiveSet:
        return ObjectiveSet([editor.objective() for editor in self.objective_editors])

    def _recompute_pareto(self) -> None:
        self._update_summary()
        if len(self.candidates) > EXACT_PARETO_LIMIT:
            self.pareto = None
            self.performance_notice.setText(
                "Large library: exact Pareto disabled; selection uses available property scores "
                "and quotas. Map loads on request (at most 5,000 molecules). "
                "Use Compare scenarios for bounded, documented ranking."
            )
            self.refresh()
            return
        first = self.first_objective.currentText()
        second = self.second_objective.currentText()
        if not first or not second or first == second:
            self.pareto = None
            self.warnings.setText("Choose two different objectives.")
            self.refresh()
            return
        try:
            self.pareto = self.ranker.rank(self.candidates, self.objectives())
        except Exception as exc:
            self.pareto = None
            self.warnings.setText(str(exc))
            self.refresh()
            return
        self.warnings.setText("\n".join(self.pareto.warnings))
        self.refresh()

    def _show_molecule(self, record_id: int) -> None:
        extras: dict[str, object] = {}
        if self.pareto is not None and record_id in self.pareto.table.index:
            extras["Pareto rank"] = int(self.pareto.table.loc[record_id, "pareto_rank"])
        if record_id in self.basket:
            state = self.basket.state(record_id)
            extras["Chemical status"] = state.chemical_status.value
            extras["Selection status"] = state.selection_status.value
            verdict = self.result.decision.decisions
            if record_id in verdict.index and not bool(verdict.loc[record_id, "selected"]):
                extras["Original run decision"] = "Not selected in the original run"
                extras["Decision stage"] = (
                    "Passed screening; later allocation/reference/count criteria excluded it"
                    if state.chemical_status.passed else "Failed original chemical screening"
                )
        if self._papyrus_path is not None and record_id in self.candidates.index:
            from smiles2select.reference.papyrus import lookup_evidence

            try:
                mol = Chem.MolFromSmiles(str(self.candidates.loc[record_id, "canonical_smiles"]))
                key = Chem.MolToInchiKey(mol) if mol is not None else None
                evidence = lookup_evidence(self._papyrus_path, [key])
                extras.update(evidence.iloc[0].dropna().to_dict())
                if not evidence.iloc[0].get("papyrus_records", 0):
                    extras["Papyrus interpretation"] = "No evidence in this index; activity unknown"
            except Exception as exc:
                extras["Papyrus query unavailable"] = str(exc)
        self.inspector.show_molecule(record_id, self.candidates, extras)

    def _shortlist_region(self, record_ids: list[int]) -> None:
        if not record_ids:
            return
        preview = self.basket.preview(record_ids)
        summary = "\n".join(f"{label}: {value}" for label, value in preview.items())
        answer = QMessageBox.question(
            self,
            "Add to shortlist",
            f"{summary}\n\nConfirm?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if answer == QMessageBox.Yes:
            self.basket.add_to_shortlist(
                record_ids, origin=SelectionOrigin.LASSO_SELECTION, source="CHEMICAL_SPACE_LASSO"
            )
            self.refresh()

    def _decide(self, record_id: int, action: str) -> None:
        try:
            if action == "shortlist":
                self.basket.add_to_shortlist([record_id])
            elif action == "final":
                self._select_with_justification(record_id)
            elif action == "exclude":
                self.basket.exclude([record_id])
            else:
                pinned = self.basket.state(record_id).pinned
                (self.basket.unpin if pinned else self.basket.pin)([record_id])
        except JustificationRequired as exc:
            QMessageBox.warning(self, "Justification required", str(exc))
            return
        self.refresh()

    def _select_with_justification(self, record_id: int) -> None:
        """A chemically rejected molecule can only be kept with a written reason."""
        state = self.basket.state(record_id)
        reason = ""
        if not state.chemical_status.passed:
            reason, accepted = QInputDialog.getText(
                self,
                "Justification",
                f"The molecule is {state.chemical_status.value}. Explain the selection:",
            )
            if not accepted or not reason.strip():
                raise JustificationRequired("selection cancelled: no justification supplied")
        self.basket.add_to_final([record_id], reason=reason)

    def constraints(self) -> SelectionConstraints:
        return SelectionConstraints(
            target_count=self.target_count.value(),
            max_per_scaffold=self.per_scaffold.value() or None,
            max_per_cluster=self.per_cluster.value() or None,
        )

    def _auto_select(self) -> None:
        if self.is_busy:
            return
        states = self.basket.states()
        constraints = self.constraints()
        strategy = Strategy(self.strategy.currentData())
        pinned_ids, excluded_ids = self.basket.pinned_ids(), self.basket.excluded_ids()
        source, pareto = self.candidates, self.pareto

        def compute():
            from smiles2select.selection_intelligence.scenarios import _reference_mask

            if self.result.zone_allocation is not None:
                raise ValueError("This run uses zone allocation. Re-run without zones before "
                                 "changing automatic selection in the workspace.")
            eligible = [state.record_id for state in states
                        if state.chemical_status.passed or (state.pinned and state.is_selected)]
            candidates = source.loc[source.index.intersection(eligible)].copy()
            candidates = candidates.loc[_reference_mask(self.result, candidates.index)]
            if pareto is not None:
                candidates = candidates.join(pareto.table, how="left")
            if constraints.max_per_scaffold and candidates["murcko_scaffold"].isna().any():
                raise ValueError("Scaffold quota requires complete cached molecular cores.")
            if constraints.max_per_cluster and candidates["cluster_id"].isna().any():
                raise ValueError("Cluster quota requires complete cached cluster IDs.")
            if strategy is Strategy.SCAFFOLD_COVERAGE:
                if candidates["murcko_scaffold"].isna().any():
                    raise ValueError("Molecular-core coverage requires complete cached scaffolds.")
                counts = candidates["murcko_scaffold"].value_counts()
                candidates = candidates.assign(scaffold_size=candidates["murcko_scaffold"].map(counts))
            return select(candidates, constraints, strategy, pinned_ids=pinned_ids,
                          excluded_ids=excluded_ids, explain_rejections=len(candidates) <= 5000)

        if len(source) > EXACT_PARETO_LIMIT:
            self._start_job(compute, lambda outcome: self._apply_selection(outcome, constraints),
                            "Selecting from the full library in the background...")
        else:
            try:
                self._apply_selection(compute(), constraints)
            except Exception as exc:
                self.warnings.setText(str(exc))

    def _apply_selection(self, outcome, constraints):
        self.basket.replace_final(
            list(outcome.selected_ids), origin=SelectionOrigin.AUTOMATIC,
            source=outcome.strategy.value, reason="automatic selection",
        )
        action_index = len(self.basket.log.applied) - 1
        self._selection_outcomes[action_index] = outcome
        self._scenario_snapshots.pop(action_index, None)
        self.warnings.setText("\n".join(outcome.warnings(constraints)))
        self.refresh()

    def _undo(self) -> None:
        self.basket.undo()
        self.refresh()

    def _redo(self) -> None:
        self.basket.redo()
        self.refresh()

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export selection", "", "Excel (*.xlsx)")
        if not path:
            return
        try:
            workbook, recipe = selection_export.export(self.build_artifacts(), path)
            snapshot = self._active_snapshot()
            if snapshot is not None:
                from smiles2select.selection_intelligence.scenario_io import save_study
                save_study(workbook.with_suffix(".scenarios.json"), (snapshot,))
        except Exception as exc:
            QMessageBox.warning(self, "Export unavailable", str(exc))
            return
        QMessageBox.information(
            self, "Exported", f"Workbook: {workbook.name}\nRecipe: {recipe.name}"
        )

    def _active_snapshot(self):
        for index in range(len(self.basket.log.applied) - 1, -1, -1):
            if self.basket.log.applied[index].action_type is ActionType.AUTOMATIC_SELECTION:
                return self._scenario_snapshots.get(index)
        return None

    def build_artifacts(self) -> selection_export.SessionArtifacts:
        """Everything the export needs, assembled from the current session."""
        selected_ids = self.basket.final_ids()
        selected_set = set(selected_ids)
        selected = self.candidates.reindex(selected_ids)
        reasons = {}
        rejections = {}
        strategy = Strategy(self.strategy.currentData())
        for index, action in enumerate(self.basket.log.applied):
            automatic = (
                self._selection_outcomes.get(index)
                if action.action_type is ActionType.AUTOMATIC_SELECTION else None
            )
            if automatic is not None:
                strategy = automatic.strategy
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
        selected_constraints = snapshot.spec.constraints if snapshot else self.constraints()
        objective_values = (tuple(obj.as_dict() for obj in snapshot.spec.objectives)
                            if snapshot else self.objectives().as_dicts())
        return selection_export.SessionArtifacts(
            result=self.result,
            basket=self.basket,
            outcome=outcome,
            constraints=selected_constraints,
            recipe=recipes.SelectionRecipe(
                name=f"scenario:{snapshot.spec.name}" if snapshot else "workspace",
                input_hash=snapshot.data_fingerprint if snapshot else self.result.config.fingerprint(),
                objectives=objective_values,
                original_thresholds={rule.id: rule.threshold for profile in self.result.profiles
                                     for rule in profile.rules if not rule.is_substructure},
                applied_thresholds=dict(snapshot.spec.thresholds) if snapshot else {},
                target_count=selected_constraints.target_count,
                strategy=strategy.value,
                max_per_scaffold=selected_constraints.max_per_scaffold,
                max_per_cluster=selected_constraints.max_per_cluster,
                pinned_ids=self.basket.pinned_ids(),
                excluded_ids=self.basket.excluded_ids(),
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
            ),
            pareto=None if snapshot else self.pareto,
            scenario_snapshot=snapshot,
            clusters=self.candidates.get("cluster_id"),
            scaffolds=self.candidates.get("murcko_scaffold"),
        )

    def refresh(self) -> None:
        """Redraw the active view and the basket from the shared state."""
        selected = self.basket.final_ids()
        ranks = self.pareto.table["pareto_rank"] if self.pareto is not None else None

        if self.views.currentIndex() == 0 and (
            len(self.candidates) <= 5000 or self._map_requested
        ):
            try:
                projection, reference_coordinates = self._map_projection()
            except Exception as exc:
                self.warnings.setText(str(exc))
                self.map_view.set_points(pd.DataFrame(columns=["x", "y"]))
                self.map_view.set_reference_points(None)
            else:
                self.map_view.set_points(
                    projection.projection.coordinates,
                    selected,
                    self._colour_values(projection.projection.coordinates, ranks),
                )
                self.map_view.set_reference_points(
                    reference_coordinates if self.reference_overlay.isChecked() else None
                )
        elif self.pareto is not None:
            self.pareto_view.show_objectives(
                self.candidates,
                self.first_objective.currentText(),
                self.second_objective.currentText(),
                self.pareto.table["pareto_rank"],
                selected,
            )
        else:
            self.pareto_view.set_points(pd.DataFrame(columns=["x", "y"]))
            self.pareto_view.clear_lasso()
        self.basket_panel.refresh(self.basket, self.candidates.get("molecule_id"))
        if self.inspector.record_id is not None:
            self._show_molecule(self.inspector.record_id)

    def _map_projection(self) -> tuple[ProjectionResult, pd.DataFrame]:
        method = str(self.projection_selector.currentData())
        overlay = bool(self.reference_overlay.isChecked() and self.result.reference_libraries)
        key = (method, overlay)
        if key not in self._projection_results:
            self._projection_results[key] = compute_map(self.candidates, self.result, method, overlay)
        return self._projection_results[key]

    def _colour_values(self, coordinates: pd.DataFrame, ranks: pd.Series | None) -> pd.Series:
        mode = self.color_selector.currentText()
        if mode == "Pareto rank" and ranks is not None:
            return ranks.reindex(coordinates.index)
        if mode == "Selection status":
            values = pd.Series(float("nan"), index=coordinates.index)
            values.loc[values.index.intersection(self.basket.final_ids())] = 1
            return values
        if "max_reference_similarity" in self.candidates.columns:
            values = self.candidates["max_reference_similarity"].reindex(coordinates.index)
            return (1 + (values.fillna(0).clip(0, 1) * 4).round()).astype(int)
        return pd.Series(float("nan"), index=coordinates.index)
