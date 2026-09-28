"""The Selection Intelligence workspace window.

Layout follows the specification: controls on the left, one dominant
visualisation in the centre, the molecule inspector on the right, the basket
below. Not a grid of equally weighted cards - at any moment there is one
question on screen, and the rest of the interface supports answering it.
"""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
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
from smiles2select.gui.workspace import criteria_state
from smiles2select.gui.workspace.guided_controls import build_layout
from smiles2select.gui.workspace.jobs import ComputationJob
from smiles2select.gui.workspace.map_compute import compute_map
from smiles2select.gui.workspace.panels import BasketPanel, InspectorPanel
from smiles2select.gui.workspace.selection_help import strategy_help_text
from smiles2select.gui.workspace.selection_provenance import AppliedSelection, run_provenance
from smiles2select.gui.workspace.session_artifacts import build_artifacts as build_session_artifacts
from smiles2select.gui.workspace.views import ChemicalSpaceView, ParetoView
from smiles2select.pipeline.runner import RunResult
from smiles2select.selection_intelligence.action_log import ActionType
from smiles2select.selection_intelligence.basket import JustificationRequired, SelectionBasket
from smiles2select.selection_intelligence.constrained_selection import (
    CORE_VERSION,
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
    SelectionStatus,
    chemical_status_from_run,
)

VIEW_MAP = "Chemical space"
VIEW_PARETO = "Pareto"
OBJECTIVE_CANDIDATES = ("qed", "mol_wt", "rdkit_wlogp", "tpsa", "sa_score", "np_score")


class WorkspaceWindow(QMainWindow):
    """Turns a finished run into an interactive selection session."""

    computation_progress = Signal(str)
    session_open_requested = Signal(str)

    def __init__(self, result: RunResult, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} - Chemical Space Hub")
        self.resize(1440, 900)

        self._job = None
        self._job_completed = None
        self._job_cancel = None
        self._transient_message = ""
        self._selection_message = ""
        self._papyrus_path = None
        self._map_requested = False
        self._pending_map_refresh = False
        self._scenarios_dialog = None
        self.result = result
        self.candidates = self._build_candidates(result)
        self.basket = self._build_basket(result)
        self.ranker = ParetoRanker()
        self.pareto = None
        self._selection_outcomes: dict[int, SelectionOutcome] = {}
        self._applied_selections = {}
        self._criteria_snapshots = {}
        self._run_provenance = None
        self._scenario_snapshots = {}
        self._model_scores_by_action: dict[int, pd.DataFrame] = {}
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
        self.cancel_job_button = QPushButton("Cancel preview", self)
        self.cancel_job_button.hide()
        self.cancel_job_button.clicked.connect(self._cancel_job)
        self.statusBar().addPermanentWidget(self.cancel_job_button)
        self._initial_criteria = criteria_state.capture(self)
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
        original_selected = set(result.decision.selected_ids())
        states = [
            MoleculeState(
                record_id=int(record_id),
                chemical_status=chemical_status_from_run(
                    valid=bool(valid),
                    passed=bool(decisions["selected"].get(record_id, False)),
                ),
                selection_status=(SelectionStatus.FINAL_SELECTED if record_id in original_selected
                                  else SelectionStatus.UNDECIDED),
                origin=SelectionOrigin.AUTOMATIC if record_id in original_selected else None,
                note="Selected by original screening pipeline" if record_id in original_selected else "",
            )
            for record_id, valid in zip(result.descriptors.index, result.descriptors["valid"])
        ]
        return SelectionBasket(states)

    def _build_layout(self) -> QWidget:
        return build_layout(self, OBJECTIVE_CANDIDATES)

    def _connect(self) -> None:
        self.computation_progress.connect(self._show_progress, Qt.QueuedConnection)
        self.view_selector.currentTextChanged.connect(self._switch_view)
        self.first_objective.currentTextChanged.connect(self._recompute_pareto)
        self.second_objective.currentTextChanged.connect(self._recompute_pareto)
        self.projection_selector.currentIndexChanged.connect(self._request_map)
        self.color_selector.currentIndexChanged.connect(lambda _index: self.refresh())
        self.reference_overlay.stateChanged.connect(self._request_map)
        self.strategy.currentTextChanged.connect(self._update_method_card)
        self.strategy.currentTextChanged.connect(self._recompute_pareto)
        for widget in (self.target_count, self.per_scaffold, self.per_cluster, self.min_scaffolds):
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
        self.basket_panel.docking_export_requested.connect(self._export_docking)
        self.basket_panel.molecule_requested.connect(self._show_molecule)

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
        if self.strategy.currentData() == "qed_only":
            objectives = "Descending QED; property objectives are ignored"
        self.criteria_summary.setText(
            f"Current criteria: {self.target_count.value():,} molecules; "
            f"{self.strategy.currentText()}. {objectives}. "
            f"Scaffold limit: {self.per_scaffold.value() or 'none'}; "
            f"Minimum molecular cores: {self.min_scaffolds.value() or 'none'}; "
            f"cluster limit: {self.per_cluster.value() or 'none'}. "
            "Chemical screening rules remain in effect."
        )
        self._update_selection_summary()

    def _update_selection_summary(self) -> None:
        if not hasattr(self, "selection_summary"):
            return
        count = len(self.basket.final_ids())
        applied = self._active_applied_selection()
        scenario = self._active_snapshot()
        target = (scenario.spec.constraints.target_count if scenario else
                  applied.constraints.target_count if applied else self.result.config.final_count)
        source = "Applied selection" if applied or scenario else "Original pipeline"
        self.basket.target_count = target
        pending = self.has_pending_criteria()
        self.selection_summary.setText(
            f"{source}: {count:,} final molecules for export. "
            + (f"Applied target: {target:,}. " if target is not None else "No original count limit. ")
            + (f"Count differs by {count - target:+,}; review quotas, pins and manual edits. "
               if target is not None and count != target else "")
            + ("Edited criteria not applied; click Create selection. " if pending else "")
            + "Clicking a point only inspects it; Select changes the final library."
        )
        criteria_state.update_export_buttons(self)

    def has_pending_criteria(self) -> bool:
        return criteria_state.capture(self) != criteria_state.active(self)

    def _record_current_criteria(self, index) -> None:
        self._criteria_snapshots[index] = criteria_state.capture(self)

    def _active_applied_selection(self):
        for index in range(len(self.basket.log.applied) - 1, -1, -1):
            action = self.basket.log.applied[index]
            if action.action_type is ActionType.AUTOMATIC_SELECTION:
                if index in self._scenario_snapshots:
                    return None
                applied = self._applied_selections.get(index)
                return applied if applied is not None and action.source == applied.strategy else None
        return None

    def _provenance(self):
        if self._run_provenance is None:
            self._run_provenance = run_provenance(self.result, self.candidates)
        return self._run_provenance

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
            f"Map: all {len(self.basket.final_ids()):,} final molecules plus up to 5,000 "
            f"context candidates of {len(self.candidates):,}; fixed seed 42. Reference overlay: at most "
            "2,000 per library. Selection still considers all candidates."
        )
        candidates, result, selected = self.candidates, self.result, self.basket.final_ids()

        def completed(value):
            self._projection_results[key] = value
            self._map_requested = True
            self.performance_notice.setText(
                f"Map ready: {len(value[0].projection.coordinates):,} projected candidates; "
                f"{len(selected):,} final molecules. Context is sampled; final selection is complete."
            )
            self.refresh()
            self.warnings.setText(self._selection_message)

        self._start_job(lambda: compute_map(candidates, result, method, overlay, selected_ids=selected), completed,
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

    def _start_job(self, function, completed, message, *, cancel=None) -> None:
        if self._job is not None:
            return
        self._job_completed = completed
        self._job_cancel = cancel
        self.cancel_job_button.setEnabled(True)
        self.cancel_job_button.setVisible(cancel is not None)
        self._show_progress(message)
        self.job_progress.show()
        self.controls_scroll.setEnabled(False)
        self.basket_panel.setEnabled(False)
        self.inspector.setEnabled(False)
        self.views.setEnabled(False)
        self._job = ComputationJob(function, self)
        self._job.completed.connect(self._job_succeeded, Qt.QueuedConnection)
        self._job.failed.connect(self._job_failed, Qt.QueuedConnection)
        self._job.finished.connect(self._finish_job, Qt.QueuedConnection)
        self._job.start()

    def _cancel_job(self) -> None:
        if self._job is None or self._job_cancel is None:
            return
        self._job_cancel()
        self.cancel_job_button.setEnabled(False)
        self._show_progress("Cancelling preview after current computation stage...")

    @Slot(str)
    def _show_progress(self, message) -> None:
        self._transient_message = message
        self.warnings.setText(message)

    @Slot(object)
    def _job_succeeded(self, value) -> None:
        try:
            self._job_completed(value)
        except Exception as exc:
            self._job_failed(str(exc))
        else:
            if self.warnings.text() == self._transient_message:
                self.warnings.clear()

    @Slot(str)
    def _job_failed(self, message) -> None:
        self._pending_map_refresh = False
        self.warnings.setText(f"Computation failed: {message}. Review the current final count before export.")

    @Slot()
    def _finish_job(self) -> None:
        job, self._job = self._job, None
        self._job_completed = None
        self._job_cancel = None
        self.cancel_job_button.hide()
        self.job_progress.hide()
        for widget in (self.controls_scroll, self.basket_panel, self.inspector, self.views):
            widget.setEnabled(True)
        if job is not None:
            job.deleteLater()
        criteria_state.update_export_buttons(self)
        if self._pending_map_refresh:
            self._pending_map_refresh = False
            QTimer.singleShot(0, self._request_map)

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
        self.model_panel.clear_preview()
        super().closeEvent(event)

    def _switch_view(self, name: str) -> None:
        self.views.setCurrentIndex(0 if name == VIEW_MAP else 1)
        self.refresh()

    def _update_method_card(self, strategy: str) -> None:
        """Keep method bias and consequence visible while changing controls."""
        strategy = self.strategy.currentData()
        for editor in self.objective_editors:
            editor.setEnabled(strategy != "qed_only")
        try:
            card = get_method_card(strategy)
        except KeyError:
            card = get_method_card("balanced")
        self.strategy_help.setText(strategy_help_text(
            strategy, large_pool=len(self.candidates) > EXACT_PARETO_LIMIT,
            scaffold_ready=self.candidates["murcko_scaffold"].notna().all(),
        ))
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
        if self.strategy.currentData() == "qed_only":
            self.pareto = None
            self.performance_notice.setText(
                "QED-only: descending QED and record_id ties at every library size; "
                "property objectives and Pareto ranking are inactive."
            )
            self.warnings.clear()
            self.refresh()
            return
        if len(self.candidates) > EXACT_PARETO_LIMIT:
            self.pareto = None
            self.performance_notice.setText(
                "Large library: exact Pareto disabled; selection uses weighted objective "
                "percentiles and quotas. Map shows all finals plus sampled context on request."
            )
            self.refresh()
            return
        first = self.first_objective.currentText()
        self.performance_notice.clear()
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
            extras["Included in docking export"] = "Yes" if state.is_selected else "No"
            extras["Map point"] = "This molecule; not a cluster centroid"
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
        self._selection_message = ""
        self.warnings.clear()
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
            min_scaffolds=self.min_scaffolds.value() or None,
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
        result = self.result
        try:
            objectives = ObjectiveSet() if strategy is Strategy.QED_ONLY else self.objectives()
        except ValueError as exc:
            self.warnings.setText(str(exc))
            return
        if strategy is Strategy.QED_ONLY:
            pareto = None
        elif len(source) <= EXACT_PARETO_LIMIT and pareto is None:
            self.warnings.setText("Valid objective ranking is required before selecting.")
            return

        def compute():
            from smiles2select.selection_intelligence.preparation import ensure_selection_scaffolds
            from smiles2select.selection_intelligence.scenarios import (
                ScenarioSpec,
                _rank,
                _reference_mask,
            )

            if result.zone_allocation is not None:
                raise ValueError("This run uses zone allocation. Re-run without zones before "
                                 "changing automatic selection in the workspace.")
            prepared = ensure_selection_scaffolds(
                source, constraints, strategy, progress=self.computation_progress.emit,
            )
            eligible = [state.record_id for state in states
                        if state.chemical_status.passed or (state.pinned and state.is_selected)]
            candidates = prepared.loc[prepared.index.intersection(eligible)].copy()
            candidates = candidates.loc[_reference_mask(result, candidates.index)]
            if pareto is not None:
                candidates = candidates.join(pareto.table, how="left")
            ranking_method = "exact_pareto" if pareto is not None else "strategy_order"
            ranking_warnings = ()
            if strategy is Strategy.QED_ONLY:
                candidates = candidates.loc[~candidates.index.isin(excluded_ids)]
                candidates, ranking_method, ranking_warnings = _rank(candidates, ScenarioSpec(
                    name="workspace", constraints=constraints, strategy=strategy,
                ))
            elif len(source) > EXACT_PARETO_LIMIT:
                ranked, ranking_method, ranking_warnings = _rank(prepared, ScenarioSpec(
                    name="workspace", objectives=tuple(objectives.active),
                    constraints=constraints, strategy=strategy,
                ))
                candidates = ranked.loc[candidates.index]
            if constraints.max_per_scaffold and candidates["murcko_scaffold"].isna().any():
                raise ValueError("Scaffold quota requires complete cached molecular cores.")
            if constraints.max_per_cluster and candidates["cluster_id"].isna().any():
                raise ValueError("Cluster quota requires complete cached cluster IDs.")
            if strategy is Strategy.SCAFFOLD_COVERAGE:
                if candidates["murcko_scaffold"].isna().any():
                    raise ValueError("Molecular-core coverage requires complete cached scaffolds.")
                counts = candidates["murcko_scaffold"].value_counts()
                candidates = candidates.assign(scaffold_size=candidates["murcko_scaffold"].map(counts))
            outcome = select(candidates, constraints, strategy, pinned_ids=pinned_ids,
                             excluded_ids=excluded_ids, explain_rejections=len(candidates) <= 5000)
            metadata = run_provenance(result, prepared)
            applied = AppliedSelection.capture(
                constraints=constraints, objectives=objectives, pareto=pareto, strategy=strategy,
                input_hash=metadata["input_hash"],
                provenance={**metadata, "source": "workspace",
                            "ranking_universe": "all evaluable candidates",
                            "ranking_method": ranking_method, "ranking_warnings": ranking_warnings,
                            "tie_break": "ascending record_id", "random_sampling": False,
                            "selection_algorithm_version": CORE_VERSION,
                            "scaffold_coverage_method": "one feasible member per core before filling; "
                            "rarity, property ranking, record_id; acyclic molecules form one group"},
            )
            return outcome, applied, prepared, metadata

        def completed(value):
            outcome, applied, prepared, metadata = value
            self.candidates = prepared
            self._run_provenance = metadata
            self._apply_selection(outcome, constraints, applied)
            self._update_method_card(strategy.value)

        if len(source) > EXACT_PARETO_LIMIT:
            self._start_job(compute, completed,
                            "Selecting from the full library in the background...")
        else:
            try:
                completed(compute())
            except Exception as exc:
                self.warnings.setText(str(exc))

    def _apply_selection(self, outcome, constraints, applied=None, *, source=None,
                         reason="automatic selection", model_scores=None):
        self.basket.replace_final(
            list(outcome.selected_ids), origin=SelectionOrigin.AUTOMATIC,
            source=source or (applied.strategy if applied else outcome.strategy.value), reason=reason,
        )
        action_index = len(self.basket.log.applied) - 1
        for attached in (
            self._selection_outcomes, self._applied_selections, self._criteria_snapshots,
            self._model_scores_by_action, self._scenario_snapshots,
        ):
            for obsolete in tuple(attached):
                if obsolete >= action_index:
                    attached.pop(obsolete)
        self._selection_outcomes[action_index] = outcome
        if applied is not None:
            self._applied_selections[action_index] = applied
        if model_scores is not None:
            self._model_scores_by_action[action_index] = model_scores.copy(deep=True)
        self._record_current_criteria(action_index)
        self._pending_map_refresh = bool(
            self._job is not None and len(self.candidates) > 5000
        )
        self.warnings.setText("\n".join([
            f"Selection applied: {len(self.basket.final_ids()):,} / {constraints.target_count:,} molecules.",
            *(applied.provenance.get("ranking_warnings", ()) if applied else ()),
            *outcome.warnings(constraints),
        ]))
        self._selection_message = self.warnings.text()
        self.refresh()

    def _undo(self) -> None:
        self.basket.undo()
        self._selection_message = ""
        criteria_state.restore(self)
        self.refresh()

    def _redo(self) -> None:
        self.basket.redo()
        self._selection_message = ""
        criteria_state.restore(self)
        self.refresh()

    def _export(self) -> None:
        if not criteria_state.allow_export(self):
            return
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

    def _export_docking(self) -> None:
        if not criteria_state.allow_export(self):
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export final molecules to SMILES2Docking", "selection-docking.csv",
            "CSV (*.csv);;Excel (*.xlsx)",
        )
        if not path:
            return
        try:
            paths = selection_export.export_docking(self.build_artifacts(), path)
        except Exception as exc:
            QMessageBox.warning(self, "Docking export unavailable", str(exc))
            return
        QMessageBox.information(self, "Docking export complete", "\n".join(str(p) for p in paths))

    def save_session(self, path):
        """Persist source-independent workspace state and its undo/redo cursor."""
        if self.is_busy:
            raise ValueError("Wait for active computation before saving the workspace")
        from smiles2select.storage.session_store import save_session

        return save_session(path, self.result, self.basket, workspace_state={
            "applied_selections": self._applied_selections,
            "criteria_snapshots": self._criteria_snapshots,
            "selection_outcomes": self._selection_outcomes,
            "model_scores_by_action": self._model_scores_by_action,
            "initial_criteria": self._initial_criteria,
            "draft_criteria": criteria_state.capture(self),
            "run_provenance": self._run_provenance,
            "scenario_snapshots": self._scenario_snapshots,
            "candidates": self.candidates.copy(deep=True),
        })

    @classmethod
    def from_session(cls, path, parent=None):
        """Open a complete saved workspace without reading the original library."""
        from smiles2select.storage.session_store import load_session

        result, basket, state = load_session(path)
        window = cls(result, parent=parent)
        candidates = state.get("candidates")
        if candidates is not None:
            if (not isinstance(candidates, pd.DataFrame)
                    or not candidates.index.equals(window.candidates.index)
                    or not candidates.columns.equals(window.candidates.columns)):
                window.close()
                raise ValueError("Saved candidates do not match workspace records or columns")
            window.candidates = candidates.copy(deep=True)
        window.basket = basket
        for key, attribute in (
            ("applied_selections", "_applied_selections"),
            ("criteria_snapshots", "_criteria_snapshots"),
            ("selection_outcomes", "_selection_outcomes"),
            ("model_scores_by_action", "_model_scores_by_action"),
            ("scenario_snapshots", "_scenario_snapshots"),
        ):
            setattr(window, attribute, dict(state.get(key, {})))
        window._run_provenance = state.get("run_provenance")
        window._initial_criteria = state.get("initial_criteria", window._initial_criteria)
        draft = state.get("draft_criteria")
        if draft is None:
            criteria_state.restore(window)
        else:
            criteria_state.apply(window, draft)
        window.refresh()
        return window

    def _save_session_dialog(self) -> None:
        if self.is_busy:
            self.warnings.setText("Wait for active computation before saving the workspace.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save workspace session", "selection.s2s.sqlite",
            "SMILES2Select session (*.s2s.sqlite *.sqlite)",
        )
        if not path:
            return
        try:
            saved = self.save_session(path)
        except Exception as exc:
            QMessageBox.warning(self, "Session not saved", str(exc))
            return
        QMessageBox.information(self, "Session saved", str(saved))

    def _open_session_dialog(self) -> None:
        if self.is_busy:
            self.warnings.setText("Wait for active computation before opening a session.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open workspace session", "",
            "SMILES2Select session (*.s2s.sqlite *.sqlite)",
        )
        if path:
            self.session_open_requested.emit(path)

    def _active_snapshot(self):
        for index in range(len(self.basket.log.applied) - 1, -1, -1):
            if self.basket.log.applied[index].action_type is ActionType.AUTOMATIC_SELECTION:
                return self._scenario_snapshots.get(index)
        return None

    def build_artifacts(self) -> selection_export.SessionArtifacts:
        """Everything the export needs, assembled from the current session."""
        return build_session_artifacts(self)

    def refresh(self) -> None:
        """Redraw the active view and the basket from the shared state."""
        selected = self.basket.final_ids()
        self._update_selection_summary()
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
        criteria_state.update_export_buttons(self)
        if self.inspector.record_id is not None:
            self._show_molecule(self.inspector.record_id)

    def _map_projection(self) -> tuple[ProjectionResult, pd.DataFrame]:
        method = str(self.projection_selector.currentData())
        overlay = bool(self.reference_overlay.isChecked() and self.result.reference_libraries)
        key = (method, overlay)
        cached = self._projection_results.get(key)
        missing = cached is None or not set(self.basket.final_ids()).issubset(
            cached[0].projection.coordinates.index
        )
        if missing:
            if len(self.candidates) > 5000:
                if not self.is_busy:
                    QTimer.singleShot(0, self._request_map)
                if cached is None:
                    raise ValueError("Building map with every final molecule...")
            else:
                self._projection_results[key] = compute_map(
                    self.candidates, self.result, method, overlay,
                    selected_ids=self.basket.final_ids(),
                )
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
