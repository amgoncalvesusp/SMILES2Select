"""The Selection Intelligence workspace window.

Layout follows the specification: controls on the left, one dominant
visualisation in the centre, the molecule inspector on the right, the basket
below. Not a grid of equally weighted cards - at any moment there is one
question on screen, and the rest of the interface supports answering it.
"""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from smiles2select.app_metadata import APP_NAME
from smiles2select.chemical_space import pca_projection
from smiles2select.chemical_space.clustering import cluster
from smiles2select.chemistry.scaffolds import scaffolds_from_smiles
from smiles2select.export import selection_export
from smiles2select.gui.workspace.panels import BasketPanel, InspectorPanel
from smiles2select.gui.workspace.views import ChemicalSpaceView, ParetoView
from smiles2select.pipeline.runner import RunResult
from smiles2select.selection_intelligence import recipes
from smiles2select.selection_intelligence.basket import JustificationRequired, SelectionBasket
from smiles2select.selection_intelligence.constrained_selection import (
    SelectionConstraints,
    SelectionOutcome,
    Strategy,
    select,
)
from smiles2select.selection_intelligence.objectives import Direction, Objective, ObjectiveSet
from smiles2select.selection_intelligence.pareto_ranking import ParetoRanker
from smiles2select.selection_intelligence.states import (
    MoleculeState,
    SelectionOrigin,
    chemical_status_from_run,
)

VIEW_MAP = "Espaço químico"
VIEW_PARETO = "Pareto"
OBJECTIVE_CANDIDATES = ("qed", "mol_wt", "rdkit_wlogp", "tpsa", "sa_score", "np_score")


class WorkspaceWindow(QMainWindow):
    """Turns a finished run into an interactive selection session."""

    def __init__(self, result: RunResult, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} — Selection Intelligence")
        self.resize(1440, 900)

        self.result = result
        self.candidates = self._build_candidates(result)
        self.basket = self._build_basket(result)
        self.ranker = ParetoRanker()
        self.pareto = None
        self._outcome: SelectionOutcome | None = None

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
            evaluable["murcko_scaffold"] = scaffolds_from_smiles(
                evaluable["canonical_smiles"].fillna("").tolist()
            )
        clusters = cluster(evaluable["canonical_smiles"].fillna("").tolist(), evaluable.index)
        evaluable["cluster_id"] = clusters.labels
        for column in ("qed", "sa_score"):
            if column not in evaluable.columns:
                evaluable[column] = pd.NA
        return evaluable

    def _build_basket(self, result: RunResult) -> SelectionBasket:
        """Seed the basket with the chemical verdict of every record."""
        decisions = result.decision.decisions
        states = [
            MoleculeState(
                record_id=int(record_id),
                chemical_status=chemical_status_from_run(
                    valid=bool(row["valid"]),
                    passed=bool(decisions["selected"].get(record_id, False)),
                ),
            )
            for record_id, row in result.descriptors.iterrows()
        ]
        return SelectionBasket(states)

    def _build_layout(self) -> QWidget:
        controls = QGroupBox("Controles")
        form = QFormLayout(controls)

        self.view_selector = QComboBox()
        self.view_selector.addItems([VIEW_MAP, VIEW_PARETO])

        numeric = [column for column in OBJECTIVE_CANDIDATES if column in self.candidates.columns]
        self.first_objective = QComboBox()
        self.first_objective.addItems(numeric)
        self.second_objective = QComboBox()
        self.second_objective.addItems(numeric)
        if len(numeric) > 1:
            self.second_objective.setCurrentIndex(1)

        self.target_count = QSpinBox()
        self.target_count.setRange(1, 100000)
        self.target_count.setValue(min(50, max(1, len(self.candidates))))
        self.per_scaffold = QSpinBox()
        self.per_scaffold.setRange(0, 100)
        self.per_scaffold.setSpecialValueText("sem cota")
        self.per_cluster = QSpinBox()
        self.per_cluster.setRange(0, 100)
        self.per_cluster.setSpecialValueText("sem cota")
        self.strategy = QComboBox()
        self.strategy.addItems([item.value for item in Strategy])

        form.addRow("Visualização:", self.view_selector)
        form.addRow("Objetivo 1 (maximizar):", self.first_objective)
        form.addRow("Objetivo 2 (minimizar):", self.second_objective)
        form.addRow("Número final:", self.target_count)
        form.addRow("Máx. por scaffold:", self.per_scaffold)
        form.addRow("Máx. por cluster:", self.per_cluster)
        form.addRow("Estratégia:", self.strategy)
        form.addRow(self.warnings)

        top = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(controls)
        left_layout.addStretch(1)
        top.addWidget(left)
        top.addWidget(self.views)
        top.addWidget(self.inspector)
        top.setSizes([280, 780, 340])

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(top)
        splitter.addWidget(self.basket_panel)
        splitter.setSizes([620, 260])

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.addWidget(splitter)
        return container

    def _connect(self) -> None:
        self.view_selector.currentTextChanged.connect(self._switch_view)
        self.first_objective.currentTextChanged.connect(self._recompute_pareto)
        self.second_objective.currentTextChanged.connect(self._recompute_pareto)

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

    def _switch_view(self, name: str) -> None:
        self.views.setCurrentIndex(0 if name == VIEW_MAP else 1)
        self.refresh()

    def objectives(self) -> ObjectiveSet:
        return ObjectiveSet(
            [
                Objective(self.first_objective.currentText(), Direction.MAXIMIZE),
                Objective(self.second_objective.currentText(), Direction.MINIMIZE),
            ]
        )

    def _recompute_pareto(self) -> None:
        first = self.first_objective.currentText()
        second = self.second_objective.currentText()
        if not first or not second or first == second:
            self.pareto = None
            self.warnings.setText("Escolha dois objetivos diferentes.")
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
            extras["Status químico"] = state.chemical_status.value
            extras["Status de seleção"] = state.selection_status.value
        self.inspector.show_molecule(record_id, self.candidates, extras)

    def _shortlist_region(self, record_ids: list[int]) -> None:
        if not record_ids:
            return
        preview = self.basket.preview(record_ids)
        summary = "\n".join(f"{label}: {value}" for label, value in preview.items())
        answer = QMessageBox.question(
            self,
            "Adicionar à shortlist",
            f"{summary}\n\nConfirmar?",
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
            QMessageBox.warning(self, "Justificativa necessária", str(exc))
            return
        self.refresh()

    def _select_with_justification(self, record_id: int) -> None:
        """A chemically rejected molecule can only be kept with a written reason."""
        state = self.basket.state(record_id)
        reason = ""
        if not state.chemical_status.passed:
            reason, accepted = QInputDialog.getText(
                self,
                "Justificativa",
                f"A molécula está como {state.chemical_status.value}. Justifique a seleção:",
            )
            if not accepted or not reason.strip():
                raise JustificationRequired("seleção cancelada: nenhuma justificativa fornecida")
        self.basket.add_to_final([record_id], reason=reason)

    def constraints(self) -> SelectionConstraints:
        return SelectionConstraints(
            target_count=self.target_count.value(),
            max_per_scaffold=self.per_scaffold.value() or None,
            max_per_cluster=self.per_cluster.value() or None,
        )

    def _auto_select(self) -> None:
        candidates = self.candidates.copy()
        if self.pareto is not None:
            candidates = candidates.join(self.pareto.table, how="left")

        constraints = self.constraints()
        outcome = select(
            candidates,
            constraints,
            Strategy(self.strategy.currentText()),
            pinned_ids=self.basket.pinned_ids(),
            excluded_ids=self.basket.excluded_ids(),
        )
        self.basket.add_to_final(
            list(outcome.selected_ids),
            origin=SelectionOrigin.AUTOMATIC,
            source=outcome.strategy.value,
            reason="seleção automática",
        )
        self._outcome = outcome
        self.warnings.setText("\n".join(outcome.warnings(constraints)))
        self.refresh()

    def _undo(self) -> None:
        self.basket.undo()
        self.refresh()

    def _redo(self) -> None:
        self.basket.redo()
        self.refresh()

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Exportar seleção", "", "Excel (*.xlsx)")
        if not path:
            return
        workbook, recipe = selection_export.export(self.build_artifacts(), path)
        QMessageBox.information(
            self, "Exportado", f"Planilha: {workbook.name}\nReceita: {recipe.name}"
        )

    def build_artifacts(self) -> selection_export.SessionArtifacts:
        """Everything the export needs, assembled from the current session."""
        outcome = self._outcome or SelectionOutcome(selected_ids=self.basket.final_ids())
        return selection_export.SessionArtifacts(
            result=self.result,
            basket=self.basket,
            outcome=outcome,
            constraints=self.constraints(),
            recipe=recipes.SelectionRecipe(
                name="workspace",
                input_hash=self.result.config.fingerprint(),
                objectives=self.objectives().as_dicts(),
                target_count=self.target_count.value(),
                strategy=self.strategy.currentText(),
                max_per_scaffold=self.per_scaffold.value() or None,
                max_per_cluster=self.per_cluster.value() or None,
                pinned_ids=self.basket.pinned_ids(),
                excluded_ids=self.basket.excluded_ids(),
            ),
            pareto=self.pareto,
            clusters=self.candidates.get("cluster_id"),
            scaffolds=self.candidates.get("murcko_scaffold"),
        )

    def refresh(self) -> None:
        """Redraw the active view and the basket from the shared state."""
        selected = self.basket.final_ids()
        ranks = self.pareto.table["pareto_rank"] if self.pareto is not None else None

        if self.views.currentIndex() == 0:
            projection = pca_projection.project(self.candidates)
            self.map_view.set_points(projection.coordinates, selected, ranks)
        elif self.pareto is not None:
            self.pareto_view.show_objectives(
                self.candidates,
                self.first_objective.currentText(),
                self.second_objective.currentText(),
                self.pareto.table["pareto_rank"],
                selected,
            )
        self.basket_panel.refresh(self.basket, self.candidates.get("molecule_id"))
