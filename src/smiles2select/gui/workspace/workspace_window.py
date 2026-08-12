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
    QCheckBox,
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
from rdkit import Chem

from smiles2select.app_metadata import APP_NAME
from smiles2select.chemical_space.clustering import ClusteringTooLargeError, cluster
from smiles2select.chemical_space.projection_manager import (
    ProjectionConfig,
    ProjectionResult,
    project,
)
from smiles2select.chemistry.descriptor_registry import default_registry
from smiles2select.chemistry.scaffolds import scaffolds_from_smiles
from smiles2select.explainability.consequences import explain_change
from smiles2select.explainability.method_cards import get_method_card
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

VIEW_MAP = "Chemical space"
VIEW_PARETO = "Pareto"
OBJECTIVE_CANDIDATES = ("qed", "mol_wt", "rdkit_wlogp", "tpsa", "sa_score", "np_score")


class WorkspaceWindow(QMainWindow):
    """Turns a finished run into an interactive selection session."""

    def __init__(self, result: RunResult, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} - Chemical Space Hub")
        self.resize(1440, 900)

        self.result = result
        self.candidates = self._build_candidates(result)
        self.basket = self._build_basket(result)
        self.ranker = ParetoRanker()
        self.pareto = None
        self._outcome: SelectionOutcome | None = None
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
            evaluable["murcko_scaffold"] = scaffolds_from_smiles(
                evaluable["canonical_smiles"].fillna("").tolist()
            )
        try:
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
        controls = QGroupBox("Chemical Space Hub")
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
        self.per_scaffold.setSpecialValueText("no quota")
        self.per_cluster = QSpinBox()
        self.per_cluster.setRange(0, 100)
        self.per_cluster.setSpecialValueText("no quota")
        self.strategy = QComboBox()
        self.strategy.addItems([item.value for item in Strategy])
        self.method_card = QLabel()
        self.method_card.setWordWrap(True)
        self.method_card.setStyleSheet("background: #eef3f8; padding: 6px;")
        self.consequence = QLabel()
        self.consequence.setWordWrap(True)
        self.projection_selector = QComboBox()
        self.projection_selector.addItem("Property PCA", "property_pca")
        self.projection_selector.addItem("Structural UMAP", "structural_umap")
        self.projection_selector.addItem("TMAP", "tmap")
        self.color_selector = QComboBox()
        self.color_selector.addItems(["Selection status", "Pareto rank", "Reference similarity"])
        self.reference_overlay = QCheckBox("Show reference overlay")
        self.reference_overlay.setChecked(bool(self.result.reference_libraries))
        reference_info = QLabel(
            f"Reference libraries: {len(self.result.reference_libraries)}\n"
            "Map distances are for visualization only. Molecular similarity and "
            "novelty use fingerprints."
        )
        reference_info.setWordWrap(True)

        form.addRow("View:", self.view_selector)
        form.addRow("Objective 1 (maximize):", self.first_objective)
        form.addRow("Objective 2 (minimize):", self.second_objective)
        form.addRow("Final count:", self.target_count)
        form.addRow("Max per scaffold:", self.per_scaffold)
        form.addRow("Max per cluster:", self.per_cluster)
        form.addRow("Strategy:", self.strategy)
        form.addRow("Method card:", self.method_card)
        form.addRow("Consequence:", self.consequence)
        form.addRow("Projection:", self.projection_selector)
        form.addRow("Color by:", self.color_selector)
        form.addRow(self.reference_overlay)
        form.addRow(reference_info)
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
        self.projection_selector.currentIndexChanged.connect(lambda _index: self.refresh())
        self.color_selector.currentIndexChanged.connect(lambda _index: self.refresh())
        self.reference_overlay.stateChanged.connect(lambda _state: self.refresh())
        self.strategy.currentTextChanged.connect(self._update_method_card)
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

    def _switch_view(self, name: str) -> None:
        self.views.setCurrentIndex(0 if name == VIEW_MAP else 1)
        self.refresh()

    def _update_method_card(self, strategy: str) -> None:
        """Keep method bias and consequence visible while changing controls."""
        try:
            card = get_method_card(strategy)
        except KeyError:
            card = get_method_card("balanced")
        self.method_card.setText(
            f"<b>{card.title}</b><br>{card.what_it_does}<br>"
            f"<i>Favors:</i> {card.what_it_favors}<br>"
            f"<i>May underrepresent:</i> {card.may_underrepresent}"
        )
        self.consequence.setText(
            explain_change("selection_strategy", "previous", strategy).message
        )

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
            reason="automatic selection",
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
        path, _ = QFileDialog.getSaveFileName(self, "Export selection", "", "Excel (*.xlsx)")
        if not path:
            return
        workbook, recipe = selection_export.export(self.build_artifacts(), path)
        QMessageBox.information(
            self, "Exported", f"Workbook: {workbook.name}\nRecipe: {recipe.name}"
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
            pareto=self.pareto,
            clusters=self.candidates.get("cluster_id"),
            scaffolds=self.candidates.get("murcko_scaffold"),
        )

    def refresh(self) -> None:
        """Redraw the active view and the basket from the shared state."""
        selected = self.basket.final_ids()
        ranks = self.pareto.table["pareto_rank"] if self.pareto is not None else None

        if self.views.currentIndex() == 0:
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
        self.basket_panel.refresh(self.basket, self.candidates.get("molecule_id"))

    def _map_projection(self) -> tuple[ProjectionResult, pd.DataFrame]:
        """Build/cache the selected map and, for structural methods, references."""
        method = str(self.projection_selector.currentData())
        overlay = bool(self.reference_overlay.isChecked() and self.result.reference_libraries)
        key = (method, overlay)
        if key in self._projection_results:
            return self._projection_results[key]
        config = ProjectionConfig(method=method, fingerprint=self.result.config.fingerprint_config)
        if overlay and method == "property_pca":
            features = tuple(ProjectionConfig().features)
            combined = self.candidates[[column for column in features if column in self.candidates]].copy()
            combined.index = [f"candidate::{record_id}" for record_id in combined.index]
            reference_frames = []
            registry = default_registry()
            for library in self.result.reference_libraries:
                rows = []
                for row_id, row in library.valid.iterrows():
                    mol = Chem.MolFromSmiles(str(row["canonical_smiles"]))
                    values = registry.compute(mol, features) if mol is not None else {}
                    rows.append(values)
                frame = pd.DataFrame(rows, index=library.valid.index)
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
            self._projection_results[key] = (
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
            result = project(self.candidates, config)
            self._projection_results[key] = (result, pd.DataFrame(columns=["x", "y"]))
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
