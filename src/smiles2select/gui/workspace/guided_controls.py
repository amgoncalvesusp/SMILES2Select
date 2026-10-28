"""Progressive disclosure for a selection workspace on notebook screens."""

from textwrap import fill

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from smiles2select.gui.pages.methods_page import experimental_notice
from smiles2select.gui.workspace import active_criteria
from smiles2select.gui.workspace.selection_help import (
    DIRECTION_HELP,
    OBJECTIVE_HELP,
    STRATEGY_HELP,
    show_selection_help,
)
from smiles2select.selection_intelligence.constrained_selection import Strategy
from smiles2select.selection_intelligence.objectives import Direction, Objective

STRATEGY_LABELS = {
    "qed_only": "Rank by QED only (drug-likeness)",
    "balanced": "Balance properties and representation",
    "pareto_first": "Prioritize favorable property trade-offs",
    "diversity_first": "Spread across property values",
    "scaffold_coverage": "Cover more molecular cores",
    "manual_assisted": "Complete my pinned choices",
}
MAP_GUIDANCE = (
    "Map distances are visual guidance, not measured molecular similarity. "
    "Map context may be sampled; selection uses the full library."
)


class StrategyCombo(QComboBox):
    """Human labels and stable strategy IDs; accept legacy programmatic IDs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        for strategy in Strategy:
            self.addItem(STRATEGY_LABELS[strategy.value], strategy.value)
            self.setItemData(self.count() - 1, STRATEGY_HELP[strategy.value], Qt.ToolTipRole)

    def setCurrentText(self, text):
        index = self.findData(text)
        if index >= 0:
            self.setCurrentIndex(index)
        else:
            super().setCurrentText(text)


def wrapped(text=""):
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextFormat(Qt.PlainText)
    return label


class ObjectiveControls(QWidget):
    """All existing objective directions, with explicit interval/value inputs."""

    def __init__(self, field, direction, parent=None):
        super().__init__(parent)
        self.field = field
        self.direction = QComboBox()
        for label, value in (
            ("Higher is better", Direction.MAXIMIZE),
            ("Lower is better", Direction.MINIMIZE),
            ("Desired interval", Direction.TARGET_RANGE),
            ("Desired value", Direction.TARGET_VALUE),
        ):
            self.direction.addItem(label, value)
        self.direction.setCurrentIndex(0 if direction is Direction.MAXIMIZE else 1)
        self.direction.setToolTip(DIRECTION_HELP)
        self.help_label = wrapped(OBJECTIVE_HELP.get(field.currentText(), ""))
        field.currentTextChanged.connect(
            lambda name: self.help_label.setText(OBJECTIVE_HELP.get(name, ""))
        )
        self.low, self.high = QDoubleSpinBox(), QDoubleSpinBox()
        for widget in (self.low, self.high):
            widget.setRange(-1_000_000, 1_000_000)
            widget.setDecimals(3)
        self.high.setValue(500)
        self.low.setPrefix("Value / lower: ")
        self.high.setPrefix("Upper: ")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for widget in (field, self.help_label, self.direction, self.low, self.high):
            layout.addWidget(widget)
        self.direction.currentIndexChanged.connect(self._visibility)
        self._visibility()

    def _visibility(self):
        mode = Direction(self.direction.currentData())
        self.low.setVisible(mode in (Direction.TARGET_RANGE, Direction.TARGET_VALUE))
        self.high.setVisible(mode is Direction.TARGET_RANGE)

    def objective(self):
        mode = Direction(self.direction.currentData())
        return Objective(
            self.field.currentText(), mode,
            target_low=self.low.value() if mode is Direction.TARGET_RANGE else None,
            target_high=self.high.value() if mode is Direction.TARGET_RANGE else None,
            target_value=self.low.value() if mode is Direction.TARGET_VALUE else None,
        )


def _chemical_controls(window):
    window.chemical_controls = QGroupBox("Chemical ranking")
    form = QFormLayout(window.chemical_controls)
    form.setRowWrapPolicy(QFormLayout.WrapAllRows)
    form.addRow(wrapped("Choose a strategy, then create a chemical selection. "
                        "Clicking a map point only inspects it."))
    window.strategy = StrategyCombo()
    window.strategy_help = wrapped(STRATEGY_HELP["balanced"])
    window.selection_help_button = QPushButton("How to choose criteria...")
    window.selection_help_button.clicked.connect(lambda: show_selection_help(window))
    window.compare_button = QPushButton("Compare scenarios A / B...")
    window.compare_button.clicked.connect(window._open_scenarios)
    window.criteria_summary = wrapped()
    form.addRow("Selection strategy", window.strategy)
    for widget in (window.strategy_help, window.selection_help_button, window.compare_button):
        form.addRow(widget)
    window.method_card, window.consequence = QLabel(), wrapped()
    window.method_card.setWordWrap(True)
    form.addRow(window.method_card)
    form.addRow(window.consequence)


def _objectives(window, form, objective_candidates):
    numeric = [name for name in objective_candidates if name in window.candidates.columns]
    window.first_objective, window.second_objective = QComboBox(), QComboBox()
    for widget in (window.first_objective, window.second_objective):
        widget.addItems(numeric)
        for index in range(widget.count()):
            widget.setItemData(index, OBJECTIVE_HELP.get(widget.itemText(index), ""), Qt.ToolTipRole)
    if len(numeric) > 1:
        window.second_objective.setCurrentIndex(1)
    window.objective_editors = (
        ObjectiveControls(window.first_objective, Direction.MAXIMIZE),
        ObjectiveControls(window.second_objective, Direction.MINIMIZE),
    )
    for i, widget in enumerate(window.objective_editors, 1):
        form.addRow(f"Chemical property objective {i}", widget)


def _quotas(window, form):
    window.per_scaffold, window.per_cluster = QSpinBox(), QSpinBox()
    for widget in (window.per_scaffold, window.per_cluster):
        widget.setRange(0, 10_000_000)
        widget.setSpecialValueText("No limit")
    window.per_scaffold.setToolTip("Maximum molecules sharing a Murcko molecular core. "
                                   "Missing cores are computed when you click Create selection. "
                                   "A strict limit may prevent reaching the requested count.")
    window.per_cluster.setToolTip("Maximum molecules in an available structural cluster.")
    if window.candidates["cluster_id"].isna().any():
        window.per_cluster.setEnabled(False)
        window.per_cluster.setToolTip("Unavailable: complete clustering is not cached; large exact "
                                     "clustering is intentionally skipped to protect memory.")
    form.addRow("Maximum per molecular core (scaffold)", window.per_scaffold)
    window.min_scaffolds = QSpinBox()
    window.min_scaffolds.setRange(0, 10_000_000)
    window.min_scaffolds.setSpecialValueText("No minimum")
    window.min_scaffolds.setToolTip(
        "Request at least this many distinct Murcko molecular cores before filling by rank. "
        "Acyclic molecules count as one core group. Missing cores are computed automatically. "
        "If the library, requested count, pins or quotas make this impossible, selection "
        "reports the shortfall; it does not guarantee the requested diversity."
    )
    form.addRow("Minimum molecular cores (scaffolds)", window.min_scaffolds)
    form.addRow("Maximum per structural cluster", window.per_cluster)


def _map_controls(window):
    window.map_controls = QWidget()
    layout = QVBoxLayout(window.map_controls)
    layout.setContentsMargins(0, 0, 0, 0)
    toolbar = QHBoxLayout()
    window.view_selector = QComboBox()
    window.view_selector.addItems(["Chemical space", "Pareto", "Table"])
    window.view_selector.setAccessibleName("Map, Pareto or candidate table view")
    window.map_button = QPushButton("Refresh map")
    window.map_button.setToolTip("Show final molecules and the disclosed map context.")
    window.map_button.clicked.connect(window._request_map)
    window.map_options_toggle = QToolButton()
    window.map_options_toggle.setText("Display...")
    window.map_options_toggle.setToolTip("Projection, color, reference overlay and local evidence.")
    window.map_options_toggle.setCheckable(True)
    for widget in (window.view_selector, window.map_button, window.map_options_toggle):
        toolbar.addWidget(widget)
    layout.addLayout(toolbar)
    options = QWidget()
    form = QFormLayout(options)
    form.setContentsMargins(0, 0, 0, 0)
    form.setRowWrapPolicy(QFormLayout.WrapLongRows)
    window.projection_selector = QComboBox()
    window.projection_selector.addItem("Property PCA (lightweight)", "property_pca")
    window.projection_selector.addItem("Structural UMAP (optional, slower)", "structural_umap")
    window.color_selector = QComboBox()
    window.color_selector.addItems(["Selection status", "Pareto rank", "Reference similarity"])
    window.reference_overlay = QCheckBox("Show reference overlay")
    window.reference_overlay.setChecked(bool(window.result.reference_libraries))
    form.addRow("Projection", window.projection_selector)
    form.addRow("Color by", window.color_selector)
    form.addRow(window.reference_overlay)
    window.papyrus_button = QPushButton("Attach local Papyrus evidence index...")
    window.papyrus_button.clicked.connect(window._attach_papyrus)
    form.addRow(window.papyrus_button)
    options.hide()
    window.map_options_toggle.toggled.connect(options.setVisible)
    layout.addWidget(options)
    window.view_notice = wrapped(MAP_GUIDANCE)
    layout.addWidget(window.view_notice)


def _method_controls(window, objective_candidates):
    from smiles2select.gui.workspace.model_decision import ModelDecisionPanel

    _chemical_controls(window)
    window.model_panel = ModelDecisionPanel(window)
    window.method_pages = QWidget()
    pages = QVBoxLayout(window.method_pages)
    pages.setContentsMargins(0, 0, 0, 0)
    window._method_widgets = (window.chemical_controls, window.model_panel,
                              window.model_panel.contextual, window.model_panel.tolerance)
    for page in window._method_widgets:
        pages.addWidget(page)
    window.advanced_toggle = QToolButton()
    window.advanced_toggle.setText("Quotas / objectives")
    window.advanced_toggle.setToolTip("Shared quotas and chemical property objectives.")
    window.advanced_toggle.setCheckable(True)
    window.advanced_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    window.advanced_toggle.setArrowType(Qt.RightArrow)
    window.advanced_content = QWidget()
    advanced = QFormLayout(window.advanced_content)
    advanced.setRowWrapPolicy(QFormLayout.WrapAllRows)
    advanced.setContentsMargins(0, 0, 0, 0)
    _quotas(window, advanced)
    _objectives(window, advanced, objective_candidates)
    advanced.addRow(window.criteria_summary)
    window.advanced_content.hide()
    window.advanced_toggle.toggled.connect(window.advanced_content.setVisible)
    window.advanced_toggle.toggled.connect(lambda on: window.advanced_toggle.setArrowType(
        Qt.DownArrow if on else Qt.RightArrow))
    controls = QWidget()
    form = QVBoxLayout(controls)
    for widget in (window.advanced_toggle, window.advanced_content, window.method_pages):
        form.addWidget(widget)
    window.save_session_button = QPushButton("Save workspace session...")
    window.save_session_button.clicked.connect(window._save_session_dialog)
    window.open_session_button = QPushButton("Open saved session...")
    window.open_session_button.clicked.connect(window._open_session_dialog)
    window.performance_notice = wrapped()
    for widget in (window.save_session_button, window.open_session_button, window.performance_notice):
        form.addWidget(widget)
    form.addStretch()
    window.controls_scroll = QScrollArea()
    window.controls_scroll.setWidgetResizable(True)
    window.controls_scroll.setMinimumWidth(380)
    window.controls_scroll.setWidget(controls)


def _compact_forms(window, container):
    for combo in container.findChildren(QComboBox):
        combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(12)
        if window.controls_scroll.isAncestorOf(combo):
            combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
    for label in window.controls_scroll.findChildren(QLabel):
        label.setWordWrap(True)
    for button in window.controls_scroll.findChildren(QPushButton):
        button.setText("\n".join(fill(line, width=24) for line in button.text().splitlines()))
    for checkbox in window.controls_scroll.findChildren(QCheckBox):
        checkbox.setText(fill(checkbox.text(), width=26))


def _show_method(window, index):
    for number, (page, actions) in enumerate(zip(window._method_widgets, window._method_actions)):
        page.setVisible(number == index)
        actions.setVisible(number == index)
    window.controls_scroll.verticalScrollBar().setValue(0)


def _selection_actions(window):
    window.selection_actions = QWidget()
    layout = QVBoxLayout(window.selection_actions)
    layout.setContentsMargins(0, 0, 0, 0)
    toolbar = QHBoxLayout()
    window.method_selector = QComboBox()
    for label, value in (("Chemical ranking", "chemical"), ("Model (experimental)", "model"),
                         ("Contextual (experimental)", "contextual"),
                         ("Tolerance (experimental)", "tolerance")):
        window.method_selector.addItem(label, value)
    window.method_selector.setToolTip("Change the form only. Preview and adoption remain explicit.")
    window.target_count.setAccessibleName("Requested number of final molecules")
    window.target_count.setMaximumWidth(120)
    toolbar.addWidget(QLabel("Requested N"))
    toolbar.addWidget(window.target_count)
    toolbar.addWidget(QLabel("Configure"))
    toolbar.addWidget(window.method_selector)
    method_actions = QWidget()
    action_layout = QVBoxLayout(method_actions)
    action_layout.setContentsMargins(0, 0, 0, 0)
    window.select_button = QPushButton("Create selection")
    window.select_button.setToolTip("Replace the chemical selection. Undo restores the previous set.")
    window.select_button.clicked.connect(window._auto_select)
    panels = (window.model_panel, window.model_panel.contextual, window.model_panel.tolerance)
    window._method_actions = tuple(QWidget() for _ in range(4))
    for row, buttons in zip(window._method_actions, ((window.select_button,), *(
            (panel.preview_button, panel.adopt_button) for panel in panels))):
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        for button in buttons:
            row_layout.addWidget(button)
        action_layout.addWidget(row)
    toolbar.addStretch()
    toolbar.addWidget(window.applied_criteria_button)
    layout.addLayout(toolbar)
    action_row = QHBoxLayout()
    action_row.addWidget(method_actions, stretch=1)
    window.molecule_compare_button = QPushButton("Compare molecules...")
    window.molecule_compare_button.setToolTip("Inspect a molecule, keep it as A, then compare another.")
    window.molecule_compare_button.clicked.connect(window._open_molecule_comparison)
    action_row.addWidget(window.molecule_compare_button)
    layout.addLayout(action_row)
    window.basket_panel.auto_button.hide()
    layout.addWidget(window.basket_panel.actions)
    window.method_selector.currentIndexChanged.connect(lambda index: _show_method(window, index))
    _show_method(window, 0)


def build_layout(window, objective_candidates):
    window.target_count = QSpinBox()
    window.target_count.setRange(1, 10_000_000)
    window.target_count.setValue(window.result.config.final_count or max(1, len(window.basket.final_ids())))
    _method_controls(window, objective_candidates)
    _map_controls(window)
    center = QWidget()
    center_layout = QVBoxLayout(center)
    center_layout.setContentsMargins(0, 0, 0, 0)
    center_layout.addWidget(window.map_controls)
    center_layout.addWidget(window.views, stretch=1)
    top = QSplitter(Qt.Horizontal)
    top.addWidget(window.controls_scroll)
    top.addWidget(center)
    top.addWidget(window.inspector)
    top.setSizes([300, 610, 330])
    splitter = QSplitter(Qt.Vertical)
    splitter.addWidget(top)
    splitter.addWidget(window.basket_panel)
    splitter.setSizes([570, 200])
    container = QWidget()
    layout = QVBoxLayout(container)
    window.experimental_notice = experimental_notice(window)
    window.selection_summary = wrapped()
    window.selection_summary.setStyleSheet("font-weight: 600;")
    window.applied_criteria_button = QPushButton("Inspect applied criteria...")
    window.applied_criteria_button.clicked.connect(lambda: active_criteria.show_details(window))
    _selection_actions(window)
    window.job_progress = QProgressBar()
    window.job_progress.setRange(0, 0)
    window.job_progress.hide()
    layout.addWidget(window.experimental_notice)
    layout.addWidget(window.selection_summary)
    layout.addWidget(window.selection_actions)
    layout.addWidget(window.warnings)
    layout.addWidget(window.job_progress)
    layout.addWidget(splitter, stretch=1)
    _compact_forms(window, container)
    return container
