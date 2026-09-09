"""Progressive disclosure for a selection workspace on notebook screens."""

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
    QSpinBox,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from smiles2select.selection_intelligence.constrained_selection import Strategy
from smiles2select.selection_intelligence.objectives import Direction, Objective

STRATEGY_HELP = {
    "balanced": "Balance property quality, spread and available evidence of threshold margin.",
    "pareto_first": "Favor molecules with favorable trade-offs between the two objectives.",
    "diversity_first": "Favor spread across objective values; this is not pairwise structural diversity.",
    "scaffold_coverage": "Represent more distinct molecular cores (Murcko scaffolds).",
    "manual_assisted": "Keep justified pinned choices, then fill remaining places by ranking.",
}

STRATEGY_LABELS = {
    "balanced": "Balance properties and representation",
    "pareto_first": "Prioritize favorable property trade-offs",
    "diversity_first": "Spread across property values",
    "scaffold_coverage": "Cover more molecular cores",
    "manual_assisted": "Complete my pinned choices",
}


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
        self.low, self.high = QDoubleSpinBox(), QDoubleSpinBox()
        for widget in (self.low, self.high):
            widget.setRange(-1_000_000, 1_000_000)
            widget.setDecimals(3)
        self.high.setValue(500)
        self.low.setPrefix("Value / lower: ")
        self.high.setPrefix("Upper: ")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for widget in (field, self.direction, self.low, self.high):
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


def build_layout(window, objective_candidates):
    controls = QGroupBox("2. Choose your final library")
    form = QFormLayout(controls)
    form.setRowWrapPolicy(QFormLayout.WrapAllRows)
    form.addRow(wrapped("Set the number of molecules, choose a strategy, then create a selection. "
                        "Compare alternatives before exporting. Click a map point to inspect it."))
    window.target_count = QSpinBox()
    window.target_count.setRange(1, 10_000_000)
    window.target_count.setValue(min(50, max(1, len(window.candidates))))
    window.strategy = StrategyCombo()
    window.strategy_help = wrapped(STRATEGY_HELP["balanced"])
    window.select_button = QPushButton("Create selection")
    window.select_button.setToolTip("Replace the current selection. Undo restores the previous set.")
    window.select_button.clicked.connect(window._auto_select)
    window.compare_button = QPushButton("Compare scenarios A / B...")
    window.compare_button.clicked.connect(window._open_scenarios)
    window.criteria_summary = wrapped()
    for label, widget in (("Number of molecules", window.target_count),
                          ("Selection strategy", window.strategy)):
        form.addRow(label, widget)
    for widget in (window.strategy_help, window.select_button, window.compare_button,
                   window.criteria_summary):
        form.addRow(widget)

    window.advanced_toggle = QToolButton()
    window.advanced_toggle.setText("Advanced criteria and map")
    window.advanced_toggle.setCheckable(True)
    window.advanced_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    window.advanced_toggle.setArrowType(Qt.RightArrow)
    window.advanced_content = QWidget()
    advanced = QFormLayout(window.advanced_content)
    advanced.setRowWrapPolicy(QFormLayout.WrapAllRows)
    advanced.setContentsMargins(0, 0, 0, 0)
    numeric = [name for name in objective_candidates if name in window.candidates.columns]
    window.first_objective, window.second_objective = QComboBox(), QComboBox()
    for widget in (window.first_objective, window.second_objective):
        widget.addItems(numeric)
        widget.setToolTip("QED: drug-likeness estimate; mol_wt: molecular weight; "
                          "WLOGP: lipophilicity; TPSA: polar surface area. These do not predict activity.")
    if len(numeric) > 1:
        window.second_objective.setCurrentIndex(1)
    window.objective_editors = (
        ObjectiveControls(window.first_objective, Direction.MAXIMIZE),
        ObjectiveControls(window.second_objective, Direction.MINIMIZE),
    )
    for i, widget in enumerate(window.objective_editors, 1):
        advanced.addRow(f"Property objective {i}", widget)
    window.per_scaffold, window.per_cluster = QSpinBox(), QSpinBox()
    for widget in (window.per_scaffold, window.per_cluster):
        widget.setRange(0, 10_000_000)
        widget.setSpecialValueText("No limit")
    window.per_scaffold.setToolTip("Maximum molecules sharing a Murcko molecular core.")
    window.per_cluster.setToolTip("Maximum molecules in an available structural cluster.")
    if window.candidates["murcko_scaffold"].isna().any():
        window.per_scaffold.setEnabled(False)
        window.per_scaffold.setToolTip("Unavailable: complete cached molecular cores are required.")
    if window.candidates["cluster_id"].isna().any():
        window.per_cluster.setEnabled(False)
        window.per_cluster.setToolTip("Unavailable: complete clustering is not cached; large exact "
                                     "clustering is intentionally skipped to protect memory.")
    advanced.addRow("Maximum per molecular core (scaffold)", window.per_scaffold)
    advanced.addRow("Maximum per structural cluster", window.per_cluster)
    window.view_selector = QComboBox()
    window.view_selector.addItems(["Chemical space", "Pareto"])
    window.projection_selector = QComboBox()
    window.projection_selector.addItem("Property PCA (lightweight)", "property_pca")
    window.projection_selector.addItem("Structural UMAP (optional, slower)", "structural_umap")
    window.color_selector = QComboBox()
    window.color_selector.addItems(["Selection status", "Pareto rank", "Reference similarity"])
    window.reference_overlay = QCheckBox("Show reference overlay")
    window.reference_overlay.setChecked(bool(window.result.reference_libraries))
    for label, widget in (("View", window.view_selector), ("Projection", window.projection_selector),
                          ("Color by", window.color_selector)):
        advanced.addRow(label, widget)
    advanced.addRow(window.reference_overlay)
    advanced.addRow(wrapped("Map distances are visual guidance, not measured molecular similarity. "
                            "Large maps use a disclosed sample; selection uses the full library."))
    window.map_button = QPushButton("Load / refresh map sample")
    window.map_button.clicked.connect(window._request_map)
    advanced.addRow(window.map_button)
    window.papyrus_button = QPushButton("Attach local Papyrus evidence index...")
    window.papyrus_button.clicked.connect(window._attach_papyrus)
    advanced.addRow(window.papyrus_button)
    window.method_card, window.consequence = QLabel(), wrapped()
    window.method_card.setWordWrap(True)
    advanced.addRow(window.method_card)
    advanced.addRow(window.consequence)
    window.advanced_content.hide()
    window.advanced_toggle.toggled.connect(window.advanced_content.setVisible)
    window.advanced_toggle.toggled.connect(lambda on: window.advanced_toggle.setArrowType(
        Qt.DownArrow if on else Qt.RightArrow))
    form.addRow(window.advanced_toggle)
    form.addRow(window.advanced_content)
    window.performance_notice = wrapped()
    window.job_progress = QProgressBar()
    window.job_progress.setRange(0, 0)
    window.job_progress.hide()
    for widget in (window.performance_notice, window.job_progress, window.warnings):
        form.addRow(widget)
    window.controls_scroll = QScrollArea()
    window.controls_scroll.setWidgetResizable(True)
    window.controls_scroll.setMinimumWidth(255)
    window.controls_scroll.setWidget(controls)
    top = QSplitter(Qt.Horizontal)
    top.addWidget(window.controls_scroll)
    top.addWidget(window.views)
    top.addWidget(window.inspector)
    top.setSizes([300, 610, 330])
    splitter = QSplitter(Qt.Vertical)
    splitter.addWidget(top)
    splitter.addWidget(window.basket_panel)
    splitter.setSizes([570, 200])
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.addWidget(splitter)
    return container
