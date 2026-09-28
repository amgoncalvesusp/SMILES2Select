"""Separate editable criteria from the criteria acknowledged by a final decision."""

from dataclasses import dataclass

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QMessageBox

from smiles2select.gui.workspace.selection_help import OBJECTIVE_HELP
from smiles2select.selection_intelligence.action_log import ActionType
from smiles2select.selection_intelligence.objectives import Direction


@dataclass(frozen=True)
class CriteriaState:
    count: int
    strategy: str
    scaffold_limit: int
    cluster_limit: int
    objectives: tuple[tuple[str, int, float, float], ...]
    minimum_scaffolds: int = 0


def capture(window) -> CriteriaState:
    return CriteriaState(
        window.target_count.value(), window.strategy.currentData(),
        window.per_scaffold.value(), window.per_cluster.value(),
        tuple((editor.field.currentText(), editor.direction.currentIndex(),
               editor.low.value(), editor.high.value()) for editor in window.objective_editors),
        window.min_scaffolds.value(),
    )


def active(window) -> CriteriaState:
    for index in range(len(window.basket.log.applied) - 1, -1, -1):
        if window.basket.log.applied[index].action_type is ActionType.AUTOMATIC_SELECTION:
            return window._criteria_snapshots.get(index, window._initial_criteria)
    return window._initial_criteria


def restore(window) -> None:
    apply(window, active(window))


def apply(window, saved: CriteriaState) -> None:
    """Restore all visible criteria atomically, then refresh their explanations."""
    widgets = [window.target_count, window.strategy, window.per_scaffold, window.per_cluster,
               window.min_scaffolds]
    widgets += [widget for editor in window.objective_editors
                for widget in (editor.field, editor.direction, editor.low, editor.high)]
    blockers = [QSignalBlocker(widget) for widget in widgets]
    window.target_count.setValue(saved.count)
    window.strategy.setCurrentText(saved.strategy)
    window.per_scaffold.setValue(saved.scaffold_limit)
    window.per_cluster.setValue(saved.cluster_limit)
    window.min_scaffolds.setValue(saved.minimum_scaffolds)
    for editor, (field, direction, low, high) in zip(window.objective_editors, saved.objectives):
        editor.field.setCurrentText(field)
        editor.direction.setCurrentIndex(direction)
        editor.low.setValue(low)
        editor.high.setValue(high)
        editor._visibility()
        editor.help_label.setText(OBJECTIVE_HELP.get(field, ""))
    del blockers
    window._update_method_card(saved.strategy)
    window._recompute_pareto()


def _spin_value(widget, value):
    if value is None or not widget.minimum() <= value <= widget.maximum():
        raise ValueError("a requested value is outside the editor's supported range")
    if hasattr(widget, "decimals") and round(value, widget.decimals()) != value:
        raise ValueError("an objective target requires more precision than the editor supports")
    return value


def from_scenario(window, spec) -> CriteriaState:
    """Reject unrepresentable studies before changing their selection or UI state."""
    constraints = spec.constraints
    if not constraints.preserve_pinned:
        raise ValueError("altered pin policies are not editable here")
    qed_only = spec.strategy.value == "qed_only"
    if not qed_only and len(spec.objectives) != len(window.objective_editors):
        raise ValueError(f"the workspace requires {len(window.objective_editors)} objectives")
    objectives = list(capture(window).objectives) if qed_only else []
    for editor, objective in zip(window.objective_editors, spec.objectives):
        direction = editor.direction.findData(objective.direction)
        if (editor.field.findText(objective.field) < 0 or direction < 0
                or objective.weight != 1. or not objective.enabled or objective.transform != "none"):
            raise ValueError("objective fields, weights or enabled states are not editable here")
        low, high = editor.low.value(), editor.high.value()
        if objective.direction is Direction.TARGET_RANGE:
            low = _spin_value(editor.low, objective.target_low)
            high = _spin_value(editor.high, objective.target_high)
        elif objective.direction is Direction.TARGET_VALUE:
            low = _spin_value(editor.low, objective.target_value)
        objectives.append((objective.field, direction, low, high))
    if window.strategy.findData(spec.strategy.value) < 0:
        raise ValueError("the selection strategy is unavailable in this workspace")
    return CriteriaState(
        _spin_value(window.target_count, constraints.target_count), spec.strategy.value,
        _spin_value(window.per_scaffold, constraints.max_per_scaffold or 0),
        _spin_value(window.per_cluster, constraints.max_per_cluster or 0), tuple(objectives),
        _spin_value(window.min_scaffolds, constraints.min_scaffolds or 0),
    )


def update_export_buttons(window) -> None:
    pending = window.has_pending_criteria()
    enabled = bool(window.basket.final_ids()) and not pending and not window.is_busy
    tooltip = ("Create selection to apply the edited criteria before exporting."
               if pending else "Export the current final molecules and their applied criteria.")
    for button in (window.basket_panel.export_button, window.basket_panel.docking_button):
        button.setEnabled(enabled)
        button.setToolTip(tooltip)


def allow_export(window) -> bool:
    """Never silently export an old result in place of the requested selection."""
    if window.is_busy:
        window.warnings.setText("Wait for the active computation before exporting.")
        return False
    count = len(window.basket.final_ids())
    if not count:
        window.warnings.setText("No final molecules to export. Create a selection first.")
        return False
    if window.has_pending_criteria():
        window.warnings.setText(
            f"Export blocked: the edited criteria have not been applied. Requested: "
            f"{window.target_count.value():,}; current final set: {count:,}. "
            "Click Create selection and wait for completion."
        )
        return False
    requested = window.target_count.value()
    if count != requested:
        return QMessageBox.question(
            window, "Final count differs from the request",
            f"Requested: {requested:,} molecules. Current final set: {count:,}.\n"
            "Review quota/pin warnings and manual edits. The export will contain exactly "
            f"{count:,} molecules, with this difference in the audit record.\n\n"
            "Export this current set anyway?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) == QMessageBox.Yes
    return True
