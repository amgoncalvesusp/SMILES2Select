"""Export of a Selection Intelligence session.

Adds the decision sheets to the chemistry report, and writes the recipe JSON
next to the workbook. Together they answer, for every molecule in the final
set: what the rules said, what the human decided, and under which settings.

A spreadsheet alone cannot be replayed. The recipe can, which is why it is
always written alongside rather than only embedded as a sheet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from smiles2select.export.excel import export_frame, sanitize_sheet_name
from smiles2select.pipeline.runner import RunResult
from smiles2select.selection_intelligence import recipes
from smiles2select.selection_intelligence.basket import SelectionBasket
from smiles2select.selection_intelligence.constrained_selection import (
    SelectionConstraints,
    SelectionOutcome,
)
from smiles2select.selection_intelligence.pareto_ranking import ParetoResult
from smiles2select.selection_intelligence.selection_explanations import (
    final_alerts,
    manual_decisions,
    selection_reason,
    summary_rows,
)
from smiles2select.selection_intelligence.states import SelectionStatus

FINAL_COLUMNS = (
    "Selection_Status",
    "Selection_Origin",
    "Selection_Reason",
    "Pareto_Rank",
    "Distance_To_Ideal",
    "Robustness_Score",
    "Minimum_Rule_Margin",
    "Borderline_Rule_Count",
    "Cluster_ID",
    "Scaffold_ID",
    "Pinned",
    "Manual_Note",
)


@dataclass(frozen=True)
class SessionArtifacts:
    """Everything a session produced, ready to be written."""

    result: RunResult
    basket: SelectionBasket
    outcome: SelectionOutcome
    constraints: SelectionConstraints
    recipe: recipes.SelectionRecipe
    pareto: ParetoResult | None = None
    margins: pd.DataFrame | None = None
    robustness: pd.DataFrame | None = None
    borderline: pd.DataFrame | None = None
    rescue: pd.DataFrame | None = None
    counterfactuals: pd.DataFrame | None = None
    clusters: pd.Series | None = None
    scaffolds: pd.Series | None = None
    unrepresented_clusters: list[int] | None = None


def final_selected_sheet(artifacts: SessionArtifacts) -> pd.DataFrame:
    """The chemistry columns of the selected molecules, plus the decision ones."""
    frame = export_frame(artifacts.result)
    frame = frame.loc[frame.index.intersection(pd.Index(list(artifacts.outcome.selected_ids)))]
    if frame.empty:
        return frame.reset_index(names="record_id")

    frame = frame.copy()
    states = {state.record_id: state for state in artifacts.basket.states()}
    frame["Selection_Status"] = [
        states[record_id].selection_status.value
        if record_id in states
        else SelectionStatus.FINAL_SELECTED.value
        for record_id in frame.index
    ]
    frame["Selection_Origin"] = [
        states[record_id].origin.value if record_id in states and states[record_id].origin else "-"
        for record_id in frame.index
    ]
    frame["Selection_Reason"] = [
        selection_reason(artifacts.outcome, int(record_id)) for record_id in frame.index
    ]
    frame["Pinned"] = [
        "yes" if record_id in states and states[record_id].pinned else "no"
        for record_id in frame.index
    ]
    frame["Manual_Note"] = [
        states[record_id].note if record_id in states else "" for record_id in frame.index
    ]

    if artifacts.pareto is not None:
        table = artifacts.pareto.table.reindex(frame.index)
        frame["Pareto_Rank"] = table["pareto_rank"]
        frame["Distance_To_Ideal"] = table["distance_to_ideal"].round(4)
    if artifacts.robustness is not None:
        table = artifacts.robustness.reindex(frame.index)
        frame["Robustness_Score"] = table.get("robustness_score")
        frame["Minimum_Rule_Margin"] = table.get("minimum_rule_margin")
        frame["Borderline_Rule_Count"] = table.get("borderline_rule_count")
    if artifacts.clusters is not None:
        frame["Cluster_ID"] = artifacts.clusters.reindex(frame.index)
    if artifacts.scaffolds is not None:
        frame["Scaffold_ID"] = artifacts.scaffolds.reindex(frame.index)

    return frame.reset_index(names="record_id")


def selection_summary_sheet(artifacts: SessionArtifacts) -> pd.DataFrame:
    """Counts, coverage, constraints and the alerts raised before export."""
    counters = artifacts.basket.counters()
    rows: list[dict[str, object]] = [
        {"section": "basket", "item": label, "value": value} for label, value in counters.as_rows()
    ]
    rows += [
        {"section": "selection", "item": label, "value": value}
        for label, value in summary_rows(artifacts.outcome, artifacts.constraints)
    ]
    rows += [
        {"section": "constraints", "item": label, "value": value}
        for label, value in artifacts.constraints.summary_rows()
    ]

    alerts = final_alerts(
        artifacts.outcome,
        artifacts.constraints,
        artifacts.basket,
        artifacts.unrepresented_clusters,
        list(artifacts.borderline["record_id"]) if artifacts.borderline is not None else None,
    )
    rows += [
        {"section": "alerts", "item": f"alert {number}", "value": text}
        for number, text in enumerate(alerts, start=1)
    ]
    return pd.DataFrame(rows)


def export(artifacts: SessionArtifacts, path: str | Path) -> tuple[Path, Path]:
    """Write the workbook and the recipe; returns both paths."""
    workbook_path = Path(path)
    workbook_path.parent.mkdir(parents=True, exist_ok=True)

    sheets: dict[str, pd.DataFrame] = {
        "SELECTION_SUMMARY": selection_summary_sheet(artifacts),
        "FINAL_SELECTED": final_selected_sheet(artifacts),
        "MANUAL_DECISIONS": manual_decisions(artifacts.basket),
        "SELECTION_RECIPE": pd.DataFrame(
            artifacts.recipe.summary_rows(), columns=["campo", "valor"]
        ),
    }
    if artifacts.pareto is not None:
        sheets["PARETO_RESULTS"] = artifacts.pareto.table.reset_index()
    if artifacts.borderline is not None:
        sheets["BORDERLINE"] = artifacts.borderline
    if artifacts.rescue is not None:
        sheets["RESCUE_ANALYSIS"] = artifacts.rescue
    if artifacts.counterfactuals is not None:
        sheets["COUNTERFACTUALS"] = artifacts.counterfactuals

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sanitize_sheet_name(name), index=False)

    recipe_path = workbook_path.with_suffix("").with_suffix(recipes.RECIPE_SUFFIX)
    recipes.save(artifacts.recipe, recipe_path)
    return workbook_path, recipe_path
