"""Excel report.

Two export modes, because the detailed one can produce dozens of sheets:

* detailed - one sheet per broken rule, plus one per failed profile;
* compact  - a single ``RULE_FAILURES`` sheet, one row per molecule x rule.

Every sheet is derived from the run result; nothing is recomputed here.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from smiles2select.alerts import engine as alert_engine
from smiles2select.app_metadata import APP_NAME, APP_VERSION, AUTHORSHIP, DISCLAIMER
from smiles2select.chemical_space.coverage import coverage
from smiles2select.decision.explanations import policy_sentence
from smiles2select.pipeline.runner import RunResult
from smiles2select.rules import explanations
from smiles2select.scores import qed as qed_scores

MAX_SHEET_NAME = 31
_INVALID_SHEET_CHARS = re.compile(r"[\[\]:*?/\\]")

DESCRIPTOR_EXPORT_COLUMNS = {
    "mol_wt": "MW",
    "rdkit_wlogp": "WLOGP",
    "mol_mr": "MR",
    "hbd_lipinski": "HBD",
    "hba_lipinski": "HBA",
    "tpsa": "TPSA",
    "rotatable_bonds": "RotB",
    "ring_count": "RingCount",
    "qed": "QED",
    "sa_score": "SA",
    "np_score": "NP",
    "max_reference_similarity": "Max_Reference_Similarity",
    "reference_novelty": "Reference_Novelty",
}


@dataclass(frozen=True)
class ExportOptions:
    """Which sheets to write."""

    detailed: bool = True
    include_profile_matrix: bool = True
    include_alerts: bool = True


def export(result: RunResult, path: str | Path, options: ExportOptions | None = None) -> Path:
    """Write the full workbook and return its path."""
    settings = options or ExportOptions(detailed=result.config.detailed_export)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        _write(writer, summary_sheet(result), "00_SUMMARY")
        _write(writer, selected_sheet(result), "01_SELECTED_FINAL")
        if result.reserve_ids:
            _write(writer, reserve_sheet(result), "02_RESERVE")
        _write(writer, excluded_sheet(result), "02_EXCLUDED_FINAL")
        if result.reference_duplicates is not None:
            _write(writer, reference_overlap_sheet(result), "05_REFERENCE_OVERLAP")
        if result.reference_similarity is not None:
            _write(writer, reference_similarity_sheet(result), "06_REFERENCE_SIMILARITY")
        if result.zone_allocation is not None:
            _write(writer, result.zone_allocation.memberships, "07_ZONE_MEMBERSHIPS")
            _write(writer, result.zone_allocation.allocation_table, "08_ZONE_ALLOCATION")
        if settings.include_profile_matrix:
            _write(writer, profile_matrix(result), "03_PROFILE_MATRIX")
        for profile in result.profiles:
            _write(writer, failed_profile_sheet(result, profile.id), f"FAIL_{profile.id.upper()}")
        if settings.detailed:
            for code, frame in rule_failure_sheets(result).items():
                _write(writer, frame, code)
        else:
            _write(writer, rule_failures_compact(result), "RULE_FAILURES")
        if settings.include_alerts:
            for sheet_name, frame in alert_sheets(result).items():
                _write(writer, frame, sheet_name)
        if result.preparability is not None and not result.preparability.empty:
            _write(writer, preparability_sheet(result), "PREPARABILITY_FLAGS")
        _write(writer, config_sheet(result), "CONFIG")
    return output


def _write(writer: pd.ExcelWriter, frame: pd.DataFrame, sheet_name: str) -> None:
    frame.to_excel(writer, sheet_name=sanitize_sheet_name(sheet_name), index=False)


def sanitize_sheet_name(name: str) -> str:
    """Excel forbids some characters and caps sheet names at 31 characters."""
    cleaned = _INVALID_SHEET_CHARS.sub("_", name).strip() or "SHEET"
    return cleaned[:MAX_SHEET_NAME]


def summary_sheet(result: RunResult) -> pd.DataFrame:
    """Counts, per-profile approval, alerts, QED distribution and the policy used."""
    blocks: list[pd.DataFrame] = [
        pd.DataFrame(
            [
                {"section": "Totals", "item": "Records processed", "value": result.total_records},
                {
                    "section": "Totals",
                    "item": "Final selected",
                    "value": result.decision.selected_count,
                },
                {
                    "section": "Totals",
                    "item": "Final excluded",
                    "value": result.decision.excluded_count,
                },
                {"section": "Totals", "item": "Invalid", "value": result.invalid_count},
                {"section": "Totals", "item": "Duplicates", "value": result.duplicate_count},
                {"section": "Totals", "item": "Evaluated", "value": result.evaluated_count},
            ]
        )
    ]

    profile_rows = result.profile_summary()
    blocks.append(
        pd.DataFrame(
            {
                "section": "Approval by profile",
                "item": profile_rows["profile"],
                "value": profile_rows["approved"],
                "extra": profile_rows["percentage"].map(lambda pct: f"{pct:.2f}%"),
            }
        )
    )

    coverage_report = coverage(result.descriptors, result.decision.selected_ids())
    blocks.append(
        pd.DataFrame(
            [
                {
                    "section": "Coverage",
                    "item": "Selected molecules",
                    "value": coverage_report.selected_count,
                },
                {
                    "section": "Coverage",
                    "item": "Unique scaffolds",
                    "value": coverage_report.unique_scaffolds,
                },
                {
                    "section": "Coverage",
                    "item": "Selected unique scaffolds",
                    "value": coverage_report.selected_unique_scaffolds,
                },
                {
                    "section": "Coverage",
                    "item": "Novel candidates",
                    "value": coverage_report.novel_candidate_count,
                },
            ]
        )
    )

    failures = result.evaluation.failures
    if not failures.empty:
        counted = failures.groupby("failure_code").size().sort_values(ascending=False)
        blocks.append(
            pd.DataFrame(
                {
                    "section": "Violations by rule",
                    "item": counted.index,
                    "value": counted.to_numpy(),
                }
            )
        )

    if not result.alerts.empty:
        alerts = result.alerts.groupby("catalog_id")["occurrence_count"].sum()
        blocks.append(
            pd.DataFrame(
                {
                    "section": "Structural alerts",
                    "item": alerts.index,
                    "value": alerts.to_numpy(),
                }
            )
        )

    if result.preparability is not None and not result.preparability.empty:
        prep_counts = result.preparability.groupby("flag_id").size().sort_values(ascending=False)
        blocks.append(
            pd.DataFrame(
                {
                    "section": "Preparability flags",
                    "item": prep_counts.index,
                    "value": prep_counts.to_numpy(),
                }
            )
        )

    if "qed" in result.scores.columns:
        histogram = qed_scores.distribution(result.scores["qed"])
        blocks.append(
            pd.DataFrame(
                {
                    "section": "QED distribution",
                    "item": [
                        f"{row.bin_lower:.1f}-{row.bin_upper:.1f}" for row in histogram.itertuples()
                    ],
                    "value": histogram["count"].to_numpy(),
                }
            )
        )

    if result.reference_similarity is not None:
        similarity = result.reference_similarity.annotations
        blocks.append(
            pd.DataFrame(
                [
                    {
                        "section": "Reference comparison",
                        "item": "Reference libraries",
                        "value": len(result.reference_libraries),
                    },
                    {
                        "section": "Reference comparison",
                        "item": "Background libraries",
                        "value": len(result.background_libraries),
                    },
                    {
                        "section": "Reference comparison",
                        "item": "Exact search method",
                        "value": result.reference_similarity.method_description,
                    },
                    {
                        "section": "Reference comparison",
                        "item": "Exact reference duplicates",
                        "value": int(
                            result.reference_duplicates.duplicate_count
                            if result.reference_duplicates is not None
                            else 0
                        ),
                    },
                    {
                        "section": "Reference comparison",
                        "item": "Candidates with a nearest reference",
                        "value": int(similarity["nearest_reference_id"].notna().sum()),
                    },
                ]
            )
        )

    blocks.append(
        pd.DataFrame(
            [
                {"section": "Policy", "item": "Final policy", "value": result.config.policy.id},
                {
                    "section": "Policy",
                    "item": "Description",
                    "value": policy_sentence(result.config.policy),
                },
                {"section": "Warning", "item": "Interpretation", "value": DISCLAIMER},
            ]
        )
    )

    summary = pd.concat(blocks, ignore_index=True)
    ordered = [
        column for column in ("section", "item", "value", "extra") if column in summary.columns
    ]
    return summary[ordered]


def export_frame(result: RunResult, *, record_ids: Sequence[int] | None = None) -> pd.DataFrame:
    """Wide per-record table shared by the selected sheet, the excluded sheet
    and the results screen of the interface."""
    descriptors = result.descriptors
    if record_ids is not None:
        descriptors = descriptors.loc[descriptors.index.intersection(record_ids)]
    decisions = result.decision.decisions
    frame = pd.DataFrame(index=descriptors.index)
    frame["ID"] = descriptors["molecule_id"]
    frame["Original_SMILES"] = descriptors["original_smiles"]
    frame["Standardized_SMILES"] = descriptors["standardized_smiles"]
    frame["Canonical_SMILES"] = descriptors["canonical_smiles"]

    for descriptor_id, label in DESCRIPTOR_EXPORT_COLUMNS.items():
        if descriptor_id in descriptors.columns:
            frame[label] = descriptors[descriptor_id]

    for profile in result.profiles:
        column = f"{profile.id}__passed"
        if column in result.evaluation.status.columns:
            verdict = result.evaluation.status[column].reindex(frame.index)
            frame[f"{profile.label}_Status"] = verdict.map({True: "PASS", False: "FAIL"}).fillna(
                "N/A"
            )

    for catalog_id, label in (("pains", "PAINS_Count"), ("brenk", "Brenk_Count")):
        if catalog_id in result.config.alert_catalogs:
            frame[label] = alert_engine.counts_by_catalog(result.alerts, frame.index, catalog_id)

    if "selection_status" in decisions:
        status = decisions["selection_status"].reindex(frame.index)
        frame["Final_Status"] = status.map(
            {"FINAL_SELECTED": "SELECTED", "RESERVE": "RESERVE", "EXCLUDED": "EXCLUDED"}
        ).fillna("NOT_EVALUATED")
    else:
        status = decisions["selected"].reindex(frame.index)
        frame["Final_Status"] = status.map({True: "SELECTED", False: "EXCLUDED"}).fillna(
            "NOT_EVALUATED"
        )
    frame["Source_File"] = descriptors["source_file"]
    frame["Source_Sheet"] = descriptors["source_sheet"]
    frame["Source_Row"] = descriptors["source_row"]
    return frame


def reference_overlap_sheet(result: RunResult) -> pd.DataFrame:
    """One row per exact candidate/reference overlap."""

    if result.reference_duplicates is None:
        return pd.DataFrame()
    return result.reference_duplicates.overlaps.copy()


def reference_similarity_sheet(result: RunResult) -> pd.DataFrame:
    """Nearest-reference results with the exact search metadata."""

    if result.reference_similarity is None:
        return pd.DataFrame()
    return result.reference_similarity.annotations.reset_index()


def selected_sheet(result: RunResult) -> pd.DataFrame:
    frame = export_frame(result)
    return frame[frame["Final_Status"] == "SELECTED"].reset_index(drop=True)


def reserve_sheet(result: RunResult) -> pd.DataFrame:
    """Reserve molecules are exported separately and never counted as final."""

    frame = export_frame(result)
    if not result.reserve_ids:
        return frame.iloc[0:0].reset_index(drop=True)
    return (
        frame.loc[frame.index.isin(result.reserve_ids)]
        .assign(Final_Status="RESERVE")
        .reset_index(drop=True)
    )


def excluded_sheet(result: RunResult) -> pd.DataFrame:
    """One row per excluded molecule, with the reasons that removed it."""
    frame = export_frame(result)
    excluded = frame[frame["Final_Status"] != "SELECTED"].copy()
    if excluded.empty:
        return excluded.reset_index(drop=True)

    decisions = result.decision.decisions
    failures = result.evaluation.failures
    rules_by_id = {rule.id: rule for profile in result.profiles for rule in profile.rules}

    excluded["Final_Exclusion_Reasons"] = decisions["exclusion_reasons"].reindex(excluded.index)
    excluded["Failed_Profiles"] = explanations.failed_profiles(failures).reindex(excluded.index)
    excluded["Failed_Rules"] = explanations.failed_rule_codes(failures).reindex(excluded.index)
    excluded["Observed_Values"] = explanations.summarise_failures(failures, rules_by_id).reindex(
        excluded.index
    )
    excluded["Alert_Count"] = result.scores["alert_count"].reindex(excluded.index).fillna(0)

    invalid = result.descriptors["invalid_reason"].reindex(excluded.index)
    excluded["Final_Exclusion_Reasons"] = excluded["Final_Exclusion_Reasons"].fillna(invalid)
    duplicate_of = result.descriptors["duplicate_of"].reindex(excluded.index)
    is_duplicate = duplicate_of.notna()
    excluded.loc[is_duplicate, "Final_Exclusion_Reasons"] = "duplicate of record " + duplicate_of[
        is_duplicate
    ].astype(str)
    return excluded.reset_index(drop=True)


def profile_matrix(result: RunResult) -> pd.DataFrame:
    """ID x profile PASS/FAIL matrix, for quick side-by-side comparison."""
    status = result.evaluation.status
    matrix = pd.DataFrame(index=status.index)
    matrix["ID"] = result.descriptors["molecule_id"].reindex(status.index)
    for profile in result.profiles:
        matrix[profile.label] = status[f"{profile.id}__passed"].map({True: "PASS", False: "FAIL"})
    return matrix.reset_index(drop=True)


def failed_profile_sheet(result: RunResult, profile_id: str) -> pd.DataFrame:
    """Molecules that failed one profile, with the rules they broke."""
    status = result.evaluation.status
    column = f"{profile_id}__passed"
    if column not in status.columns:
        return pd.DataFrame(columns=["ID", "SMILES", "Violations", "Failed_Rules"])

    failed_index = status.index[~status[column].astype(bool)]
    failures = result.evaluation.failures
    subset = failures[failures["profile_id"] == profile_id] if not failures.empty else failures

    frame = pd.DataFrame(index=failed_index)
    frame["ID"] = result.descriptors["molecule_id"].reindex(failed_index)
    frame["SMILES"] = result.descriptors["canonical_smiles"].reindex(failed_index)
    frame["Violations"] = status[f"{profile_id}__violations"].reindex(failed_index)
    frame["Failed_Rules"] = explanations.failed_rule_codes(subset).reindex(failed_index)
    return frame.reset_index(drop=True)


def rule_failures_compact(result: RunResult) -> pd.DataFrame:
    """One row per molecule x broken rule (compact export)."""
    failures = result.evaluation.failures
    if failures.empty:
        return pd.DataFrame(columns=["record_id", "ID", "profile_id", "rule_id", "failure_code"])
    frame = failures.copy()
    frame.insert(1, "ID", frame["record_id"].map(result.descriptors["molecule_id"]))
    return frame


def rule_failure_sheets(result: RunResult) -> dict[str, pd.DataFrame]:
    """One sheet per failure code (detailed export)."""
    failures = result.evaluation.failures
    if failures.empty:
        return {}
    identifiers = result.descriptors["molecule_id"]
    smiles = result.descriptors["canonical_smiles"]
    sheets: dict[str, pd.DataFrame] = {}
    for code, group in failures.groupby("failure_code"):
        frame = group.copy()
        frame.insert(1, "ID", frame["record_id"].map(identifiers))
        frame.insert(2, "SMILES", frame["record_id"].map(smiles))
        sheets[str(code)] = frame.reset_index(drop=True)
    return sheets


def alert_sheets(result: RunResult) -> dict[str, pd.DataFrame]:
    """PAINS, Brenk and custom SMARTS sheets."""
    if result.alerts.empty:
        return {}
    identifiers = result.descriptors["molecule_id"]
    sheets: dict[str, pd.DataFrame] = {}
    for catalog_id, group in result.alerts.groupby("catalog_id"):
        frame = group.copy()
        frame.insert(1, "ID", frame["record_id"].map(identifiers))
        sheets[f"{str(catalog_id).upper()}_ALERTS"] = frame.reset_index(drop=True)
    return sheets


def config_sheet(result: RunResult) -> pd.DataFrame:
    """Profiles, versions, thresholds, methods and hashes used in this run."""
    rows: list[dict[str, object]] = [
        {"section": "application", "key": "name", "value": APP_NAME},
        {"section": "application", "key": "version", "value": APP_VERSION},
        {"section": "application", "key": "authorship", "value": AUTHORSHIP},
    ]
    rows.extend(
        {"section": "run", "key": key, "value": value}
        for key, value in result.config.summary_rows()
    )
    for profile in result.profiles:
        section = f"profile:{profile.id}"
        rows.append(
            {"section": section, "key": "version", "value": f"{profile.version} ({profile.name})"}
        )
        rows.append({"section": section, "key": "pass_policy", "value": profile.policy_label()})
        if profile.logp_method:
            rows.append({"section": section, "key": "logp_method", "value": profile.logp_method})
        if profile.atom_count_definition:
            rows.append(
                {"section": section, "key": "atom_count", "value": profile.atom_count_definition}
            )
        if profile.notes:
            rows.append({"section": section, "key": "notes", "value": profile.notes})
        for rule in profile.rules:
            rows.append({"section": section, "key": rule.id, "value": rule.describe()})
    for descriptor_id in result.plan.descriptor_ids:
        rows.append(
            {
                "section": "descriptors",
                "key": descriptor_id,
                "value": "computed once per canonical structure",
            }
        )
    return pd.DataFrame(rows)


def preparability_sheet(result: RunResult) -> pd.DataFrame:
    """Docking preparability flags sheet."""
    if result.preparability is None or result.preparability.empty:
        return pd.DataFrame()
    frame = result.preparability.copy()
    identifiers = result.descriptors["molecule_id"]
    smiles = result.descriptors["canonical_smiles"]
    frame.insert(1, "ID", frame["record_id"].map(identifiers))
    frame.insert(2, "SMILES", frame["record_id"].map(smiles))
    return frame.reset_index(drop=True)
