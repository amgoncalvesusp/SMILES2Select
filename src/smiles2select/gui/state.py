"""Mutable state shared by the wizard pages.

The pages edit this object; only :meth:`WizardState.build_config` turns it into
the immutable :class:`RunConfig` the pipeline consumes. Keeping the mutable and
the immutable forms apart means a half-filled screen can never start a run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from smiles2select.alerts.custom_smarts import SmartsAlert
from smiles2select.alerts.policies import DEFAULT_ACTIONS, AlertAction, AlertPolicy
from smiles2select.chemistry.standardization import StandardizationConfig
from smiles2select.decision.policies import DecisionPolicy, ProfileRole
from smiles2select.io.importer import ColumnMapping, SourceFile
from smiles2select.pipeline.config import RunConfig
from smiles2select.pipeline.presets import RunPreset
from smiles2select.scores.qed import QedSelection

DEFAULT_ROLES: dict[str, ProfileRole] = {
    "lipinski": "mandatory",
    "veber": "mandatory",
    "ghose": "informative",
    "egan": "informative",
    "muegge": "informative",
}


@dataclass
class FileSelection:
    """One chosen input file plus how its columns map."""

    path: Path
    sheet: str | None = None
    smiles_column: str = ""
    id_column: str | None = None

    def to_source(self) -> SourceFile:
        return SourceFile(
            path=self.path,
            mapping=ColumnMapping(smiles=self.smiles_column, molecule_id=self.id_column),
            sheet=self.sheet,
        )


@dataclass
class WizardState:
    """Everything the user has chosen so far."""

    files: list[FileSelection] = field(default_factory=list)
    standardization: StandardizationConfig = field(default_factory=StandardizationConfig)
    roles: dict[str, ProfileRole] = field(default_factory=lambda: dict(DEFAULT_ROLES))
    consensus_min_pass: int | None = None
    expression: str | None = None
    qed: QedSelection = field(default_factory=lambda: QedSelection(mode="rank"))
    alert_actions: dict[str, AlertAction] = field(default_factory=lambda: dict(DEFAULT_ACTIONS))
    active_catalogs: list[str] = field(default_factory=lambda: ["pains", "brenk"])
    custom_alerts: list[SmartsAlert] = field(default_factory=list)
    compute_qed: bool = True
    drop_duplicates: bool = True
    detailed_export: bool = True
    n_jobs: int = -1
    chunk_size: int = 2000
    database_path: Path | None = None
    excel_path: Path | None = None

    def selected_profiles(self) -> list[str]:
        return sorted(self.roles)

    def set_role(self, profile_id: str, role: ProfileRole | None) -> None:
        """Assign a role, or remove the profile from the run when ``role`` is None."""
        if role is None:
            self.roles.pop(profile_id, None)
        else:
            self.roles[profile_id] = role

    def build_policy(self) -> DecisionPolicy:
        return DecisionPolicy(
            id="gui",
            roles=dict(self.roles),
            consensus_min_pass=self.consensus_min_pass,
            expression=self.expression or None,
            qed=self.qed,
            alert_policy=AlertPolicy(actions=dict(self.alert_actions)),
        )

    def validation_errors(self) -> list[str]:
        """Problems that must be fixed before the run can start."""
        problems: list[str] = []
        if not self.files:
            problems.append("Nenhum arquivo selecionado.")
        for selection in self.files:
            if not selection.smiles_column:
                problems.append(f"{selection.path.name}: coluna de SMILES não definida.")
        if not self.roles:
            problems.append("Nenhum perfil selecionado.")
        if self.qed.excludes and not self.compute_qed:
            problems.append("A política de QED exclui moléculas, mas o QED está desativado.")
        try:
            self.build_policy()
        except ValueError as exc:
            problems.append(str(exc))
        return problems

    def to_preset(self, name: str = "preset") -> RunPreset:
        """Capture the current choices, without files or output paths."""
        return RunPreset(
            name=name,
            roles=dict(self.roles),
            consensus_min_pass=self.consensus_min_pass,
            expression=self.expression,
            qed=self.qed,
            active_catalogs=tuple(self.active_catalogs),
            alert_actions=dict(self.alert_actions),
            custom_alerts=tuple(self.custom_alerts),
            standardization=self.standardization,
            compute_qed=self.compute_qed,
            drop_duplicates=self.drop_duplicates,
            detailed_export=self.detailed_export,
        )

    def apply_preset(self, preset: RunPreset, available_profiles: tuple[str, ...] = ()) -> None:
        """Adopt a saved preset, leaving the chosen files untouched.

        Validation runs first: a preset naming a profile this installation does
        not have is rejected whole, rather than applied in part.
        """
        if available_profiles:
            problems = preset.validate(available_profiles)
            if problems:
                raise ValueError("\n".join(problems))
        self.roles = dict(preset.roles)
        self.consensus_min_pass = preset.consensus_min_pass
        self.expression = preset.expression
        self.qed = preset.qed
        self.active_catalogs = list(preset.active_catalogs)
        self.alert_actions = dict(preset.alert_actions)
        self.custom_alerts = list(preset.custom_alerts)
        self.standardization = preset.standardization
        self.compute_qed = preset.compute_qed
        self.drop_duplicates = preset.drop_duplicates
        self.detailed_export = preset.detailed_export

    def build_config(self) -> RunConfig:
        problems = self.validation_errors()
        if problems:
            raise ValueError("\n".join(problems))
        return RunConfig(
            sources=tuple(selection.to_source() for selection in self.files),
            profile_ids=tuple(self.selected_profiles()),
            policy=self.build_policy(),
            standardization=self.standardization,
            alert_catalogs=tuple(self.active_catalogs),
            custom_alerts=tuple(self.custom_alerts),
            compute_qed=self.compute_qed,
            drop_duplicates=self.drop_duplicates,
            n_jobs=self.n_jobs,
            chunk_size=self.chunk_size,
            database_path=self.database_path,
            excel_path=self.excel_path,
            detailed_export=self.detailed_export,
        )
