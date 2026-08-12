"""Run configuration.

One immutable object describes an entire run: inputs, profiles, policy,
standardization, alerts and parallelism. It is hashed into the CONFIG sheet and
the cache key, so results can always be traced back to the settings that
produced them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from smiles2select.alerts.custom_smarts import SmartsAlert
from smiles2select.app_metadata import APP_VERSION, config_hash, rdkit_version
from smiles2select.chemistry.fingerprints import FingerprintConfig
from smiles2select.chemistry.standardization import StandardizationConfig
from smiles2select.decision.policies import DecisionPolicy, recommended_policy
from smiles2select.io.importer import SourceFile
from smiles2select.pipeline.diagnostics import ParallelDiagnosticsConfig


@dataclass(frozen=True)
class RunConfig:
    """Everything needed to reproduce a run."""

    sources: tuple[SourceFile, ...]
    profile_ids: tuple[str, ...]
    #: Omitted, the recommended policy is derived from the selected profiles.
    policy: DecisionPolicy | None = None
    standardization: StandardizationConfig = field(default_factory=StandardizationConfig)
    alert_catalogs: tuple[str, ...] = ("pains", "brenk")
    custom_alerts: tuple[SmartsAlert, ...] = ()
    compute_qed: bool = True
    #: Synthetic accessibility, reported and rankable; never a pass/fail rule.
    compute_sa: bool = False
    #: Natural-product likeness, likewise a ranking score only.
    compute_np: bool = False
    #: Post-selection diversity picking. These act on the selected set, after
    #: every rule has been applied - they narrow a selection, never widen it.
    diversity_pick: int | None = None
    per_scaffold_limit: int | None = None
    # Not named `fingerprint`: that name is already the configuration hash
    # method below, and a field would silently shadow it.
    fingerprint_config: FingerprintConfig = field(default_factory=FingerprintConfig)
    drop_duplicates: bool = True
    #: Process only the first N records. Used by the sample impact analysis;
    #: None means the whole library.
    max_records: int | None = None
    #: -1 or 0 means "decide automatically from available memory and CPU
    #: count" (see pipeline.resource_estimation); it is never passed to
    #: joblib as a literal -1, which would ignore memory entirely.
    n_jobs: int = -1
    chunk_size: int = 2000
    database_path: Path | None = None
    excel_path: Path | None = None
    parquet_path: Path | None = None
    #: Persistent descriptor cache reused across runs; None disables caching.
    cache_path: Path | None = None
    detailed_export: bool = True
    profile_directories: tuple[Path, ...] = ()
    diagnostics: ParallelDiagnosticsConfig = field(default_factory=ParallelDiagnosticsConfig)
    #: Per-chunk checkpoint for resuming a crashed run. None derives a path
    #: next to database_path (or a temp file) and discards it on success;
    #: set explicitly to keep it around for a later --resume.
    checkpoint_path: Path | None = None
    #: Optional libraries used to characterize candidate/reference overlap.
    #: They are never selected as candidates.
    reference_sources: tuple[SourceFile, ...] = ()
    background_sources: tuple[SourceFile, ...] = ()
    reference_search: str = "exact"
    exclude_reference_duplicates: bool = False
    reference_min_similarity: float = 0.0
    reference_max_similarity: float = 1.0
    #: Optional Hub strategy layer applied after chemical eligibility.
    selection_strategy: str = "traditional"
    final_count: int | None = None
    reserve_count: int | None = None
    selection_seed: int = 0xF00D
    #: Optional overlapping chemical-space zones with explicit quotas.
    zones: tuple[dict[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not self.profile_ids:
            raise ValueError("at least one profile must be selected")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if self.reference_search not in {"exact", "fast"}:
            raise ValueError("reference_search must be 'exact' or 'fast'")
        if not 0.0 <= self.reference_min_similarity <= self.reference_max_similarity <= 1.0:
            raise ValueError("reference similarity window must satisfy 0 <= minimum <= maximum <= 1")
        if self.selection_strategy not in {
            "traditional",
            "balanced",
            "diversity_first",
            "reference_novelty",
            "reference_neighborhood",
            "reference_aware_diversity",
            "stratified",
            "manual_assisted",
        }:
            raise ValueError(f"unknown selection_strategy '{self.selection_strategy}'")
        if self.final_count is not None and self.final_count < 0:
            raise ValueError("final_count must not be negative")
        if self.reserve_count is not None and self.reserve_count < 0:
            raise ValueError("reserve_count must not be negative")
        if self.policy is None:
            object.__setattr__(self, "policy", recommended_policy(self.profile_ids))
        missing_roles = [
            profile_id
            for profile_id in self.policy.evaluated_profiles()
            if profile_id not in self.profile_ids
        ]
        if missing_roles:
            raise ValueError(
                f"the policy references profiles that are not selected: {missing_roles}"
            )

    def score_ids(self) -> tuple[str, ...]:
        scores = []
        if self.compute_qed:
            scores.append("qed")
        if self.compute_sa:
            scores.append("sa_score")
        if self.compute_np:
            scores.append("np_score")
        return tuple(scores)

    @property
    def needs_scaffolds(self) -> bool:
        """True when a scaffold column has to be computed for this run."""
        return self.per_scaffold_limit is not None or self.diversity_pick is not None

    def fingerprint(self) -> str:
        """Hash covering settings and toolkit version, used to invalidate caches."""
        return config_hash(
            {
                "app_version": APP_VERSION,
                "rdkit": rdkit_version(),
                "profiles": sorted(self.profile_ids),
                "policy": self.policy.id,
                "roles": dict(self.policy.roles),
                "consensus_min_pass": self.policy.consensus_min_pass,
                "expression": self.policy.expression,
                "qed_mode": self.policy.qed.mode,
                "standardization": self.standardization.fingerprint(),
                "alerts": sorted(self.alert_catalogs),
                "custom_alerts": sorted(alert.smarts for alert in self.custom_alerts),
                "compute_qed": self.compute_qed,
                "drop_duplicates": self.drop_duplicates,
                "reference_sources": [source.label for source in self.reference_sources],
                "background_sources": [source.label for source in self.background_sources],
                "reference_search": self.reference_search,
                "exclude_reference_duplicates": self.exclude_reference_duplicates,
                "reference_min_similarity": self.reference_min_similarity,
                "reference_max_similarity": self.reference_max_similarity,
                "selection_strategy": self.selection_strategy,
                "final_count": self.final_count,
                "reserve_count": self.reserve_count,
                "selection_seed": self.selection_seed,
                "zones": [dict(zone) for zone in self.zones],
            }
        )

    def summary_rows(self) -> list[tuple[str, str]]:
        """Key/value rows for the CONFIG sheet."""
        return [
            ("app_version", APP_VERSION),
            ("rdkit_version", rdkit_version()),
            ("profiles", ", ".join(self.profile_ids)),
            ("decision_policy", self.policy.id),
            ("mandatory_profiles", ", ".join(self.policy.mandatory_profiles()) or "-"),
            ("informative_profiles", ", ".join(self.policy.informative_profiles()) or "-"),
            ("consensus_min_pass", str(self.policy.consensus_min_pass or "-")),
            ("expression", self.policy.expression or "-"),
            ("qed_mode", self.policy.qed.describe()),
            ("alert_catalogs", ", ".join(self.alert_catalogs) or "-"),
            ("custom_alerts", str(len(self.custom_alerts))),
            ("standardization", str(self.standardization)),
            ("drop_duplicates", str(self.drop_duplicates)),
            ("sa_score", "computed" if self.compute_sa else "disabled"),
            ("np_score", "computed" if self.compute_np else "disabled"),
            ("diversity_pick", str(self.diversity_pick or "-")),
            ("per_scaffold_limit", str(self.per_scaffold_limit or "-")),
            ("fingerprint", self.fingerprint_config.label()),
            ("cache", str(self.cache_path) if self.cache_path else "disabled"),
            ("n_jobs", str(self.n_jobs) if self.n_jobs and self.n_jobs > 0 else "automatic"),
            ("chunk_size_inicial", str(self.chunk_size)),
            ("diagnostics_mode", "safe" if self.diagnostics.enabled else "standard"),
            ("config_hash", self.fingerprint()),
            ("reference_libraries", ", ".join(source.label for source in self.reference_sources) or "-"),
            ("background_libraries", ", ".join(source.label for source in self.background_sources) or "-"),
            ("reference_search", self.reference_search),
            ("exclude_reference_duplicates", str(self.exclude_reference_duplicates)),
            ("reference_similarity_window", f"{self.reference_min_similarity:g}-{self.reference_max_similarity:g}"),
            ("selection_strategy", self.selection_strategy),
            ("final_count", str(self.final_count or "-")),
            ("reserve_count", str(self.reserve_count or "-")),
            ("selection_seed", str(self.selection_seed)),
            ("zones", str(len(self.zones))),
        ]


def default_config(
    sources: Sequence[SourceFile],
    profile_ids: Sequence[str] = ("lipinski", "veber", "ghose", "egan", "muegge"),
) -> RunConfig:
    """The recommended starting configuration described in the specification."""
    return RunConfig(
        sources=tuple(sources),
        profile_ids=tuple(profile_ids),
        policy=recommended_policy(profile_ids),
    )
