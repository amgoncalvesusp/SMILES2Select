"""Multi-metric coverage and exploration summaries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from smiles2select.chemistry.fingerprints import mean_pairwise_similarity
from smiles2select.chemistry.scaffolds import scaffolds_from_smiles


@dataclass(frozen=True)
class CoverageReport:
    """Before/after metrics; no single number is presented as diversity."""

    candidate_count: int
    selected_count: int
    unique_scaffolds: int
    selected_unique_scaffolds: int
    scaffold_coverage: float
    mean_pairwise_similarity: float
    selected_mean_pairwise_similarity: float
    reference_similarity_threshold: float | None = None
    novel_candidate_count: int | None = None
    selected_novel_candidate_count: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_count": self.candidate_count,
            "selected_count": self.selected_count,
            "unique_scaffolds": self.unique_scaffolds,
            "selected_unique_scaffolds": self.selected_unique_scaffolds,
            "scaffold_coverage": self.scaffold_coverage,
            "mean_pairwise_similarity": self.mean_pairwise_similarity,
            "selected_mean_pairwise_similarity": self.selected_mean_pairwise_similarity,
            "reference_similarity_threshold": self.reference_similarity_threshold,
            "novel_candidate_count": self.novel_candidate_count,
            "selected_novel_candidate_count": self.selected_novel_candidate_count,
        }


def coverage(
    candidates: pd.DataFrame,
    selected_ids: Sequence[object],
    *,
    smiles_column: str = "canonical_smiles",
    scaffold_column: str = "murcko_scaffold",
    reference_novelty_column: str = "reference_novelty",
    novelty_threshold: float | None = None,
) -> CoverageReport:
    """Calculate complementary coverage metrics before and after selection."""

    if smiles_column not in candidates.columns:
        raise KeyError(f"missing SMILES column '{smiles_column}'")
    frame = candidates.copy()
    if scaffold_column in frame.columns:
        scaffolds = frame[scaffold_column].fillna("").astype(str)
    else:
        scaffolds = pd.Series(
            scaffolds_from_smiles(frame[smiles_column].fillna("").tolist()), index=frame.index
        )
    selected = frame.loc[frame.index.intersection(pd.Index(selected_ids))]
    selected_scaffolds = scaffolds.reindex(selected.index).fillna("")
    unique = int(scaffolds[scaffolds != ""].nunique())
    selected_unique = int(selected_scaffolds[selected_scaffolds != ""].nunique())

    novel_count = selected_novel_count = None
    if novelty_threshold is not None:
        if not 0.0 <= novelty_threshold <= 1.0:
            raise ValueError("novelty_threshold must lie in [0, 1]")
        if reference_novelty_column not in frame.columns:
            raise KeyError(f"missing novelty column '{reference_novelty_column}'")
        novel = frame[reference_novelty_column] >= novelty_threshold
        novel_count = int(novel.fillna(False).sum())
        selected_novel_count = int(novel.reindex(selected.index).fillna(False).sum())

    return CoverageReport(
        candidate_count=len(frame),
        selected_count=len(selected),
        unique_scaffolds=unique,
        selected_unique_scaffolds=selected_unique,
        scaffold_coverage=round(selected_unique / unique, 4) if unique else 0.0,
        mean_pairwise_similarity=mean_pairwise_similarity_from_frame(frame, smiles_column),
        selected_mean_pairwise_similarity=mean_pairwise_similarity_from_frame(
            selected, smiles_column
        ),
        reference_similarity_threshold=novelty_threshold,
        novel_candidate_count=novel_count,
        selected_novel_candidate_count=selected_novel_count,
    )


def mean_pairwise_similarity_from_frame(frame: pd.DataFrame, smiles_column: str) -> float:
    from smiles2select.chemistry.fingerprints import fingerprints_from_smiles

    vectors, _ = fingerprints_from_smiles(frame[smiles_column].fillna("").tolist())
    return mean_pairwise_similarity(vectors)
