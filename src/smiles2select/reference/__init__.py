"""Reference-library analysis for the Chemical Space Selection Hub.

Reference compounds describe covered chemistry; they are not silently mixed
with candidates or treated as a second eligibility filter.
"""

from smiles2select.reference.duplicates import (
    ExactDuplicateReport,
    compare_exact_duplicates,
)
from smiles2select.reference.libraries import (
    LibraryRole,
    LibrarySpec,
    ReferenceLibrary,
    prepare_library_frame,
)
from smiles2select.reference.similarity import (
    ReferenceSimilarityResult,
    SimilaritySearchUnavailable,
    compute_reference_similarity,
)

__all__ = [
    "ExactDuplicateReport",
    "LibraryRole",
    "LibrarySpec",
    "ReferenceLibrary",
    "ReferenceSimilarityResult",
    "SimilaritySearchUnavailable",
    "compare_exact_duplicates",
    "compute_reference_similarity",
    "prepare_library_frame",
]
