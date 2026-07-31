"""Shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from smiles2select.chemistry.descriptor_registry import default_registry  # noqa: E402
from smiles2select.profiles.loader import builtin_registry  # noqa: E402

#: Reference molecules with well-known properties.
REFERENCE_SMILES = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "caffeine": "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "benzene": "c1ccccc1",
    "long_alkane": "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
    "salicylic_sodium_salt": "OC(=O)c1ccccc1O.[Na+]",
    "chalcone": "Oc1ccc(cc1)C(=O)C=Cc1ccc(O)cc1",
}


@pytest.fixture(scope="session")
def descriptors():
    return default_registry()


@pytest.fixture(scope="session")
def profiles():
    return builtin_registry()


@pytest.fixture
def library_csv(tmp_path: Path) -> Path:
    """Small library covering valid, invalid, duplicated and salt cases."""
    path = tmp_path / "library.csv"
    pd.DataFrame(
        [
            ("MOL001", REFERENCE_SMILES["aspirin"]),
            ("MOL002", REFERENCE_SMILES["caffeine"]),
            ("MOL003", REFERENCE_SMILES["ibuprofen"]),
            ("MOL004", "not_a_smiles"),
            ("MOL005", REFERENCE_SMILES["aspirin"]),
            ("MOL006", REFERENCE_SMILES["long_alkane"]),
            ("MOL007", ""),
            ("MOL008", REFERENCE_SMILES["salicylic_sodium_salt"]),
            ("MOL009", REFERENCE_SMILES["chalcone"]),
        ],
        columns=["ID", "SMILES"],
    ).to_csv(path, index=False)
    return path


def descriptor_frame(rows: dict[str, list[float]]) -> pd.DataFrame:
    """Synthetic descriptor table indexed like the pipeline's."""
    frame = pd.DataFrame(rows)
    frame.index = pd.RangeIndex(start=1, stop=len(frame) + 1, name="record_id")
    return frame
