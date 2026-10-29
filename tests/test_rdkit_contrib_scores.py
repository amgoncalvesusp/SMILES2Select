"""Contrib scores use their installed modules and data, not an external RDKit tree."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest


@pytest.mark.parametrize("score", ("sa_score", "np_score"))
def test_contrib_score_matches_rdkit_with_external_base_unavailable(tmp_path, score):
    root = Path(__file__).resolve().parents[1]
    process = subprocess.run(
        [sys.executable, "-c", dedent("""
            import json
            import sys
            from rdkit import Chem
            from rdkit.Contrib.NP_Score import npscorer
            from rdkit.Contrib.SA_Score import sascorer
            from smiles2select.chemistry import rdkit_descriptors

            molecule = Chem.MolFromSmiles('CCO')
            score = sys.argv[1]
            actual = getattr(rdkit_descriptors, score)(molecule)
            expected = (sascorer.calculateScore(molecule) if score == 'sa_score'
                        else npscorer.scoreMol(molecule, npscorer.readNPModel()))
            print(json.dumps({'actual': actual, 'expected': expected}))
        """), score],
        cwd=tmp_path,
        env={**os.environ, "RDBASE": str(tmp_path / "missing-rdkit"),
             "PYTHONPATH": str(root / "src")},
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    scores = json.loads(process.stdout)
    assert math.isfinite(scores["actual"])
    assert scores["actual"] == pytest.approx(scores["expected"], rel=1e-12, abs=1e-12)
