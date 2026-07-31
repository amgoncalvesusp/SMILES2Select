"""Natural-product likeness."""

from __future__ import annotations

import pandas as pd
import pytest
from rdkit import Chem

from smiles2select.chemistry.descriptor_planner import DescriptorPlanner
from smiles2select.export.excel import export_frame
from smiles2select.io.importer import ColumnMapping, SourceFile
from smiles2select.pipeline.config import RunConfig
from smiles2select.pipeline.runner import run
from tests.conftest import REFERENCE_SMILES

pytestmark = pytest.mark.unit

# Morphine: a natural product. Ibuprofen: a synthetic drug.
MORPHINE = "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5"


def test_np_score_is_registered_as_a_ranking_score(descriptors):
    definition = descriptors.get("np_score")
    assert "neither a rule nor a classification" in definition.compatibility
    assert "npscorer" in definition.function


def test_natural_product_scores_above_a_synthetic_drug(descriptors):
    natural = descriptors.compute(Chem.MolFromSmiles(MORPHINE), ("np_score",))["np_score"]
    synthetic = descriptors.compute(
        Chem.MolFromSmiles(REFERENCE_SMILES["ibuprofen"]), ("np_score",)
    )["np_score"]
    assert natural > synthetic
    assert -5.0 <= synthetic <= 5.0
    assert -5.0 <= natural <= 5.0


def test_np_score_is_only_computed_when_requested(descriptors, profiles):
    planner = DescriptorPlanner(descriptors)
    without = planner.resolve(profiles=[profiles.get("lipinski")], scores=["qed"])
    with_np = planner.resolve(profiles=[profiles.get("lipinski")], scores=["qed", "np_score"])
    assert "np_score" not in without.descriptor_ids
    assert "np_score" in with_np.descriptor_ids


def test_config_lists_np_among_the_scores():
    config = RunConfig(
        sources=(SourceFile(path=__file__, mapping=ColumnMapping(smiles="SMILES")),),
        profile_ids=("lipinski",),
        compute_np=True,
    )
    assert "np_score" in config.score_ids()
    assert ("np_score", "calculado") in config.summary_rows()


@pytest.mark.integration
def test_np_score_reaches_the_report(tmp_path):
    csv = tmp_path / "library.csv"
    pd.DataFrame(
        [("MOL001", REFERENCE_SMILES["ibuprofen"]), ("MOL002", MORPHINE)],
        columns=["ID", "SMILES"],
    ).to_csv(csv, index=False)

    result = run(
        RunConfig(
            sources=(
                SourceFile(path=csv, mapping=ColumnMapping(smiles="SMILES", molecule_id="ID")),
            ),
            profile_ids=("lipinski",),
            alert_catalogs=(),
            compute_np=True,
            compute_sa=True,
            n_jobs=1,
            chunk_size=2,
        )
    )
    assert result.descriptors["np_score"].notna().all()

    frame = export_frame(result)
    assert "NP" in frame.columns
    assert "SA" in frame.columns
