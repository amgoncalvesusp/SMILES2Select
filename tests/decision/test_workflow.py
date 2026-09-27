"""Public CLI checks; synthetic labels test software, not scientific efficacy."""

import json

import numpy as np
import pandas as pd
import pytest

from s2s_decision.artifacts import read_bundle, write_bundle
from s2s_decision.cli import main
from s2s_decision.schema import FeatureSet, fingerprint_matrix


def test_prepare_select_cli_preserves_source_and_reports_counts(tmp_path):
    source = tmp_path / "pool.csv"
    pd.DataFrame({"ID": ["NA", "NULL", "001"], "SMILES": ["CCO", "c1ccccc1", "bad"]}).to_csv(
        source, index=False
    )
    folder = tmp_path / "features"
    assert main(["prepare", "--input", str(source), "--output", str(folder)]) == 0
    bundle = read_bundle(folder)
    assert bundle.records.molecule_id.tolist() == ["NA", "NULL", "001"]
    assert fingerprint_matrix(bundle.records[bundle.records.valid]).shape == (2, 2048)
    frame = bundle.records.assign(priority_score=[0.9, 0.2, np.nan])
    scores = tmp_path / "scores"
    write_bundle(FeatureSet(frame, bundle.manifest), scores)
    chosen = tmp_path / "chosen"
    assert main(["select", "--input", str(scores), "--output", str(chosen), "--n", "1"]) == 0
    assert read_bundle(chosen).records.query("is_final").molecule_id.tolist() == ["NA"]
    assert (chosen / "final.csv").exists()
    assert main(["prepare", "--input", str(source), "--output", str(folder)]) == 2


def test_bundle_hash_and_strict_json(tmp_path):
    folder = tmp_path / "bundle"
    write_bundle(FeatureSet(pd.DataFrame({"record_id": [1], "molecule_id": ["001"]}), {}), folder)
    assert read_bundle(folder).records.molecule_id.iloc[0] == "001"
    with pytest.raises(ValueError, match="exists"):
        write_bundle(FeatureSet(pd.DataFrame(), {}), folder)
    (folder / "records.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        read_bundle(folder)


def test_audit_cli_and_help(tmp_path, capsys):
    source = tmp_path / "measured.csv"
    pd.DataFrame(
        {
            "target_id": ["T"] * 3,
            "endpoint": ["Ki"] * 3,
            "pactivity": [5.0, 7.0, np.nan],
            "SMILES": ["CCO", "CCN", "CCC"],
        }
    ).to_csv(source, index=False)
    output = tmp_path / "audit.json"
    assert main(["audit", "--input", str(source), "--output", str(output)]) == 0
    audit = json.loads(output.read_text())
    assert audit["unknown_labels"] == 1
    assert (
        main(["prepare", "--input", str(tmp_path / "missing.csv"), "--output", str(tmp_path / "x")])
        == 2
    )
    assert "ERROR" in capsys.readouterr().err


def test_selection_csv_neutralizes_formulas_without_mutating_raw_ids(tmp_path):
    frame = pd.DataFrame(
        {
            "record_id": [1],
            "molecule_id": ["=1+1"],
            "original_smiles": ["CCO"],
            "valid": [True],
            "eligible": [True],
            "priority_score": [0.9],
            "identity": ["test-only"],
            "murcko_scaffold": [""],
            "=column": ["@SUM(1)"],
        }
    )
    source, target = tmp_path / "input", tmp_path / "output"
    write_bundle(FeatureSet(frame, {}), source)
    assert main(["select", "--input", str(source), "--output", str(target), "--n", "1"]) == 0
    assert read_bundle(target).records.molecule_id.iloc[0] == "=1+1"
    table = pd.read_csv(target / "final.csv")
    assert table.molecule_id.iloc[0] == "'=1+1"
    assert "'=column" in table
