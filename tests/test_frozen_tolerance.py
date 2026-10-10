"""CI smoke: distributed statistical profiles run through the frozen worker."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from s2s_decision.artifacts import file_hash
from s2s_decision.tolerance_profile import read_tolerance_package

MODES = ("general", "hierarchical", "contextual")
STATES = {"base_eligible", "supported_relaxation", "review", "excluded"}


def candidates() -> pd.DataFrame:
    """Distinct unsorted IDs catch lost records and positional reassignment."""
    return pd.DataFrame({
        "record_id": [91, 14, 507, 901],
        "molecule_id": ["ethanol-α", "caffeine-β", "alkane-γ", "invalid-δ"],
        "SMILES": ["CCO", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "C" * 40, "not-a-smiles"],
    })


@pytest.fixture(scope="module")
def frozen_tolerance_bundle():
    bundle_root = os.environ.get("S2S_FROZEN_BUNDLE")
    if not bundle_root:
        pytest.skip("Set S2S_FROZEN_BUNDLE when testing a PyInstaller bundle")
    bundle = Path(bundle_root).resolve()
    executable = bundle / ("SMILES2Select.exe" if sys.platform == "win32" else "SMILES2Select")
    assert executable.is_file()
    directory = bundle / "_internal" / "s2s_decision" / "bundled_tolerance_models"
    assert directory.is_dir(), "Frozen application has no distributed tolerance profiles"
    profiles = {}
    for path in sorted(directory.glob("*.json")):
        profile = json.loads(path.read_text(encoding="utf-8"))
        if (isinstance(profile, dict) and profile.get("schema") == "s2s-tolerance-profile/1"
                and profile.get("recipe") == "A" and profile.get("mode") in MODES):
            profiles.setdefault(profile["mode"], path)
    assert set(profiles) == set(MODES), "Distribute recipe A profiles for all three modes"
    return executable, profiles


def run_worker(executable: Path, tmp_path: Path, command: str, *args: str) -> None:
    completed = subprocess.run(
        [str(executable), "--s2s-worker", "-m", "s2s_decision", command, *args],
        cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=180, check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def assert_prediction_contract(source, records, spreadsheet, report, profile_hash):
    for frame in (records, spreadsheet):
        assert len(frame) == len(source) == 4
        assert frame.record_id.tolist() == source.record_id.tolist()
        assert frame.molecule_id.tolist() == source.molecule_id.tolist()
        assert frame.original_smiles.tolist() == source.SMILES.tolist()
        assert set(frame.tolerance_state).issubset(STATES)
        assert frame.tolerance_profile_sha256.tolist() == [profile_hash] * 4
        assert {"tolerance_evidence", "tolerance_support", "tolerance_applied_context",
                "tolerance_nonconformity", "tolerance_fallback"}.issubset(frame.columns)
    assert records.valid.tolist() == [True, True, True, False]
    invalid = records.loc[records.record_id.eq(901)].iloc[0]
    assert invalid.tolerance_state in {"excluded", "review"}
    assert not bool(invalid.tolerance_eligible)
    assert not bool(invalid.tolerance_supported)
    assert "invalid_structure" in invalid.tolerance_evidence["reasons"]
    assert records.tolerance_evidence.map(lambda item: isinstance(item, dict)).all()
    assert report["rows"] == len(source)
    assert report["state_counts"] == records.tolerance_state.value_counts().to_dict()
    assert "chemical compatibility" in report["meaning"].lower()
    assert "not activity" in report["meaning"].lower()
    assert not {"activity_score", "activity_probability", "predicted_pactivity"} & set(records)


def test_frozen_tolerance_smoke_contract_matches_source_worker(tmp_path: Path) -> None:
    """Exercise smoke assertions locally even when no frozen bundle is configured."""
    from s2s_decision.tolerance_profile import fit_tolerance, write_tolerance_package
    from s2s_decision.tolerance_workflows import predict_tolerance_file

    training = pd.DataFrame({"model_smiles": ["CCN", "CCC"], "pactivity": [7., 7.]})
    profile = fit_tolerance(training, recipe="A", support_settings={
        "min_identities": 1, "min_scaffolds": 1, "min_documents": 0})
    package = write_tolerance_package(profile, tmp_path / "source-profile.json")
    source = candidates()
    input_path = tmp_path / "candidate-identities.csv"
    source.to_csv(input_path, index=False)
    output = tmp_path / "prediction"
    predict_tolerance_file(input_path, output, package)
    records = pd.read_json(output / "records.jsonl", orient="records", lines=True)
    spreadsheet = pd.read_csv(output / "decisions.csv")
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert_prediction_contract(source, records, spreadsheet, report, profile["profile_sha256"])


@pytest.mark.parametrize("mode", MODES)
def test_frozen_bundle_predicts_with_distributed_tolerance_profile(
        tmp_path: Path, frozen_tolerance_bundle, mode: str) -> None:
    executable, profiles = frozen_tolerance_bundle
    package = profiles[mode]
    profile = read_tolerance_package(package)
    package_hash = file_hash(package)
    source = candidates()
    input_path = tmp_path / "candidate-identities.csv"
    source.to_csv(input_path, index=False)
    output = tmp_path / "tolerance-prediction"
    run_worker(executable, tmp_path, "tolerance-predict", "--input", str(input_path),
               "--model", str(package), "--output", str(output))
    records = pd.read_json(output / "records.jsonl", orient="records", lines=True)
    spreadsheet = pd.read_csv(output / "decisions.csv")
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert_prediction_contract(source, records, spreadsheet, report, profile["profile_sha256"])
    inspection_path = tmp_path / "profile-inspection.json"
    run_worker(executable, tmp_path, "tolerance-inspect", "--input", str(package),
               "--output", str(inspection_path))
    inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
    assert inspection["schema"] == profile["schema"]
    assert inspection["mode"] == mode
    assert inspection["recipe"] == "A"
    assert inspection["context"] == profile["context"]
    assert inspection["training_counts"] == profile["training_counts"]
    assert inspection["calibration"] == profile["policy"]
    assert "not activity" in inspection["meaning"].lower()
    assert file_hash(package) == package_hash, "Inference or inspection modified the package"
