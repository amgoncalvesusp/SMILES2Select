import json

import pandas as pd

from s2s_decision.cli import main, parser


def test_baseline_cli_exposes_explicit_estimator_and_features():
    args = parser().parse_args(
        [
            "train-baseline",
            "--input",
            "dataset",
            "--output",
            "new-model",
            "--estimator",
            "gradient_boosting",
            "--input-layout",
            "scalar_fingerprint",
            "--seed",
            "23",
            "--threads",
            "2",
        ]
    )
    assert args.estimator == "gradient_boosting"
    assert args.input_layout == "scalar_fingerprint"
    assert args.seed == 23


def test_chemical_preview_does_not_export_until_adoption(tmp_path):
    source = tmp_path / "library.csv"
    pd.DataFrame({"ID": ["001", "NA", "NULL"], "SMILES": ["CCO", "c1ccccc1", "bad"]}).to_csv(
        source, index=False
    )
    preview, adopted = tmp_path / "preview", tmp_path / "adopted"
    assert main(["preview", "--input", str(source), "--output", str(preview), "--n", "1"]) == 0
    assert (preview / "comparison.json").is_file()
    assert not (preview / "final.csv").exists()
    assert main(["adopt", "--input", str(preview), "--output", str(adopted)]) == 0
    assert (adopted / "final.csv").exists()
    assert main(["adopt", "--input", str(preview), "--output", str(adopted)]) == 2


def test_discovery_empty_directory_explains_no_available_model(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    output = tmp_path / "models.json"
    assert main(["models", "--input", str(root), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["models"] == []
