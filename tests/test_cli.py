"""Command-line interface."""

from __future__ import annotations

import pandas as pd
import pytest

from smiles2select.cli import build_parser, main
from tests.conftest import REFERENCE_SMILES

pytestmark = pytest.mark.integration


@pytest.fixture
def library(tmp_path):
    path = tmp_path / "library.csv"
    pd.DataFrame(
        [
            ("MOL001", REFERENCE_SMILES["aspirin"]),
            ("MOL002", REFERENCE_SMILES["long_alkane"]),
            ("MOL003", REFERENCE_SMILES["ibuprofen"]),
        ],
        columns=["ID", "SMILES"],
    ).to_csv(path, index=False)
    return path


def test_list_profiles_exits_cleanly(capsys):
    assert main(["--list-profiles"]) == 0
    output = capsys.readouterr().out
    assert "lipinski" in output
    assert "SwissADME" in output


def test_run_writes_database_and_excel(library, tmp_path, capsys):
    database = tmp_path / "run.sqlite"
    workbook = tmp_path / "run.xlsx"
    exit_code = main(
        [
            str(library),
            "--smiles-column",
            "SMILES",
            "--id-column",
            "ID",
            "--profiles",
            "lipinski,veber",
            "--mandatory",
            "lipinski,veber",
            "--database",
            str(database),
            "--excel",
            str(workbook),
            "--jobs",
            "1",
        ]
    )
    assert exit_code == 0
    assert database.exists()
    assert workbook.exists()

    output = capsys.readouterr().out
    assert "Records processed:   3" in output
    assert "Final selected:      2" in output


def test_consensus_option(library, capsys):
    exit_code = main(
        [
            str(library),
            "--smiles-column",
            "SMILES",
            "--profiles",
            "lipinski,veber,ghose",
            "--mandatory",
            "",
            "--consensus",
            "2:lipinski,veber,ghose",
            "--jobs",
            "1",
            "--quiet",
        ]
    )
    assert exit_code == 0
    assert capsys.readouterr().out.strip().isdigit()


def test_mandatory_profile_outside_selection_is_rejected(library, capsys):
    exit_code = main(
        [
            str(library),
            "--smiles-column",
            "SMILES",
            "--profiles",
            "lipinski",
            "--mandatory",
            "veber",
        ]
    )
    assert exit_code == 2
    assert "not in --profiles" in capsys.readouterr().err


def test_invalid_custom_smarts_is_rejected(library, capsys):
    exit_code = main(
        [
            str(library),
            "--smiles-column",
            "SMILES",
            "--profiles",
            "lipinski",
            "--mandatory",
            "lipinski",
            "--custom-smarts",
            "Broken=C1CC",
        ]
    )
    assert exit_code == 2
    assert "invalid SMARTS" in capsys.readouterr().err


def test_alert_action_can_be_set_to_exclude(library):
    exit_code = main(
        [
            str(library),
            "--smiles-column",
            "SMILES",
            "--profiles",
            "lipinski",
            "--mandatory",
            "lipinski",
            "--alerts",
            "brenk",
            "--alert-action",
            "brenk=exclude",
            "--jobs",
            "1",
            "--quiet",
        ]
    )
    assert exit_code == 0


def test_no_qed_skips_the_score(library, tmp_path):
    workbook = tmp_path / "no_qed.xlsx"
    exit_code = main(
        [
            str(library),
            "--smiles-column",
            "SMILES",
            "--profiles",
            "lipinski",
            "--mandatory",
            "lipinski",
            "--no-qed",
            "--excel",
            str(workbook),
            "--jobs",
            "1",
            "--quiet",
        ]
    )
    assert exit_code == 0
    assert workbook.exists()


def test_failure_returns_nonzero(tmp_path, capsys):
    missing = tmp_path / "does_not_exist.csv"
    exit_code = main(
        [
            str(missing),
            "--smiles-column",
            "SMILES",
            "--profiles",
            "lipinski",
            "--mandatory",
            "lipinski",
        ]
    )
    assert exit_code == 1
    assert "run failed" in capsys.readouterr().err


def test_parser_defaults_match_the_recommended_configuration():
    args = build_parser().parse_args(["library.csv"])
    assert args.profiles == "lipinski,veber,ghose,egan,muegge"
    assert args.mandatory == "lipinski,veber"
    assert args.alerts == "pains,brenk"
    assert args.qed_mode == "rank"


def test_cli_saves_and_replays_a_selection_plan(library, tmp_path, capsys):
    plan = tmp_path / "selection.selection.json"
    common = [
        str(library),
        "--smiles-column",
        "SMILES",
        "--id-column",
        "ID",
        "--profiles",
        "lipinski",
        "--mandatory",
        "lipinski",
        "--selection-strategy",
        "diversity_first",
        "--final-count",
        "1",
        "--reserve-count",
        "1",
        "--jobs",
        "1",
        "--quiet",
    ]
    assert main([*common, "--save-selection-plan", str(plan)]) == 0
    assert plan.exists()
    capsys.readouterr()
    assert main([str(library), "--smiles-column", "SMILES", "--id-column", "ID",
                 "--selection-plan", str(plan), "--jobs", "1", "--quiet"]) == 0
