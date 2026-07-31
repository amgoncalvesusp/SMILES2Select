"""Saved run presets: round trip, validation and CLI/GUI integration."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from smiles2select.alerts.custom_smarts import SmartsAlert
from smiles2select.chemistry.standardization import StandardizationConfig
from smiles2select.cli import main
from smiles2select.pipeline import presets
from smiles2select.profiles.loader import builtin_registry
from smiles2select.scores.qed import QedSelection
from tests.conftest import REFERENCE_SMILES

pytestmark = pytest.mark.unit

AVAILABLE = builtin_registry().ids()


def sample_preset() -> presets.RunPreset:
    return presets.RunPreset(
        name="grupo",
        roles={"lipinski": "mandatory", "veber": "mandatory", "ghose": "informative"},
        qed=QedSelection(mode="threshold", threshold=0.4),
        active_catalogs=("brenk",),
        alert_actions={"brenk": "exclude"},
        custom_alerts=(SmartsAlert(id="nitro", name="Nitro", smarts="[N+](=O)[O-]"),),
        standardization=StandardizationConfig(neutralize=True),
        detailed_export=False,
    )


def test_preset_round_trips_through_json(tmp_path):
    path = presets.save(sample_preset(), tmp_path / "grupo.json")
    assert presets.load(path) == sample_preset()


def test_preset_stores_no_files_or_output_paths(tmp_path):
    payload = json.loads(presets.save(sample_preset(), tmp_path / "p.json").read_text("utf-8"))
    assert "sources" not in payload
    assert "excel_path" not in payload
    assert "database_path" not in payload


def test_preset_builds_the_expected_policy():
    policy = sample_preset().build_policy()
    assert policy.mandatory_profiles() == ("lipinski", "veber")
    assert policy.informative_profiles() == ("ghose",)
    assert policy.alert_policy.excluding_catalogs() == ("brenk",)
    assert policy.qed.threshold == 0.4


def test_preset_naming_an_unknown_profile_is_rejected():
    preset = presets.RunPreset(roles={"not_a_profile": "mandatory"})
    assert any("desconhecidos" in problem for problem in preset.validate(AVAILABLE))


def test_empty_preset_is_rejected():
    assert any("nenhum perfil" in problem for problem in presets.RunPreset().validate(AVAILABLE))


def test_preset_with_qed_exclusion_but_no_qed_is_rejected():
    preset = presets.RunPreset(
        roles={"lipinski": "mandatory"},
        qed=QedSelection(mode="threshold", threshold=0.5),
        compute_qed=False,
    )
    assert any("QED" in problem for problem in preset.validate(AVAILABLE))


def test_unsupported_format_is_rejected():
    with pytest.raises(presets.PresetError, match="format"):
        presets.from_dict({"format": "9.9", "roles": {}})


def test_broken_smarts_fails_on_load():
    with pytest.raises(presets.PresetError, match="custom alert"):
        presets.from_dict({"custom_alerts": [{"id": "bad", "smarts": "C1CC"}]})


def test_unknown_standardization_option_is_rejected():
    with pytest.raises(presets.PresetError, match="standardization"):
        presets.from_dict({"standardization": {"boil_it": True}})


def test_invalid_json_is_reported(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(presets.PresetError, match="invalid JSON"):
        presets.load(path)


def test_wizard_state_round_trips_a_preset():
    pytest.importorskip("PySide6")
    from smiles2select.gui.state import WizardState

    state = WizardState()
    state.apply_preset(sample_preset(), AVAILABLE)
    assert state.roles["ghose"] == "informative"
    assert state.active_catalogs == ["brenk"]
    assert state.standardization.neutralize is True
    assert state.detailed_export is False
    assert state.to_preset(name="grupo") == sample_preset()


def test_applying_an_invalid_preset_leaves_the_state_untouched():
    pytest.importorskip("PySide6")
    from smiles2select.gui.state import WizardState

    state = WizardState()
    before = dict(state.roles)
    with pytest.raises(ValueError, match="desconhecidos"):
        state.apply_preset(presets.RunPreset(roles={"nope": "mandatory"}), AVAILABLE)
    assert state.roles == before


@pytest.mark.integration
def test_cli_saves_and_reloads_a_preset(tmp_path, capsys):
    library = tmp_path / "library.csv"
    pd.DataFrame(
        [
            ("MOL001", REFERENCE_SMILES["aspirin"]),
            ("MOL002", REFERENCE_SMILES["long_alkane"]),
        ],
        columns=["ID", "SMILES"],
    ).to_csv(library, index=False)

    preset_path = tmp_path / "meu.json"
    exit_code = main(
        [
            str(library),
            "--smiles-column",
            "SMILES",
            "--profiles",
            "lipinski,veber",
            "--mandatory",
            "lipinski,veber",
            "--save-preset",
            str(preset_path),
            "--jobs",
            "1",
            "--quiet",
        ]
    )
    assert exit_code == 0
    assert preset_path.exists()
    assert presets.load(preset_path).roles == {"lipinski": "mandatory", "veber": "mandatory"}

    capsys.readouterr()
    reloaded = main(
        [str(library), "--smiles-column", "SMILES", "--preset", str(preset_path), "--jobs", "1"]
    )
    assert reloaded == 0
    assert "preset:meu" in capsys.readouterr().out


@pytest.mark.integration
def test_cli_rejects_a_preset_with_unknown_profiles(tmp_path, capsys):
    library = tmp_path / "library.csv"
    pd.DataFrame([("MOL001", "CCO")], columns=["ID", "SMILES"]).to_csv(library, index=False)
    preset_path = tmp_path / "bad.json"
    presets.save(presets.RunPreset(roles={"ghost": "mandatory"}), preset_path)

    assert main([str(library), "--smiles-column", "SMILES", "--preset", str(preset_path)]) == 2
    assert "desconhecidos" in capsys.readouterr().err
