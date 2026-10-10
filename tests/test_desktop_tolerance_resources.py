"""Real workspace acceptance for every distributed experimental tolerance profile."""

from pathlib import Path

import pytest

from smiles2select.gui import desktop_self_check
from tests import test_workspace

qapp = test_workspace.qapp
result = test_workspace.result

PROFILE_MODES = {
    "Contextual_S3_A_experimental.json": "contextual",
    "General_S1_A_experimental.json": "general",
    "Hierarchical_S2_A_experimental.json": "hierarchical",
}


def test_real_workspace_discovers_and_loads_included_tolerance_profiles(qapp, result):
    report = desktop_self_check._check_tolerance_gui(result)
    gui = report["tolerance_gui"]
    assert gui["mounted"] and gui["rendered"]
    assert gui["workflow_choices"] == ["General", "Target-conditioned", "Custom"]
    assert gui["experimental"] and gui["basket_unchanged"]
    assert gui["activity_model_count"] == 12
    assert gui["tiny_model_count"] == 3
    assert gui["contextual_model_count"] == 9
    assert gui["criteria_count"] > 0
    assert gui["about_checked"] and gui["about_action"] == "About SMILES2Select"
    assert all(value in gui["authorship"] for value in (
        "Adriano", "UNIARA", "Daniel Andres Grajales Ruiz", "IQ-UNESP",
    ))
    models = report["tolerance_models"]
    assert {Path(item["resource"]).name: item["mode"] for item in models} == PROFILE_MODES
    for item in models:
        assert len(item["sha256"]) == 64
        assert item["recipe"] == "A"
        assert item["policy_status"] == "infeasible"
        assert item["automatic_rescue_enabled"] is False
        assert item["training_counts"]["identities"] > 0
        assert item["preview_enabled"] and not item["adopt_enabled"]


@pytest.mark.parametrize("corrupt", [False, True])
def test_missing_or_corrupt_included_tolerance_profiles_fail(qapp, result, tmp_path, monkeypatch, corrupt):
    from smiles2select.gui.workspace import tolerance_controls

    directory = tmp_path / "bundled_tolerance_models"
    directory.mkdir()
    if corrupt:
        for name in PROFILE_MODES:
            (directory / name).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(tolerance_controls, "files", lambda _name: tmp_path)
    message = "failed to load" if corrupt else "included tolerance profiles"
    with pytest.raises(ValueError, match=message):
        desktop_self_check._check_tolerance_gui(result)
