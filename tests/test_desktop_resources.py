"""Fixed desktop diagnostics exercise real resources without launching the workspace."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from smiles2select.gui import app, desktop_self_check


def test_real_desktop_resources_render_and_score(qapp):
    report = desktop_self_check.verify_desktop_resources()
    assert report["status"] == "ok"
    assert report["schema"] == "smiles2select-desktop-resources/1"
    assert report["methods"]["section_count"] >= 5
    assert "experimental" in report["methods"]["rendered_snippet"].lower()
    assert report["methods"]["rendered_characters"] > 1000
    assert len(report["contextual_models"]) == 9
    assert len(report["risk_models"]) == 1
    root = Path(__file__).resolve().parents[1] / "src"
    for entry in [report["methods"], *report["contextual_models"], *report["risk_models"]]:
        resource = root / entry["resource"]
        assert entry["sha256"] == hashlib.sha256(resource.read_bytes()).hexdigest()
    for entry in [*report["contextual_models"], *report["risk_models"]]:
        assert entry["record_ids"] == [1, 2, 3]
        assert len(entry["probabilities"]) == 3
        assert np.isfinite(entry["probabilities"]).all()
        assert all(0 <= value <= 1 for value in entry["probabilities"])
    assert report["risk_models"][0]["endpoint"] == "shsy5y_atp_viability_48h"
    package_root = root / "s2s_decision" / "bundled_contextual_models"
    provenance = json.loads((package_root / "info" / "provenance.json").read_text("utf-8"))
    expected_hashes = {item["path"]: item["sha256"] for item in provenance["files"]}
    actual_hashes = {
        item["resource"].split("bundled_contextual_models/", 1)[1]: item["sha256"]
        for item in [*report["contextual_models"], *report["risk_models"]]
    }
    assert actual_hashes == expected_hashes


@pytest.fixture
def qapp(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize("values", [[0.1, 0.2], [0.1, 0.2, np.nan],
                                     [0.1, 0.2, np.inf], [-0.1, 0.2, 0.3],
                                     [0.1, 0.2, 1.1], [[0.1, 0.2, 0.3]], None])
def test_rejects_invalid_inference_results(values):
    with pytest.raises(ValueError, match="three finite probabilities"):
        desktop_self_check._probabilities(values)


def test_probability_boundaries_are_valid():
    assert desktop_self_check._probabilities([0, 0.5, 1]) == [0.0, 0.5, 1.0]


def test_missing_contextual_model_fails_before_success(tmp_path, monkeypatch):
    monkeypatch.setattr(desktop_self_check, "_bundle_root", lambda: tmp_path)
    with pytest.raises(FileNotFoundError):
        desktop_self_check._check_contextual_models(None)


def test_invalid_contextual_model_fails_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(desktop_self_check, "_bundle_root", lambda: tmp_path)
    for name in desktop_self_check.CONTEXTUAL_MODEL_NAMES:
        (tmp_path / name).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        desktop_self_check._check_contextual_models(None)


def test_changed_row_identity_fails(monkeypatch):
    import pandas as pd

    from s2s_decision import contextual_model

    monkeypatch.setattr(contextual_model, "score_frame", lambda *_: pd.DataFrame({
        "record_id": [3, 2, 1], "activity_score": [0.1, 0.2, 0.3],
    }))
    with pytest.raises(ValueError, match="record identity"):
        desktop_self_check._check_contextual_models(None)


def test_missing_methods_does_not_accept_fallback_page(qapp, monkeypatch):
    from smiles2select.gui.pages import methods_page

    def missing():
        raise FileNotFoundError("Methods resource missing")

    monkeypatch.setattr(methods_page, "methods_text", missing)
    with pytest.raises((FileNotFoundError, ValueError)):
        desktop_self_check._check_methods()


def test_unrendered_methods_is_not_a_success(qapp, monkeypatch):
    from smiles2select.gui.pages import methods_page

    monkeypatch.setattr(methods_page, "methods_text", lambda: "# Empty")
    with pytest.raises(ValueError, match="Methods"):
        desktop_self_check._check_methods()


def test_risk_chemistry_mismatch_fails(monkeypatch):
    from s2s_decision import risk_model

    original = risk_model.load_risk_bundle
    monkeypatch.setattr(risk_model, "load_risk_bundle", lambda path: {
        **original(path), "chemistry_version": "wrong",
    })
    with pytest.raises(ValueError, match="chemistry"):
        desktop_self_check._check_risk_models(None, "expected")


@pytest.mark.parametrize("arguments", [[], ["--help"], ["--s2s-worker", "-m", "s2s_decision"]])
def test_non_diagnostic_arguments_are_not_dispatched(arguments):
    assert desktop_self_check.desktop_resources_dispatch(arguments) is None


@pytest.mark.parametrize("arguments", [
    ["--verify-desktop-resources", "-m", "arbitrary"],
    ["--verify-desktop-resources", "--output"],
    ["--verify-desktop-resources", "--output", ""],
    ["--verify-desktop-resources", "--output", "--other"],
    ["--verify-desktop-resources", "--output", "a", "extra"],
    ["prefix", "--verify-desktop-resources"],
])
def test_fixed_dispatch_rejects_other_arguments(arguments, capsys):
    assert desktop_self_check.desktop_resources_dispatch(arguments) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "error"


def test_stdout_report(monkeypatch, capsys):
    monkeypatch.setattr(desktop_self_check, "verify_desktop_resources", lambda: {"status": "ok"})
    assert desktop_self_check.desktop_resources_dispatch(["--verify-desktop-resources"]) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "ok"}


def test_explicit_report_works_for_windowed_executable(tmp_path, monkeypatch):
    output = tmp_path / "relatório Ω.json"
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(desktop_self_check, "verify_desktop_resources", lambda: {"status": "ok"})
    assert desktop_self_check.desktop_resources_dispatch([
        "--verify-desktop-resources", "--output", str(output),
    ]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "ok"


def test_report_refuses_to_overwrite_existing_file(tmp_path, monkeypatch, capsys):
    output = tmp_path / "existing.json"
    output.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(desktop_self_check, "verify_desktop_resources", lambda: pytest.fail("ran"))
    assert desktop_self_check.desktop_resources_dispatch([
        "--verify-desktop-resources", "--output", str(output),
    ]) == 1
    assert output.read_text(encoding="utf-8") == "keep"
    assert json.loads(capsys.readouterr().out)["status"] == "error"


def test_resource_failure_is_reported_with_nonzero_exit(tmp_path, monkeypatch, capsys):
    def broken():
        raise ValueError("Bundled model failed validation")

    monkeypatch.setattr(desktop_self_check, "verify_desktop_resources", broken)
    output = tmp_path / "failed.json"
    assert desktop_self_check.desktop_resources_dispatch([
        "--verify-desktop-resources", "--output", str(output),
    ]) == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "error"
    assert "failed validation" in report["error"]
    assert json.loads(capsys.readouterr().out) == report


@pytest.mark.parametrize("use_file", [False, True])
def test_broken_stdout_requires_a_written_report(tmp_path, monkeypatch, use_file):
    class BrokenOutput:
        def write(self, _):
            raise OSError("closed output")

    monkeypatch.setattr(sys, "stdout", BrokenOutput())
    monkeypatch.setattr(desktop_self_check, "verify_desktop_resources", lambda: {"status": "ok"})
    output = tmp_path / "report.json"
    arguments = ["--verify-desktop-resources"]
    if use_file:
        arguments += ["--output", str(output)]
    assert desktop_self_check.desktop_resources_dispatch(arguments) == (0 if use_file else 1)
    if use_file:
        assert json.loads(output.read_text("utf-8"))["status"] == "ok"


def test_app_dispatches_diagnostic_without_workspace(monkeypatch):
    monkeypatch.setattr(app, "desktop_resources_dispatch", lambda _: 7)
    assert app.main(["--verify-desktop-resources"]) == 7
