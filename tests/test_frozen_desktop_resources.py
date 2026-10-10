"""Run the real frozen desktop's fixed resource check, including Qt rendering."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_frozen_desktop_resources(tmp_path):
    configured = os.environ.get("S2S_FROZEN_BUNDLE")
    if not configured:
        pytest.skip("Set S2S_FROZEN_BUNDLE when testing a PyInstaller bundle")
    bundle = Path(configured).resolve()
    executable = bundle / ("SMILES2Select.exe" if sys.platform == "win32" else "SMILES2Select")
    assert executable.is_file()
    report_path = tmp_path / "desktop-resources.json"
    process = subprocess.run(
        [str(executable), "--verify-desktop-resources", "--output", str(report_path)],
        cwd=tmp_path, env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True, text=True, timeout=180, check=False,
    )
    assert process.returncode == 0, process.stderr or process.stdout
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "ok"
    assert report["frozen"] is True
    assert report["schema"] == "smiles2select-desktop-resources/1"
    assert len(report["contextual_models"]) == 9
    assert len(report["risk_models"]) == 1
    assert len(report["tolerance_models"]) == 3
    assert report["tolerance_gui"]["mounted"] and report["tolerance_gui"]["rendered"]
    assert report["tolerance_gui"]["workflow_choices"] == ["General", "Target-conditioned", "Custom"]
    assert report["tolerance_gui"]["experimental"] and report["tolerance_gui"]["basket_unchanged"]
    assert report["tolerance_gui"]["activity_model_count"] == 12
    assert report["tolerance_gui"]["tiny_model_count"] == 3
    assert report["tolerance_gui"]["contextual_model_count"] == 9
    assert report["tolerance_gui"]["about_checked"]
    assert report["tolerance_gui"]["about_action"] == "About SMILES2Select"
    assert all(value in report["tolerance_gui"]["authorship"] for value in (
        "Adriano", "UNIARA", "Daniel Andres Grajales Ruiz", "IQ-UNESP",
    ))
    assert {item["mode"] for item in report["tolerance_models"]} == {
        "general", "hierarchical", "contextual",
    }
    for item in report["tolerance_models"]:
        assert item["policy_status"] == "infeasible"
        assert item["automatic_rescue_enabled"] is False
        assert item["training_counts"]["identities"] > 0
        assert item["recipe"] == "A"
        assert item["preview_enabled"] and not item["adopt_enabled"]
    assert report["methods"]["section_count"] >= 5
    assert report["methods"]["rendered_characters"] > 1000
    assert "experimental" in report["methods"]["rendered_snippet"].lower()
    resources = bundle / "_internal"
    for item in [report["methods"], *report["contextual_models"], *report["risk_models"],
                 *report["tolerance_models"]]:
        resource = resources / item["resource"]
        assert item["sha256"] == hashlib.sha256(resource.read_bytes()).hexdigest()
    for item in [*report["contextual_models"], *report["risk_models"]]:
        assert item["record_ids"] == [1, 2, 3]
        assert len(item["probabilities"]) == 3
        assert all(0 <= value <= 1 for value in item["probabilities"])
    provenance_path = resources / "s2s_decision/bundled_contextual_models/info/provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected_hashes = {item["path"]: item["sha256"] for item in provenance["files"]}
    actual_hashes = {
        item["resource"].split("bundled_contextual_models/", 1)[1]: item["sha256"]
        for item in [*report["contextual_models"], *report["risk_models"]]
    }
    assert actual_hashes == expected_hashes
