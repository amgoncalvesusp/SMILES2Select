"""Tolerance persistence and frozen worker contracts remain distinct from activity."""

import json
import os
import sys
from pathlib import Path

import pytest

from smiles2select.gui.worker_launch import worker_command, worker_dispatch
from smiles2select.selection_intelligence import recipes

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def tolerance_recipe():
    return recipes.SelectionRecipe(strategy="chemical_tolerance", model={
        "model_sha256": "fixture", "preview_sha256": "fixture",
        "context": {"mode": "general", "endpoint": "IC50", "species": "Homo sapiens"},
        "tolerance_settings": {"reconsidered_criteria": ["rule:lipinski:lip_mw_max"]},
    })


def test_tolerance_recipe_roundtrip_and_schema_identity():
    payload = tolerance_recipe().as_dict()
    assert payload["schema_version"] == "6.0"
    assert recipes.from_dict(payload).strategy == "chemical_tolerance"
    with pytest.raises(recipes.RecipeError, match="tolerance"):
        recipes.from_dict({**payload, "model": {"model_sha256": "fixture"}})
    with pytest.raises(recipes.RecipeError, match="schema"):
        recipes.from_dict({**payload, "final_selection": {"strategy": "contextual_policy"}})


def test_frozen_tolerance_training_requires_external_runtime(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    with pytest.raises(RuntimeError, match="configured Python runtime"):
        worker_command("s2s_decision", ["tolerance-train"])
    assert worker_dispatch(["--s2s-worker", "-m", "s2s_decision", "tolerance-train"]) == 2
    assert worker_command("s2s_decision", ["tolerance-train"], python_executable="python311")[0] == "python311"


def test_custom_tolerance_training_uses_existing_studio_fields():
    from PySide6.QtWidgets import QApplication

    from s2s_decision.gui import Studio

    app = QApplication.instance() or QApplication([])
    studio = Studio()
    try:
        studio.mode.setCurrentText("Chemical tolerance")
        studio.action.setCurrentText("tolerance-train")
        studio.input_path.setText("input.csv")
        studio.output_path.setText("new-package")
        studio.fields["target"].setText("P22303")
        studio.fields["species"].setText("Homo sapiens")
        args = studio.command_arguments()
        assert args[2] == "tolerance-train"
        assert args[args.index("--target") + 1] == "P22303"
        assert args[args.index("--species") + 1] == "Homo sapiens"
        assert "--epochs" not in args
        assert "--endpoint" not in args
        assert studio.fields["mode"].currentText() == "general"
        assert studio.fields["recipe"].currentText() == "A"
        studio.fields["reconsider-criteria"].setText("rule:lipinski:lip_mw_max rule:veber:veber_tpsa_max")
        scoped = studio.command_arguments()
        assert scoped[scoped.index("--reconsider-criteria") + 1:][:2] == [
            "rule:lipinski:lip_mw_max", "rule:veber:veber_tpsa_max"]
        studio.fields["positive-cal"].setText("positives.csv")
        with pytest.raises(ValueError, match="together"):
            studio.command_arguments()
        studio.fields["background-cal"].setText("background.csv")
        calibrated_args = studio.command_arguments()
        assert calibrated_args[calibrated_args.index("--positive-cal") + 1] == "positives.csv"
        assert calibrated_args[calibrated_args.index("--background-cal") + 1] == "background.csv"
        studio.action.setCurrentText("tolerance-prepare")
        prepare = studio.command_arguments()
        assert prepare[prepare.index("--target") + 1] == "P22303"
        assert prepare[prepare.index("--species") + 1] == "Homo sapiens"
        assert "--endpoint" not in prepare
        assert "--reconsider-criteria" not in prepare
    finally:
        studio.close()
    assert app is not None


def test_custom_adaptation_requires_base_and_keeps_scratch_explicit():
    from PySide6.QtWidgets import QApplication

    from s2s_decision.gui import Studio

    app = QApplication.instance() or QApplication([])
    studio = Studio()
    try:
        studio.mode.setCurrentText("Chemical tolerance")
        studio.action.setCurrentText("tolerance-train")
        studio.input_path.setText("input.csv")
        studio.output_path.setText("new-package")
        assert studio.custom_fit.currentText() == "Scratch"
        assert "--base-profile" not in studio.command_arguments()
        studio.custom_fit.setCurrentText("Adapt shared profile")
        with pytest.raises(ValueError, match="base"):
            studio.command_arguments()
        studio.base_profile.setText("shared-profile.json")
        args = studio.command_arguments()
        assert args[args.index("--base-profile") + 1] == "shared-profile.json"
        assert "recalibration" in studio.help_text.text().lower()
        studio.custom_fit.setCurrentText("Scratch")
        assert "--base-profile" not in studio.command_arguments()
    finally:
        studio.close()
    assert app is not None


def test_custom_training_accepts_prepared_folder_and_external_worker(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QFileDialog

    import s2s_decision.gui as gui_module

    app = QApplication.instance() or QApplication([])
    source = tmp_path / "prepared"
    source.mkdir()
    source.joinpath("manifest.json").write_text("{}", encoding="utf-8")
    studio = gui_module.Studio()
    try:
        studio.mode.setCurrentText("Chemical tolerance")
        studio.action.setCurrentText("tolerance-train")
        studio.prepared_tolerance.setChecked(True)
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_a: str(source))
        studio.browse_input()
        assert studio.input_path.text() == str(source)
        studio.output_path.setText(str(tmp_path / "profile"))
        studio.training_python.setText("configured-python")
        checked, started = [], []
        monkeypatch.setattr(gui_module, "_verify_training_python",
                            lambda path, **kwargs: checked.append((path, kwargs)))
        monkeypatch.setattr(studio.process, "start", lambda program, args: started.append((program, args)))
        studio.run_command()
        assert checked == [("configured-python", {"tolerance": True})]
        assert started[0][0] == "configured-python"
        assert "tolerance-train" in started[0][1]
        assert started[0][1][started[0][1].index("--input") + 1] == str(source)
    finally:
        studio.close()
    assert app is not None


@pytest.mark.parametrize("recipe,dimension", [("A", 6), ("B", 2055), ("C", 322), ("D", 2055)])
def test_tolerance_resource_estimate_uses_only_bounded_prepared_metadata(
        tmp_path, monkeypatch, recipe, dimension):
    from s2s_decision.gui import tolerance_resource_estimate

    source = tmp_path / "prepared"
    source.mkdir()
    records = source / "records.jsonl"
    records.write_bytes(b"do not parse this corpus")
    metadata = {"schema": "s2s-tolerance-custom-data/1", "row_count": 1000,
                "audit": {"split_counts": {"fit": 600}, "exact_positive_rows": 120},
                "annotation_schema": {"numeric_columns": ["a", "b"], "binary_columns": ["c"]}}
    (source / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
    original_open = Path.open

    def metadata_only(path, *_args, **kwargs):
        assert path.name != "records.jsonl", "Resource estimate must never parse the corpus"
        return original_open(path, *_args, **kwargs)

    monkeypatch.setattr(Path, "open", metadata_only)
    estimate = tolerance_resource_estimate(source, recipe)

    assert estimate["row_count"] == 1000
    assert estimate["fit_row_upper_bound"] == 120
    assert estimate["feature_dimension"] == dimension
    assert estimate["matrix_bytes_upper_bound"] == 1000 * dimension * 4
    assert estimate["fit_matrix_bytes_upper_bound"] == 120 * dimension * 4
    assert estimate["input_bytes"] == len(b"do not parse this corpus")
    assert estimate["provisional"] is True


def test_raw_tolerance_estimate_reports_bytes_and_unknown_rows_before_launch(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    from s2s_decision.gui import Studio, tolerance_resource_estimate

    source = tmp_path / "raw.csv"
    source.write_bytes(b"SMILES,pActivity\nCCO,7\n")
    estimate = tolerance_resource_estimate(source, "B")
    assert estimate["input_bytes"] == source.stat().st_size
    assert estimate["row_count"] is None
    assert estimate["feature_dimension"] is None
    assert estimate["representation_extra_dimension"] == 2049
    app = QApplication.instance() or QApplication([])
    studio = Studio()
    try:
        studio.mode.setCurrentText("Chemical tolerance")
        studio.action.setCurrentText("tolerance-train")
        studio.input_path.setText(str(source))
        studio.output_path.setText(str(tmp_path / "profile"))
        studio.fields["recipe"].setCurrentText("B")
        studio.refresh_resource_estimate()
        assert "not measured" in studio.resource_estimate.text().lower()
        assert "rows: unknown" in studio.resource_estimate.text().lower()
        started = []
        monkeypatch.setattr(studio.process, "start", lambda *args: started.append(args))
        studio.run_command()
        assert started
        assert "Preliminary resource estimate" in studio.log.toPlainText()
        assert "not peak ram" in studio.log.toPlainText().lower()
        studio.action.setCurrentText("tolerance-prepare")
        assert studio.resource_estimate.isHidden()
    finally:
        studio.close()
    assert app is not None


@pytest.mark.parametrize("payload,advisory", [
    (b" " * (1024 * 1024 + 1), "1 MiB preview limit"),
    (b'{"schema":"unrelated"}', "unavailable"),
    (b"[" * 1200 + b"0" + b"]" * 1200, "unavailable"),
], ids=["oversized", "unsupported-schema", "deeply-nested"])
def test_resource_metadata_limits_do_not_invent_counts(tmp_path, payload, advisory):
    from s2s_decision.gui import tolerance_resource_estimate

    source = tmp_path / "prepared"
    source.mkdir()
    (source / "records.jsonl").write_bytes(b"do not parse")
    (source / "manifest.json").write_bytes(payload)
    estimate = tolerance_resource_estimate(source, "A")
    assert estimate["row_count"] is None
    assert estimate["feature_dimension"] is None
    assert estimate["matrix_bytes_upper_bound"] is None
    assert advisory in estimate["advisory"]
