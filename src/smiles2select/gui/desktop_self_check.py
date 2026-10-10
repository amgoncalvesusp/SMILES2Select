"""Fixed, offline verification of distributed desktop resources and inference."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from importlib.resources import files
from pathlib import Path
from unittest.mock import patch

FLAG = "--verify-desktop-resources"
SCHEMA = "smiles2select-desktop-resources/1"
CONTEXTUAL_MODEL_NAMES = tuple(
    f"{task}-seed{seed}.json"
    for task in ("P00918_HUMAN_Ki", "P22303_HUMAN_IC50", "P56817_HUMAN_IC50")
    for seed in (42, 43, 44)
)
RISK_MODEL_NAME = "shsy5y_atp_viability_48h.json"
TOLERANCE_MODEL_MODES = {
    "Contextual_S3_A_experimental.json": "contextual",
    "General_S1_A_experimental.json": "general",
    "Hierarchical_S2_A_experimental.json": "hierarchical",
}
RECORD_IDS = (1, 2, 3)


def _bundle_root() -> Path:
    return Path(str(files("s2s_decision").joinpath("bundled_contextual_models")))


def _resource_info(path: Path, resource: str) -> dict:
    return {"resource": resource, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _probabilities(values) -> list[float]:
    import numpy as np

    probabilities = np.asarray(values, dtype=float)
    if (probabilities.shape != (3,) or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()):
        raise ValueError("Inference must return three finite probabilities in [0, 1]")
    return probabilities.tolist()


def _check_methods() -> dict:
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import QApplication

    from smiles2select.gui.pages.methods_page import MethodsPage, methods_text

    source = methods_text()  # Reject unreadable resources even if the page offers a fallback.
    application = QApplication.instance() or QApplication([])
    page = MethodsPage()
    try:
        page.resize(1000, 740)
        page.layout().activate()
        surface = QPixmap(page.size())
        page.render(surface)  # Exercise Qt painting without showing a window.
        application.processEvents()
        rendered = page.browser.toPlainText()
        section_count = page.sections.count()
        if (len(rendered) <= 1000 or section_count < 5 or "Methods" not in rendered
                or "experimental" not in rendered.lower() or not source.startswith("# ")):
            raise ValueError("Methods did not render the complete English document")
        if page.browser.document().documentLayout().documentSize().height() <= 0:
            raise ValueError("Methods document layout is empty")
        marker = rendered.lower().index("experimental")
        path = Path(str(files("smiles2select").joinpath("assets", "methods.md")))
        return {
            **_resource_info(path, "smiles2select/assets/methods.md"),
            "section_count": section_count,
            "rendered_characters": len(rendered),
            "rendered_snippet": rendered[:160] + "\n" + rendered[max(0, marker - 100):marker + 260],
        }
    finally:
        page.close()
        page.deleteLater()


def _check_contextual_models(frame) -> list[dict]:
    from s2s_decision.contextual_model import load_bundle, score_frame

    results = []
    for name in CONTEXTUAL_MODEL_NAMES:
        path = _bundle_root() / name
        bundle = load_bundle(path)
        scored = score_frame(frame, bundle)
        if scored.record_id.tolist() != list(RECORD_IDS):
            raise ValueError("Contextual inference changed record identity or order")
        results.append({
            **_resource_info(path, f"s2s_decision/bundled_contextual_models/{name}"),
            "task_id": bundle["task_id"],
            "record_ids": list(RECORD_IDS),
            "probabilities": _probabilities(scored.activity_score),
        })
    return results


def _check_risk_models(frame, chemistry_hash: str) -> list[dict]:
    from s2s_decision.risk_model import load_risk_bundle, predict_risk

    path = _bundle_root() / "risk" / RISK_MODEL_NAME
    bundle = load_risk_bundle(path)
    if bundle.get("chemistry_version") != chemistry_hash:
        raise ValueError("Bundled risk model chemistry does not match inference")
    return [{
        **_resource_info(path, f"s2s_decision/bundled_contextual_models/risk/{RISK_MODEL_NAME}"),
        "endpoint": bundle["risk_scope"],
        "record_ids": list(RECORD_IDS),
        "probabilities": _probabilities(predict_risk(frame, bundle)),
    }]


def _check_about() -> dict:
    from PySide6.QtWidgets import QMessageBox

    from smiles2select.app_metadata import APP_NAME, APP_VERSION, AUTHORSHIP
    from smiles2select.gui.main_window import MainWindow

    window = MainWindow()
    try:
        action = window.about_action
        if action.text() != f"About {APP_NAME}" or not action.isEnabled():
            raise ValueError("About action is missing or disabled")
        # Capture the actual callback without blocking this diagnostic in a modal dialog.
        with patch.object(QMessageBox, "about") as dialog:
            action.trigger()
        if dialog.call_count != 1:
            raise ValueError("About action did not open the authorship dialog")
        _parent, title, message = dialog.call_args.args
        if APP_NAME not in title or APP_VERSION not in message or AUTHORSHIP not in message:
            raise ValueError("About dialog does not contain current version and authorship")
        return {"about_action": action.text(), "about_checked": True, "authorship": AUTHORSHIP}
    finally:
        window.close()
        window.deleteLater()


def _check_tolerance_profile(controls, name: str, mode: str) -> dict:
    path = Path(controls.included_profiles.currentData()).resolve()
    controls.load_included_button.click()
    profile = controls._profile
    if controls._package != path or profile is None:
        raise ValueError(f"Included tolerance profile failed to load: {name}")
    policy = profile["policy"]
    if profile["mode"] != mode or profile["recipe"] != "A":
        raise ValueError(f"Included tolerance profile has unexpected mode or recipe: {name}")
    if (policy["status"] != "infeasible" or policy.get("tau") is not None
            or policy.get("rescue_enabled") is not False):
        raise ValueError(f"Included experimental tolerance policy changed: {name}")
    label = controls.package_label.text()
    if not all(value in label for value in ("Training:", "Support gates:", "infeasible")):
        raise ValueError(f"Tolerance GUI did not explain training and policy: {name}")
    if not controls.preview_button.isEnabled() or controls.adopt_button.isEnabled():
        raise ValueError(f"Tolerance GUI did not require explicit preview before adoption: {name}")
    return {
        **_resource_info(path, f"s2s_decision/bundled_tolerance_models/{name}"),
        "mode": profile["mode"], "recipe": profile["recipe"],
        "profile_sha256": profile["profile_sha256"],
        "policy_status": policy["status"], "automatic_rescue_enabled": False,
        "training_counts": profile["training_counts"],
        "preview_enabled": True, "adopt_enabled": False,
    }


def _check_tolerance_gui(result) -> dict:
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import QApplication

    from smiles2select.gui.workspace.workspace_window import WorkspaceWindow

    application = QApplication.instance() or QApplication([])
    window = WorkspaceWindow(result)
    try:
        before = tuple(window.basket.final_ids())
        window.resize(1400, 900)
        window.show()
        panel, controls = window.model_panel, window.model_panel.tolerance
        panel.tolerance_toggle.click()
        application.processEvents()
        window.controls_scroll.ensureWidgetVisible(controls)
        if not window.isAncestorOf(controls) or not controls.isVisible():
            raise ValueError("Tolerance controls are not mounted and visible in the workspace")
        choices = [controls.workflow.itemText(index) for index in range(controls.workflow.count())]
        if choices != ["General", "Target-conditioned", "Custom"]:
            raise ValueError("Tolerance GUI is missing a supported workflow")
        if "experimental" not in panel.tolerance_toggle.text().lower():
            raise ValueError("Tolerance GUI must mark included models as experimental")
        names = [Path(controls.included_profiles.itemData(index)).name
                 for index in range(1, controls.included_profiles.count())]
        if names != list(TOLERANCE_MODEL_MODES):
            raise ValueError("GUI did not discover the three included tolerance profiles")
        if controls._package is not None or controls.load_included_button.isEnabled():
            raise ValueError("Tolerance GUI loaded a package without explicit selection")
        models = []
        for index, name in enumerate(names, 1):
            controls.included_profiles.setCurrentIndex(index)
            if not controls.load_included_button.isEnabled():
                raise ValueError("Tolerance GUI cannot load the selected included profile")
            models.append(_check_tolerance_profile(controls, name, TOLERANCE_MODEL_MODES[name]))
        application.processEvents()
        controls.render(QPixmap(controls.size()))
        catalog = [item for item in panel._catalog if item.get("origin") == "bundled"
                   and item.get("compatible")]
        tiny_count = sum("tiny" in item["name"].lower() for item in catalog)
        contextual_count = panel.contextual.included_models.count() - 1
        if len(catalog) != 12 or tiny_count != 3 or contextual_count != 9:
            raise ValueError("Experimental activity/contextual model inventory is incomplete")
        if tuple(window.basket.final_ids()) != before:
            raise ValueError("Loading tolerance profiles changed the native basket")
        return {"tolerance_gui": {
            "mounted": True, "rendered": True, "workflow_choices": choices,
            "experimental": True, "basket_unchanged": True,
            "criteria_count": controls.criteria.rowCount(),
            "activity_model_count": len(catalog), "tiny_model_count": tiny_count,
            "contextual_model_count": contextual_count, **_check_about(),
        }, "tolerance_models": models}
    finally:
        window.close()
        window.deleteLater()


def verify_desktop_resources() -> dict:
    """Render real desktop controls and check distributed models with illustrative structures.

    This checks distribution integrity and execution, not predictive performance or
    biological activity. It constructs offscreen windows without an application event loop.
    """
    import pandas as pd
    from PySide6.QtWidgets import QApplication

    from s2s_decision.features import featurize
    from smiles2select.app_metadata import APP_VERSION
    from smiles2select.io.importer import ColumnMapping, SourceFile
    from smiles2select.pipeline.config import RunConfig
    from smiles2select.pipeline.runner import run

    application = QApplication.instance() or QApplication([])
    methods = _check_methods()
    features = featurize(pd.DataFrame({
        "record_id": list(RECORD_IDS),
        "original_smiles": ["CCO", "CCN", "CC(=O)O"],
        "eligible": [True, True, True],
    }))
    with tempfile.TemporaryDirectory(prefix="s2s-desktop-check-") as temporary:
        source = Path(temporary) / "library.csv"
        pd.DataFrame({"ID": list(RECORD_IDS), "SMILES": ["CCO", "CCN", "CC(=O)O"]}).to_csv(
            source, index=False,
        )
        result = run(RunConfig(
            sources=(SourceFile(path=source, mapping=ColumnMapping(smiles="SMILES", molecule_id="ID")),),
            profile_ids=("lipinski", "veber"), alert_catalogs=(), compute_sa=True,
            n_jobs=1, chunk_size=3,
        ))
        tolerance = _check_tolerance_gui(result)
    application.processEvents()
    return {
        "schema": SCHEMA,
        "status": "ok",
        "app_version": APP_VERSION,
        "frozen": bool(getattr(sys, "frozen", False)),
        "scope": "Resource integrity and illustrative inference only; no biological validation.",
        "methods": methods,
        "contextual_models": _check_contextual_models(features.records),
        "risk_models": _check_risk_models(features.records, features.manifest["chemistry_hash"]),
        **tolerance,
    }


def _error(exc: Exception) -> dict:
    return {"schema": SCHEMA, "status": "error", "error": f"{type(exc).__name__}: {exc}"}


def _run_check() -> tuple[dict, int]:
    try:
        return verify_desktop_resources(), 0
    except Exception as exc:
        # The diagnostic boundary reports any failed check and exits unsuccessfully.
        return _error(exc), 1


def _emit(report: dict) -> bool:
    if sys.stdout is None:  # PyInstaller's Windows windowed executable has no stdout.
        return True
    try:
        sys.stdout.write(json.dumps(report, ensure_ascii=True, allow_nan=False) + "\n")
        sys.stdout.flush()
        return True
    except OSError:
        return False


def desktop_resources_dispatch(arguments: list[str]) -> int | None:
    """Accept only the fixed check and an optional, exclusively created report path."""
    if FLAG not in arguments:
        return None
    valid = arguments == [FLAG] or (
        len(arguments) == 3 and arguments[:2] == [FLAG, "--output"]
        and bool(arguments[2].strip()) and not arguments[2].startswith("--")
    )
    if not valid:
        _emit(_error(ValueError(f"Usage: {FLAG} [--output NEW_REPORT.json]")))
        return 2
    output_written = False
    if len(arguments) == 3:
        try:
            with Path(arguments[2]).open("x", encoding="utf-8") as destination:
                report, status = _run_check()
                json.dump(report, destination, ensure_ascii=True, allow_nan=False, indent=2)
                destination.write("\n")
            output_written = True
        except (OSError, ValueError) as exc:
            report, status = _error(exc), 1
    else:
        report, status = _run_check()
    emitted = _emit(report)
    return status if emitted or output_written else 1
