"""Fixed, offline verification of distributed desktop resources and inference."""

from __future__ import annotations

import hashlib
import json
import sys
from importlib.resources import files
from pathlib import Path

FLAG = "--verify-desktop-resources"
SCHEMA = "smiles2select-desktop-resources/1"
CONTEXTUAL_MODEL_NAMES = tuple(
    f"{task}-seed{seed}.json"
    for task in ("P00918_HUMAN_Ki", "P22303_HUMAN_IC50", "P56817_HUMAN_IC50")
    for seed in (42, 43, 44)
)
RISK_MODEL_NAME = "shsy5y_atp_viability_48h.json"
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


def verify_desktop_resources() -> dict:
    """Render Methods and infer three illustrative structures with every included JSON model.

    This checks distribution integrity and execution, not predictive performance or
    biological activity. It never starts the main window or the application event loop.
    """
    import pandas as pd

    from s2s_decision.features import featurize
    from smiles2select.app_metadata import APP_VERSION

    methods = _check_methods()
    features = featurize(pd.DataFrame({
        "record_id": list(RECORD_IDS),
        "original_smiles": ["CCO", "CCN", "CC(=O)O"],
        "eligible": [True, True, True],
    }))
    return {
        "schema": SCHEMA,
        "status": "ok",
        "app_version": APP_VERSION,
        "frozen": bool(getattr(sys, "frozen", False)),
        "scope": "Resource integrity and illustrative inference only; no biological validation.",
        "methods": methods,
        "contextual_models": _check_contextual_models(features.records),
        "risk_models": _check_risk_models(features.records, features.manifest["chemistry_hash"]),
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
