"""Exercise actual Qt selection/map/export jobs at the reported library size.

Run with ``python benchmarks/benchmark_gui_selection.py``. This repeats cached
rows from six real SMILES; it is a software regression/scale smoke check, NOT
a unique-chemistry throughput benchmark or validation of scientific efficacy.
Only file-dialog destinations are patched. Message boxes use normal Qt events.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from smiles2select.app_metadata import APP_VERSION  # noqa: E402
from smiles2select.gui.workspace.workspace_window import WorkspaceWindow  # noqa: E402
from smiles2select.io.importer import ColumnMapping, SourceFile  # noqa: E402
from smiles2select.pipeline.config import RunConfig  # noqa: E402
from smiles2select.pipeline.runner import run  # noqa: E402
from smiles2select.selection_intelligence.constrained_selection import CORE_VERSION  # noqa: E402
from smiles2select.selection_intelligence.scenarios import selected_ids_fingerprint  # noqa: E402

SMILES = (
    "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CC(=O)Nc1ccc(O)cc1",
    "Oc1ccc(cc1)C(=O)C=Cc1ccc(O)cc1", "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
)


def make_result(directory, total, evaluable, original):
    source = directory / "six-real-smiles.csv"
    pd.DataFrame({"ID": range(1, 7), "SMILES": SMILES}).to_csv(source, index=False)
    base = run(RunConfig(
        sources=(SourceFile(path=source, mapping=ColumnMapping(smiles="SMILES", molecule_id="ID")),),
        profile_ids=("lipinski", "veber"), alert_catalogs=(), compute_sa=True,
        n_jobs=1, chunk_size=6,
    ))

    def expand(frame, count):
        expanded = frame.iloc[np.arange(count) % len(frame)].copy()
        expanded.index = pd.RangeIndex(1, count + 1, name="record_id")
        return expanded

    descriptors = expand(base.descriptors, total).drop(columns="murcko_scaffold", errors="ignore")
    descriptors = descriptors.assign(
        molecule_id=[f"REC{i:07d}" for i in descriptors.index],
        evaluable=np.arange(total) < evaluable, valid=np.arange(total) < evaluable,
    )
    records = expand(base.records, total).assign(molecule_id=descriptors.molecule_id)
    decisions = expand(base.decision.decisions, evaluable)
    eligible = decisions.index[decisions.selected.astype(bool)]
    if original > len(eligible):
        raise ValueError(f"Original selection exceeds fixture eligibility ({len(eligible)})")
    decisions = decisions.assign(selected=decisions.index.isin(eligible[:original]))
    return replace(
        base, records=records, descriptors=descriptors, scores=expand(base.scores, evaluable),
        evaluation=replace(base.evaluation, status=expand(base.evaluation.status, evaluable)),
        decision=replace(base.decision, decisions=decisions),
    )


def wait_for_jobs(window, app, timeout=300):
    deadline = time.monotonic() + timeout
    stable = 0
    while time.monotonic() < deadline:
        app.processEvents()
        stable = stable + 1 if not window.is_busy else 0
        if stable >= 3:
            return
        time.sleep(0.01)
    raise AssertionError(f"Timed out: {window.warnings.text()}")


def export_and_check(window, app, directory, selected, suffix):
    path = directory / f"final-selection.{suffix}"
    messages = []

    def close_messages():
        for widget in app.topLevelWidgets():
            if isinstance(widget, QMessageBox) and widget.isVisible():
                messages.append((widget.windowTitle(), widget.text()))
                widget.accept()

    timer = QTimer()
    timer.timeout.connect(close_messages)
    timer.start(20)
    started = time.perf_counter()
    try:
        with patch(
            "smiles2select.gui.workspace.workspace_window.QFileDialog.getSaveFileName",
            return_value=(str(path), "CSV (*.csv)" if suffix == "csv" else "Excel (*.xlsx)"),
        ):
            window._export_docking()
    finally:
        timer.stop()
    assert path.exists(), messages
    frame = pd.read_csv(path) if suffix == "csv" else pd.read_excel(path)
    expected = [f"REC{i:07d}" for i in selected]
    assert frame.access_code.tolist() == expected, "Exported identities/order differ from final basket"
    return {"rows": len(frame), "seconds": round(time.perf_counter() - started, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total", type=int, default=203390)
    parser.add_argument("--evaluable", type=int, default=199243)
    parser.add_argument("--original", type=int, default=73968)
    parser.add_argument("--target", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("build/gui-selection-v3.3.1"))
    args = parser.parse_args()
    if not 0 < args.target <= args.original <= args.evaluable <= args.total:
        parser.error("Require 0 < target <= original <= evaluable <= total")
    args.output.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    started = time.perf_counter()
    result = make_result(args.output, args.total, args.evaluable, args.original)
    window = WorkspaceWindow(result)
    setup_seconds = time.perf_counter() - started
    try:
        assert len(window.basket.final_ids()) == args.original
        assert len(window.candidates) == args.evaluable
        window.target_count.setValue(args.target)
        window.strategy.setCurrentText("scaffold_coverage")
        durations, selections = [], []
        for _ in range(2):
            started = time.perf_counter()
            window.select_button.click()
            wait_for_jobs(window, app)
            selected = list(window.basket.final_ids())
            assert len(selected) == args.target, window.warnings.text()
            shown = [int(point.data()) for point in window.map_view.selected_scatter.points()]
            assert set(shown) == set(selected)
            projected = len(window.map_view._coordinates)
            assert projected == min(args.evaluable, max(5000, args.target))
            assert "Building" not in window.warnings.text()
            durations.append(round(time.perf_counter() - started, 3))
            selections.append(selected)
        assert selections[0] == selections[1], "Repeated selection differs"
        report = {
            "app_version": APP_VERSION, "selection_algorithm_version": CORE_VERSION,
            "input_rows": args.total, "evaluable_rows": args.evaluable,
            "original_selected": args.original, "requested": args.target, "final_selected": len(selected),
            "map_final_points": len(shown), "map_projected_points": projected,
            "setup_seconds": round(setup_seconds, 3),
            "selection_and_map_seconds": durations, "repeat_same_ids_and_order": True,
            "selected_ids_sha256": selected_ids_fingerprint(selected),
            "exports": {suffix: export_and_check(window, app, args.output, selected, suffix)
                        for suffix in ("csv", "xlsx")},
            "limitations": "Synthetic repeated cached rows from six real SMILES; validates GUI, "
            "selection identity/count and export. Not a unique-chemistry throughput benchmark, "
            "biological efficacy assessment, or the user's unavailable original dataset.",
        }
        try:
            import psutil
            report["rss_mib_at_end"] = round(psutil.Process().memory_info().rss / 1024 ** 2, 1)
        except ImportError:
            report["rss_mib_at_end"] = None
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        wait_for_jobs(window, app)
        window.close()


if __name__ == "__main__":
    main()
