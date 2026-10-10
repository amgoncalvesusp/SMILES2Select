"""Responsive desktop client for the authoritative S2S-Decision CLI."""

import codecs
import json
import subprocess
import sys
from pathlib import Path
from typing import TypedDict

from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from smiles2select.app_metadata import APP_VERSION
from smiles2select.gui.pages.methods_page import experimental_notice, install_methods_action
from smiles2select.gui.worker_launch import worker_command


def _verify_training_python(executable: str, *, tolerance: bool = False) -> None:
    path = Path(executable)
    if not path.is_file() or (
        getattr(sys, "frozen", False) and path.resolve() == Path(sys.executable).resolve()
    ):
        raise ValueError("Choose an external Python 3.11+ executable for training")
    dependencies = ("numpy", "pandas", "rdkit") if tolerance else (
        "torch", "sklearn", "onnx", "onnxscript", "onnxruntime", "skl2onnx")
    script = (
        "import importlib.util,json,sys; from importlib import metadata; "
        "print(json.dumps({'python':sys.version_info[:2],"
        "'version':metadata.version('smiles2select'),"
        "'missing':[name for name in "
        f"{dependencies!r} "
        "if importlib.util.find_spec(name) is None]}))"
    )
    try:
        check = subprocess.run(
            [str(path), "-c", script],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("Could not inspect the configured training Python") from exc
    if check.returncode != 0:
        raise ValueError("Configured Python must have smiles2select[train] installed")
    try:
        details = json.loads(check.stdout)
        version = tuple(details["python"])
        missing = details["missing"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError("Configured Python returned an invalid runtime report") from exc
    if version < (3, 11):
        raise ValueError("Training requires Python 3.11 or newer")
    if details["version"] != APP_VERSION:
        raise ValueError(f"Training Python must contain SMILES2Select {APP_VERSION}")
    if missing:
        raise ValueError(f"Training Python is missing dependencies: {', '.join(missing)}")


class ToleranceResourceEstimate(TypedDict):
    recipe: str
    provisional: bool
    input_bytes: int | None
    row_count: int | None
    fit_row_upper_bound: int | None
    feature_dimension: int | None
    representation_extra_dimension: int
    matrix_bytes_upper_bound: int | None
    fit_matrix_bytes_upper_bound: int | None
    advisory: str


def _resource_count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 10**12 else None


def tolerance_resource_estimate(source: str | Path, recipe: str) -> ToleranceResourceEstimate:
    """Read at most 1 MiB of metadata; never scan input rows on the GUI thread."""
    extra = {"A": 0, "B": 2049, "C": 316, "D": 2049}[recipe]
    estimate: ToleranceResourceEstimate = {"recipe": recipe, "provisional": True, "input_bytes": None,
        "row_count": None, "fit_row_upper_bound": None, "feature_dimension": None,
        "representation_extra_dimension": extra, "matrix_bytes_upper_bound": None,
        "fit_matrix_bytes_upper_bound": None, "advisory": "Prepare data to obtain row and base-feature counts."}
    path = Path(source)
    try:
        data = path / "records.jsonl" if path.is_dir() else path
        if data.is_file():
            estimate["input_bytes"] = data.stat().st_size
        if not path.is_dir():
            return estimate
        manifest = path / "manifest.json"
        with manifest.open("rb") as stream:
            payload = stream.read(1024 * 1024 + 1)
        if len(payload) > 1024 * 1024:
            estimate["advisory"] = "Manifest exceeds the 1 MiB preview limit; counts await worker validation."
            return estimate
        metadata = json.loads(payload)
        if not isinstance(metadata, dict) or metadata.get("schema") != "s2s-tolerance-custom-data/1":
            raise ValueError("not prepared tolerance metadata")
        rows = _resource_count(metadata.get("row_count"))
        audit = metadata.get("audit", {})
        fit = _resource_count(audit.get("split_counts", {}).get("fit"))
        positives = _resource_count(audit.get("exact_positive_rows"))
        bounds = [value for value in (rows, fit, positives) if value is not None]
        schema = metadata.get("annotation_schema", {})
        numeric, binary = schema.get("numeric_columns"), schema.get("binary_columns")
        dimension = 2 * (len(numeric) + len(binary)) + extra if (
            isinstance(numeric, list) and isinstance(binary, list)) else None
        estimate = {**estimate, "row_count": rows,
                    "fit_row_upper_bound": min(bounds) if bounds else None,
                    "feature_dimension": dimension,
                    "advisory": "Counts come from metadata, not a GUI row scan; "
                    "the worker reports actual fit rows after validity, activity, conflict and context filtering."}
        if rows is not None and dimension is not None:
            estimate["matrix_bytes_upper_bound"] = rows * dimension * 4
        if estimate["fit_row_upper_bound"] is not None and dimension is not None:
            estimate["fit_matrix_bytes_upper_bound"] = estimate["fit_row_upper_bound"] * dimension * 4
    except (OSError, ValueError, TypeError, AttributeError, RecursionError) as exc:
        estimate["advisory"] = f"Metadata estimate unavailable ({exc}); inspect the worker validation report."
    return estimate


def resource_estimate_text(estimate: ToleranceResourceEstimate) -> str:
    def size(value: int | None) -> str:
        return "unknown" if value is None else f"{value:,} bytes ({value / 1024**2:.2f} MiB)"

    rows, fit, dimension = estimate["row_count"], estimate["fit_row_upper_bound"], estimate["feature_dimension"]
    return ("Preliminary resource estimate (provisional)\n"
        f"Input data: {size(estimate['input_bytes'])}; rows: {rows if rows is not None else 'unknown'}; "
        f"custom fit rows: at most {fit if fit is not None else 'unknown'}.\n"
        f"Recipe {estimate['recipe']}: {dimension if dimension is not None else 'unknown'} total features; "
        f"representation adds {estimate['representation_extra_dimension']} values (with a missingness mask when present).\n"
        f"One float32 matrix for all input rows: at most {size(estimate['matrix_bytes_upper_bound'])}; "
        f"custom fit-only matrix: at most {size(estimate['fit_matrix_bytes_upper_bound'])}. "
        "This is not peak RAM; chemistry, pandas, copies, support references, shared-profile adaptation "
        "and calibration add memory. Training time and peak RAM are not measured.\n" + estimate["advisory"])


class Studio(QMainWindow):
    MODES = {
        "Selection": ("prepare", "predict", "select"),
        "Training": ("audit", "dataset", "train", "train-baseline"),
        "Benchmark": ("benchmark", "evaluate"),
        "Chemical tolerance": ("tolerance-prepare", "tolerance-train"),
    }
    OPTIONS = {
        "threshold": ("audit", "dataset"),
        "fingerprint-bits": ("prepare", "dataset"),
        "target": ("dataset", "tolerance-train", "tolerance-prepare"),
        "endpoint": ("dataset", "tolerance-train", "tolerance-prepare"),
        "species": ("tolerance-train", "tolerance-prepare"),
        "mode": ("tolerance-train",),
        "recipe": ("tolerance-train",),
        "positive-cal": ("tolerance-train",),
        "background-cal": ("tolerance-train",),
        "target-retention": ("tolerance-train",),
        "max-expansion": ("tolerance-train",),
        "reconsider-criteria": ("tolerance-train",),
        "split": ("dataset",),
        "max-rows": ("dataset",),
        "seed": ("dataset", "train", "train-baseline"),
        "estimator": ("train-baseline",),
        "input-layout": ("train-baseline",),
        "epochs": ("train",),
        "patience": ("train",),
        "learning-rate": ("train",),
        "weight-decay": ("train",),
        "threads": ("train", "train-baseline"),
        "resume": ("train",),
        "batch-size": ("train", "predict"),
        "n": ("benchmark", "evaluate", "select"),
        "max-per-scaffold": ("select",),
        "min-scaffolds": ("select",),
        "max-per-cluster": ("select",),
        "pins": ("select",),
        "exclude": ("select",),
    }
    FILE_INPUTS = {"audit", "prepare", "dataset", "tolerance-prepare", "tolerance-train"}
    FILE_OUTPUTS = {"audit", "benchmark", "evaluate"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SMILES2Select — Advanced model tools")
        self.resize(900, 850)
        install_methods_action(self)
        self.cancelled = False
        self.close_after_finish = False
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.finished.connect(self.process_finished)
        self.process.errorOccurred.connect(self.process_error)
        self.kill_timer = QTimer(self)
        self.kill_timer.setSingleShot(True)
        self.kill_timer.timeout.connect(self.kill_process)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        self.experimental_notice = experimental_notice(self)
        layout.addWidget(self.experimental_notice)
        self.editor = QWidget()
        form = QFormLayout(self.editor)
        self.mode = QComboBox()
        self.mode.addItems(list(self.MODES))
        self.action = QComboBox()
        form.addRow("Workflow", self.mode)
        form.addRow("Action", self.action)
        self.input_path = QLineEdit()
        self.output_path = QLineEdit()
        self.model_path = QLineEdit()
        self.input_label = QLabel()
        self.output_label = QLabel()
        form.addRow(self.input_label, self.path_row(self.input_path, self.browse_input))
        self.prepared_tolerance = QCheckBox("Browse a prepared tolerance dataset folder")
        form.addRow(self.prepared_tolerance)
        self.custom_fit = QComboBox()
        self.custom_fit.addItems(["Scratch", "Adapt shared profile"])
        self.custom_fit_label = QLabel("Custom training")
        form.addRow(self.custom_fit_label, self.custom_fit)
        self.base_profile = QLineEdit()
        self.base_profile.setPlaceholderText("Shared tolerance package directory or profile JSON")
        self.base_profile_row = self.path_row(self.base_profile, self.browse_base_profile)
        self.base_profile_label = QLabel("Base tolerance package")
        form.addRow(self.base_profile_label, self.base_profile_row)
        form.addRow(self.output_label, self.path_row(self.output_path, self.browse_output))
        self.model_row = self.path_row(self.model_path, self.browse_model)
        self.model_label = QLabel("Model directory")
        form.addRow(self.model_label, self.model_row)
        self.training_python = QLineEdit()
        self.training_python.setPlaceholderText("Python 3.11+ with smiles2select[train]")
        self.training_runtime_row = self.path_row(
            self.training_python, self.browse_training_python
        )
        self.training_runtime_label = QLabel("Training Python (optional in source install)")
        form.addRow(self.training_runtime_label, self.training_runtime_row)
        self.fields = {}
        self.labels = {}
        for name in self.OPTIONS:
            field = self.make_field(name)
            label = QLabel(name.replace("-", " ").capitalize())
            label.setBuddy(field)
            self.fields[name] = field
            self.labels[name] = label
            form.addRow(label, field)
        layout.addWidget(self.editor)
        self.help_text = QLabel()
        self.help_text.setWordWrap(True)
        layout.addWidget(self.help_text)
        self.resource_estimate = QLabel()
        self.resource_estimate.setWordWrap(True)
        self.resource_estimate_button = QPushButton("Refresh preliminary resource estimate")
        self.resource_estimate_button.clicked.connect(self.refresh_resource_estimate)
        form.addRow(self.resource_estimate_button)
        form.addRow(self.resource_estimate)
        self.resource_timer = QTimer(self)
        self.resource_timer.setSingleShot(True)
        self.resource_timer.setInterval(350)
        self.resource_timer.timeout.connect(self.refresh_resource_estimate)
        self.input_path.textChanged.connect(lambda: self.resource_timer.start())
        self.fields["recipe"].currentTextChanged.connect(lambda: self.resource_timer.start())
        buttons = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.cancel_button = QPushButton("Cancel job")
        self.cancel_button.setEnabled(False)
        self.run_button.clicked.connect(self.run_command)
        self.cancel_button.clicked.connect(self.cancel_process)
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.status = QLabel("Ready")
        layout.addWidget(self.status)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)
        layout.addWidget(self.log, 1)
        self.mode.currentTextChanged.connect(self.change_mode)
        self.action.currentTextChanged.connect(self.change_action)
        self.prepared_tolerance.toggled.connect(lambda: self.change_action(self.action.currentText()))
        self.custom_fit.currentTextChanged.connect(lambda: self.change_action(self.action.currentText()))
        self.change_mode(self.mode.currentText())

    @staticmethod
    def path_row(field, callback):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(field)
        button = QPushButton("Browse…")
        button.clicked.connect(callback)
        layout.addWidget(button)
        return row

    @staticmethod
    def make_field(name):
        choices = {
            "estimator": ("logistic", "gradient_boosting"),
            "input-layout": ("scalar_fingerprint", "scalar"),
            "fingerprint-bits": ("2048", "1024"),
            "endpoint": ("Ki", "Kd", "IC50", "EC50"),
            "split": ("scaffold", "temporal", "random"),
            "mode": ("general", "fixed", "hierarchical", "contextual"),
            "recipe": ("A", "B", "C", "D"),
        }
        if name in choices:
            field = QComboBox()
            field.addItems(choices[name])
            return field
        if name in {"target", "species", "pins", "exclude", "positive-cal", "background-cal", "reconsider-criteria"}:
            field = QLineEdit()
            if name in {"pins", "exclude"}:
                field.setPlaceholderText("Optional integer IDs, separated by commas or spaces")
            elif name == "species":
                field.setText("unknown")
            elif name in {"positive-cal", "background-cal"}:
                field.setPlaceholderText("Optional separate calibration CSV; supply both files together")
            elif name == "reconsider-criteria":
                field.setPlaceholderText("Optional criterion IDs separated by spaces; blank uses standard calibration scope")
            return field
        if name == "resume":
            return QCheckBox("Resume existing model checkpoint")
        defaults = {"threshold": 6.0, "learning-rate": 0.001, "weight-decay": 0.0001,
                    "target-retention": .95, "max-expansion": .1}
        if name in defaults:
            field = QDoubleSpinBox()
            field.setDecimals(6)
            field.setRange(0.000001 if name == "learning-rate" else 0.0, 100.0)
            if name in {"target-retention", "max-expansion"}:
                field.setMaximum(1.)
        else:
            field = QSpinBox()
            field.setRange(
                0
                if name in {"seed", "max-per-scaffold", "min-scaffolds", "max-per-cluster"}
                else 1,
                2_000_000_000,
            )
            defaults = {
                "max-rows": 250000,
                "seed": 42,
                "epochs": 200,
                "patience": 20,
                "threads": 4,
                "batch-size": 1024,
                "n": 100,
            }
            if name in {"max-per-scaffold", "min-scaffolds", "max-per-cluster"}:
                field.setSpecialValueText("Not set")
        field.setValue(defaults.get(name, 0))
        return field

    def change_mode(self, mode):
        self.action.clear()
        self.action.addItems(self.MODES[mode])

    def change_action(self, action):
        if not action:
            return
        endpoint = self.fields["endpoint"]
        index = endpoint.findText("Keep input endpoints")
        tolerance = action in {"tolerance-train", "tolerance-prepare"}
        if tolerance and index < 0:
            endpoint.addItem("Keep input endpoints")
            endpoint.setCurrentIndex(endpoint.count() - 1)
        elif not tolerance and index >= 0:
            endpoint.removeItem(index)
        for name, field in self.fields.items():
            relevant = action in self.OPTIONS[name]
            field.setVisible(relevant)
            field.setEnabled(relevant)
            self.labels[name].setVisible(relevant)
        model = action in {"predict", "evaluate"}
        self.model_row.setVisible(model)
        self.model_row.setEnabled(model)
        self.model_label.setVisible(model)
        training = action in {"train", "train-baseline", "benchmark", "tolerance-train"}
        self.training_runtime_row.setVisible(training)
        self.training_runtime_row.setEnabled(training)
        self.training_runtime_label.setVisible(training)
        self.prepared_tolerance.setVisible(action == "tolerance-train")
        self.custom_fit.setVisible(action == "tolerance-train")
        self.custom_fit_label.setVisible(action == "tolerance-train")
        adapting = action == "tolerance-train" and self.custom_fit.currentText() == "Adapt shared profile"
        self.base_profile_row.setVisible(adapting)
        self.base_profile_row.setEnabled(adapting)
        self.base_profile_label.setVisible(adapting)
        self.resource_estimate.setVisible(action == "tolerance-train")
        self.resource_estimate_button.setVisible(action == "tolerance-train")
        self.input_label.setText(
            "Input CSV or prepared tolerance folder" if action == "tolerance-train"
            else "Input table file" if action in self.FILE_INPUTS else "Input bundle directory"
        )
        self.output_label.setText(
            "New report JSON file" if action in self.FILE_OUTPUTS else "New output directory"
        )
        messages = {
            "prepare": "Import a SMILES2Select table. Molecular data and upstream selection provenance are preserved.",
            "predict": "Input: prepared candidate bundle. Model: trained experimental model with references.",
            "select": "Input: scored candidate bundle. N and diversity constraints apply to eligible candidates.",
            "audit": "Inspect measured data and label availability before preparing a training dataset.",
            "dataset": "Input: measured table with structures. Unknown measurements are not inactive labels.",
            "train": "Input: prepared measured dataset. Resume requires its original dataset and model directory.",
            "train-baseline": "Input: prepared measured dataset. Fit logistic regression or gradient boosting, then export a portable model for guided selection.",
            "benchmark": "Input: prepared measured dataset. Compare baselines; results are not biological proof.",
            "evaluate": "Evaluate held-out data with an existing model. Keep test data out of model selection.",
            "tolerance-prepare": "Validate measured CSV rows before training. Include SMILES and exact pActivity "
                "or activity values with units, endpoint, relation and provenance. Unknown activity remains unknown. "
                "Set context defaults before preparing a folder; its curated rows and split are then frozen. "
                "Inspect the validation report before choosing a fit partition.",
            "tolerance-train": "Fit chemical compatibility from observed actives at pActivity >= 6.5. "
                "The CLI fixes scaffold partitions before fitting and reports curation and counts. "
                "Scratch learns a new profile; Adapt shared profile keeps the base recipe and transforms, "
                "adds custom fit references, and requires recalibration rather than inheriting its threshold. "
                "Separate positive and unlabeled-background calibration CSVs are optional and must be supplied together. "
                "A profile without held-out policy calibration cannot automatically rescue exclusions. "
                "The optional criterion IDs bind calibration to this exact selection scope; blank uses "
                "Lipinski, Veber, PAINS and Brenk entries. "
                "Load the completed package into the workspace to inspect it.",
        }
        self.help_text.setText(messages[action])
        if action == "tolerance-train":
            self.refresh_resource_estimate()

    def refresh_resource_estimate(self) -> str:
        if self.action.currentText() != "tolerance-train":
            return ""
        report = resource_estimate_text(tolerance_resource_estimate(
            self.input_path.text().strip(), self.fields["recipe"].currentText()))
        self.resource_estimate.setText(report)
        return report

    def browse_input(self):
        if self.action.currentText() == "tolerance-train" and self.prepared_tolerance.isChecked():
            path = QFileDialog.getExistingDirectory(self, "Select prepared tolerance dataset")
        elif self.action.currentText() in self.FILE_INPUTS:
            path, _ = QFileDialog.getOpenFileName(self, "Select input table")
        else:
            path = QFileDialog.getExistingDirectory(self, "Select input bundle directory")
        if path:
            self.input_path.setText(path)

    def browse_model(self):
        path = QFileDialog.getExistingDirectory(self, "Select trained model directory")
        if path:
            self.model_path.setText(path)

    def browse_base_profile(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose base tolerance package manifest or profile",
                                            "", "Tolerance JSON (*.json)")
        if path:
            selected = Path(path)
            if selected.name == "manifest.json" or (selected.name == "profile.json"
                    and (selected.parent / "manifest.json").is_file()):
                selected = selected.parent
            self.base_profile.setText(str(selected))
            try:
                from s2s_decision.tolerance_profile import read_tolerance_package

                profile = read_tolerance_package(selected)
                self.fields["mode"].setCurrentText(profile["mode"])
                self.fields["recipe"].setCurrentText(profile["recipe"])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.log.appendPlainText(f"Cannot inspect the base tolerance package: {exc}")

    def browse_training_python(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select training Python executable")
        if path:
            self.training_python.setText(path)

    def browse_output(self):
        action = self.action.currentText()
        if action in self.FILE_OUTPUTS:
            path, _ = QFileDialog.getSaveFileName(self, "Choose new report", "", "JSON (*.json)")
        else:
            path = QFileDialog.getExistingDirectory(
                self, "Select parent directory (edit new folder name below)"
            )
            if path:
                path = str(Path(path) / f"s2s_{action}")
        if path:
            self.output_path.setText(path)

    def command_arguments(self):
        action = self.action.currentText()
        tolerance = action in {"tolerance-train", "tolerance-prepare"}
        source, output = self.input_path.text().strip(), self.output_path.text().strip()
        if not source or not output:
            raise ValueError("Input and output paths are required")
        if action == "tolerance-train" and bool(self.fields["positive-cal"].text().strip()) != bool(self.fields["background-cal"].text().strip()):
            raise ValueError("Supply positive and background calibration CSVs together")
        args = ["-m", "s2s_decision", action, "--input", source, "--output", output]
        if action == "tolerance-train" and self.custom_fit.currentText() == "Adapt shared profile":
            base = self.base_profile.text().strip()
            if not base:
                raise ValueError("Adaptation requires a base tolerance package")
            args += ["--base-profile", base]
        if action in {"predict", "evaluate"}:
            if not self.model_path.text().strip():
                raise ValueError("Model directory is required")
            args += ["--model", self.model_path.text().strip()]
        for name, actions in self.OPTIONS.items():
            if action not in actions:
                continue
            field = self.fields[name]
            if isinstance(field, QCheckBox):
                if field.isChecked():
                    args.append(f"--{name}")
                continue
            value = (
                field.currentText()
                if isinstance(field, QComboBox)
                else field.text().strip()
                if isinstance(field, QLineEdit)
                else str(field.value())
            )
            if name in {"pins", "exclude"}:
                if value:
                    args += [f"--{name}", *[str(int(v)) for v in value.replace(",", " ").split()]]
                continue
            if name == "reconsider-criteria":
                if value:
                    args += ["--reconsider-criteria", *value.replace(",", " ").split()]
                continue
            if tolerance and name in {"target", "endpoint"} and not value:
                continue
            if tolerance and name == "endpoint" and value == "Keep input endpoints":
                continue
            if name in {"positive-cal", "background-cal"} and not value:
                continue
            if name in {"max-per-scaffold", "min-scaffolds", "max-per-cluster"} and value == "0":
                continue
            if not value:
                raise ValueError(f"{name} is required")
            args += [f"--{name}", value]
        return args

    def set_running(self, running):
        self.editor.setEnabled(not running)
        self.run_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self.progress.setRange(0, 0 if running else 1)
        if not running:
            self.progress.setValue(0)

    def run_command(self):
        if self.process.state() != QProcess.ProcessState.NotRunning:
            return
        training = self.action.currentText() in {"train", "train-baseline", "benchmark", "tolerance-train"}
        runtime_check = False
        try:
            args = self.command_arguments()
            source, output = (
                Path(self.input_path.text().strip()),
                Path(self.output_path.text().strip()),
            )
            file_input = self.action.currentText() in self.FILE_INPUTS
            tolerance_input = self.action.currentText() == "tolerance-train"
            exists = (source.is_file() or source.is_dir()) if tolerance_input else (
                source.is_file() if file_input else source.is_dir())
            if not exists:
                raise ValueError(
                    "Input must be an existing CSV or prepared tolerance dataset folder" if tolerance_input
                    else "Input must be an existing table file"
                    if file_input
                    else "Input must be an existing bundle directory"
                )
            resume = self.action.currentText() == "train" and self.fields["resume"].isChecked()
            if output.exists() and not resume:
                raise ValueError(
                    "output already exists; choose a new path (train supports --resume)"
                )
            if tolerance_input:
                self.log.appendPlainText(self.refresh_resource_estimate())
            runtime_check = True
            external_python = (self.training_python.text().strip() or None) if training else None
            if external_python:
                if self.action.currentText() == "tolerance-train":
                    _verify_training_python(external_python, tolerance=True)
                else:
                    _verify_training_python(external_python)
            program, worker_args = worker_command(
                args=args[2:], python_executable=external_python
            )
        except (ValueError, RuntimeError) as exc:
            self.status.setText(
                "Training runtime unavailable" if training and runtime_check else "Invalid input"
            )
            self.log.appendPlainText(str(exc))
            return
        self.cancelled = False
        self.decoder.reset()
        self.set_running(True)
        self.status.setText("Running — progress reported by CLI")
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        environment.insert("PYTHONIOENCODING", "utf-8")
        self.process.setProcessEnvironment(environment)
        self.process.start(program, worker_args)

    def read_output(self):
        chunk = self.decoder.decode(bytes(self.process.readAllStandardOutput()))
        # ponytail: retain only recent diagnostics; structured results remain in output artifacts.
        self.log.appendPlainText(chunk[-64000:])
        if len(self.log.toPlainText()) > 200000:
            self.log.setPlainText(self.log.toPlainText()[-200000:])

    def process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.set_running(False)
            self.status.setText("Failed to start CLI")
            self.log.appendPlainText(self.process.errorString())
            if self.close_after_finish:
                self.close()

    def process_finished(self, code, exit_status):
        self.kill_timer.stop()
        self.read_output()
        self.set_running(False)
        self.status.setText(
            "Cancelled — partial artifacts retained"
            if self.cancelled
            else "Completed"
            if code == 0 and exit_status == QProcess.ExitStatus.NormalExit
            else f"Failed (exit {code}) — inspect log"
        )
        if self.close_after_finish:
            self.close()

    def cancel_process(self):
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.cancelled = True
            self.cancel_button.setEnabled(False)
            self.status.setText("Cancelling — partial artifacts will be retained")
            self.process.terminate()
            self.kill_timer.start(3000)

    def kill_process(self):
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()

    def closeEvent(self, event):
        if self.process.state() == QProcess.ProcessState.NotRunning:
            event.accept()
            return
        answer = QMessageBox.question(
            self,
            "Job still running",
            "Cancel the job and close after it stops? Partial artifacts will remain.\n"
            "Choose No to keep this window open and wait.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        event.ignore()
        if answer == QMessageBox.StandardButton.Yes:
            self.close_after_finish = True
            self.cancel_process()


def main():
    from .decision_gui import DecisionWindow

    app = QApplication.instance() or QApplication(sys.argv)
    window = DecisionWindow()
    window.show()
    return app.exec()
