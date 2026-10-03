"""Responsive desktop client for the authoritative S2S-Decision CLI."""

import codecs
import json
import subprocess
import sys
from pathlib import Path

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


def _verify_training_python(executable: str) -> None:
    path = Path(executable)
    if not path.is_file() or (
        getattr(sys, "frozen", False) and path.resolve() == Path(sys.executable).resolve()
    ):
        raise ValueError("Choose an external Python 3.11+ executable for training")
    script = (
        "import importlib.util,json,sys; from importlib import metadata; "
        "print(json.dumps({'python':sys.version_info[:2],"
        "'version':metadata.version('smiles2select'),"
        "'missing':[name for name in "
        "('torch','sklearn','onnx','onnxscript','onnxruntime','skl2onnx') "
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


class Studio(QMainWindow):
    MODES = {
        "Selection": ("prepare", "predict", "select"),
        "Training": ("audit", "dataset", "train", "train-baseline"),
        "Benchmark": ("benchmark", "evaluate"),
    }
    OPTIONS = {
        "threshold": ("audit", "dataset"),
        "fingerprint-bits": ("prepare", "dataset"),
        "target": ("dataset",),
        "endpoint": ("dataset",),
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
    FILE_INPUTS = {"audit", "prepare", "dataset"}
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
        }
        if name in choices:
            field = QComboBox()
            field.addItems(choices[name])
            return field
        if name in {"target", "pins", "exclude"}:
            field = QLineEdit()
            if name in {"pins", "exclude"}:
                field.setPlaceholderText("Optional integer IDs, separated by commas or spaces")
            return field
        if name == "resume":
            return QCheckBox("Resume existing model checkpoint")
        defaults = {"threshold": 6.0, "learning-rate": 0.001, "weight-decay": 0.0001}
        if name in defaults:
            field = QDoubleSpinBox()
            field.setDecimals(6)
            field.setRange(0.000001 if name == "learning-rate" else 0.0, 100.0)
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
        for name, field in self.fields.items():
            relevant = action in self.OPTIONS[name]
            field.setVisible(relevant)
            field.setEnabled(relevant)
            self.labels[name].setVisible(relevant)
        model = action in {"predict", "evaluate"}
        self.model_row.setVisible(model)
        self.model_row.setEnabled(model)
        self.model_label.setVisible(model)
        training = action in {"train", "train-baseline", "benchmark"}
        self.training_runtime_row.setVisible(training)
        self.training_runtime_row.setEnabled(training)
        self.training_runtime_label.setVisible(training)
        self.input_label.setText(
            "Input table file" if action in self.FILE_INPUTS else "Input bundle directory"
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
        }
        self.help_text.setText(messages[action])

    def browse_input(self):
        if self.action.currentText() in self.FILE_INPUTS:
            path, _ = QFileDialog.getOpenFileName(self, "Select input table")
        else:
            path = QFileDialog.getExistingDirectory(self, "Select input bundle directory")
        if path:
            self.input_path.setText(path)

    def browse_model(self):
        path = QFileDialog.getExistingDirectory(self, "Select trained model directory")
        if path:
            self.model_path.setText(path)

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
        source, output = self.input_path.text().strip(), self.output_path.text().strip()
        if not source or not output:
            raise ValueError("Input and output paths are required")
        args = ["-m", "s2s_decision", action, "--input", source, "--output", output]
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
        training = self.action.currentText() in {"train", "train-baseline", "benchmark"}
        runtime_check = False
        try:
            args = self.command_arguments()
            source, output = (
                Path(self.input_path.text().strip()),
                Path(self.output_path.text().strip()),
            )
            file_input = self.action.currentText() in self.FILE_INPUTS
            if not (source.is_file() if file_input else source.is_dir()):
                raise ValueError(
                    "Input must be an existing table file"
                    if file_input
                    else "Input must be an existing bundle directory"
                )
            resume = self.action.currentText() == "train" and self.fields["resume"].isChecked()
            if output.exists() and not resume:
                raise ValueError(
                    "output already exists; choose a new path (train supports --resume)"
                )
            runtime_check = True
            external_python = (self.training_python.text().strip() or None) if training else None
            if external_python:
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
