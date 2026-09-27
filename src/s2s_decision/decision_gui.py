"""Guided local selection: inspect a proposal before explicitly exporting it."""

import codecs
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
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
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from smiles2select.gui.worker_launch import worker_command


class DecisionWindow(QMainWindow):
    """QProcess keeps model loading and chemistry outside the GUI event loop."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("S2S-Decision — Comparar e selecionar")
        self.resize(1100, 850)
        self.workspace = TemporaryDirectory(prefix="s2s-decision-", ignore_cleanup_errors=True)
        self.models = []
        self.preview_dir = None
        self.cancelled = False
        self.close_after_finish = False
        self.operation = None
        self.operation_output = None
        self.studio = None
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.process_error)
        self.kill_timer = QTimer(self)
        self.kill_timer.setSingleShot(True)
        self.kill_timer.timeout.connect(self.process.kill)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        note = QLabel(
            "Compare sua seleção com uma proposta antes de exportar. Modelos são "
            "experimentais; pontuação não comprova atividade nem probabilidade de avanço."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.editor = QWidget()
        form = QFormLayout(self.editor)
        self.input_path = QLineEdit()
        form.addRow(
            "Biblioteca do SMILES2Select", self.path_row(self.input_path, self.browse_input)
        )
        self.model_root = QLineEdit()
        root_row = self.path_row(self.model_root, self.browse_models)
        self.scan_button = QPushButton("Buscar modelos")
        root_row.layout().addWidget(self.scan_button)
        self.scan_button.clicked.connect(self.scan_models)
        form.addRow("Pasta de modelos", root_row)
        self.target_filter = QLineEdit()
        self.target_filter.setPlaceholderText("Opcional: identificador exato do alvo")
        self.endpoint_filter = QComboBox()
        self.endpoint_filter.addItems(["Todos", "Ki", "Kd", "IC50", "EC50"])
        filters = QWidget()
        filters_layout = QHBoxLayout(filters)
        filters_layout.setContentsMargins(0, 0, 0, 0)
        filters_layout.addWidget(self.target_filter)
        filters_layout.addWidget(self.endpoint_filter)
        form.addRow("Alvo e medida experimental", filters)
        self.model_choice = QComboBox()
        form.addRow("Método da proposta", self.model_choice)
        self.n = QSpinBox()
        self.n.setRange(1, 1000000)
        self.n.setValue(50)
        form.addRow("Quantidade desejada", self.n)
        self.max_per_scaffold = QSpinBox()
        self.max_per_scaffold.setRange(0, 1000000)
        self.max_per_scaffold.setSpecialValueText("Sem limite")
        form.addRow("Máximo por scaffold", self.max_per_scaffold)
        self.min_scaffolds = QSpinBox()
        self.min_scaffolds.setRange(0, 1000000)
        self.min_scaffolds.setSpecialValueText("Sem mínimo")
        form.addRow("Mínimo de scaffolds", self.min_scaffolds)
        self.pins = QLineEdit()
        self.exclude = QLineEdit()
        for label, field in (("Fixar moléculas", self.pins), ("Excluir moléculas", self.exclude)):
            field.setPlaceholderText("IDs internos da tabela, separados por vírgula; opcional")
            form.addRow(label, field)
        layout.addWidget(self.editor)
        buttons = QHBoxLayout()
        self.preview_button = QPushButton("Gerar prévia")
        self.export_button = QPushButton("Exportar cesta…")
        self.export_button.setEnabled(False)
        self.comparison_button = QPushButton("Comparação completa (CSV)")
        self.comparison_button.setEnabled(False)
        self.cancel_button = QPushButton("Cancelar")
        self.cancel_button.setEnabled(False)
        self.studio_button = QPushButton("Treinamento avançado…")
        for button in (
            self.preview_button,
            self.export_button,
            self.comparison_button,
            self.cancel_button,
            self.studio_button,
        ):
            buttons.addWidget(button)
        self.preview_button.clicked.connect(self.preview)
        self.export_button.clicked.connect(self.browse_export)
        self.comparison_button.clicked.connect(self.open_comparison)
        self.cancel_button.clicked.connect(self.cancel)
        self.studio_button.clicked.connect(self.open_studio)
        layout.addLayout(buttons)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)
        self.status = QLabel("Importe uma biblioteca e escolha o método da proposta.")
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.status.setWordWrap(True)
        self.summary.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        layout.addWidget(self.summary)
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        layout.addWidget(self.table, 1)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setMaximumHeight(110)
        layout.addWidget(self.log)
        self.log.hide()
        details = QPushButton("Detalhes técnicos")
        details.setCheckable(True)
        details.toggled.connect(self.log.setVisible)
        layout.addWidget(details)
        for field in (self.input_path, self.model_root, self.pins, self.exclude):
            field.textChanged.connect(self.invalidate_preview)
        for field in (self.n, self.max_per_scaffold, self.min_scaffolds):
            field.valueChanged.connect(self.invalidate_preview)
        self.model_choice.currentIndexChanged.connect(self.invalidate_preview)
        self.target_filter.textChanged.connect(self.filter_models)
        self.endpoint_filter.currentTextChanged.connect(self.filter_models)
        self.filter_models()

    @staticmethod
    def path_row(field, callback):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(field)
        button = QPushButton("Procurar…")
        button.clicked.connect(callback)
        layout.addWidget(button)
        return row

    def browse_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Abrir biblioteca do SMILES2Select",
            "",
            "Bibliotecas (*.sqlite *.db *.xlsx *.csv *.tsv *.parquet);;Todos (*)",
        )
        if path:
            self.input_path.setText(path)

    def browse_models(self):
        path = QFileDialog.getExistingDirectory(self, "Pasta com modelos treinados")
        if path:
            self.model_root.setText(path)

    def invalidate_preview(self, *_):
        if self.preview_dir is not None:
            self.status.setText("Configuração alterada. Gere outra prévia antes de exportar.")
        self.preview_dir = None
        self.export_button.setEnabled(False)
        self.comparison_button.setEnabled(False)

    def populate_models(self, models):
        if not isinstance(models, list):
            raise ValueError("Lista de modelos inválida.")
        self.models = models
        self.filter_models()

    def filter_models(self, *_):
        self.model_choice.clear()
        self.model_choice.addItem("Escolha um método…", None)
        self.model_choice.addItem(
            "Somente seleção química (QED) — sem previsão de atividade", "chemical"
        )
        target, endpoint = self.target_filter.text().strip(), self.endpoint_filter.currentText()
        for model in self.models:
            if target and model.get("target_id") != target:
                continue
            if endpoint != "Todos" and model.get("endpoint") != endpoint:
                continue
            layout = {
                "scalar": "descritores + contexto",
                "scalar_fingerprint": "descritores + contexto + Morgan",
                "tiny_branches": "descritores + contexto + Morgan",
            }.get(model.get("input_layout"), "entradas não informadas")
            label = (
                f"{model.get('target_id')} · {model.get('endpoint')} · "
                f"{model.get('estimator')} · pActivity ≥ {model.get('threshold', '?')} · "
                f"{layout} · {model.get('name')}"
            )
            self.model_choice.addItem(label, model["path"])
            item = self.model_choice.model().item(self.model_choice.count() - 1)
            if not model.get("compatible"):
                item.setEnabled(False)
                item.setToolTip(str(model.get("reason") or "Modelo incompatível"))
        self.invalidate_preview()

    def scan_models(self):
        root = Path(self.model_root.text().strip())
        if not self.model_root.text().strip() or not root.is_dir():
            self.status.setText("Escolha uma pasta de modelos existente.")
            return
        output = Path(self.workspace.name) / f"models-{uuid4().hex}.json"
        self.start("models", output, ["--input", str(root)])

    def preview(self):
        source = self.input_path.text().strip()
        method = self.model_choice.currentData()
        if not source or not Path(source).exists():
            self.status.setText("Escolha uma biblioteca existente.")
            return
        if method is None:
            self.status.setText("Escolha um modelo ou somente seleção química.")
            return
        args = ["--input", source, "--n", str(self.n.value())]
        if method != "chemical":
            args += ["--model", method]
        for name, field in (
            ("max-per-scaffold", self.max_per_scaffold),
            ("min-scaffolds", self.min_scaffolds),
        ):
            if field.value():
                args += [f"--{name}", str(field.value())]
        try:
            for name, field in (("pins", self.pins), ("exclude", self.exclude)):
                ids = [str(int(value)) for value in field.text().replace(",", " ").split()]
                if ids:
                    args += [f"--{name}", *ids]
        except ValueError:
            self.status.setText("IDs devem ser números inteiros, separados por vírgula ou espaço.")
            return
        self.invalidate_preview()
        self.start("preview", Path(self.workspace.name) / f"preview-{uuid4().hex}", args)

    def browse_export(self):
        destination, _ = QFileDialog.getSaveFileName(
            self, "Nome de uma nova pasta para a cesta", "cesta-s2s", "Pasta (*)"
        )
        if destination:
            self.export_to(Path(destination))

    def export_to(self, destination):
        if self.preview_dir is None:
            self.status.setText("Gere e examine uma prévia antes de exportar.")
            return
        destination = Path(destination)
        if destination.exists():
            self.status.setText("Destino já existe. Escolha uma nova pasta.")
            return
        self.start("adopt", destination, ["--input", str(self.preview_dir)])

    def open_comparison(self):
        if self.preview_dir is None:
            return
        path = self.preview_dir / "comparison.csv"
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve()))):
            self.status.setText(f"Não foi possível abrir a comparação. Arquivo: {path}")

    def start(self, operation, output, args):
        if self.process.state() != QProcess.ProcessState.NotRunning:
            return
        self.operation, self.operation_output = operation, output
        self.cancelled = False
        self.decoder.reset()
        self.set_running(True)
        self.status.setText(
            {
                "models": "Verificando modelos…",
                "preview": "Preparando proposta…",
                "adopt": "Exportando cesta…",
            }[operation]
        )
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        environment.insert("PYTHONIOENCODING", "utf-8")
        self.process.setProcessEnvironment(environment)
        program, worker_args = worker_command(
            args=[operation, *args, "--output", str(output)]
        )
        self.process.start(program, worker_args)

    def set_running(self, running):
        self.editor.setEnabled(not running)
        self.preview_button.setEnabled(not running)
        self.export_button.setEnabled(not running and self.preview_dir is not None)
        self.comparison_button.setEnabled(not running and self.preview_dir is not None)
        self.cancel_button.setEnabled(running)
        self.progress.setRange(0, 0 if running else 1)
        if not running:
            self.progress.setValue(0)

    def read_output(self):
        chunk = self.decoder.decode(bytes(self.process.readAllStandardOutput()))
        if chunk:
            self.log.appendPlainText(chunk[-16000:])

    def process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.set_running(False)
            self.status.setText("Não foi possível iniciar a operação.")
            self.log.appendPlainText(self.process.errorString())

    def finished(self, code, status):
        self.kill_timer.stop()
        self.read_output()
        self.set_running(False)
        if self.cancelled:
            self.status.setText("Operação interrompida. Verifique o destino antes de reutilizá-lo.")
        elif code != 0 or status != QProcess.ExitStatus.NormalExit:
            self.status.setText("Operação falhou. Consulte os detalhes abaixo.")
        else:
            try:
                if self.operation == "models":
                    report = json.loads(self.operation_output.read_text(encoding="utf-8"))
                    self.populate_models(report["models"])
                    self.status.setText("Busca concluída. Escolha modelo ou seleção química.")
                elif self.operation == "preview":
                    self.load_preview(self.operation_output)
                else:
                    self.status.setText(f"Cesta exportada: {self.operation_output}")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.invalidate_preview()
                self.status.setText("Resultado inválido. Consulte os detalhes abaixo.")
                self.log.appendPlainText(str(exc))
        if self.close_after_finish:
            self.close()

    def load_preview(self, directory):
        summary = json.loads((directory / "comparison.json").read_text(encoding="utf-8"))
        original = summary.get("original_count")
        calibration = summary.get("calibration_status", "unknown")
        calibrated = calibration == "fitted_held_out_not_prospectively_validated"
        model = summary.get("model") or {}
        text = f"Cesta original: {original if original is not None else 'desconhecida'} · Proposta: {summary['proposed_count']}"
        if original is not None:
            text += " · " + " · ".join(
                f"{label}: {len(summary.get(key) or [])}"
                for key, label in (
                    ("retained_ids", "Mantidas"),
                    ("added_ids", "Incluídas"),
                    ("removed_ids", "Retiradas"),
                )
            )
        if model:
            text += (
                f" · Referências de treino: {model.get('train_reference_count', 'não informado')}"
            )
            text += (
                " · Calibração em dados separados; sem validação prospectiva."
                if calibrated
                else " · Calibração não estabelecida."
            )
        else:
            text += " · Ordenação por QED; sem previsão de atividade."
        if summary.get("warnings"):
            translations = {
                "Selection and rule violations are not experimental activity labels.": "Seleção e violações de regras não são medidas experimentais de atividade.",
                "Selected-only scope does not recover all chemically eligible upstream leftovers.": "A proposta usa elegíveis identificados na entrada; não recupera candidatos ausentes ou excluídos anteriormente.",
            }
            text += "\n" + "\n".join(
                translations.get(str(warning), str(warning)) for warning in summary["warnings"]
            )
        self.summary.setText(text)
        self.load_table(directory / "comparison.csv")
        self.preview_dir = directory
        self.export_button.setEnabled(True)
        self.comparison_button.setEnabled(True)
        self.status.setText("Prévia pronta. Examine diferenças e exporte quando desejar.")

    def load_table(self, path):
        columns = [
            ("record_id", "ID interno"),
            ("molecule_id", "ID de origem"),
            ("original_selected", "Original"),
            ("proposed_selected", "Proposta"),
            ("change", "Mudança"),
            ("priority_score", "Pontuação"),
            ("reference_similarity_max", "Proximidade máxima às referências"),
        ]
        translations = {
            "True": "Sim",
            "False": "Não",
            "added": "Incluída",
            "removed": "Retirada",
            "retained": "Mantida",
            "unselected": "Não selecionada",
            "unknown_original": "Original desconhecida",
        }
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = [(key, title) for key, title in columns if key in (reader.fieldnames or [])]
            self.table.setColumnCount(len(columns))
            self.table.setHorizontalHeaderLabels([title for _, title in columns])
            self.table.setRowCount(0)
            # ponytail: cap widget rows; exported comparison keeps every molecule.
            for index, row in enumerate(reader):
                if index == 2000:
                    self.summary.setText(
                        self.summary.text()
                        + "\nTabela mostra primeiras 2.000 moléculas. Use Comparação completa (CSV) para examinar todas antes de exportar."
                    )
                    break
                self.table.insertRow(index)
                for column, (key, _) in enumerate(columns):
                    value = row.get(key, "")
                    if key in {"priority_score", "reference_similarity_max"} and value:
                        value = f"{float(value):.6f}"
                    item = QTableWidgetItem(translations.get(value, value))
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.table.setItem(index, column, item)
        self.table.resizeColumnsToContents()

    def cancel(self):
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.cancelled = True
            self.cancel_button.setEnabled(False)
            self.process.terminate()
            self.kill_timer.start(3000)

    def open_studio(self):
        from .gui import Studio

        if self.studio is None:
            self.studio = Studio()
        self.studio.show()
        self.studio.raise_()

    def closeEvent(self, event):
        if self.process.state() != QProcess.ProcessState.NotRunning:
            event.ignore()
            answer = QMessageBox.question(
                self,
                "Operação em andamento",
                "Cancelar operação e fechar?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer == QMessageBox.StandardButton.Yes:
                self.close_after_finish = True
                self.cancel()
            return
        event.accept()
        self.workspace.cleanup()
