"""Explain task-specific model choices without changing their ranking or evidence."""

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QPlainTextEdit, QVBoxLayout

_FAMILIES = {
    ("tiny", "tiny_branches"): (
        "Tiny — neural network",
        "Combines molecular properties, Morgan fingerprints and reference context in three "
        "neural-network branches. It learns nonlinear combinations of these inputs.",
        "Compare with logistic regression and both gradient-boosting alternatives on the same "
        "target, endpoint, candidate pool and quotas. Use as an experimental alternative when "
        "you want to test whether these combined inputs change useful selections. More model "
        "complexity does not establish better recovery. Its pActivity output, when supported, "
        "does not determine the ranking: the activity score does.",
    ),
    ("logistic", "scalar_fingerprint"): (
        "Logistic regression + Morgan",
        "Uses a linear combination of molecular properties, reference context and Morgan "
        "fingerprint bits, followed by a logistic transformation.",
        "Use as a simple supervised baseline for this task. Compare its basket with Tiny or "
        "gradient boosting before accepting extra complexity. Linear effects cannot represent "
        "every interaction; correlated descriptors and fingerprint collisions affect interpretation.",
    ),
    ("gradient_boosting", "scalar"): (
        "Gradient boosting — properties + context",
        "Combines decision trees over molecular properties and reference context. Morgan bits "
        "are not direct inputs, but reference-context similarities still use fingerprints.",
        "Use to test nonlinear property and reference-context relationships without direct "
        "Morgan-bit inputs. Compare with the Morgan variant to assess its incremental effect. "
        "It can miss structural distinctions that its property/context inputs do not capture.",
    ),
    ("gradient_boosting", "scalar_fingerprint"): (
        "Gradient boosting + Morgan",
        "Combines decision trees over molecular properties, reference context and structural "
        "fingerprints (Morgan bits).",
        "Use to compare nonlinear structural fingerprints and property/context combinations "
        "against the scalar variant. High-dimensional fingerprint inputs can overfit small "
        "datasets; inspect out-of-domain candidates and scaffold concentration.",
    ),
}


def _family(item):
    return _FAMILIES.get((item.get("estimator"), item.get("input_layout")), (
        str(item.get("estimator") or "Model"),
        "See the imported model card for its estimator, input layout and training design.",
        "Confirm task compatibility and independent evaluation before using this package.",
    ))


def _role(item):
    if item.get("origin") != "bundled":
        return "Imported package; qualification must be checked in its model card."
    if item.get("validation_selected"):
        return "Original validation-selected baseline; selection was frozen before test evaluation."
    return "Bundled experimental comparator; not the original validation-selected baseline."


def model_label(item):
    label = _family(item)[0]
    if item.get("origin") == "bundled":
        role = "validation selection" if item.get("validation_selected") else "experimental comparator"
        return f"{label} ({role})"
    return f"{label} — {item.get('name', 'imported package')}"


def model_guidance_text(item):
    """Build guidance from the chosen package, keeping unknown provenance explicit."""
    label, explanation, tips = _family(item)
    task = f"{item.get('target_id') or 'not declared'} / {item.get('endpoint') or 'not declared'}"
    threshold = item.get("threshold")
    threshold_text = f"pActivity >= {threshold:g}" if isinstance(threshold, (int, float)) else "not declared"
    source = "Papyrus++ 05.7" if item.get("origin") == "bundled" else "See imported model card"
    status = "Compatible" if item.get("compatible") else f"Unavailable: {item.get('reason') or 'not verified'}"
    return (
        f"{label}\n{status}\nTask: {task} | active label: {threshold_text}\n"
        f"{_role(item)}\n\nHow it works\n{explanation}\n\nSelection tips\n{tips}\n\n"
        "Choose this model only for its declared target, endpoint and activity threshold. "
        "A different target or endpoint requires its own compatible model. Scores do not "
        "guarantee activity and cannot be compared across different tasks. Calibration on the "
        "training study does not establish calibrated probabilities for a new chemical library.\n\n"
        "Keep the candidate pool, count, pins and scaffold/cluster quotas fixed when comparing "
        "models. Inspect retained, added and removed molecules before Adopt proposal. "
        "Shared quotas can prevent the requested count or minimum core coverage; inspect warnings. "
        "Use reference similarity as context, not proof of activity.\n\n"
        "Create selection applies the native chemical strategy and replaces the current basket. "
        "Preview model selection ranks model scores under shared quotas; the property-strategy "
        "dropdown does not change those scores. Adopt proposal applies that preview in one "
        "undoable action.\n\n"
        f"Package: {item.get('name', 'not declared')}\n"
        f"Inputs: {item.get('input_layout') or 'not declared'}\n"
        f"Calibration: {item.get('calibration_status') or 'not declared'} | "
        f"training references: {item.get('train_reference_count', 'not declared')}\n"
        f"Origin: {item.get('origin', 'external')} | source: {source}\n"
        f"Source SHA-256: {item.get('source_sha256') or 'not declared'}\n"
        f"Quality: {item.get('quality_policy') or 'not declared'} | "
        f"split: {item.get('split_method') or 'not declared'}"
    )


class ModelGuideDialog(QDialog):
    """Scrollable guidance for the currently selected model; no selection side effects."""

    def __init__(self, item, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model guide and selection tips")
        self.resize(680, 580)
        layout = QVBoxLayout(self)
        self.text = QPlainTextEdit(model_guidance_text(item))
        self.text.setReadOnly(True)
        layout.addWidget(self.text)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
