"""Research adapter safety and descriptive benchmark checks; no model download."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def module():
    path = Path(__file__).parents[2] / "benchmarks" / "von_assay.py"
    spec = importlib.util.spec_from_file_location("von_assay", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize("description,expected", [
    ("Antagonist activity in receptor reporter assay", "antagonism"),
    ("Agonistic activity measured by transcription", "agonism"),
    ("Displacement of radioligand from receptor", "binding"),
    ("Inhibitory activity at receptor, IC50", "unknown"),
    ("Receptor activity measured as EC50", "unknown"),
    ("Inhibition of enzyme catalytic activity", "enzyme_inhibition"),
    ("Receptor degradation measured by western blot", "degradation"),
    ("Cell viability after compound treatment", "other"),
    ("Binding and agonist activity", "unknown"),
    ("No agonist activity is measured", "unknown"),
    ("Ligand-binding domain assay EC50", "unknown"),
    ("Inhibition of receptor reporter transcription", "unknown"),
    ("Suppression of target immunostaining", "unknown"),
    ("Transactivation of human nuclear receptor in reporter assay", "agonism"),
    ("Activation of nuclear receptor measured by reporter", "agonism"),
    ("Inhibition of receptor mediated transactivation", "unknown"),
    ("Inhibition of agonist-induced receptor activation", "antagonism"),
])
def test_conservative_rules(description, expected):
    assert module().rule_label(description) == expected


def response(mod, **overrides):
    probs = {label: float(label == "binding") for label in mod.CRITERIA}
    answer = SimpleNamespace(choice="binding", probabilities=probs, confidence=1.)
    return SimpleNamespace(answers={"mechanism": answer}, truncation=None,
                           usage=SimpleNamespace(input_tokens=40), **overrides)


def test_response_boundary_and_rounding():
    mod = module()
    value = response(mod)
    assert mod.validate_response(value)["choice"] == "binding"
    value.answers["mechanism"].probabilities = dict.fromkeys(mod.CRITERIA, .1429)
    parsed = mod.validate_response(value)
    assert sum(parsed["probabilities"].values()) == pytest.approx(1.)
    assert parsed["shipped_probabilities"]["binding"] == .1429
    for probs in ({"binding": 1}, dict.fromkeys(mod.CRITERIA, .5),
                  dict.fromkeys(mod.CRITERIA, float("nan"))):
        value.answers["mechanism"].probabilities = probs
        with pytest.raises(ValueError):
            mod.validate_response(value)
    value = response(mod)
    value.answers["mechanism"].choice = "alien"
    with pytest.raises(ValueError):
        mod.validate_response(value)
    value = response(mod)
    value.truncation = {"kept_tokens": 10}
    with pytest.raises(ValueError, match="truncation"):
        mod.validate_response(value)


def snapshot(tmp_path, mod):
    root = tmp_path / "snapshot"
    root.mkdir()
    for name in mod.REQUIRED_FILES:
        (root / name).write_text('{}', encoding="utf-8")
    (root / "marker_calibration.json").write_text('{"independent_options": true}')
    manifest = {"files": {p.name: mod.sha256(p) for p in root.iterdir()}}
    return root, manifest


def test_snapshot_is_complete_and_safe(tmp_path):
    mod = module()
    root, manifest = snapshot(tmp_path, mod)
    assert mod.verify_snapshot(root, manifest) == manifest["files"]
    (root / "unexpected.bin").write_bytes(b"extra")
    with pytest.raises(ValueError, match="inventory"):
        mod.verify_snapshot(root, manifest)
    (root / "unexpected.bin").unlink()
    (root / "config.json").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        mod.verify_snapshot(root, manifest)
    with pytest.raises(ValueError):
        mod.verify_snapshot(root, {"files": {"../escape": "0" * 64}})


def test_reference_contract(tmp_path):
    mod = module()
    path = tmp_path / "reference.jsonl"
    row = {"id": "a1", "description": "Binding", "label": "binding"}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert mod.read_reference(path) == [row]
    path.write_text(json.dumps(row) + "\n" + json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        mod.read_reference(path)
    for bad in ([], [{**row, "label": "bad"}], [{**row, "description": ""}]):
        path.write_text("\n".join(map(json.dumps, bad)), encoding="utf-8")
        with pytest.raises(ValueError):
            mod.read_reference(path)


def test_metrics_macro_support_and_brier():
    mod = module()
    truth = ["binding", "agonism"]
    probs = [{label: float(label == pred) for label in mod.CRITERIA}
             for pred in ["binding", "binding"]]
    result = mod.metrics(truth, ["binding", "binding"], probs)
    assert result["accuracy"] == .5
    assert result["macro_f1_supported"] == pytest.approx(1 / 3)
    assert result["multiclass_brier"] == 1
    assert result["ece_10_equal_width"] == .5
    assert result["supports"]["agonism"] == 1
    assert result["supports"]["degradation"] == 0
    assert result["per_class_precision"]["binding"] == .5
    assert result["per_class_recall"]["binding"] == 1
    assert result["selective_coverage"] == 1


def test_predict_receives_description_only_and_order():
    mod = module()
    calls = []
    backend = SimpleNamespace(evaluate=lambda **kw: calls.append(kw) or response(mod))
    parsed = mod.predict(backend, lambda **kw: kw, "Binding assay", reverse=True)
    assert parsed["choice"] == "binding"
    assert list(calls[0]["questions"]["mechanism"]["criteria"]) == list(mod.CRITERIA)[::-1]
    assert calls[0]["state"] == "Binding assay"


def test_full_runner_fake_backend(tmp_path, monkeypatch):
    mod = module()
    root, manifest = snapshot(tmp_path, mod)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    reference = tmp_path / "reference.jsonl"
    reference.write_text(json.dumps({"id": "a1", "description": "Binding assay", "label": "binding"}))
    backend = SimpleNamespace(evaluate=lambda **kw: response(mod))
    monkeypatch.setattr(mod, "load_backend", lambda _: (backend, lambda **kw: kw))
    output = tmp_path / "results"
    result = mod.run(reference, root, manifest_path, output)
    assert result["von"]["accuracy"] == 1
    assert result["option_order_agreement"] == 1
    assert len((output / "predictions.jsonl").read_text().splitlines()) == 1
    assert json.loads((output / "summary.json").read_text())["n"] == 1
    with pytest.raises(FileExistsError):
        mod.run(reference, root, manifest_path, output)


def test_snapshot_configuration_and_manifest_boundaries(tmp_path):
    mod = module()
    root, manifest = snapshot(tmp_path, mod)
    for name in ("../escape", "C:/escape", "folder\\escape"):
        with pytest.raises(ValueError, match="Unsafe"):
            mod.verify_snapshot(root, {"files": {**manifest["files"], name: "0" * 64}})
    with pytest.raises(ValueError, match="SHA256"):
        mod.verify_snapshot(root, {"files": {**manifest["files"], "config.json": "bad"}})
    calibration = root / "marker_calibration.json"
    calibration.write_text('{"independent_options": false}')
    changed = {"files": {**manifest["files"], calibration.name: mod.sha256(calibration)}}
    with pytest.raises(ValueError, match="independent_options"):
        mod.verify_snapshot(root, changed)
    with pytest.raises(ValueError, match="required"):
        mod.verify_snapshot(root, {})


def test_extra_response_boundaries():
    mod = module()
    for problem in ("schema", "confidence", "argmax", "negative", "bool"):
        value = response(mod)
        answer = value.answers["mechanism"]
        if problem == "schema":
            value.answers = {}
        elif problem == "confidence":
            answer.confidence = float("nan")
        elif problem == "argmax":
            answer.choice = "unknown"
        elif problem == "negative":
            answer.probabilities["binding"] = -1.
        else:
            answer.probabilities["binding"] = True
        with pytest.raises(ValueError):
            mod.validate_response(value)
    with pytest.raises(ValueError):
        mod.metrics([], [], [])
    result = mod.metrics(["unknown"], ["binding"], [response(mod).answers["mechanism"].probabilities])
    assert result["unsafe_unknown_count"] == 1


def test_loader_pins_cpu_offline_and_version(tmp_path, monkeypatch):
    mod = module()
    calls = []
    torch = SimpleNamespace(set_num_threads=lambda n: calls.append(n))
    backend = SimpleNamespace(_get_model=lambda: calls.append("loaded"),
                              chain_runner=None, _independent_options=True)
    von = SimpleNamespace(__version__="1.3.5", choice=lambda **kw: kw)
    module_backend = SimpleNamespace(OptionMarkerBackend=lambda **kw: calls.append(kw) or backend)
    for name, value in {"torch": torch, "von": von,
                        "von.backends.option_marker_backend": module_backend}.items():
        monkeypatch.setitem(sys.modules, name, value)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "VON_CHAINS_DIR",
                 "VON_ON_OVERFLOW", "VON_MAX_STATE_TOKENS"):
        monkeypatch.setenv(name, "test")
    assert mod.load_backend(tmp_path)[0] is backend
    assert calls == [4, {"checkpoint_dir": str(tmp_path.resolve()), "device": "cpu"}, "loaded"]
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["VON_CHAINS_DIR"] == "off"
    assert os.environ["VON_ON_OVERFLOW"] == "refuse"
    backend._independent_options = False
    with pytest.raises(ValueError, match="configuration"):
        mod.load_backend(tmp_path)
    von.__version__ = "1.2"
    with pytest.raises(ValueError, match="1.3.5"):
        mod.load_backend(tmp_path)


def test_failed_run_records_failure_and_cli_dispatch(tmp_path, monkeypatch, capsys):
    mod = module()
    root, manifest = snapshot(tmp_path, mod)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    reference = tmp_path / "reference.jsonl"
    reference.write_text(json.dumps({"id": "x", "description": "Unclear", "label": "unknown"}))

    def fail(_):
        raise RuntimeError("offline snapshot failed")

    monkeypatch.setattr(mod, "load_backend", fail)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="offline"):
        mod.run(reference, root, manifest_path, output)
    assert json.loads((output / "failure.json").read_text())["error_type"] == "RuntimeError"
    assert not (output / "summary.json").exists()
    monkeypatch.setattr(mod, "run", lambda *args: {"ok": len(args) == 4})
    monkeypatch.setattr(sys, "argv", ["von_assay.py", "--reference", str(reference),
                                    "--snapshot", str(root), "--manifest", str(manifest_path),
                                    "--output", str(tmp_path / "new")])
    mod.main()
    assert json.loads(capsys.readouterr().out) == {"ok": True}
