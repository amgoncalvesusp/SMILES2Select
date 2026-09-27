"""The verified ONNX bytes must be the bytes executed by the runtime."""

import onnxruntime as ort

from s2s_decision.artifacts import read_bundle
from s2s_decision.inference import predict
from s2s_decision.model_catalog import bundled_model_root


def test_inference_loads_verified_model_bytes(monkeypatch):
    package = bundled_model_root() / "Q72547_WT_IC50"
    records = read_bundle(package / "references").records.iloc[:1]
    load_session = ort.InferenceSession

    def verified_bytes_only(model, *args, **kwargs):
        assert isinstance(model, bytes)
        return load_session(model, *args, **kwargs)

    monkeypatch.setattr(ort, "InferenceSession", verified_bytes_only)
    assert predict(records, package).activity_probability.between(0, 1).all()
