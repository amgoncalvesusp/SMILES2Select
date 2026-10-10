"""Optional Torch/safetensors research tests; real Laya smoke evidence is separate.

Unit contracts use explicit tiny stubs. The opt-in real integration also requires
Transformers and the verified local model assets.
"""

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from benchmarks import laya_backend as module  # noqa: E402
from benchmarks.laya_backend import (  # noqa: E402
    BinaryQuestion,
    LayaBackend,
    validate_indices,
    verify_assets,
)


@pytest.mark.parametrize("values", [("", "yes", "no"), (None, "yes", "no"),
                                     ("Question", "same", "same"), ("Question", 1, "no")])
def test_question_requires_distinct_nonempty_text(values):
    with pytest.raises(ValueError):
        BinaryQuestion(*values)


def test_binary_choice_is_ordered_positive_first():
    question = BinaryQuestion("Active?", "positive", "negative")
    assert question.internal() == {"t": "choice", "ins": "Active?",
                                   "crit": {"A": "positive", "B": "negative"}}


@pytest.mark.parametrize("indices", [[], [-1], [3], [0, 0], [True], [0.5], [[0]], None])
def test_invalid_training_indices_fail(indices):
    with pytest.raises(ValueError):
        validate_indices(indices, 3)


def test_indices_preserve_explicit_order():
    assert validate_indices([2, 0], 3).tolist() == [2, 0]


def test_manifest_requires_fixed_revisions_and_inventory(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"code_revision": "main"}), encoding="utf-8")
    with pytest.raises(ValueError):
        verify_assets(path)


class TokenizerStub:
    mask_token, mask_token_id, cls_token_id, sep_token_id, pad_token_id = "[MASK]", 3, 1, 2, 0

    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(10, 10 + len(text.split())))}


class HeadStub(torch.nn.Module):
    """Deliberately tiny unit fixture, not the Laya architecture."""

    def __init__(self):
        super().__init__()
        self.scorer = torch.nn.Linear(2, 2)
        self.act_head = torch.nn.Linear(2, 2)

    def forward(self, input_ids, **kwargs):
        return self.scorer(input_ids[:, 0]), self.act_head(input_ids[:, 0])


class EncoderStub(torch.nn.Module):
    config = SimpleNamespace(hidden_size=2)

    def forward(self, input_ids, attention_mask):
        hidden = torch.stack((input_ids.float() / 10, -input_ids.float() / 10), -1)
        return SimpleNamespace(last_hidden_state=hidden)


def build_sequence_stub(tok, state, question, **kwargs):
    words = len(state.split())
    used = min(words, 505)
    return ([1, 3, 10, 3, 11, 2] + [12] * used + [2], [1, 3],
            {"options_distinct": 2}, {"state_tokens_dropped": words - used})


def collate_stub(groups, pad_id):
    rows = [row for group in groups for row in group]
    length = max(len(row["ids"]) for row in rows)
    ids = torch.tensor([row["ids"] + [pad_id] * (length-len(row["ids"])) for row in rows])
    return {"input_ids": ids, "attention_mask": ids.ne(pad_id).long(),
            "marker_pos": torch.tensor([row["markers"] for row in rows]),
            "marker_mask": torch.ones(len(rows), 2, dtype=torch.bool),
            "qtype": torch.zeros(len(rows), dtype=torch.long)}


@pytest.fixture
def backend():
    common = SimpleNamespace(build_sequence=build_sequence_stub, collate_items=collate_stub)
    return LayaBackend(common, EncoderStub(), HeadStub(), TokenizerStub(), {"signature": "stub"})


@pytest.mark.parametrize("states", [[], None, [""], [None], ["bad [MASK] input"]])
def test_invalid_state_inputs_fail(backend, states):
    with pytest.raises(ValueError):
        backend.tokenize(states, BinaryQuestion("Active?", "yes", "no"))


@pytest.mark.parametrize("field", ["state", "instruction", "option"])
def test_any_truncation_is_rejected(backend, field):
    long = "word " * 600
    question = BinaryQuestion(long if field == "instruction" else "Active?",
                              long if field == "option" else "yes", "no")
    with pytest.raises(ValueError, match="budget|truncat"):
        backend.tokenize([long if field == "state" else "MW 100"], question)


def test_offline_cache_predictions_and_training_preserve_encoder(tmp_path, backend):
    question = BinaryQuestion("Active?", "yes", "no")
    states = ["MW 100", "MW 200", "MW 300", "MW 400"]
    raw = backend.predict_logits(states, question, batch_size=2)
    cache = backend.encode_cache(states, question, tmp_path / "cache",
                                 row_ids=["α", "b", "c", "d"], batch_size=2)
    assert cache.row_ids == ("α", "b", "c", "d")
    assert raw.shape == (4, 2)
    np.testing.assert_allclose(raw, backend.predict_cached(cache), rtol=1e-6)
    initial = {key: value.clone() for key, value in backend.encoder.state_dict().items()}
    history = backend.train_head(cache, np.array([1, 0, 1, 0]), indices=[0, 1, 2, 3],
                                 epochs=2, learning_rate=0.01, seed=3, batch_size=2)
    assert len(history) == 2 and all(np.isfinite(row["train_loss"]) for row in history)
    assert all(torch.equal(initial[key], value) for key, value in backend.encoder.state_dict().items())
    assert all(not param.requires_grad for param in backend.encoder.parameters())
    assert all(not param.requires_grad for param in backend.head.act_head.parameters())
    trained = backend.predict_cached(cache)
    assert not np.allclose(raw, trained)
    backend.reset_head()
    np.testing.assert_allclose(raw, backend.predict_cached(cache), rtol=1e-6)
    path = tmp_path / "head.safetensors"
    receipt = backend.save_head(path)
    assert path.is_file() and len(receipt["sha256"]) == 64


@pytest.mark.parametrize("labels", [[1], [1, np.nan], [1, 2], [0, 0]])
def test_invalid_training_labels_fail(tmp_path, backend, labels):
    cache = backend.encode_cache(["one", "two"], BinaryQuestion("Active?", "yes", "no"),
                                 tmp_path / "cache", row_ids=["a", "b"])
    with pytest.raises(ValueError):
        backend.train_head(cache, labels, indices=[0, 1])


def test_cache_identity_and_directory_are_protected(tmp_path, backend):
    question = BinaryQuestion("Active?", "yes", "no")
    with pytest.raises(ValueError):
        backend.encode_cache(["one", "two"], question, tmp_path / "new", row_ids=["a", "a"])
    with pytest.raises(FileExistsError):
        backend.encode_cache(["one"], question, tmp_path, row_ids=["a"])


def test_cache_corruption_fails_before_use(tmp_path, backend):
    cache = backend.encode_cache(["one"], BinaryQuestion("Active?", "yes", "no"),
                                 tmp_path / "cache", row_ids=["a"])
    (cache.root / "hidden.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        backend.predict_cached(cache)


def test_past_deadline_stops_before_head_changes(tmp_path, backend):
    cache = backend.encode_cache(["one", "two"], BinaryQuestion("Active?", "yes", "no"),
                                 tmp_path / "cache", row_ids=["a", "b"])
    before = backend.predict_cached(cache)
    with pytest.raises(TimeoutError, match="deadline"):
        backend.train_head(cache, [0, 1], indices=[0, 1], deadline_monotonic=time.monotonic()-1)
    np.testing.assert_array_equal(before, backend.predict_cached(cache))


def test_continuation_keeps_optimizer_and_checkpoint_restore(tmp_path, backend):
    cache = backend.encode_cache(["one", "two"], BinaryQuestion("Active?", "yes", "no"),
                                 tmp_path / "cache", row_ids=["a", "b"])
    backend.train_head(cache, [0, 1], indices=[0, 1], epochs=1, seed=3)
    before = backend.predict_cached(cache)
    checkpoint = tmp_path / "selected.safetensors"
    backend.save_head(checkpoint)
    history = backend.train_head(cache, [0, 1], indices=[0, 1], epochs=2, seed=3, reset=False)
    assert [row["epoch"] for row in history] == [2, 3]
    backend.load_head(checkpoint)
    np.testing.assert_array_equal(before, backend.predict_cached(cache))
    with pytest.raises(FileExistsError):
        backend.save_head(checkpoint)
    receipt_path = checkpoint.with_suffix(".json")
    receipt = json.loads(receipt_path.read_text("utf-8"))
    receipt_path.write_text(json.dumps({**receipt, "sha256": "invalid"}), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        backend.load_head(checkpoint)


@pytest.fixture
def asset_manifest(tmp_path, monkeypatch):
    """Synthetic bytes test integrity boundaries only, never loaded as a real model."""
    hashes = {}
    for name in module.REQUIRED_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("unit fixture\n", encoding="utf-8")
        hashes[name] = module.sha256(path)
    monkeypatch.setattr(module, "COMMON_SHA256", module.hashlib.sha256(b"unit fixture\n").hexdigest())
    monkeypatch.setattr(module, "MODEL_SHA256", {key: value for key, value in hashes.items()
                                               if key.startswith("model/")}, raising=False)
    manifest = tmp_path / "assets.json"
    manifest.write_text(json.dumps({"code_revision": module.CODE_REVISION,
        "model_revision": module.MODEL_REVISION, "files": hashes}), encoding="utf-8")
    return manifest


def test_verified_asset_inventory_is_recorded(asset_manifest):
    assets = verify_assets(asset_manifest)
    assert assets["signature"] == module.sha256(asset_manifest)
    assert set(assets["files"]) == module.REQUIRED_FILES


@pytest.mark.parametrize("name", ["model/model.safetensors", "upstream/laya/common.py"])
def test_rewriting_receipt_cannot_disguise_changed_pinned_assets(asset_manifest, name):
    path = asset_manifest.parent / name
    path.write_text("changed", encoding="utf-8")
    receipt = json.loads(asset_manifest.read_text("utf-8"))
    receipt["files"][name] = module.sha256(path)
    asset_manifest.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="pinned|reviewed"):
        verify_assets(asset_manifest)


def test_asset_checksum_and_malformed_digest_fail(asset_manifest):
    path = asset_manifest.parent / "model/model.safetensors"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        verify_assets(asset_manifest)
    receipt = json.loads(asset_manifest.read_text("utf-8"))
    receipt["files"]["model/model.safetensors"] = "bad"
    asset_manifest.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        verify_assets(asset_manifest)


@pytest.mark.parametrize("batch_size", [0, -1, True, 0.5])
def test_batch_size_rejects_invalid_values(backend, batch_size):
    with pytest.raises(ValueError):
        backend.predict_logits(["MW 100"], BinaryQuestion("Active?", "yes", "no"),
                               batch_size=batch_size)


def test_nonfinite_model_output_is_rejected(backend, monkeypatch):
    monkeypatch.setattr(backend.head, "forward", lambda **_: (torch.full((1, 2), float("nan")), None))
    with pytest.raises(ValueError, match="logits"):
        backend.predict_logits(["MW 100"], BinaryQuestion("Active?", "yes", "no"))


@pytest.mark.parametrize("field,value", [("assets", "other"), ("shape", [2, 10, 999])])
def test_cache_contract_cannot_change(tmp_path, backend, field, value):
    cache = backend.encode_cache(["one", "two"], BinaryQuestion("Active?", "yes", "no"),
                                 tmp_path / "cache", row_ids=["a", "b"])
    metadata = json.loads((cache.root / "cache.json").read_text("utf-8"))
    (cache.root / "cache.json").write_text(json.dumps({**metadata, field: value}), encoding="utf-8")
    with pytest.raises(ValueError):
        backend.predict_cached(cache)


@pytest.mark.integration
def test_real_pinned_head_matches_official_forward_when_requested():
    configured = os.environ.get("S2S_LAYA_ASSET_MANIFEST")
    if not configured:
        pytest.skip("Set S2S_LAYA_ASSET_MANIFEST for the actual pinned offline integration test")
    torch.set_num_threads(4)
    real = LayaBackend.load(Path(configured))
    question = BinaryQuestion("Active against human AChE at IC50 <=1000 nM?", "active", "inactive")
    states = ["MW=310 LogP=2.1 HBD=1 HBA=4", "MW=180 LogP=1.0 HBD=2 HBA=3"]
    raw = real.predict_logits(states, question, batch_size=2)
    reference = real.common.DecisionModel(real.encoder, head_layers=2, n_act=2, no_init=True)
    reference.load_state_dict({**{"encoder."+key: value for key, value in real.encoder.state_dict().items()},
                               **real.head.state_dict()}, strict=True)
    reference.eval()
    with torch.no_grad():
        official, _ = reference(**real._batch(real.tokenize(states, question)))
    np.testing.assert_allclose(raw, official.numpy(), rtol=1e-5, atol=1e-5)
    assert sum(p.numel() for p in real.head.parameters() if p.requires_grad) == 26248193
    assert sum(p.numel() for p in real.encoder.parameters()) == 394781696
