"""Research-only offline Laya backend: frozen encoder, original trainable decision head.

Inference returns raw A-positive/B-negative logits. No checkpoint temperature or
act/escalate score is used. Calibration and split ownership belong to the pilot runner.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

CODE_REVISION = "79f671b235418ff6f2f572aee7709bac30392f51"
MODEL_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
COMMON_SHA256 = "0e4020cc844db80b28e446f3d5fed5a927ae6a8dc4af806fdd334107855211c2"
MODEL_SHA256 = {
    "model/model.safetensors": "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c",
    "model/rl_agent_config.json": "ae287b56bbcf5f8c4f4541ae9dfd00c914c4c48b940b8398c3058af37ba92bbd",
    "model/encoder/config.json": "bf3ab80598fdccf414855a2ce80f22859e4492d06ca8a62ddd1cfb63972f8979",
    "model/tokenizer/tokenizer.json": "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30",
    "model/tokenizer/tokenizer_config.json": "50044de60daaa73df97d262e15a40d4faf0160e7d742df64b377877a1320dd12",
}
REQUIRED_FILES = frozenset({
    "upstream/laya/common.py", "model/model.safetensors", "model/rl_agent_config.json",
    "model/encoder/config.json", "model/tokenizer/tokenizer.json",
    "model/tokenizer/tokenizer_config.json",
})
INPUT_KEYS = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def verify_assets(manifest_path: Path) -> dict:
    """Verify pinned, data-only model assets and the one reviewed executable source file."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if (not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict)
            or manifest.get("code_revision") != CODE_REVISION
            or manifest.get("model_revision") != MODEL_REVISION
            or set(manifest.get("files", {})) != REQUIRED_FILES):
        raise ValueError("Laya assets require the fixed revisions and exact required inventory")
    root = manifest_path.parent
    for name, digest in manifest["files"].items():
        path = root / name
        if (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or path.is_symlink() or not path.resolve().is_relative_to(root)):
            raise ValueError("Unsafe Laya asset path or digest")
        if sha256(path) != digest:
            raise ValueError(f"Laya asset checksum mismatch: {name}")
    if any(manifest["files"][name] != digest for name, digest in MODEL_SHA256.items()):
        raise ValueError("Model assets differ from the pinned upstream checkpoint")
    source = (root / "upstream/laya/common.py").read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(source).hexdigest() != COMMON_SHA256:
        raise ValueError("Executable Laya source is not the reviewed pinned common.py")
    return {**manifest, "root": str(root), "signature": sha256(manifest_path)}


@dataclass(frozen=True)
class BinaryQuestion:
    instructions: str
    positive: str
    negative: str

    def __post_init__(self):
        if (any(not isinstance(value, str) or not value.strip()
                for value in (self.instructions, self.positive, self.negative))
                or self.positive == self.negative):
            raise ValueError("Binary question requires distinct, nonempty text")

    def internal(self) -> dict:
        return {"t": "choice", "ins": self.instructions,
                "crit": {"A": self.positive, "B": self.negative}}


@dataclass(frozen=True)
class RepresentationCache:
    root: Path
    row_ids: tuple[str, ...]
    signature: str


def validate_indices(indices, size: int) -> np.ndarray:
    raw = np.asarray(indices)
    if (raw.ndim != 1 or not len(raw) or raw.dtype.kind not in "iu"
            or (raw < 0).any() or (raw >= size).any() or len(np.unique(raw)) != len(raw)):
        raise ValueError("Indices must be nonempty unique integer rows inside the cache")
    return raw.astype(np.int64, copy=True)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _check_deadline(deadline):
    if deadline is not None:
        if not np.isfinite(deadline):
            raise ValueError("Training deadline must be finite")
        if time.monotonic() >= deadline:
            raise TimeoutError("Head training reached the prespecified deadline")


def _head_from_model(common, model):
    class CachedEncoder(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=width)

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(last_hidden_state=input_ids)

    head = common.DecisionModel(
        CachedEncoder(model.encoder.config.hidden_size), head_layers=2,
        n_act=model.act_head[-1].out_features, no_init=True,
    )
    head.load_state_dict({key: value for key, value in model.state_dict().items()
                          if not key.startswith("encoder.")}, strict=True)
    return head


class LayaBackend:
    """Load with ``load`` for real inference; dependency injection supports declared unit stubs."""

    def __init__(self, common, encoder, head, tokenizer, assets: dict):
        self.common, self.encoder, self.head, self.tokenizer = common, encoder, head, tokenizer
        self.assets = copy.deepcopy(assets)
        self.encoder.requires_grad_(False).eval()
        self.head.act_head.requires_grad_(False)
        self.head.eval()
        self._initial_head = {key: value.detach().clone() for key, value in head.state_dict().items()}
        self._optimizer = None
        self._epochs = 0

    @classmethod
    def load(cls, manifest_path: Path):
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        assets = verify_assets(manifest_path)
        root = Path(assets["root"])
        spec = importlib.util.spec_from_file_location("_s2s_reviewed_laya_common",
                                                    root / "upstream/laya/common.py")
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load the verified local Laya source")
        common = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(common)
        config = json.loads((root / "model/rl_agent_config.json").read_text("utf-8"))
        if (config.get("max_len") != 512 or config.get("head_max_len") != 192
                or config.get("head_layers") != 2):
            raise ValueError("Laya checkpoint differs from the frozen 512/192 two-layer contract")
        tokenizer = AutoTokenizer.from_pretrained(root / "model/tokenizer", local_files_only=True,
                                                  trust_remote_code=False)
        encoder_config = AutoConfig.from_pretrained(root / "model/encoder", local_files_only=True,
                                                    trust_remote_code=False)
        if encoder_config.model_type != "modernbert":
            raise ValueError("The pinned Laya encoder must be ModernBERT")
        common._apply_rope_config(encoder_config)
        encoder_config.reference_compile = False
        with common._no_init_weights():
            encoder = AutoModel.from_config(encoder_config, attn_implementation="sdpa",
                                            trust_remote_code=False)
        model = common.DecisionModel(encoder, head_layers=2, n_act=2, no_init=True)
        model.load_state_dict(load_file(root / "model/model.safetensors"), strict=True)
        return cls(common, model.encoder, _head_from_model(common, model), tokenizer, assets)

    def tokenize(self, states: list[str], question: BinaryQuestion) -> list[dict]:
        if (not isinstance(states, list) or not states or not isinstance(question, BinaryQuestion)
                or any(not isinstance(state, str) or not state.strip() for state in states)):
            raise ValueError("States must be a nonempty list of nonempty strings")
        all_text = [*states, question.instructions, question.positive, question.negative]
        if any(self.tokenizer.mask_token in text for text in all_text):
            raise ValueError("Literal mask tokens are forbidden in pilot inputs")
        def tokenize(text):
            return self.tokenizer(text, add_special_tokens=False)["input_ids"]
        options = [tokenize(" A: " + question.positive), tokenize(" B: " + question.negative)]
        instruction = tokenize("choice question: " + question.instructions)
        if max(map(len, options)) > 48 or len(instruction) > 192-sum(len(o)+1 for o in options):
            raise ValueError("Question exceeds the untruncated 192-token head budget")
        items = []
        for state in states:
            ids, markers, stats, truncation = self.common.build_sequence(
                self.tokenizer, state, question.internal(), max_len=512, head_max_len=192,
                return_stats=True, return_truncation_stats=True,
            )
            if (truncation["state_tokens_dropped"] or len(ids) > 512 or len(markers) != 2
                    or stats["options_distinct"] != 2):
                raise ValueError("State exceeds token budget or choice was truncated")
            items.append({"ids": ids, "markers": markers, "qtype": 0,
                          "state_tokens_dropped": 0})
        return items

    def _batch(self, items):
        data = self.common.collate_items([[item] for item in items], self.tokenizer.pad_token_id)
        return {key: data[key] for key in INPUT_KEYS}

    def _head_logits(self, hidden, data):
        logits, _unused_act = self.head(input_ids=hidden,
                                       **{key: data[key] for key in INPUT_KEYS if key != "input_ids"})
        if logits.shape != (len(hidden), 2) or not torch.isfinite(logits).all():
            raise ValueError("Laya returned invalid binary raw logits")
        return logits

    def predict_logits(self, states, question, *, batch_size=4) -> np.ndarray:
        _positive_integer(batch_size, "batch_size")
        items = self.tokenize(states, question)
        self.head.eval()
        output = []
        with torch.no_grad():
            for start in range(0, len(items), batch_size):
                data = self._batch(items[start:start+batch_size])
                hidden = self.encoder(input_ids=data["input_ids"],
                                      attention_mask=data["attention_mask"]).last_hidden_state
                output.append(self._head_logits(hidden, data).numpy())
        return np.concatenate(output)

    def encode_cache(self, states, question, output_dir, *, row_ids, batch_size=4):
        from safetensors.torch import save_file

        _positive_integer(batch_size, "batch_size")
        items = self.tokenize(states, question)
        if (len(row_ids) != len(items) or any(not isinstance(row, str) or not row for row in row_ids)
                or len(set(row_ids)) != len(row_ids)):
            raise ValueError("Cache requires unique nonempty row identities")
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=False)
        data = self._batch(items)
        shape = (len(items), data["input_ids"].shape[1], self.encoder.config.hidden_size)
        hidden = np.lib.format.open_memmap(root / "hidden.npy", mode="w+", dtype="float32", shape=shape)
        with torch.no_grad():
            for start in range(0, len(items), batch_size):
                stop = min(len(items), start+batch_size)
                encoded = self.encoder(input_ids=data["input_ids"][start:stop],
                                       attention_mask=data["attention_mask"][start:stop]).last_hidden_state
                if not torch.isfinite(encoded).all():
                    raise ValueError("Encoder produced nonfinite cached representations")
                hidden[start:stop] = encoded.float().numpy()
        hidden.flush()
        del hidden
        save_file(data, root / "inputs.safetensors")
        signature = hashlib.sha256(json.dumps({"assets": self.assets["signature"],
            "states": states, "question": question.internal(), "row_ids": row_ids},
            sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        metadata = {"schema": "s2s-laya-cache/1", "signature": signature,
                    "assets": self.assets["signature"], "row_ids": list(row_ids), "shape": list(shape),
                    "question": question.internal(), "dtype": "float32",
                    "files": {name: sha256(root / name) for name in ("hidden.npy", "inputs.safetensors")}}
        _json(root / "cache.json", metadata)
        return RepresentationCache(root, tuple(row_ids), signature)

    def _read_cache(self, cache):
        from safetensors.torch import load_file

        metadata = json.loads((cache.root / "cache.json").read_text("utf-8"))
        if (metadata.get("assets") != self.assets["signature"]
                or metadata.get("signature") != cache.signature
                or metadata.get("row_ids") != list(cache.row_ids)):
            raise ValueError("Cache model or row identity mismatch")
        for name in ("hidden.npy", "inputs.safetensors"):
            if sha256(cache.root / name) != metadata["files"][name]:
                raise ValueError("Cache checksum mismatch")
        hidden = np.load(cache.root / "hidden.npy", mmap_mode="r", allow_pickle=False)
        if list(hidden.shape) != metadata["shape"] or hidden.dtype != np.float32:
            raise ValueError("Cache hidden shape or dtype mismatch")
        return hidden, load_file(cache.root / "inputs.safetensors")

    def _cached_batches(self, hidden, data, indices, batch_size):
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start+batch_size]
            yield (torch.from_numpy(np.array(hidden[batch], copy=True)),
                   {key: value[batch] for key, value in data.items()}, batch)

    def predict_cached(self, cache, *, indices=None, batch_size=4):
        _positive_integer(batch_size, "batch_size")
        chosen = validate_indices(np.arange(len(cache.row_ids)) if indices is None else indices,
                                  len(cache.row_ids))
        hidden, data = self._read_cache(cache)
        self.head.eval()
        with torch.no_grad():
            output = [self._head_logits(h, batch).numpy()
                      for h, batch, _ in self._cached_batches(hidden, data, chosen, batch_size)]
        return np.concatenate(output)

    def reset_head(self):
        self.head.load_state_dict(self._initial_head, strict=True)
        self.head.eval()
        self._optimizer, self._epochs = None, 0

    def train_head(self, cache, labels, *, indices, epochs=1, learning_rate=1e-4,
                   seed=42, batch_size=4, reset=True, deadline_monotonic=None):
        """Train original type embedding, two transformer layers and scorer using cross-entropy.

        Biological label 1 maps to option A (index 0). Frozen act/escalate parameters
        receive no loss. This is supervised head adaptation, not a reproduction of RLCD.
        ``reset=False`` preserves AdamW state for sequential validation checkpoints.
        """
        _check_deadline(deadline_monotonic)
        for value, name in ((epochs, "epochs"), (batch_size, "batch_size")):
            _positive_integer(value, name)
        chosen = validate_indices(indices, len(cache.row_ids))
        target = np.asarray(labels, dtype=float)
        if (target.shape != (len(cache.row_ids),) or not np.isfinite(target).all()
                or not np.isin(target, [0, 1]).all() or len(np.unique(target[chosen])) != 2):
            raise ValueError("Training requires aligned binary labels and both classes in train rows")
        if not np.isfinite(learning_rate) or learning_rate <= 0 or not isinstance(seed, int):
            raise ValueError("Training requires positive finite learning rate and integer seed")
        hidden, data = self._read_cache(cache)
        if reset:
            self.reset_head()
        if self._optimizer is None:
            self._optimizer = torch.optim.AdamW([p for p in self.head.parameters() if p.requires_grad],
                                                lr=learning_rate, weight_decay=0)
        elif self._optimizer.param_groups[0]["lr"] != learning_rate:
            raise ValueError("Continuing head training must preserve the learning rate")
        history = []
        for _ in range(epochs):
            _check_deadline(deadline_monotonic)
            started, loss_sum = time.perf_counter(), 0.0
            with torch.random.fork_rng():
                torch.manual_seed(seed + self._epochs)
                order = chosen[torch.randperm(len(chosen)).numpy()]
                self.head.train()
                for h, batch, rows in self._cached_batches(hidden, data, order, batch_size):
                    _check_deadline(deadline_monotonic)
                    self._optimizer.zero_grad(set_to_none=True)
                    logits = self._head_logits(h, batch)
                    truth = torch.tensor(1-target[rows], dtype=torch.long)
                    loss = torch.nn.functional.cross_entropy(logits, truth)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.head.parameters(), 1.0, error_if_nonfinite=True)
                    self._optimizer.step()
                    loss_sum += float(loss.detach()) * len(rows)
            _check_deadline(deadline_monotonic)
            self._epochs += 1
            history.append({"epoch": self._epochs, "train_loss": loss_sum/len(chosen),
                            "elapsed_seconds": time.perf_counter()-started})
        self.head.eval()
        return history

    def save_head(self, path):
        from safetensors.torch import save_file

        path = Path(path)
        if path.exists() or path.with_suffix(".json").exists():
            raise FileExistsError("Head checkpoint destination exists")
        save_file({key: value.detach().contiguous() for key, value in self.head.state_dict().items()}, path)
        receipt = {"schema": "s2s-laya-head/1", "sha256": sha256(path),
                   "assets": self.assets["signature"], "epochs": self._epochs,
                   "file": path.name, "adaptation": "supervised original decision head; frozen encoder"}
        _json(path.with_suffix(".json"), receipt)
        return receipt

    def load_head(self, path):
        from safetensors.torch import load_file

        path = Path(path)
        receipt = json.loads(path.with_suffix(".json").read_text("utf-8"))
        if receipt.get("assets") != self.assets["signature"] or receipt.get("sha256") != sha256(path):
            raise ValueError("Head checkpoint model or checksum mismatch")
        self.head.load_state_dict(load_file(path), strict=True)
        self.head.eval()
        self._optimizer, self._epochs = None, int(receipt["epochs"])
