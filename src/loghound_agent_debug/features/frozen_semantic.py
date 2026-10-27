"""Frozen, full-text MiniLM features with a data-independent 128-d projection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Callable, Sequence

import numpy as np

from loghound_agent_debug.features.semantic import semantic_document
from loghound_agent_debug.schema import TrajectoryInput

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_FILES = ("config.json", "model.safetensors", "tokenizer.json",
               "tokenizer_config.json", "special_tokens_map.json", "vocab.txt",
               "sentence_bert_config.json")
ENCODING_VERSION = "minilm_fulltext_projection128_v1"
CHUNK_TOKENS = 254


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return (vector / norm if norm > 0 else vector).astype(np.float32)


def projection() -> np.ndarray:
    """Fixed orthonormal columns; never fit to the experimental corpus."""
    matrix = np.random.default_rng(0).standard_normal((384, 128))
    q, r = np.linalg.qr(matrix)
    return (q * np.where(np.diag(r) < 0, -1, 1)).astype(np.float32)


def chunk_ids(ids: Sequence[int], width: int = CHUNK_TOKENS) -> list[list[int]]:
    if width < 1:
        raise ValueError("Chunk width must be positive")
    return [list(ids[i:i + width]) for i in range(0, len(ids), width)]


def item_texts(item: TrajectoryInput) -> list[str]:
    """Same admitted fields and ordering as attention_inputs.build_inputs."""
    failure = "\n".join(
        [f"{t.code}\n{t.error_description}" for t in item.diagnostic_tests]
        + [o.content for o in item.failure_observations]
    )
    return [item.task, failure] + [
        text for step in item.steps
        for text in (semantic_document(step, include_observation=False), step.observation_text)
    ]


def replace_text_features(x: np.ndarray, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep every numeric/position/length feature byte-for-byte unchanged."""
    if x.ndim != 2 or x.shape[1] != 521 or vectors.shape != (2 + 2 * len(x), 128):
        raise ValueError("Unexpected lexical/semantic feature dimensions")
    if not np.isfinite(vectors).all():
        raise ValueError("Nonfinite semantic representation")
    task, failure = vectors[:2]
    decision, observation = vectors[2::2], vectors[3::2]
    result = x.copy()
    result[:, :512] = np.concatenate(
        [decision, observation, decision * failure, decision * task], axis=1)
    return result, np.concatenate([task, failure]).astype(np.float32)


class FrozenTextEncoder:
    """Lazy transformer import keeps lexical-only installs lightweight."""

    def __init__(self, model_dir: Path, device: str = "cpu") -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            model_dir, local_files_only=True, use_safetensors=True,
            attn_implementation="eager").to(device).eval()
        self.model.requires_grad_(False)
        self.device = device
        self.projection = projection()
        self.identity = {
            "version": ENCODING_VERSION, "model": MODEL_ID, "revision": REVISION,
            "chunk_tokens": CHUNK_TOKENS, "device": device,
            "projection_sha256": hashlib.sha256(self.projection.tobytes()).hexdigest(),
            "files": {name: hashlib.sha256((model_dir / name).read_bytes()).hexdigest()
                      for name in MODEL_FILES},
            "torch": torch.__version__,
        }

    def encode(self, texts: Sequence[str], cache: Path, batch_size: int = 32,
               progress: Callable[[dict], None] | None = None) -> tuple[np.ndarray, dict]:
        import torch
        from torch.nn import functional as F

        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        cache.mkdir(parents=True, exist_ok=True)
        identity = {**self.identity, "batch_size": batch_size}
        identity_path = cache / "identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError("Encoder cache identity changed; use a new cache directory")
        identity_path.write_text(json.dumps(identity, indent=2))
        start = time.monotonic()
        unique = dict.fromkeys(texts)
        pending = [text for text in unique if not (cache / (text_key(text) + ".npz")).exists()]
        new_chunks = 0
        # Bounded groups avoid retaining all corpus tokens/hidden states in RAM.
        for offset in range(0, len(pending), 128):
            group = pending[offset:offset + 128]
            token_ids = self.tokenizer(group, add_special_tokens=False,
                                       truncation=False, verbose=False)["input_ids"]
            sums = np.zeros((len(group), 384), dtype=np.float64)
            chunks = [(owner, ids) for owner, tokens in enumerate(token_ids)
                      for ids in chunk_ids(tokens)]
            new_chunks += len(chunks)
            for chunk_offset in range(0, len(chunks), batch_size):
                batch = chunks[chunk_offset:chunk_offset + batch_size]
                inputs = [{"input_ids": self.tokenizer.build_inputs_with_special_tokens(ids)}
                          for _, ids in batch]
                encoded = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                with torch.inference_mode():
                    hidden = self.model(**encoded).last_hidden_state
                    mask = encoded["attention_mask"].unsqueeze(-1)
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                    pooled = F.normalize(pooled, p=2, dim=1).cpu().numpy()
                for (owner, ids), vector in zip(batch, pooled):
                    sums[owner] += len(ids) * vector
            for index, text in enumerate(group):
                full = normalize(sums[index])
                projected = normalize(full @ self.projection)
                path = cache / (text_key(text) + ".npz")
                temp = path.with_suffix(".tmp")
                with temp.open("wb") as handle:
                    np.savez_compressed(handle, vector=projected, full_vector=full,
                                        tokens=len(token_ids[index]),
                                        chunks=len(chunk_ids(token_ids[index])))
                temp.replace(path)
            if progress is not None:
                progress({"encoded_texts": min(offset + 128, len(pending)),
                          "pending_texts": len(pending), "chunks": new_chunks,
                          "seconds": round(time.monotonic() - start, 1)})
        vectors = {}
        tokens = chunks_count = 0
        for text in unique:
            with np.load(cache / (text_key(text) + ".npz"), allow_pickle=False) as z:
                vector = z["vector"]
                if vector.shape != (128,) or not np.isfinite(vector).all():
                    raise ValueError("Invalid cached vector")
                vectors[text] = vector
                tokens += int(z["tokens"])
                chunks_count += int(z["chunks"])
        stats = {"texts": len(texts), "unique_texts": len(unique), "new_texts": len(pending),
                 "content_tokens": tokens, "chunks": chunks_count,
                 "seconds": time.monotonic() - start, "identity": identity,
                 "trainable_encoder_parameters": 0}
        return np.stack([vectors[text] for text in texts]), stats
