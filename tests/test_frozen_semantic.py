from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from loghound_agent_debug.features.frozen_semantic import (
    FrozenTextEncoder, chunk_ids, projection, replace_text_features,
)


def test_chunking_covers_every_token_including_tail():
    ids = list(range(1021))
    chunks = chunk_ids(ids)
    assert [value for chunk in chunks for value in chunk] == ids
    assert all(len(chunk) <= 254 for chunk in chunks)
    assert len(chunks[-1]) == 5 and chunk_ids([]) == []
    with pytest.raises(ValueError):
        chunk_ids(ids, 0)


def test_projection_is_fixed_and_orthonormal():
    a = projection()
    assert a.shape == (384, 128)
    assert np.array_equal(a, projection())
    np.testing.assert_allclose(a.T @ a, np.eye(128), atol=1e-6)


def test_replacement_keeps_numeric_features_and_dimensions():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(3, 521)).astype(np.float32)
    before = x.copy()
    vectors = rng.normal(size=(8, 128)).astype(np.float32)
    result, failure = replace_text_features(x, vectors)
    assert np.array_equal(x, before)
    assert np.array_equal(result[:, 512:], x[:, 512:])
    assert np.array_equal(result[:, :128], vectors[2::2])
    assert np.array_equal(result[:, 256:384], vectors[2::2] * vectors[1])
    assert np.array_equal(failure, vectors[:2].flatten())
    with pytest.raises(ValueError):
        replace_text_features(x, vectors[:-1])
    vectors[0, 0] = np.nan
    with pytest.raises(ValueError):
        replace_text_features(x, vectors)


class ToyTokenizer:
    def __call__(self, texts, **kwargs):
        assert kwargs["truncation"] is False
        return {"input_ids": [[ord(c) for c in text] for text in texts]}

    def build_inputs_with_special_tokens(self, ids):
        return [1, *ids, 2]

    def pad(self, inputs, **kwargs):
        width = max(len(row["input_ids"]) for row in inputs)
        ids = [row["input_ids"] + [0] * (width - len(row["input_ids"])) for row in inputs]
        masks = [[1] * len(row["input_ids"]) + [0] * (width - len(row["input_ids"])) for row in inputs]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}


class ToyModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, input_ids, attention_mask):
        assert not torch.is_grad_enabled()
        self.calls += 1
        hidden = torch.sin(input_ids.float()[..., None] * torch.arange(1, 385)[None, None, :] / 100)
        return SimpleNamespace(last_hidden_state=hidden)


def toy_encoder():
    encoder = FrozenTextEncoder.__new__(FrozenTextEncoder)
    encoder.tokenizer = ToyTokenizer()
    encoder.model = ToyModel()
    encoder.device = "cpu"
    encoder.projection = projection()
    encoder.identity = {"test_encoder": 1}
    return encoder


def test_encoder_tail_empty_padding_and_deduplicated_resume(tmp_path: Path):
    encoder = toy_encoder()
    texts = ["a" * 254, "a" * 254 + "zzzz", "", "xy", "xy"]
    vectors, stats = encoder.encode(texts, tmp_path / "cache", batch_size=3)
    assert not np.allclose(vectors[0], vectors[1])
    assert not vectors[2].any()
    assert np.array_equal(vectors[3], vectors[4])
    np.testing.assert_allclose(np.linalg.norm(vectors[[0, 1, 3]], axis=1), 1, atol=1e-6)
    # Padding must not change a sentence when it is batched with longer inputs.
    single, _ = encoder.encode(["xy"], tmp_path / "single", batch_size=3)
    np.testing.assert_allclose(vectors[3], single[0], atol=1e-6)
    before = encoder.model.calls
    repeated, cached = encoder.encode(texts, tmp_path / "cache", batch_size=3)
    assert encoder.model.calls == before and cached["new_texts"] == 0
    assert np.array_equal(vectors, repeated)
    assert stats["chunks"] == 4 and stats["content_tokens"] == 514
    encoder.identity["test_encoder"] = 2
    with pytest.raises(ValueError, match="identity"):
        encoder.encode(texts, tmp_path / "cache", batch_size=3)
