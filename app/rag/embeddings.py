"""Pluggable embedding backends.

* ``fastembed`` (default in Docker): local multilingual ONNX model, no API key, ~220 MB.
* ``openai``: text-embedding-3-small via API.
* ``hash``: deterministic feature-hashing embedder (tokens + char n-grams). Zero
  dependencies, used in tests and as an offline fallback. Quality is lower, but the hybrid
  retriever still has BM25 to lean on.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Protocol

import numpy as np

from app.config import Settings
from app.rag.text import tokenize

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def _l2(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class HashEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hash-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        tokens = tokenize(text)
        feats = tokens + [t[i:i + 4] for t in tokens if len(t) > 5 for i in range(len(t) - 3)]
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 8) % 2 else -1.0
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _l2(np.stack([self._vec(t) for t in texts])) if texts else np.zeros((0, self.dim))

    def embed_query(self, text: str) -> np.ndarray:
        return _l2(self._vec(text)[None, :])[0]


class FastEmbedEmbedder:
    def __init__(self, model_name: str, cache_dir: str | None = None):
        from fastembed import TextEmbedding  # lazy: heavy import

        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self.name = model_name
        self.dim = len(next(iter(self._model.embed(["probe"]))))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _l2(np.array(list(self._model.embed(texts, batch_size=32)), dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return _l2(np.array(list(self._model.embed([text])), dtype=np.float32))[0]


class OpenAIEmbedder:
    def __init__(self, model: str, api_key: str | None):
        from langchain_openai import OpenAIEmbeddings

        self._client = OpenAIEmbeddings(model=model, api_key=api_key)
        self.name = model
        self.dim = len(self._client.embed_query("probe"))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _l2(np.array(self._client.embed_documents(texts), dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return _l2(np.array([self._client.embed_query(text)], dtype=np.float32))[0]


def build_embedder(settings: Settings) -> Embedder:
    backend = settings.embedding_backend
    try:
        if backend == "fastembed":
            return FastEmbedEmbedder(settings.fastembed_model, settings.fastembed_cache_dir)
        if backend == "openai":
            return OpenAIEmbedder(settings.openai_embedding_model, settings.openai_api_key)
    except Exception:  # noqa: BLE001
        logger.exception("embedder_init_failed_falling_back_to_hash", extra={"fields": {"backend": backend}})
    return HashEmbedder()
