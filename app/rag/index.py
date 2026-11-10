"""Hybrid (BM25 + dense) vector index persisted to disk.

Storage is intentionally simple — a NumPy matrix plus JSONL metadata — because the corpus
is small (hundreds of chunks). The ``HybridIndex`` interface (build / save / load / search)
is the seam to swap in pgvector, Qdrant or OpenSearch without touching the agents.

Retrieval = BM25 (exact terms: product names, error codes like "ADQ 1-06") fused with dense
cosine similarity (semantics, cross-language EN↔PT) using Reciprocal Rank Fusion.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from app.rag.chunking import Chunk
from app.rag.embeddings import Embedder
from app.rag.text import tokenize

logger = logging.getLogger(__name__)
RRF_K = 60


@dataclass
class SearchHit:
    chunk: Chunk
    score: float
    bm25_rank: int | None
    dense_rank: int | None
    dense_score: float


class HybridIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder: Embedder):
        self.chunks = chunks
        self.vectors = vectors
        self.embedder = embedder
        self._bm25 = BM25Okapi([tokenize(c.embedding_text) for c in chunks]) if chunks else None

    # ------------------------------------------------------------------ build / persist
    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> HybridIndex:
        vectors = embedder.embed_documents([c.embedding_text for c in chunks])
        return cls(chunks, vectors, embedder)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        with open(directory / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        (directory / "meta.json").write_text(
            json.dumps({"embedder": self.embedder.name, "dim": int(self.embedder.dim), "chunks": len(self.chunks)})
        )

    @classmethod
    def load(cls, directory: Path, embedder: Embedder) -> HybridIndex | None:
        meta_path = directory / "meta.json"
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text())
        if meta.get("embedder") != embedder.name:
            logger.warning("index_embedder_mismatch", extra={"fields": {"index": meta.get("embedder"), "runtime": embedder.name}})
            return None
        with open(directory / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
        return cls(chunks, np.load(directory / "vectors.npy"), embedder)

    # ------------------------------------------------------------------ search
    def search(self, query: str, k: int = 5, tags: list[str] | None = None, candidates: int = 20) -> list[SearchHit]:
        if not self.chunks:
            return []
        allowed = np.ones(len(self.chunks), dtype=bool)
        if tags:
            allowed = np.array([bool(set(tags) & set(c.tags)) for c in self.chunks])
            if not allowed.any():
                allowed[:] = True

        q_tokens = tokenize(query)
        bm25_scores = self._bm25.get_scores(q_tokens) if q_tokens else np.zeros(len(self.chunks))
        bm25_scores = np.where(allowed, bm25_scores, -np.inf)
        dense_scores = self.vectors @ self.embedder.embed_query(query)
        dense_scores = np.where(allowed, dense_scores, -np.inf)

        bm25_order = [i for i in np.argsort(-bm25_scores)[:candidates] if bm25_scores[i] > 0]
        dense_order = list(np.argsort(-dense_scores)[:candidates])
        bm25_rank = {int(i): r for r, i in enumerate(bm25_order)}
        dense_rank = {int(i): r for r, i in enumerate(dense_order)}

        fused: dict[int, float] = {}
        for ranks in (bm25_rank, dense_rank):
            for idx, r in ranks.items():
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + r + 1)

        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        hits, per_url = [], {}
        for idx, score in ordered:
            chunk = self.chunks[idx]
            # diversity: at most 2 chunks from the same page
            if per_url.get(chunk.url, 0) >= 2:
                continue
            per_url[chunk.url] = per_url.get(chunk.url, 0) + 1
            hits.append(SearchHit(chunk, round(score, 5), bm25_rank.get(idx), dense_rank.get(idx), float(dense_scores[idx])))
            if len(hits) >= k:
                break
        return hits
