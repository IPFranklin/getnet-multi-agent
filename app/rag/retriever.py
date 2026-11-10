"""Retrieval façade used by the agents' tools."""

from __future__ import annotations

from dataclasses import dataclass

from app.observability import RETRIEVAL_HITS
from app.rag.index import HybridIndex


@dataclass
class RetrievedChunk:
    ref: int
    text: str
    title: str
    section: str
    url: str
    score: float

    def as_source(self) -> dict:
        return {"title": self.title, "section": self.section, "url": self.url, "score": self.score}


class KnowledgeRetriever:
    def __init__(self, index: HybridIndex, top_k: int = 5, min_score: float = 0.025, relative_cutoff: float = 0.6):
        self.index = index
        self.top_k = top_k
        self.min_score = min_score
        self.relative_cutoff = relative_cutoff  # drop the long tail: keep hits >= cutoff * best score

    def retrieve(self, query: str, k: int | None = None, tags: list[str] | None = None) -> list[RetrievedChunk]:
        hits = self.index.search(query, k=k or self.top_k, tags=tags)
        floor = max(self.min_score, (hits[0].score * self.relative_cutoff) if hits else 0.0)
        relevant = [h for h in hits if h.score >= floor]
        RETRIEVAL_HITS.observe(len(relevant))
        return [
            RetrievedChunk(i + 1, h.chunk.text, h.chunk.title, h.chunk.section, h.chunk.url, h.score)
            for i, h in enumerate(relevant)
        ]

    @staticmethod
    def format_context(chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "NO_RELEVANT_DOCUMENTS_FOUND"
        return "\n\n".join(
            f"[{c.ref}] {c.title} — {c.section}\nURL: {c.url}\n{c.text}" for c in chunks
        )
