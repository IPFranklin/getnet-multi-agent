"""Ingestion pipeline: load → clean → chunk → dedupe → embed → persist.

Run manually with ``python -m app.rag.ingest [--crawl]``. The API also builds the index on
startup when none exists (or when the embedding backend changed).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import time

from app.config import Settings, get_settings
from app.observability import configure_logging, log_event
from app.rag.chunking import Chunk, Document, chunk_document
from app.rag.embeddings import Embedder, build_embedder
from app.rag.index import HybridIndex
from app.rag.loaders import crawl_sources, load_markdown_dir, load_sources

logger = logging.getLogger(__name__)


def collect_documents(settings: Settings, crawl: bool | None = None) -> list[Document]:
    docs = load_markdown_dir(settings.knowledge_base_dir)
    log_event(logger, "loaded_curated_docs", count=len(docs))
    if settings.ingest_crawl if crawl is None else crawl:
        sources = load_sources(settings.sources_file)
        crawled = crawl_sources(sources, timeout=settings.http_timeout_s * 2)
        log_event(logger, "crawled_docs", requested=len(sources), fetched=len(crawled))
        docs.extend(crawled)
    return docs


def dedupe(chunks: list[Chunk]) -> list[Chunk]:
    seen, out = set(), []
    for c in chunks:
        key = hashlib.sha1(" ".join(c.text.lower().split()).encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def build_index(settings: Settings, embedder: Embedder | None = None, crawl: bool | None = None) -> HybridIndex:
    t0 = time.perf_counter()
    embedder = embedder or build_embedder(settings)
    docs = collect_documents(settings, crawl)
    chunks = dedupe([c for d in docs for c in chunk_document(d, settings.chunk_size, settings.chunk_overlap)])
    index = HybridIndex.build(chunks, embedder)
    index.save(settings.index_dir)
    log_event(
        logger, "index_built", documents=len(docs), chunks=len(chunks), embedder=embedder.name,
        seconds=round(time.perf_counter() - t0, 2), path=str(settings.index_dir),
    )
    return index


def load_or_build_index(settings: Settings, embedder: Embedder | None = None) -> HybridIndex:
    embedder = embedder or build_embedder(settings)
    index = HybridIndex.load(settings.index_dir, embedder)
    if index is None or not index.chunks:
        index = build_index(settings, embedder)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Getnet knowledge index")
    parser.add_argument("--crawl", action="store_true", help="Also crawl live URLs from data/sources.yaml")
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    build_index(settings, crawl=args.crawl or None)


if __name__ == "__main__":
    main()
