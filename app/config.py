"""Centralised, typed configuration loaded from environment variables / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App -----------------------------------------------------------------
    app_name: str = "getnet-multi-agent"
    environment: Literal["local", "test", "staging", "production"] = "local"
    log_level: str = "INFO"
    timezone: str = "America/Sao_Paulo"

    # --- LLM -----------------------------------------------------------------
    # "fake" runs a deterministic, offline heuristic model (demo + tests, no API key needed).
    llm_provider: Literal["anthropic", "openai", "fake"] = "fake"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    # Small/fast model for routing & classification, larger model for answering.
    router_model: str | None = None
    agent_model: str | None = None
    llm_temperature: float | None = None  # only sent if set; newer Claude models reject `temperature`
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2
    llm_max_tokens: int = 1024

    # --- RAG -----------------------------------------------------------------
    embedding_backend: Literal["fastembed", "openai", "hash"] = "hash"
    fastembed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    fastembed_cache_dir: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    knowledge_base_dir: Path = DATA_DIR / "knowledge_base"
    sources_file: Path = DATA_DIR / "sources.yaml"
    index_dir: Path = DATA_DIR / "index"
    ingest_crawl: bool = False  # crawl live URLs from sources.yaml during ingestion
    chunk_size: int = 900
    chunk_overlap: int = 150
    retrieval_top_k: int = 5
    retrieval_min_score: float = 0.025  # fused RRF score; >0.0164 requires agreement of BM25 and dense rankers

    # --- Tools ---------------------------------------------------------------
    tavily_api_key: str | None = None  # if set, Tavily is used; otherwise DuckDuckGo
    web_search_max_results: int = 5
    http_timeout_s: float = 10.0

    # --- Orchestration -------------------------------------------------------
    max_agent_iterations: int = 5
    router_confidence_threshold: float = 0.45
    history_max_turns: int = 6
    max_message_chars: int = 2000

    # --- Human handoff -------------------------------------------------------
    handoff_webhook_url: str | None = None  # Teams/Slack incoming webhook
    handoff_store_path: Path = DATA_DIR / "runtime" / "handoffs.jsonl"
    tickets_store_path: Path = DATA_DIR / "runtime" / "tickets.jsonl"

    expose_trace: bool = Field(default=True, description="Return per-node trace in API responses")

    def resolved_router_model(self) -> str:
        if self.router_model:
            return self.router_model
        return {"anthropic": "claude-haiku-5-5", "openai": "gpt-4o-mini"}.get(self.llm_provider, "fake")

    def resolved_agent_model(self) -> str:
        if self.agent_model:
            return self.agent_model
        return {"anthropic": "claude-sonnet-5-5", "openai": "gpt-4o"}.get(self.llm_provider, "fake")


@lru_cache
def get_settings() -> Settings:
    return Settings()
