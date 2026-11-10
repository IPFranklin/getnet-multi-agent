"""Shared fixtures: an isolated, fully offline system (fake LLM, hash embeddings, fixed date,
temporary stores) plus stubs for the external web tools."""

from __future__ import annotations

import json
from datetime import date

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.container import build_container
from app.graph import Orchestrator
from app.main import create_app
from app.rag.embeddings import HashEmbedder
from app.services.merchant_repository import MerchantRepository
from app.tools import web_tools

FIXED_TODAY = date(2026, 10, 23)  # a Friday


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        llm_provider="fake",
        embedding_backend="hash",
        index_dir=tmp_path / "index",
        handoff_store_path=tmp_path / "handoffs.jsonl",
        tickets_store_path=tmp_path / "tickets.jsonl",
        ingest_crawl=False,
        handoff_webhook_url=None,
        log_level="WARNING",
    )


@pytest.fixture
def repository() -> MerchantRepository:
    return MerchantRepository(today_fn=lambda: FIXED_TODAY)


@pytest.fixture(autouse=True)
def stub_web(monkeypatch):
    """External APIs are stubbed in every test: deterministic and no network."""

    async def fake_weather(city, settings, day_offset=1):
        return json.dumps({"source": "Open-Meteo (https://open-meteo.com)", "location": f"{city}, RS, Brasil",
                           "date": "2026-10-24", "condition": "rain showers", "temp_min_c": 15.2, "temp_max_c": 22.8,
                           "precipitation_probability_pct": 80, "precipitation_mm": 12.4, "wind_max_kmh": 25.0})

    async def fake_fx(base, settings, quote="BRL"):
        return json.dumps({"source": "AwesomeAPI (economia.awesomeapi.com.br)", "pair": f"{base}/{quote}",
                           "bid": 6.2345, "ask": 6.2360, "high": 6.25, "low": 6.20, "change_pct": 0.12,
                           "timestamp": "2026-10-23 15:00:00"})

    async def fake_search(query, settings, max_results=None):
        return json.dumps({"provider": "stub", "results": [
            {"title": "Example result", "url": "https://example.com/a", "snippet": f"Result about {query}"}]})

    monkeypatch.setattr(web_tools, "get_weather_forecast", fake_weather)
    monkeypatch.setattr(web_tools, "get_exchange_rate", fake_fx)
    monkeypatch.setattr(web_tools, "web_search", fake_search)


@pytest.fixture
def container(settings, repository):
    return build_container(settings, embedder=HashEmbedder(), repository=repository)


@pytest.fixture
def orchestrator(container) -> Orchestrator:
    return Orchestrator(container)


@pytest.fixture
def client(settings, container):
    app = create_app(settings, container)
    with TestClient(app) as c:
        yield c
