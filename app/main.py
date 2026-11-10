"""FastAPI application: HTTP interface of the multi-agent system."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.container import Container, build_container
from app.graph import Orchestrator
from app.observability import configure_logging, log_event
from app.rag.ingest import build_index
from app.rag.retriever import KnowledgeRetriever

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- schemas
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, examples=["What's the difference between the Get Clássica and the Get Smart?"])
    user_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.@-]+$", examples=["cliente1988"])
    session_id: str | None = Field(default=None, max_length=64, description="Optional: separate conversations of the same user.")


class Source(BaseModel):
    title: str
    url: str


class ToolUse(BaseModel):
    agent: str
    tool: str
    status: str


class ChatResponse(BaseModel):
    response: str
    agent: str | None
    route: dict | None
    sources: list[Source]
    tools_used: list[ToolUse]
    handoff: dict | None
    tickets: list[dict]
    guardrails: dict | None
    trace_id: str
    latency_ms: float
    llm_provider: str
    trace: list[dict] | None = None
    errors: list[str] | None = None


# --------------------------------------------------------------------------- app
def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        c = container or build_container(settings)
        app.state.container = c
        app.state.orchestrator = Orchestrator(c)
        log_event(logger, "startup_complete", llm_provider=settings.llm_provider,
                  embedder=c.retriever.index.embedder.name, chunks=len(c.retriever.index.chunks))
        yield

    app = FastAPI(
        title="Getnet Multi-Agent Support",
        version="1.0.0",
        description="Router → Knowledge (RAG + web) / Customer Support (tools) / Human Escalation agents.",
        lifespan=lifespan,
    )

    def orchestrator(request: Request) -> Orchestrator:
        return request.app.state.orchestrator

    @app.post("/chat", response_model=ChatResponse, response_model_exclude_none=True, tags=["chat"])
    @app.post("/v1/chat", response_model=ChatResponse, response_model_exclude_none=True, tags=["chat"])
    async def chat(req: ChatRequest, orch: Orchestrator = Depends(orchestrator)) -> dict:
        return await orch.handle(req.message, req.user_id, req.session_id)

    @app.get("/health", tags=["ops"])
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready", tags=["ops"])
    async def ready(request: Request) -> dict:
        c: Container | None = getattr(request.app.state, "container", None)
        if not c or not c.retriever.index.chunks:
            raise HTTPException(503, "knowledge index not loaded")
        return {"status": "ready", "chunks": len(c.retriever.index.chunks), "embedder": c.retriever.index.embedder.name,
                "llm_provider": c.settings.llm_provider}

    @app.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/admin/reindex", tags=["admin"])
    async def reindex(request: Request, crawl: bool = False) -> dict:
        c: Container = request.app.state.container
        index = build_index(c.settings, c.retriever.index.embedder, crawl=crawl)
        c.retriever = KnowledgeRetriever(index, c.settings.retrieval_top_k, c.settings.retrieval_min_score)
        request.app.state.orchestrator = Orchestrator(c)
        return {"status": "reindexed", "chunks": len(index.chunks)}

    @app.get("/admin/handoffs", tags=["admin"])
    async def handoffs(request: Request, user_id: str | None = None) -> list[dict]:
        return request.app.state.container.desk.list_handoffs(user_id)

    @app.get("/admin/tickets", tags=["admin"])
    async def tickets(request: Request, user_id: str | None = None) -> list[dict]:
        return request.app.state.container.desk.list_tickets(user_id)

    return app


app = create_app()
