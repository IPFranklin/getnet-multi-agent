"""Dependency container: builds and wires every component once at startup."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import Settings
from app.llm import build_chat_model
from app.rag.embeddings import Embedder
from app.rag.ingest import load_or_build_index
from app.rag.retriever import KnowledgeRetriever
from app.services.memory import ConversationMemory
from app.services.merchant_repository import MerchantRepository
from app.services.support_desk import SupportDesk


@dataclass
class Container:
    settings: Settings
    router_model: BaseChatModel
    agent_model: BaseChatModel
    retriever: KnowledgeRetriever
    repository: MerchantRepository
    desk: SupportDesk
    memory: ConversationMemory


def build_container(
    settings: Settings,
    embedder: Embedder | None = None,
    repository: MerchantRepository | None = None,
    router_model: BaseChatModel | None = None,
    agent_model: BaseChatModel | None = None,
) -> Container:
    index = load_or_build_index(settings, embedder)
    return Container(
        settings=settings,
        router_model=router_model or build_chat_model(settings, "router"),
        agent_model=agent_model or build_chat_model(settings, "agent"),
        retriever=KnowledgeRetriever(index, settings.retrieval_top_k, settings.retrieval_min_score),
        repository=repository or MerchantRepository(),
        desk=SupportDesk(settings.tickets_store_path, settings.handoff_store_path, settings.handoff_webhook_url),
        memory=ConversationMemory(settings.history_max_turns),
    )
