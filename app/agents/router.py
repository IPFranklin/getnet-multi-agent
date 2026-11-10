"""Agent 1 — Router: classifies the message and decides the workflow."""

from __future__ import annotations

import asyncio
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.base import history_to_messages
from app.agents.heuristics import heuristic_route
from app.agents.schemas import GraphState, RouteDecision
from app.container import Container
from app.observability import log_event
from app.prompts.templates import ROUTER_SYSTEM, UNSUPPORTED_MESSAGE

logger = logging.getLogger(__name__)


class RouterAgent:
    name = "router"

    def __init__(self, container: Container):
        self.c = container
        self._structured = container.router_model.with_structured_output(RouteDecision)

    async def classify(self, message: str, history: list[dict]) -> tuple[dict, str]:
        messages = [SystemMessage(content=ROUTER_SYSTEM), *history_to_messages(history[-4:]), HumanMessage(content=message)]
        try:
            decision: RouteDecision = await asyncio.wait_for(
                self._structured.ainvoke(messages), timeout=self.c.settings.llm_timeout_s
            )
            return decision.model_dump(), "llm"
        except Exception as exc:  # noqa: BLE001 - degrade gracefully to deterministic routing
            log_event(logger, "router_llm_failed_using_heuristic", level=logging.WARNING, error=str(exc))
            return heuristic_route(message), "heuristic_fallback"

    async def __call__(self, state: GraphState) -> dict:
        route, method = await self.classify(state["message"], state.get("history", []))
        route["method"] = method
        update: dict = {"route": route}
        threshold = self.c.settings.router_confidence_threshold
        lang = route.get("language", "pt")
        if route["confidence"] < threshold and route.get("clarifying_question"):
            update |= {"agent": "router", "response": route["clarifying_question"]}
            route["intent_final"] = "clarify"
        elif route["intent"] == "unsupported":
            update |= {"agent": "router", "response": UNSUPPORTED_MESSAGE.get(lang, UNSUPPORTED_MESSAGE["en"])}
            route["intent_final"] = "unsupported"
        else:
            route["intent_final"] = route["intent"]
        return update
