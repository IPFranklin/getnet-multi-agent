"""Agent 2 — Knowledge: RAG over Getnet's public content + web/general tools."""

from __future__ import annotations

from app.agents.base import run_tool_agent
from app.agents.schemas import GraphState
from app.container import Container
from app.prompts.templates import KNOWLEDGE_SYSTEM
from app.services.calendar import today
from app.tools.registry import ToolContext, build_knowledge_tools

LANG_NAMES = {"pt": "Portuguese (Brazil)", "en": "English", "es": "Spanish"}


class KnowledgeAgent:
    name = "knowledge"

    def __init__(self, container: Container):
        self.c = container

    async def __call__(self, state: GraphState) -> dict:
        ctx = ToolContext()
        tools = build_knowledge_tools(self.c.retriever, self.c.settings, ctx)
        lang = state.get("route", {}).get("language", "pt")
        prompt = KNOWLEDGE_SYSTEM.format(today=today(self.c.settings.timezone).isoformat(), language=LANG_NAMES.get(lang, lang))
        run = await run_tool_agent(
            self.name, self.c.agent_model, prompt, state.get("history", []), state["message"], tools,
            max_iterations=self.c.settings.max_agent_iterations,
        )
        return {"agent": self.name, "response": run.answer, "sources": ctx.sources, "tools_used": run.tool_calls}
