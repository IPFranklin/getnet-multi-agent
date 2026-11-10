"""Agent 3 — Customer Support: resolves account-specific issues using the merchant's data."""

from __future__ import annotations

from app.agents.base import run_tool_agent
from app.agents.knowledge import LANG_NAMES
from app.agents.schemas import GraphState
from app.container import Container
from app.prompts.templates import SUPPORT_SYSTEM
from app.tools.registry import ToolContext, build_support_tools


class SupportAgent:
    name = "support"

    def __init__(self, container: Container):
        self.c = container

    async def __call__(self, state: GraphState) -> dict:
        ctx = ToolContext()
        user_id = state["user_id"]
        tools = build_support_tools(user_id, self.c.repository, self.c.desk, self.c.retriever, ctx)
        lang = state.get("route", {}).get("language", "pt")
        prompt = SUPPORT_SYSTEM.format(
            today=self.c.repository.today().strftime("%A, %Y-%m-%d"), user_id=user_id, language=LANG_NAMES.get(lang, lang)
        )
        run = await run_tool_agent(
            self.name, self.c.agent_model, prompt, state.get("history", []), state["message"], tools,
            max_iterations=self.c.settings.max_agent_iterations,
        )
        return {
            "agent": self.name,
            "response": run.answer,
            "sources": ctx.sources,
            "tools_used": run.tool_calls,
            "tickets": ctx.tickets,
            "handoff_requested": ctx.handoff_reason is not None,
            "handoff_reason": ctx.handoff_reason,
        }
