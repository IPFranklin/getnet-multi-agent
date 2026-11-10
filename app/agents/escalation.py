"""Agent 4 (bonus) — Human Escalation: summarises the case and hands it to a human operator."""

from __future__ import annotations

import asyncio
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.schemas import GraphState, HandoffSummary
from app.container import Container
from app.observability import HANDOFFS, log_event
from app.prompts.templates import ESCALATION_SYSTEM, HANDOFF_MESSAGE

logger = logging.getLogger(__name__)


class EscalationAgent:
    name = "escalation"

    def __init__(self, container: Container):
        self.c = container
        self._structured = container.router_model.with_structured_output(HandoffSummary)

    async def summarise(self, state: GraphState) -> HandoffSummary:
        transcript = "\n".join(f"{t['role']}: {t['content']}" for t in state.get("history", [])[-6:])
        tools = "; ".join(f"{t['tool']}({t['args']})" for t in state.get("tools_used", []))
        content = (
            f"Escalation reason: {state.get('handoff_reason') or state.get('route', {}).get('reasoning')}\n"
            f"Previous turns:\n{transcript or '-'}\nCurrent customer message: {state['message']}\n"
            f"Tools already used: {tools or '-'}\nAssistant draft answer: {state.get('response') or '-'}"
        )
        try:
            return await asyncio.wait_for(
                self._structured.ainvoke([SystemMessage(content=ESCALATION_SYSTEM), HumanMessage(content=content)]),
                timeout=self.c.settings.llm_timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 - a handoff must never fail because of the LLM
            log_event(logger, "handoff_summary_failed", level=logging.WARNING, error=str(exc))
            return HandoffSummary(summary=content[:800], category="other", priority="high")

    async def __call__(self, state: GraphState) -> dict:
        summary = await self.summarise(state)
        reason = state.get("handoff_reason") or state.get("route", {}).get("reasoning") or "customer request"
        handoff = await self.c.desk.create_handoff(
            user_id=state["user_id"], reason=reason, summary=summary.summary, priority=summary.priority,
            category=summary.category,
            transcript=[*state.get("history", []), {"role": "user", "content": state["message"]}],
        )
        HANDOFFS.labels(reason_category=summary.category).inc()
        lang = state.get("route", {}).get("language", "pt")
        handoff_msg = HANDOFF_MESSAGE.get(lang, HANDOFF_MESSAGE["en"]).format(id=handoff["handoff_id"], sla=handoff["sla"])
        previous = (state.get("response") or "").strip()
        # Keep useful partial work from the specialist agent (e.g. data already checked).
        response = f"{previous}\n\n{handoff_msg}" if previous and state.get("agent") == "support" else handoff_msg
        return {
            "agent": self.name,
            "response": response,
            "handoff": {k: handoff[k] for k in ("handoff_id", "priority", "category", "sla", "status", "notified")},
        }
