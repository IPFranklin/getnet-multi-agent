"""Orchestration graph (LangGraph).

                 ┌──────────────┐ blocked
   message ──▶   │ input_guard  │──────────────────────────────┐
                 └──────┬───────┘                              │
                        ▼                                      │
                 ┌──────────────┐ clarify / unsupported        │
                 │    router    │──────────────────────────────┤
                 └──┬────┬────┬─┘                              │
          knowledge │    │    │ escalation                     │
                    ▼    │    ▼                                │
            ┌──────────┐ │ ┌────────────┐                      │
            │knowledge │ │ │ escalation │◀──┐ handoff          │
            └────┬─────┘ │ └─────┬──────┘   │ requested        │
                 │       ▼       │          │                  │
                 │  ┌─────────┐  │          │                  │
                 │  │ support │──┼──────────┘                  │
                 │  └────┬────┘  │                             │
                 ▼       ▼       ▼                             ▼
                 └───────┴───────┴────────▶ output_guard ──▶ END

Agents communicate through a typed shared state (``GraphState``) — each node reads what it
needs and writes its results; conditional edges implement the routing/handoff policy.
"""

from __future__ import annotations

import time

from langgraph.graph import END, START, StateGraph

from app.agents.escalation import EscalationAgent
from app.agents.knowledge import KnowledgeAgent
from app.agents.router import RouterAgent
from app.agents.schemas import GraphState
from app.agents.support import SupportAgent
from app.container import Container
from app.guardrails import check_input, check_output
from app.observability import REQUEST_LATENCY, REQUESTS, new_trace_id, trace_id_var, traced_node


class Orchestrator:
    def __init__(self, container: Container):
        self.c = container
        self.router = RouterAgent(container)
        self.knowledge = KnowledgeAgent(container)
        self.support = SupportAgent(container)
        self.escalation = EscalationAgent(container)
        self.graph = self._build()

    # ------------------------------------------------------------------ nodes
    async def _input_guard(self, state: GraphState) -> dict:
        result = check_input(state["message"], self.c.settings.max_message_chars)
        update = {
            "message": result.sanitized_text,
            "guard": {"allowed": result.allowed, "categories": result.categories, "reason": result.reason},
            "warnings": result.warnings,
        }
        if not result.allowed:
            update |= {"agent": "guardrail", "response": result.response}
        return update

    async def _output_guard(self, state: GraphState) -> dict:
        text, categories = check_output(state.get("response", ""))
        warnings = state.get("warnings") or []
        if warnings and state.get("agent") != "guardrail":
            text = f"{text}\n\n_{' '.join(warnings)}_"
        guard = dict(state.get("guard") or {})
        guard["output_categories"] = categories
        return {"response": text, "guard": guard}

    # ------------------------------------------------------------------ edges
    @staticmethod
    def _after_input(state: GraphState) -> str:
        return "router" if (state.get("guard") or {}).get("allowed") else "output_guard"

    @staticmethod
    def _after_router(state: GraphState) -> str:
        if state.get("handoff_requested"):  # router crashed -> safest path is a human
            return "escalation"
        intent = (state.get("route") or {}).get("intent_final")
        return {"knowledge": "knowledge", "support": "support", "escalation": "escalation"}.get(intent, "output_guard")

    @staticmethod
    def _after_specialist(state: GraphState) -> str:
        return "escalation" if state.get("handoff_requested") else "output_guard"

    def _build(self):
        g = StateGraph(GraphState)
        g.add_node("input_guard", traced_node("input_guard")(self._input_guard))
        g.add_node("router", traced_node("router")(self.router))
        g.add_node("knowledge", traced_node("knowledge")(self.knowledge))
        g.add_node("support", traced_node("support")(self.support))
        g.add_node("escalation", traced_node("escalation")(self.escalation))
        g.add_node("output_guard", traced_node("output_guard")(self._output_guard))

        g.add_edge(START, "input_guard")
        g.add_conditional_edges("input_guard", self._after_input, ["router", "output_guard"])
        g.add_conditional_edges("router", self._after_router, ["knowledge", "support", "escalation", "output_guard"])
        g.add_conditional_edges("knowledge", self._after_specialist, ["escalation", "output_guard"])
        g.add_conditional_edges("support", self._after_specialist, ["escalation", "output_guard"])
        g.add_edge("escalation", "output_guard")
        g.add_edge("output_guard", END)
        return g.compile()

    # ------------------------------------------------------------------ entrypoint
    async def handle(self, message: str, user_id: str, session_id: str | None = None) -> dict:
        trace_id = new_trace_id()
        token = trace_id_var.set(trace_id)
        start = time.perf_counter()
        key = self.c.memory.key(user_id, session_id)
        try:
            state: GraphState = await self.graph.ainvoke({
                "trace_id": trace_id, "user_id": user_id, "session_key": key, "message": message,
                "history": self.c.memory.history(key), "tools_used": [], "errors": [], "steps": [],
            })
            agent = state.get("agent", "unknown")
            if (state.get("guard") or {}).get("allowed"):
                self.c.memory.append(key, state["message"], state["response"], agent)
            intent = (state.get("route") or {}).get("intent_final", agent)
            REQUESTS.labels(route=intent, status="error" if state.get("errors") else "ok").inc()
            return self._to_response(state, trace_id, time.perf_counter() - start)
        finally:
            REQUEST_LATENCY.observe(time.perf_counter() - start)
            trace_id_var.reset(token)

    @staticmethod
    def _unique_sources(sources: list[dict], limit: int = 10) -> list[dict]:
        """Sources in citation order: ``ref`` n matches the ``[n]`` markers in the answer.

        The tools already de-duplicate by URL and number sources in order of appearance
        (see ``ToolContext.add_sources``), so the position here *is* the citation number.
        """
        seen, out = set(), []
        for x in sources:
            if x["url"] not in seen:
                seen.add(x["url"])
                out.append({"ref": len(out) + 1, "title": x["title"], "url": x["url"]})
        return out[:limit]

    def _to_response(self, s: GraphState, trace_id: str, elapsed: float) -> dict:
        route = s.get("route") or {}
        body = {
            "response": s.get("response", ""),
            "agent": s.get("agent"),
            "route": {k: route.get(k) for k in ("intent", "intent_final", "confidence", "language", "reasoning", "method")} if route else None,
            "sources": self._unique_sources(s.get("sources") or []),
            "tools_used": [{"agent": t["agent"], "tool": t["tool"], "status": t["status"]} for t in s.get("tools_used", [])],
            "handoff": s.get("handoff"),
            "tickets": [{"ticket_id": t["ticket_id"], "sla": t["sla"]} for t in (s.get("tickets") or [])],
            "guardrails": s.get("guard"),
            "trace_id": trace_id,
            "latency_ms": round(elapsed * 1000, 1),
            "llm_provider": self.c.settings.llm_provider,
        }
        if self.c.settings.expose_trace:
            body["trace"] = s.get("steps", [])
            # What the specialists actually saw (truncated tool outputs) — lets the eval judge check
            # faithfulness against evidence instead of just source titles. Debug/eval only.
            body["evidence"] = [{"tool": t["tool"], "output": t.get("output", "")} for t in s.get("tools_used", [])]
        if s.get("errors"):
            body["errors"] = s["errors"]
        return body
