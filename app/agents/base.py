"""Shared tool-calling loop (ReAct style) used by the Knowledge and Support agents.

Written explicitly instead of using a prebuilt agent so we control: iteration budget, tool
error isolation, per-tool metrics/logging and a forced final answer when the budget runs out.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from app.observability import TOOL_CALLS, log_event, record_usage

logger = logging.getLogger(__name__)
EVIDENCE_CHARS = 4000  # tool output kept per call for traces / eval judge (not sent to customers)


def message_text(msg: BaseMessage) -> str:
    content = msg.content
    if isinstance(content, list):  # Anthropic returns content blocks
        return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text").strip()
    return str(content).strip()


def history_to_messages(history: list[dict], current_agent: str | None = None) -> list[BaseMessage]:
    """Convert stored turns into chat messages, keeping track of *who* answered.

    All specialists share one conversation memory. Without attribution, a specialist reads another
    agent's reply as if it had written it — e.g. the Knowledge agent "corrects" account figures that
    only the Support agent could see, and the two contradict each other on the next turn. Replies from
    a different agent are therefore labelled; the prompts tell each specialist how to treat them.
    """
    out: list[BaseMessage] = []
    for turn in history:
        if turn["role"] == "user":
            out.append(HumanMessage(content=turn["content"]))
            continue
        agent = turn.get("agent")
        content = turn["content"]
        if agent and agent != current_agent:
            content = f"[Earlier reply by the {agent} specialist]\n{content}"
        out.append(AIMessage(content=content))
    return out


@dataclass
class AgentRun:
    answer: str
    tool_calls: list[dict] = field(default_factory=list)
    iterations: int = 0


async def run_tool_agent(
    agent_name: str,
    model: BaseChatModel,
    system_prompt: str,
    history: list[dict],
    user_message: str,
    tools: list[BaseTool],
    max_iterations: int = 5,
    tool_timeout_s: float = 20.0,
) -> AgentRun:
    by_name = {t.name: t for t in tools}
    bound = model.bind_tools(tools)
    messages: list[BaseMessage] = [SystemMessage(content=system_prompt), *history_to_messages(history, agent_name),
                                   HumanMessage(content=user_message)]
    records: list[dict] = []

    for iteration in range(1, max_iterations + 1):
        ai: AIMessage = await bound.ainvoke(messages)
        record_usage(agent_name, ai)
        messages.append(ai)
        if not ai.tool_calls:
            return AgentRun(message_text(ai), records, iteration)

        async def execute(tc: dict) -> ToolMessage:
            start = time.perf_counter()
            tool = by_name.get(tc["name"])
            status = "ok"
            if tool is None:
                status, content = "unknown_tool", json.dumps({"error": f"unknown tool {tc['name']}"})
            else:
                try:
                    content = await asyncio.wait_for(tool.ainvoke(tc["args"]), timeout=tool_timeout_s)
                except TimeoutError:
                    status, content = "timeout", json.dumps({"error": "tool timed out"})
                except Exception as exc:  # noqa: BLE001 - surface to the model, never crash
                    status, content = "error", json.dumps({"error": f"{type(exc).__name__}: {exc}"})
            ms = round((time.perf_counter() - start) * 1000, 1)
            TOOL_CALLS.labels(agent=agent_name, tool=tc["name"], status=status).inc()
            log_event(logger, "tool_called", agent=agent_name, tool=tc["name"], args=tc["args"], status=status, latency_ms=ms)
            records.append({"agent": agent_name, "tool": tc["name"], "args": tc["args"], "status": status, "latency_ms": ms,
                            "output": str(content)[:EVIDENCE_CHARS]})
            return ToolMessage(content=str(content), tool_call_id=tc["id"], name=tc["name"])

        # independent tool calls in the same turn run concurrently
        messages.extend(await asyncio.gather(*(execute(tc) for tc in ai.tool_calls)))

    # Budget exhausted: ask for a final answer with what we have, without tools.
    messages.append(HumanMessage(content="Tool budget reached. Answer the customer now using only the information gathered."))
    # (kept bound: some providers reject histories with tool_use blocks when no tools are declared)
    ai = await bound.ainvoke(messages)
    record_usage(agent_name, ai)
    return AgentRun(message_text(ai), records, max_iterations)
