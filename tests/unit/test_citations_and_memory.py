"""Regressions for issues found when running the system against the real Claude API.

* Citations: the [n] in a passage must equal the position of its source in the API ``sources`` list,
  even when the same page appears in several passages or across several searches.
* Shared memory: replies from another specialist must be labelled, so an agent does not mistake
  (and "correct") account data that only the Support agent could see.
"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage

from app.agents.base import history_to_messages
from app.graph import Orchestrator
from app.tools.registry import ToolContext, build_knowledge_tools


def test_tool_context_numbers_sources_by_url_in_order():
    ctx = ToolContext()
    a = {"title": "A", "section": "s1", "url": "kb://a", "score": 1}
    a2 = {"title": "A", "section": "s2", "url": "kb://a", "score": 1}
    b = {"title": "B", "section": "s1", "url": "kb://b", "score": 1}
    assert ctx.add_sources([a, b]) == [1, 2]
    assert ctx.add_sources([b, a2]) == [2, 1]  # second search reuses the same numbers
    assert [s["url"] for s in ctx.sources] == ["kb://a", "kb://b"]


async def test_passage_refs_match_api_source_refs(container):
    ctx = ToolContext()
    search = next(t for t in build_knowledge_tools(container.retriever, container.settings, ctx)
                  if t.name == "search_getnet_knowledge_base")
    first = await search.ainvoke({"query": "diferença Get Clássica Get Smart"})
    second = await search.ainvoke({"query": "Pix na maquininha"})
    sources = Orchestrator._unique_sources(ctx.sources)
    by_ref = {s["ref"]: s["url"] for s in sources}
    for text in (first, second):
        for ref, url in re.findall(r"\[(\d+)\] .*\nURL: (\S+)", text):
            assert by_ref[int(ref)] == url


def test_history_labels_replies_from_other_agents():
    history = [
        {"role": "user", "content": "Quanto vou receber hoje?"},
        {"role": "assistant", "content": "Você recebe R$ 1.234,56 hoje.", "agent": "support"},
        {"role": "user", "content": "E como funciona a antecipação?"},
        {"role": "assistant", "content": "A antecipação…", "agent": "knowledge"},
    ]
    msgs = history_to_messages(history, current_agent="knowledge")
    assert isinstance(msgs[0], HumanMessage) and isinstance(msgs[1], AIMessage)
    assert msgs[1].content.startswith("[Earlier reply by the support specialist]")
    assert not msgs[3].content.startswith("[Earlier reply")  # its own reply stays as-is


async def test_multi_turn_conversation_keeps_agent_attribution(orchestrator, container):
    user, session = "cliente1988", "mt-1"
    await orchestrator.handle("Quanto vou receber hoje?", user, session_id=session)
    r = await orchestrator.handle("Como funciona a antecipação de recebíveis?", user, session_id=session)
    history = container.memory.history(container.memory.key(user, session))
    assert [t.get("agent") for t in history if t["role"] == "assistant"] == ["support", "knowledge"]
    assert r["agent"] == "knowledge"
    assert all(s.get("ref") for s in r["sources"])
