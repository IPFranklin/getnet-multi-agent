"""Deterministic offline chat model (``LLM_PROVIDER=fake``).

It implements the same LangChain interface the agents use — ``bind_tools`` (emits real
``tool_calls``), ``with_structured_output`` and plain generation — with keyword heuristics
and templated answers rendered from tool outputs. This lets the whole orchestration (routing,
tool loops, RAG, handoff, guardrails, API) run end-to-end in CI and in a no-API-key demo,
exercising the exact same code paths as a real model.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from app.agents import heuristics as h
from app.guardrails import normalize


def _text(m: BaseMessage) -> str:
    c = m.content
    if isinstance(c, list):
        return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return str(c)


def _call(name: str, **args: Any) -> dict:
    return {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"}


def _brl(v: float) -> str:
    return "R$ " + f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


class FakeChatModel(BaseChatModel):
    bound_tools: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "fake-heuristic"

    def bind_tools(self, tools: list, **kwargs: Any) -> FakeChatModel:
        names = [t.name if hasattr(t, "name") else t["name"] for t in tools]
        return self.model_copy(update={"bound_tools": names})

    def with_structured_output(self, schema: type[BaseModel], **kwargs: Any):  # type: ignore[override]
        def run(messages: list[BaseMessage]) -> BaseModel:
            human = [m for m in messages if isinstance(m, HumanMessage)]
            text = _text(human[-1]) if human else ""
            if schema.__name__ == "RouteDecision":
                return schema(**h.heuristic_route(text))
            if schema.__name__ == "HandoffSummary":
                m = re.search(r"Current customer message: (.*)", text)
                text = m.group(1) if m else text
                n = normalize(text)
                category = "fraud" if re.search(r"fraud|roub|stolen|reconhe", n) else (
                    "terminal" if re.search(r"maquin|machine|terminal", n) else "complaint" if re.search(r"reclam|complain", n) else "other")
                priority = "urgent" if category == "fraud" else "high" if category == "terminal" else "medium"
                return schema(summary=text[:600], category=category, priority=priority, customer_sentiment="neutral")
            raise NotImplementedError(schema.__name__)

        return RunnableLambda(run)

    # ------------------------------------------------------------------ generation
    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        last_h = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        user_text = _text(messages[last_h])
        tool_results = [m for m in messages[last_h + 1:] if isinstance(m, ToolMessage)]
        calls = self._plan(user_text, tool_results) if self.bound_tools else []
        if calls:
            msg = AIMessage(content="", tool_calls=calls)
        else:
            msg = AIMessage(content=self._answer(user_text, tool_results))
        msg.usage_metadata = {"input_tokens": sum(len(_text(m)) // 4 for m in messages),
                              "output_tokens": len(_text(msg)) // 4, "total_tokens": 0}
        return ChatResult(generations=[ChatGeneration(message=msg)])

    # ------------------------------------------------------------------ planning (which tool next)
    def _plan(self, text: str, done: list[ToolMessage]) -> list[dict]:
        n = normalize(text)
        done_names = [t.name for t in done]
        has = set(self.bound_tools).__contains__

        if any('"merchant_not_found"' in _text(t) for t in done) and "escalate_to_human" not in done_names:
            return [_call("escalate_to_human", reason="merchant account not found")]

        if has("search_getnet_knowledge_base"):  # knowledge agent
            if done:
                last = done[-1]
                if last.name == "search_getnet_knowledge_base" and "NO_RELEVANT_DOCUMENTS_FOUND" in _text(last) \
                        and "web_search" not in done_names:
                    return [_call("web_search", query=text)]
                return []
            if re.search(r"weather|forecast|previsao|tempo em|clima|chuva|rain|temperatura", n):
                return [_call("get_weather_forecast", city=h.extract_city(text), day_offset=h.extract_day_offset(text))]
            if re.search(r"exchange|cambio|cotacao|euro|dolar|dollar|currency|moeda", n):
                return [_call("get_exchange_rate", base_currency=h.extract_currency(text), quote_currency="BRL")]
            return [_call("search_getnet_knowledge_base", query=text)]

        # support agent
        if not done:
            if re.search(r"human|humano|atendente|fraud|fraude|roubad|stolen", n):
                return [_call("escalate_to_human", reason="customer request")]
            if re.search(r"deposit|cair|caiu|receber|recebo|payout|money|dinheiro|repasse|paid|pago", n):
                day = "today" if re.search(r"\b(today|hoje)\b", n) and not re.search(r"yesterday|ontem", n) else "yesterday"
                return [_call("get_receivables", start_date=day)]
            if re.search(r"connect|conecta|internet|wi-?fi|offline|sinal|signal|rede", n):
                return [_call("get_terminal_status"),
                        _call("search_help_center", query="maquininha não conecta internet configurar wi-fi chip")]
            if re.search(r"declin|recusad|negad|error|erro|codigo|code", n):
                return [_call("get_transactions", start_date="yesterday", status="declined", limit=5)]
            if re.search(r"pix|plano|plan|taxa|fee|aluguel|rent|conta|account", n):
                return [_call("get_merchant_profile")]
            return [_call("get_merchant_profile"), _call("search_help_center", query=text)]

        if done_names == ["get_transactions"]:
            try:
                txs = json.loads(_text(done[0]))["transactions"]
            except (ValueError, KeyError):
                return []
            codes = list(dict.fromkeys(t["response_code"] for t in txs))[:2]
            return [_call("lookup_error_code", code=c) for c in codes]
        return []

    # ------------------------------------------------------------------ answer rendering
    def _answer(self, text: str, results: list[ToolMessage]) -> str:
        pt = h.detect_language(text) == "pt"
        if not results:
            return ("Posso ajudar com produtos Getnet, sua conta e suas vendas." if pt
                    else "I can help with Getnet products, your account and your sales.")
        parts: list[str] = []
        lookups = []
        for r in results:
            raw = _text(r)
            try:
                data = json.loads(raw)
            except ValueError:
                data = None
            name = r.name
            if name in {"search_getnet_knowledge_base", "search_help_center"}:
                parts.append(self._render_kb(raw, pt))
            elif name == "get_weather_forecast" and data and "error" not in data:
                parts.append(
                    f"**{data['location']} — {data['date']}**: {data['condition']}, "
                    f"{data['temp_min_c']}°C–{data['temp_max_c']}°C, "
                    f"{'chance de chuva' if pt else 'chance of rain'} {data['precipitation_probability_pct']}%, "
                    f"{'vento até' if pt else 'wind up to'} {data['wind_max_kmh']} km/h. ({data['source']})")
            elif name == "get_exchange_rate" and data and "error" not in data:
                rate = data.get("bid", data.get("rate"))
                parts.append(f"**{data['pair']}**: 1 {data['pair'][:3]} = {_brl(rate)} "
                             f"({data['source']}{', ' + str(data.get('timestamp') or data.get('date')) if data.get('timestamp') or data.get('date') else ''}).")
            elif name == "web_search" and data and data.get("results"):
                lines = [f"- [{x['title']}]({x['url']}): {x['snippet'][:160]}" for x in data["results"][:3]]
                parts.append(("Resultados da web:\n" if pt else "From the web:\n") + "\n".join(lines))
            elif name == "get_receivables" and data and "error" not in data:
                parts.append(self._render_receivables(data, pt))
            elif name == "get_transactions" and data:
                txs = data.get("transactions", [])
                lines = [f"- {t['date']} {t['time']} · {_brl(t['amount_brl'])} · {t['card_brand'] or 'Pix'} "
                         f"····{t['card_last4'] or ''} · {'código' if pt else 'code'} **{t['response_code']}**" for t in txs[:5]]
                parts.append(("Transações recusadas recentes:\n" if pt else "Recent declined transactions:\n") + "\n".join(lines))
            elif name == "lookup_error_code" and data and "meaning" in data:
                lookups.append(f"- **{data['code']}** — {data['meaning']} ({'origem' if pt else 'source'}: {data['who']}). {data['action']}")
            elif name == "get_terminal_status" and data:
                parts.append(self._render_terminals(data["terminals"], pt))
            elif name == "get_merchant_profile" and data and "error" not in data:
                parts.append(f"{'Plano' if pt else 'Plan'}: **{data['plan']['name']}** — {data['plan']['description']} "
                             f"Pix: {'habilitado' if data['pix']['enabled'] else 'não habilitado'}."
                             if pt else
                             f"Plan: **{data['plan']['name']}** — {data['plan']['description']} "
                             f"Pix: {'enabled' if data['pix']['enabled'] else 'not enabled'}.")
            elif name == "open_support_ticket" and data:
                parts.append(f"{'Chamado aberto' if pt else 'Ticket opened'}: **{data['ticket_id']}** (SLA {data['sla']}).")
            elif name == "escalate_to_human":
                parts.append("Vou te transferir para um especialista." if pt else "I'll transfer you to a specialist.")
            elif data and "error" in data:
                parts.append(("Não consegui consultar essa informação agora." if pt
                              else "I couldn't retrieve that information right now.") + f" ({data['error']})")
        if lookups:
            parts.append(("**O que significa e o que fazer:**\n" if pt else "**What it means and what to do:**\n") + "\n".join(lookups))
        return "\n\n".join(p for p in parts if p)

    @staticmethod
    def _render_kb(raw: str, pt: bool) -> str:
        if "NO_RELEVANT_DOCUMENTS_FOUND" in raw:
            return ("Não encontrei essa informação no material oficial da Getnet." if pt
                    else "I couldn't find this in Getnet's official material.")
        blocks = re.split(r"\n\n(?=\[\d+\] )", raw)
        out = []
        for b in blocks[:2]:
            m = re.match(r"\[(\d+)\] (.*?)\nURL: (.*?)\n(.*)", b, re.DOTALL)
            if m:
                body = " ".join(m.group(4).split())
                out.append(f"{body[:420]}{'…' if len(body) > 420 else ''} [{m.group(1)}]")
        return "\n\n".join(out)

    @staticmethod
    def _render_receivables(d: dict, pt: bool) -> str:
        acct = d["destination_account"]
        lines = []
        for p in d["payout_schedule"][:6]:
            status = {"paid": "pago" if pt else "paid", "credited_today": "creditado hoje" if pt else "credited today",
                      "scheduled": "agendado" if pt else "scheduled"}[p["status"]]
            lines.append(f"- **{p['payment_date']}** ({p['weekday']}): {_brl(p['net_amount_brl'])} — {', '.join(p['methods'])} — {status}")
        period = d["sale_period"]["start"]
        head = (f"Suas vendas aprovadas de {period} somam {_brl(d['gross_amount_brl'])} (líquido {_brl(d['net_amount_brl'])}). Recebimentos:"
                if pt else
                f"Your approved sales from {period} total {_brl(d['gross_amount_brl'])} gross ({_brl(d['net_amount_brl'])} net). Payouts:")
        tail = (f"Regra do plano *{d['plan']}*: {d['settlement_rule']} Conta de destino: {acct['bank']}, ag. {acct['agency']}, cc {acct['account']}."
                if pt else
                f"Your plan *{d['plan']}*: {d['settlement_rule']} Destination account: {acct['bank']}, branch {acct['agency']}, account {acct['account']}.")
        return "\n".join([head, *lines, "", tail])

    @staticmethod
    def _render_terminals(terminals: list[dict], pt: bool) -> str:
        out = []
        for t in terminals:
            c = t["connectivity"]
            if t["status"] == "online":
                out.append(f"- {t['model']} {t['serial']}: online ✅")
                continue
            tips = []
            if c.get("wifi_signal") == "weak":
                tips.append("sinal de Wi-Fi fraco: aproxime a maquininha do roteador ou reinicie o roteador" if pt
                            else "weak Wi-Fi signal: move the machine closer to the router or restart the router")
            if not c.get("fallback_to_sim_enabled"):
                tips.append("ative o uso do chip (4G) como alternativa nas configurações de Conexões" if pt
                            else "enable the SIM (4G) fallback in the Connections settings")
            if t.get("firmware") != t.get("firmware_latest"):
                tips.append(f"atualize o sistema ({t['firmware']} → {t['firmware_latest']}) assim que conectar" if pt
                            else f"update the firmware ({t['firmware']} → {t['firmware_latest']}) once connected")
            tips.append("reinicie a maquininha" if pt else "restart the machine")
            out.append(f"- {t['model']} {t['serial']}: **offline** ({t['last_error']}, "
                       f"{'visto há' if pt else 'last seen'} {t['last_seen_hours_ago']}h). "
                       + ("Recomendo: " if pt else "Try: ") + "; ".join(tips) + ".")
        return ("Status das suas maquininhas:\n" if pt else "Your card machines:\n") + "\n".join(out)
