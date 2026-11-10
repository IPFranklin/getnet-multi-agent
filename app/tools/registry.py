"""Tool factories for the Knowledge and Customer Support agents.

Tools are built *per request* so the authenticated ``user_id`` is bound in a closure and never
exposed as a tool argument — the LLM cannot query another merchant's data even if a prompt
injection asks it to (prevents IDOR through the model).

A ``ToolContext`` collects side effects (retrieved sources, handoff requests, tickets) that the
orchestrator turns into structured fields of the API response.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import timedelta

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from app.config import Settings
from app.rag.retriever import KnowledgeRetriever
from app.services.merchant_repository import MerchantNotFound, MerchantRepository, parse_date
from app.services.support_desk import SupportDesk
from app.tools import error_codes, web_tools


@dataclass
class ToolContext:
    sources: list[dict] = field(default_factory=list)
    handoff_reason: str | None = None
    tickets: list[dict] = field(default_factory=list)

    def add_sources(self, items: list[dict]) -> None:
        seen = {(s["url"], s.get("section")) for s in self.sources}
        for s in items:
            if (s["url"], s.get("section")) not in seen:
                self.sources.append(s)
                seen.add((s["url"], s.get("section")))


def _dump(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


# --------------------------------------------------------------------------- arg schemas
class NoArgs(BaseModel):
    pass


class KBQuery(BaseModel):
    query: str = Field(description="Self-contained search query in Portuguese (the knowledge base is mostly PT-BR), "
                                   "including product names, e.g. 'diferença Get Clássica Get Smart'.")


class WebQuery(BaseModel):
    query: str = Field(description="Concise web search query.")


class WeatherArgs(BaseModel):
    city: str = Field(description="City name, e.g. 'Porto Alegre'.")
    day_offset: int = Field(default=1, ge=0, le=6, description="0=today, 1=tomorrow, ... up to 6.")


class FxArgs(BaseModel):
    base_currency: str = Field(description="ISO code of the currency to price, e.g. EUR, USD.")
    quote_currency: str = Field(default="BRL", description="ISO code to quote against (default BRL).")


class DateRange(BaseModel):
    start_date: str = Field(default="yesterday", description="Sale date range start: ISO date (YYYY-MM-DD), 'today' or 'yesterday'.")
    end_date: str | None = Field(default=None, description="Range end (inclusive). Defaults to start_date.")


class TxArgs(DateRange):
    status: str | None = Field(default=None, description="Filter: 'approved' or 'declined'.")
    limit: int = Field(default=15, ge=1, le=50)


class TerminalArgs(BaseModel):
    serial: str | None = Field(default=None, description="Terminal serial number; omit to list all terminals.")


class CodeArgs(BaseModel):
    code: str = Field(description="Response/error code shown on the terminal or transaction, e.g. '51', '05', 'ADQ 1-06'.")


class TicketArgs(BaseModel):
    category: str = Field(description="One of: terminal_maintenance, settlement_dispute, account_change, other.")
    summary: str = Field(description="Clear description of the problem and what was already tried.")
    priority: str = Field(default="medium", description="low | medium | high | urgent")


class EscalateArgs(BaseModel):
    reason: str = Field(description="Why a human is needed (e.g. 'customer asked for a human', 'possible fraud', "
                                    "'issue persists after troubleshooting').")


# --------------------------------------------------------------------------- knowledge tools
def build_knowledge_tools(retriever: KnowledgeRetriever, settings: Settings, ctx: ToolContext) -> list[BaseTool]:
    async def search_kb(query: str) -> str:
        chunks = retriever.retrieve(query)
        ctx.add_sources([c.as_source() for c in chunks])
        return KnowledgeRetriever.format_context(chunks)

    async def search_web(query: str) -> str:
        out = await web_tools.web_search(query, settings)
        try:
            for r in json.loads(out).get("results", [])[:3]:
                ctx.add_sources([{"title": r["title"], "section": "web", "url": r["url"], "score": None}])
        except (ValueError, KeyError, TypeError):
            pass
        return out

    async def weather(city: str, day_offset: int = 1) -> str:
        out = await web_tools.get_weather_forecast(city, settings, day_offset)
        if '"error"' not in out:
            ctx.add_sources([{"title": "Open-Meteo forecast", "section": city, "url": "https://open-meteo.com", "score": None}])
        return out

    async def fx(base_currency: str, quote_currency: str = "BRL") -> str:
        out = await web_tools.get_exchange_rate(base_currency, settings, quote_currency)
        if '"error"' not in out:
            src = json.loads(out).get("source", "FX API")
            ctx.add_sources([{"title": src, "section": f"{base_currency}/{quote_currency}", "url": "https://economia.awesomeapi.com.br", "score": None}])
        return out

    return [
        StructuredTool.from_function(
            coroutine=search_kb, name="search_getnet_knowledge_base", args_schema=KBQuery,
            description="Search Getnet's official knowledge base (products, card machines, fees policy, Pix, "
                        "antecipação, crediário, payment link, WhatsApp, manuals, troubleshooting). ALWAYS use for any "
                        "question about Getnet. Returns numbered passages [n] with URLs to cite.",
        ),
        StructuredTool.from_function(
            coroutine=search_web, name="web_search", args_schema=WebQuery,
            description="General web search for questions NOT about Getnet or that need fresh public information "
                        "(news, general facts). Do not use for the merchant's private data.",
        ),
        StructuredTool.from_function(
            coroutine=weather, name="get_weather_forecast", args_schema=WeatherArgs,
            description="Daily weather forecast for a city (today up to 6 days ahead).",
        ),
        StructuredTool.from_function(
            coroutine=fx, name="get_exchange_rate", args_schema=FxArgs,
            description="Current exchange rate between two currencies (e.g. EUR to BRL).",
        ),
    ]


# --------------------------------------------------------------------------- support tools
def build_support_tools(
    user_id: str, repo: MerchantRepository, desk: SupportDesk, retriever: KnowledgeRetriever, ctx: ToolContext
) -> list[BaseTool]:
    def _guard(fn):
        async def inner(**kwargs):
            try:
                return _dump(fn(**kwargs))
            except MerchantNotFound:
                return _dump({"error": "merchant_not_found", "detail": "No Getnet account is linked to this user."})
        return inner

    def profile() -> dict:
        return repo.get_profile(user_id)

    def receivables(start_date: str = "yesterday", end_date: str | None = None) -> dict:
        ref = repo.today()
        start = parse_date(start_date, ref - timedelta(days=1), ref)
        end = parse_date(end_date, start, ref) if end_date else start
        return repo.get_receivables(user_id, start, end)

    def transactions(start_date: str = "yesterday", end_date: str | None = None, status: str | None = None, limit: int = 15) -> dict:
        ref = repo.today()
        start = parse_date(start_date, ref - timedelta(days=1), ref)
        end = parse_date(end_date, ref, ref) if end_date else ref
        txs = repo.list_transactions(user_id, start, end, status=status, limit=limit)
        return {"count": len(txs), "transactions": txs}

    def terminal(serial: str | None = None) -> dict:
        return {"terminals": repo.get_terminals(user_id, serial)}

    async def code_lookup(code: str) -> str:
        info = error_codes.lookup(code)
        return _dump({"code": code, **info} if info else {"code": code, "error": "unknown_code"})

    async def help_center(query: str) -> str:
        chunks = retriever.retrieve(query)
        ctx.add_sources([c.as_source() for c in chunks])
        return KnowledgeRetriever.format_context(chunks)

    async def ticket(category: str, summary: str, priority: str = "medium") -> str:
        t = desk.open_ticket(user_id, category, summary, priority)
        ctx.tickets.append(t)
        return _dump({"ticket_id": t["ticket_id"], "sla": t["sla"], "status": t["status"]})

    async def escalate(reason: str) -> str:
        ctx.handoff_reason = reason
        return _dump({"status": "handoff_scheduled", "note": "A human specialist will take over after your reply. "
                                                              "Tell the customer you are transferring them."})

    return [
        StructuredTool.from_function(coroutine=_guard(profile), name="get_merchant_profile", args_schema=NoArgs,
                                     description="Merchant's registration: plan, settlement rule, fees, bank domicile, Pix status, enabled products and terminals."),
        StructuredTool.from_function(coroutine=_guard(receivables), name="get_receivables", args_schema=DateRange,
                                     description="Expected payout schedule (agenda de recebíveis) for sales made in a date range: payment dates, net amounts, status (paid/credited_today/scheduled) and destination account."),
        StructuredTool.from_function(coroutine=_guard(transactions), name="get_transactions", args_schema=TxArgs,
                                     description="Recent card/Pix transactions with status and authorization response codes (use status='declined' to investigate declines)."),
        StructuredTool.from_function(coroutine=_guard(terminal), name="get_terminal_status", args_schema=TerminalArgs,
                                     description="Live status of the merchant's card machines: online/offline, last error, Wi-Fi/SIM connectivity, firmware, battery."),
        StructuredTool.from_function(coroutine=code_lookup, name="lookup_error_code", args_schema=CodeArgs,
                                     description="Explain a decline/response or terminal error code and the recommended action."),
        StructuredTool.from_function(coroutine=help_center, name="search_help_center", args_schema=KBQuery,
                                     description="Search Getnet manuals and help articles (troubleshooting steps, how-tos)."),
        StructuredTool.from_function(coroutine=ticket, name="open_support_ticket", args_schema=TicketArgs,
                                     description="Open a support ticket (e.g. terminal replacement/maintenance) when the issue needs back-office action."),
        StructuredTool.from_function(coroutine=escalate, name="escalate_to_human", args_schema=EscalateArgs,
                                     description="Transfer the conversation to a human specialist. Use when the customer asks for a person, for suspected fraud, disputes, or when you cannot resolve the issue."),
    ]
