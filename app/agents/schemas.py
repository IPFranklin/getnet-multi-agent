"""Structured-output contracts and the shared LangGraph state."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

Intent = Literal["knowledge", "support", "escalation", "unsupported"]


class RouteDecision(BaseModel):
    """Router output (LLM structured output)."""

    intent: Intent = Field(description="Which specialist should handle the message.")
    confidence: float = Field(ge=0, le=1, description="Confidence in the routing decision (0-1).")
    language: str = Field(default="pt", description="ISO 639-1 code of the customer's language: pt, en or es.")
    reasoning: str = Field(description="One short sentence explaining the decision.")
    clarifying_question: str | None = Field(
        default=None, description="Only when confidence is low: a short question to disambiguate, in the customer's language."
    )


class HandoffSummary(BaseModel):
    """Escalation agent output (LLM structured output)."""

    summary: str = Field(description="2-4 sentence factual summary for the human specialist.")
    category: Literal["fraud", "complaint", "settlement", "terminal", "account", "information", "other"]
    priority: Literal["low", "medium", "high", "urgent"]
    customer_sentiment: Literal["positive", "neutral", "frustrated", "angry"] = "neutral"


class GraphState(TypedDict, total=False):
    trace_id: str
    user_id: str
    session_key: str
    message: str  # sanitized (PII redacted) user message
    history: list[dict]
    guard: dict
    route: dict
    agent: str
    response: str
    sources: list[dict]
    tools_used: Annotated[list[dict], operator.add]
    tickets: list[dict]
    handoff_requested: bool
    handoff_reason: str | None
    handoff: dict | None
    warnings: list[str]
    errors: Annotated[list[str], operator.add]
    steps: Annotated[list[dict], operator.add]
