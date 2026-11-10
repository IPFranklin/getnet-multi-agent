"""LLM provider factory (Anthropic / OpenAI / offline fake)."""

from __future__ import annotations

from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import Settings

Role = Literal["router", "agent"]


def _temperature_kwargs(settings: Settings) -> dict:
    return {} if settings.llm_temperature is None else {"temperature": settings.llm_temperature}


def build_chat_model(settings: Settings, role: Role = "agent") -> BaseChatModel:
    model = settings.resolved_router_model() if role == "router" else settings.resolved_agent_model()
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # Newer Claude models reject `temperature` (HTTP 400 "temperature is deprecated for this model"),
        # so it is only sent when explicitly configured via LLM_TEMPERATURE.
        return ChatAnthropic(
            model=model, api_key=settings.anthropic_api_key,
            max_tokens=settings.llm_max_tokens, timeout=settings.llm_timeout_s, max_retries=settings.llm_max_retries,
            **_temperature_kwargs(settings),
        )
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model, api_key=settings.openai_api_key,
            max_tokens=settings.llm_max_tokens, timeout=settings.llm_timeout_s, max_retries=settings.llm_max_retries,
            **_temperature_kwargs(settings),
        )
    from app.llm_fake import FakeChatModel

    return FakeChatModel()
