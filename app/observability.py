"""Structured logging, request tracing and Prometheus metrics.

Every request gets a ``trace_id`` stored in a context variable, so every log line emitted
while handling it (by any agent, tool or node) carries the same id. Logs are single-line
JSON, ready for Loki / Datadog / CloudWatch. Metrics are exposed at ``GET /metrics``.
"""

from __future__ import annotations

import functools
import json
import logging
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from prometheus_client import Counter, Histogram

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")

# --------------------------------------------------------------------------- metrics
REQUESTS = Counter("chat_requests_total", "Chat requests", ["route", "status"])
REQUEST_LATENCY = Histogram(
    "chat_request_latency_seconds", "End-to-end chat latency",
    buckets=(0.25, 0.5, 1, 2, 4, 8, 15, 30, 60),
)
NODE_LATENCY = Histogram("agent_node_latency_seconds", "Latency per graph node", ["node"])
NODE_ERRORS = Counter("agent_node_errors_total", "Unhandled errors per graph node", ["node"])
TOOL_CALLS = Counter("agent_tool_calls_total", "Tool invocations", ["agent", "tool", "status"])
LLM_TOKENS = Counter("llm_tokens_total", "LLM tokens consumed", ["agent", "kind"])
GUARDRAIL_EVENTS = Counter("guardrail_events_total", "Guardrail detections", ["stage", "category", "action"])
HANDOFFS = Counter("human_handoffs_total", "Escalations to a human operator", ["reason_category"])
RETRIEVAL_HITS = Histogram("rag_retrieved_chunks", "Relevant chunks per retrieval", buckets=(0, 1, 2, 3, 5, 8))


# --------------------------------------------------------------------------- logging
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": trace_id_var.get(),
        }
        extra = getattr(record, "fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "urllib3", "primp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_event(logger: logging.Logger, msg: str, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, msg, extra={"fields": fields})


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


# --------------------------------------------------------------------------- node tracing
_logger = logging.getLogger("app.trace")

NodeFn = Callable[[dict], Awaitable[dict]]


def traced_node(name: str) -> Callable[[NodeFn], NodeFn]:
    """Wrap a LangGraph node: time it, log it, record metrics and append a trace step.

    A node that raises is converted into a graceful failure state instead of crashing the
    whole graph, so the user always gets an answer (and the error is observable).
    """

    def decorator(fn: NodeFn) -> NodeFn:
        @functools.wraps(fn)
        async def wrapper(state: dict) -> dict:
            start = time.perf_counter()
            status = "ok"
            try:
                update = await fn(state)
            except Exception as exc:  # noqa: BLE001 - last-resort safety net
                status = "error"
                NODE_ERRORS.labels(node=name).inc()
                _logger.exception("node_failed", extra={"fields": {"node": name}})
                update = {
                    "errors": [f"{name}: {type(exc).__name__}: {exc}"],
                    "handoff_requested": name != "escalation",
                    "handoff_reason": f"internal_error_in_{name}",
                }
            elapsed = time.perf_counter() - start
            NODE_LATENCY.labels(node=name).observe(elapsed)
            step = {"node": name, "latency_ms": round(elapsed * 1000, 1), "status": status}
            for key in ("route", "tools_used"):
                if key in update and update[key]:
                    step[key] = update[key] if key != "route" else update[key].get("intent")
            log_event(_logger, "node_completed", **step)
            update.setdefault("steps", [])
            update["steps"] = [step]
            return update

        return wrapper

    return decorator


def record_usage(agent: str, ai_message: Any) -> None:
    usage = getattr(ai_message, "usage_metadata", None) or {}
    for kind in ("input_tokens", "output_tokens"):
        if usage.get(kind):
            LLM_TOKENS.labels(agent=agent, kind=kind).inc(usage[kind])
