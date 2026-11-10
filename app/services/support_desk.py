"""Support tickets and human-handoff queue.

Tickets/handoffs are appended to JSONL files (a stand-in for Zendesk/Salesforce Service
Cloud). When ``HANDOFF_WEBHOOK_URL`` is configured, every handoff is also posted to a
Microsoft Teams / Slack incoming webhook so an operator is paged in real time.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx

from app.observability import log_event, trace_id_var

logger = logging.getLogger(__name__)

SLA_BY_PRIORITY = {"urgent": "15 minutes", "high": "1 hour", "medium": "4 business hours", "low": "1 business day"}


class SupportDesk:
    def __init__(self, tickets_path: Path, handoffs_path: Path, webhook_url: str | None = None, timeout: float = 5.0):
        self.tickets_path = tickets_path
        self.handoffs_path = handoffs_path
        self.webhook_url = webhook_url
        self.timeout = timeout
        self._lock = threading.Lock()
        for p in (tickets_path, handoffs_path):
            p.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, path: Path, record: dict) -> None:
        with self._lock, open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _read(path: Path, user_id: str | None = None) -> list[dict]:
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        return [r for r in rows if user_id is None or r.get("user_id") == user_id]

    # ------------------------------------------------------------------ tickets
    def open_ticket(self, user_id: str, category: str, summary: str, priority: str = "medium") -> dict:
        now = datetime.now(UTC)
        ticket = {
            "ticket_id": f"TCK-{now:%Y%m%d}-{uuid.uuid4().hex[:6].upper()}",
            "user_id": user_id, "category": category, "summary": summary[:1000],
            "priority": priority, "status": "open", "created_at": now.isoformat(),
            "sla": SLA_BY_PRIORITY.get(priority, SLA_BY_PRIORITY["medium"]), "trace_id": trace_id_var.get(),
        }
        self._append(self.tickets_path, ticket)
        log_event(logger, "ticket_opened", ticket_id=ticket["ticket_id"], category=category, priority=priority)
        return ticket

    def list_tickets(self, user_id: str | None = None) -> list[dict]:
        return self._read(self.tickets_path, user_id)

    # ------------------------------------------------------------------ handoffs
    async def create_handoff(self, user_id: str, reason: str, summary: str, priority: str,
                             category: str, transcript: list[dict]) -> dict:
        now = datetime.now(UTC)
        handoff = {
            "handoff_id": f"HO-{now:%Y%m%d}-{uuid.uuid4().hex[:6].upper()}",
            "user_id": user_id, "reason": reason, "category": category, "priority": priority,
            "summary": summary[:2000], "transcript": transcript[-10:], "status": "waiting_agent",
            "created_at": now.isoformat(), "sla": SLA_BY_PRIORITY.get(priority, SLA_BY_PRIORITY["medium"]),
            "trace_id": trace_id_var.get(), "notified": False,
        }
        handoff["notified"] = await self._notify(handoff)
        self._append(self.handoffs_path, handoff)
        log_event(logger, "handoff_created", handoff_id=handoff["handoff_id"], reason=reason, priority=priority)
        return handoff

    def list_handoffs(self, user_id: str | None = None) -> list[dict]:
        return self._read(self.handoffs_path, user_id)

    async def _notify(self, handoff: dict) -> bool:
        if not self.webhook_url:
            return False
        text = (
            f"🚨 Human handoff requested ({handoff['priority'].upper()})\n"
            f"Merchant: {handoff['user_id']} | Reason: {handoff['reason']}\n"
            f"Summary: {handoff['summary']}\nTicket: {handoff['handoff_id']} | trace: {handoff['trace_id']}"
        )
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(self.webhook_url, json={"text": text})
                resp.raise_for_status()
            return True
        except httpx.HTTPError as exc:
            logger.warning("handoff_webhook_failed", extra={"fields": {"error": str(exc)}})
            return False
