"""Merchant data access layer (mocked).

In production this would call Getnet's internal APIs (merchant registry, transactions,
settlement/agenda de recebíveis, terminal management). Here the data is generated
deterministically *relative to today* so questions like "when will yesterday's sales be
deposited?" always have realistic answers. The interface is what matters: tools only talk
to this repository, never to raw storage, and every method is scoped by ``user_id``.
"""

from __future__ import annotations

import random
import zlib
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta

from app.services.calendar import add_business_days, add_calendar_days_then_business
from app.services.calendar import today as tz_today

# --------------------------------------------------------------------------- static registry
MERCHANTS: dict[str, dict] = {
    "cliente1988": {
        "merchant_id": "EC-1029384",
        "trade_name": "Padaria Pão Dourado",
        "legal_name": "Pão Dourado Panificadora LTDA",
        "document": "CNPJ **.***.678/0001-**",
        "segment": "Bakery",
        "city": "Porto Alegre - RS",
        "since": "2021-03-15",
        "plan": {
            "name": "Get Smart — Receba em 1 dia útil",
            "settlement": "d1",
            "description": "Debit, credit and installment sales are credited 1 business day after the sale "
                           "(installments are automatically anticipated). Pix is credited instantly.",
            "fees": {"pix": 0.0099, "debit": 0.0119, "credit_1x": 0.0349, "credit_installments": 0.0549},
            "monthly_rent_brl": 0.0,
            "rent_note": "Rent waived: monthly volume above the plan threshold.",
        },
        "bank_domicile": {"bank": "Santander (033)", "agency": "1234", "account": "*****678-9"},
        "pix": {"enabled": True, "key_type": "CNPJ"},
        "products": {"crediario": False, "link_get_pay": True, "whatsapp_pay": False, "get_tap": False},
        "terminals": [
            {
                "serial": "GS7781-2203", "model": "Get Smart", "status": "offline",
                "last_seen_hours_ago": 3, "last_error": "NO_NETWORK",
                "connectivity": {"mode": "wifi", "wifi_ssid": "PaoDourado_2G", "wifi_signal": "weak",
                                  "sim_carrier": "Vivo", "sim_status": "active", "sim_signal": "no_signal",
                                  "fallback_to_sim_enabled": False},
                "firmware": "4.2.1", "firmware_latest": "4.3.0", "battery_pct": 64,
            },
            {
                "serial": "GC5520-0911", "model": "Get Clássica", "status": "online",
                "last_seen_hours_ago": 0, "last_error": None,
                "connectivity": {"mode": "sim", "wifi_ssid": None, "wifi_signal": None,
                                  "sim_carrier": "Claro", "sim_status": "active", "sim_signal": "good",
                                  "fallback_to_sim_enabled": True},
                "firmware": "3.8.0", "firmware_latest": "3.8.0", "battery_pct": 91,
            },
        ],
    },
    "cliente2024": {
        "merchant_id": "EC-5566771",
        "trade_name": "Studio Bella Forma",
        "legal_name": "Bella Forma Estética ME",
        "document": "CNPJ **.***.221/0001-**",
        "segment": "Health & Beauty",
        "city": "São Paulo - SP",
        "since": "2023-08-01",
        "plan": {
            "name": "Get Clássica — Receba conforme as parcelas",
            "settlement": "installments",
            "description": "Debit is credited in 1 business day; credit in 30 days; installment sales are "
                           "credited monthly, one installment every 30 days. Anticipation available on demand.",
            "fees": {"pix": 0.0099, "debit": 0.0099, "credit_1x": 0.0299, "credit_installments": 0.0399},
            "monthly_rent_brl": 79.90,
            "rent_note": "Rent charged: monthly volume below the waiver threshold.",
        },
        "bank_domicile": {"bank": "Itaú (341)", "agency": "0987", "account": "*****321-0"},
        "pix": {"enabled": False, "key_type": None, "reason": "Pix on the terminal requires a Santander bank domicile."},
        "products": {"crediario": True, "link_get_pay": True, "whatsapp_pay": True, "get_tap": False},
        "terminals": [
            {
                "serial": "GC9001-4455", "model": "Get Clássica", "status": "online",
                "last_seen_hours_ago": 0, "last_error": None,
                "connectivity": {"mode": "sim", "wifi_ssid": None, "wifi_signal": None,
                                  "sim_carrier": "TIM", "sim_status": "active", "sim_signal": "good",
                                  "fallback_to_sim_enabled": True},
                "firmware": "3.8.0", "firmware_latest": "3.8.0", "battery_pct": 77,
            },
        ],
    },
}

DECLINE_CODES = ["51", "05", "55", "54", "57", "91", "59"]


@dataclass
class Transaction:
    id: str
    date: str
    time: str
    terminal: str
    method: str  # pix | debit | credit_1x | credit_installments
    installments: int
    amount_brl: float
    status: str  # approved | declined
    response_code: str
    card_brand: str | None
    card_last4: str | None


class MerchantNotFound(LookupError):
    pass


class MerchantRepository:
    def __init__(self, today_fn: Callable[[], date] | None = None, history_days: int = 35):
        self._today = today_fn or tz_today
        self.history_days = history_days

    def today(self) -> date:
        return self._today()

    # ------------------------------------------------------------------ registry
    def get_profile(self, user_id: str) -> dict:
        m = MERCHANTS.get(user_id)
        if not m:
            raise MerchantNotFound(user_id)
        profile = {k: v for k, v in m.items() if k != "terminals"}
        profile["terminals"] = [{"serial": t["serial"], "model": t["model"], "status": t["status"]} for t in m["terminals"]]
        return profile

    def get_terminals(self, user_id: str, serial: str | None = None) -> list[dict]:
        m = MERCHANTS.get(user_id)
        if not m:
            raise MerchantNotFound(user_id)
        terminals = m["terminals"]
        if serial:
            terminals = [t for t in terminals if t["serial"].lower() == serial.lower()]
        return terminals

    # ------------------------------------------------------------------ transactions
    def _generate_day(self, user_id: str, day: date) -> list[Transaction]:
        m = MERCHANTS[user_id]
        rng = random.Random(f"{user_id}:{day.isoformat()}")
        terminals = [t["serial"] for t in m["terminals"]]
        n = rng.randint(8, 14) if day.weekday() < 6 else rng.randint(4, 8)
        is_today = day == self._today()
        txs = []
        for i in range(n):
            method = rng.choices(["pix", "debit", "credit_1x", "credit_installments"], [35, 35, 20, 10])[0]
            inst = rng.randint(2, 6) if method == "credit_installments" else 1
            amount = round(rng.uniform(12, 90) * (3 if inst > 1 else 1), 2)
            hour = rng.randint(7, 11 if is_today else 20)
            declined = method != "pix" and rng.random() < 0.08
            txs.append(Transaction(
                id=f"TX{day.strftime('%y%m%d')}{i:03d}{zlib.crc32(user_id.encode()) % 97:02d}",
                date=day.isoformat(), time=time(hour, rng.randint(0, 59)).strftime("%H:%M"),
                terminal=rng.choice(terminals), method=method, installments=inst, amount_brl=amount,
                status="declined" if declined else "approved",
                response_code=rng.choice(DECLINE_CODES) if declined else "00",
                card_brand=None if method == "pix" else rng.choice(["Visa", "Mastercard", "Elo", "Amex", "Hipercard"]),
                card_last4=None if method == "pix" else f"{rng.randint(0, 9999):04d}",
            ))
        # Guarantee recent declines exist so troubleshooting scenarios are reproducible.
        if (self._today() - day).days in (0, 1):
            code = "51" if day == self._today() else "05"
            txs.append(Transaction(
                id=f"TX{day.strftime('%y%m%d')}9{len(txs):02d}", date=day.isoformat(), time="10:42",
                terminal=terminals[-1], method="credit_1x", installments=1, amount_brl=148.50,
                status="declined", response_code=code, card_brand="Visa", card_last4="4821",
            ))
        return sorted(txs, key=lambda t: t.time)

    def list_transactions(self, user_id: str, start: date, end: date, status: str | None = None,
                          method: str | None = None, limit: int = 50) -> list[dict]:
        if user_id not in MERCHANTS:
            raise MerchantNotFound(user_id)
        start = max(start, self._today() - timedelta(days=self.history_days))
        end = min(end, self._today())
        out: list[Transaction] = []
        d = start
        while d <= end:
            out.extend(self._generate_day(user_id, d))
            d += timedelta(days=1)
        if status:
            out = [t for t in out if t.status == status]
        if method:
            out = [t for t in out if t.method == method]
        out.sort(key=lambda t: (t.date, t.time), reverse=True)
        return [asdict(t) for t in out[:limit]]

    # ------------------------------------------------------------------ settlement
    def _payments_for(self, tx: Transaction, plan: dict) -> list[tuple[date, float, str]]:
        sale_day = date.fromisoformat(tx.date)
        fee = plan["fees"][tx.method]
        net = round(tx.amount_brl * (1 - fee), 2)
        if tx.method == "pix":
            return [(sale_day, net, "pix")]
        if plan["settlement"] == "d1" or tx.method == "debit":
            return [(add_business_days(sale_day, 1), net, tx.method)]
        if tx.method == "credit_1x":
            return [(add_calendar_days_then_business(sale_day, 30), net, tx.method)]
        per = round(net / tx.installments, 2)
        return [(add_calendar_days_then_business(sale_day, 30 * i), per, f"installment {i}/{tx.installments}")
                for i in range(1, tx.installments + 1)]

    def get_receivables(self, user_id: str, start: date, end: date) -> dict:
        """Expected payouts for sales made between start and end (sale dates)."""
        m = MERCHANTS.get(user_id)
        if not m:
            raise MerchantNotFound(user_id)
        plan = m["plan"]
        today = self._today()
        txs = [Transaction(**t) for t in self.list_transactions(user_id, start, end, status="approved", limit=10_000)]
        by_payment: dict[str, dict] = defaultdict(lambda: {"gross_brl": 0.0, "net_brl": 0.0, "count": 0, "methods": set()})
        total_gross = total_net = 0.0
        for tx in txs:
            total_gross += tx.amount_brl
            for pay_day, amount, label in self._payments_for(tx, plan):
                bucket = by_payment[pay_day.isoformat()]
                bucket["net_brl"] += amount
                bucket["count"] += 1
                bucket["methods"].add(label.split(" ")[0])
                total_net += amount
        schedule = []
        for day_iso in sorted(by_payment):
            b = by_payment[day_iso]
            pay_day = date.fromisoformat(day_iso)
            schedule.append({
                "payment_date": day_iso,
                "weekday": pay_day.strftime("%A"),
                "net_amount_brl": round(b["net_brl"], 2),
                "items": b["count"],
                "methods": sorted(b["methods"]),
                "status": "paid" if pay_day < today else ("credited_today" if pay_day == today else "scheduled"),
            })
        return {
            "sale_period": {"start": start.isoformat(), "end": end.isoformat()},
            "today": today.isoformat(),
            "plan": plan["name"],
            "settlement_rule": plan["description"],
            "destination_account": m["bank_domicile"],
            "approved_sales": len(txs),
            "gross_amount_brl": round(total_gross, 2),
            "net_amount_brl": round(total_net, 2),
            "payout_schedule": schedule,
        }


_REPO: MerchantRepository | None = None


def get_repository() -> MerchantRepository:
    global _REPO
    if _REPO is None:
        _REPO = MerchantRepository()
    return _REPO


def set_repository(repo: MerchantRepository) -> None:
    global _REPO
    _REPO = repo


def parse_date(value: str | None, default: date, ref: date) -> date:
    """Accept ISO dates and a few relative keywords (today/yesterday, hoje/ontem)."""
    if not value:
        return default
    v = value.strip().lower()
    if v in {"today", "hoje"}:
        return ref
    if v in {"yesterday", "ontem"}:
        return ref - timedelta(days=1)
    try:
        return datetime.fromisoformat(v).date()
    except ValueError:
        return default
