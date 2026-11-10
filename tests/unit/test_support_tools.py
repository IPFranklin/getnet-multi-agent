import json
from datetime import date

import pytest

from app.services.calendar import add_business_days, add_calendar_days_then_business
from app.tools.error_codes import lookup
from app.tools.registry import ToolContext, build_support_tools
from tests.conftest import FIXED_TODAY


def test_business_days_skip_weekends_and_holidays():
    assert add_business_days(date(2026, 10, 23), 1) == date(2026, 10, 26)  # Fri -> Mon
    assert add_business_days(date(2026, 10, 30), 1) == date(2026, 11, 3)   # Fri -> Tue (Nov 2 holiday)
    assert add_calendar_days_then_business(date(2026, 10, 3), 30) == date(2026, 11, 3)


def test_receivables_d1_plan(repository):
    yesterday = date(2026, 10, 22)
    r = repository.get_receivables("cliente1988", yesterday, yesterday)
    pay_dates = {p["payment_date"]: p for p in r["payout_schedule"]}
    assert set(pay_dates) <= {"2026-10-22", "2026-10-23"}  # pix same day, cards D+1
    assert pay_dates["2026-10-23"]["status"] == "credited_today"
    assert r["net_amount_brl"] < r["gross_amount_brl"]


def test_receivables_installments_plan_spreads_payments(repository):
    start, end = date(2026, 10, 1), date(2026, 10, 22)
    r = repository.get_receivables("cliente2024", start, end)
    assert any(p["payment_date"] > "2026-11-15" for p in r["payout_schedule"])
    assert all(p["status"] in {"paid", "credited_today", "scheduled"} for p in r["payout_schedule"])


def test_data_is_deterministic(repository):
    a = repository.list_transactions("cliente1988", FIXED_TODAY, FIXED_TODAY)
    b = repository.list_transactions("cliente1988", FIXED_TODAY, FIXED_TODAY)
    assert a == b and any(t["status"] == "declined" for t in a)


def test_error_code_lookup():
    assert lookup("51")["who"] == "issuer"
    assert lookup("código 05")["meaning"].startswith("Not authorized")
    assert lookup("ADQ 1-06")["meaning"] == "Terminal not configured"
    assert lookup("XYZ") is None


@pytest.fixture
def tools(container):
    ctx = ToolContext()
    ts = build_support_tools("cliente1988", container.repository, container.desk, container.retriever, ctx)
    return {t.name: t for t in ts}, ctx


async def test_at_least_two_tools_and_no_user_id_argument(tools):
    by_name, _ = tools
    assert len(by_name) >= 2
    for t in by_name.values():
        assert "user_id" not in t.args, f"{t.name} must not accept user_id (IDOR protection)"


async def test_tools_are_scoped_to_bound_user(container):
    ctx = ToolContext()
    ts = {t.name: t for t in build_support_tools("cliente2024", container.repository, container.desk, container.retriever, ctx)}
    profile = json.loads(await ts["get_merchant_profile"].ainvoke({}))
    assert profile["trade_name"] == "Studio Bella Forma"


async def test_unknown_merchant_is_reported_not_raised(container):
    ctx = ToolContext()
    ts = {t.name: t for t in build_support_tools("ghost", container.repository, container.desk, container.retriever, ctx)}
    out = json.loads(await ts["get_receivables"].ainvoke({"start_date": "yesterday"}))
    assert out["error"] == "merchant_not_found"


async def test_receivables_tool_understands_relative_dates(tools):
    by_name, _ = tools
    out = json.loads(await by_name["get_receivables"].ainvoke({"start_date": "yesterday"}))
    assert out["sale_period"]["start"] == "2026-10-22" and out["today"] == "2026-10-23"


async def test_ticket_and_escalation_side_effects(tools, container):
    by_name, ctx = tools
    t = json.loads(await by_name["open_support_ticket"].ainvoke({"category": "terminal_maintenance", "summary": "screen broken"}))
    assert t["ticket_id"].startswith("TCK-") and ctx.tickets
    assert container.desk.list_tickets("cliente1988")[0]["ticket_id"] == t["ticket_id"]
    await by_name["escalate_to_human"].ainvoke({"reason": "customer asked"})
    assert ctx.handoff_reason == "customer asked"


async def test_help_center_collects_sources(tools):
    by_name, ctx = tools
    out = await by_name["search_help_center"].ainvoke({"query": "maquininha não conecta wi-fi"})
    assert "[1]" in out and ctx.sources
