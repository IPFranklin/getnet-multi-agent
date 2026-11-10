"""Brazilian business-day arithmetic used for settlement (D+N) calculations."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

# National bank holidays (fixed + moveable) for 2025-2027. In production this comes from the
# settlement engine / FEBRABAN calendar API.
BANK_HOLIDAYS = {
    date(2025, 1, 1), date(2025, 3, 3), date(2025, 3, 4), date(2025, 4, 18), date(2025, 4, 21),
    date(2025, 5, 1), date(2025, 6, 19), date(2025, 9, 7), date(2025, 10, 12), date(2025, 11, 2),
    date(2025, 11, 15), date(2025, 11, 20), date(2025, 12, 25),
    date(2026, 1, 1), date(2026, 2, 16), date(2026, 2, 17), date(2026, 4, 3), date(2026, 4, 21),
    date(2026, 5, 1), date(2026, 6, 4), date(2026, 9, 7), date(2026, 10, 12), date(2026, 11, 2),
    date(2026, 11, 15), date(2026, 11, 20), date(2026, 12, 25),
    date(2027, 1, 1), date(2027, 2, 8), date(2027, 2, 9), date(2027, 3, 26), date(2027, 4, 21),
    date(2027, 5, 1), date(2027, 5, 27), date(2027, 9, 7), date(2027, 10, 12), date(2027, 11, 2),
    date(2027, 11, 15), date(2027, 11, 20), date(2027, 12, 25),
}


def today(tz: str = "America/Sao_Paulo") -> date:
    return datetime.now(ZoneInfo(tz)).date()


def is_business_day(d: date) -> bool:
    return d.weekday() < 5 and d not in BANK_HOLIDAYS


def add_business_days(d: date, n: int) -> date:
    if n == 0:
        return d
    current = d
    added = 0
    while added < n:
        current += timedelta(days=1)
        if is_business_day(current):
            added += 1
    return current


def add_calendar_days_then_business(d: date, days: int) -> date:
    """D+30 style: add calendar days, roll forward to next business day."""
    current = d + timedelta(days=days)
    while not is_business_day(current):
        current += timedelta(days=1)
    return current
