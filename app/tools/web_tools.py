"""External, general-purpose tools: web search, weather forecast and FX rates.

All tools return compact JSON strings (LLM-friendly) and never raise: network failures are
reported back to the model as ``{"error": ...}`` so it can fall back to another tool or tell
the user the information is temporarily unavailable.
"""

from __future__ import annotations

import asyncio
import json
import logging

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "rime fog",
    51: "light drizzle", 53: "drizzle", 55: "dense drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers", 81: "heavy rain showers",
    82: "violent rain showers", 95: "thunderstorm", 96: "thunderstorm with hail", 99: "severe thunderstorm with hail",
}


def _dump(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


async def web_search(query: str, settings: Settings, max_results: int | None = None) -> str:
    n = max_results or settings.web_search_max_results
    try:
        if settings.tavily_api_key:
            async with httpx.AsyncClient(timeout=settings.http_timeout_s) as client:
                resp = await client.post(
                    "https://api.tavily.com/search",
                    json={"api_key": settings.tavily_api_key, "query": query, "max_results": n, "include_answer": True},
                )
                resp.raise_for_status()
                data = resp.json()
            results = [{"title": r.get("title"), "url": r.get("url"), "snippet": r.get("content", "")[:500]}
                       for r in data.get("results", [])]
            return _dump({"provider": "tavily", "answer": data.get("answer"), "results": results})

        from ddgs import DDGS

        def _search() -> list[dict]:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=n))

        raw = await asyncio.wait_for(asyncio.to_thread(_search), timeout=settings.http_timeout_s * 1.5)
        results = [{"title": r.get("title"), "url": r.get("href"), "snippet": (r.get("body") or "")[:500]} for r in raw]
        return _dump({"provider": "duckduckgo", "results": results})
    except Exception as exc:  # noqa: BLE001 - tool must never raise
        logger.warning("web_search_failed", extra={"fields": {"error": str(exc)}})
        return _dump({"error": f"web search unavailable: {type(exc).__name__}"})


async def get_weather_forecast(city: str, settings: Settings, day_offset: int = 1) -> str:
    """Open-Meteo (free, no key): geocode the city then fetch the daily forecast."""
    try:
        async with httpx.AsyncClient(timeout=settings.http_timeout_s) as client:
            geo = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1, "language": "pt", "format": "json"},
            )
            geo.raise_for_status()
            places = geo.json().get("results") or []
            if not places:
                return _dump({"error": f"city not found: {city}"})
            p = places[0]
            fc = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": p["latitude"], "longitude": p["longitude"], "timezone": "auto", "forecast_days": 7,
                    "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum,wind_speed_10m_max",
                },
            )
            fc.raise_for_status()
            daily = fc.json()["daily"]
        idx = max(0, min(day_offset, len(daily["time"]) - 1))
        return _dump({
            "source": "Open-Meteo (https://open-meteo.com)",
            "location": f"{p['name']}, {p.get('admin1', '')}, {p.get('country', '')}",
            "date": daily["time"][idx],
            "condition": WEATHER_CODES.get(daily["weather_code"][idx], "unknown"),
            "temp_min_c": daily["temperature_2m_min"][idx], "temp_max_c": daily["temperature_2m_max"][idx],
            "precipitation_probability_pct": daily["precipitation_probability_max"][idx],
            "precipitation_mm": daily["precipitation_sum"][idx], "wind_max_kmh": daily["wind_speed_10m_max"][idx],
        })
    except Exception as exc:  # noqa: BLE001
        logger.warning("weather_failed", extra={"fields": {"error": str(exc)}})
        return _dump({"error": f"weather service unavailable: {type(exc).__name__}"})


async def get_exchange_rate(base: str, settings: Settings, quote: str = "BRL") -> str:
    """AwesomeAPI (BR market quotes, free) with Frankfurter (ECB reference rates) as fallback."""
    base, quote = base.upper()[:3], quote.upper()[:3]
    try:
        async with httpx.AsyncClient(timeout=settings.http_timeout_s) as client:
            try:
                resp = await client.get(f"https://economia.awesomeapi.com.br/json/last/{base}-{quote}")
                resp.raise_for_status()
                q = resp.json()[f"{base}{quote}"]
                return _dump({
                    "source": "AwesomeAPI (economia.awesomeapi.com.br)", "pair": f"{base}/{quote}",
                    "bid": float(q["bid"]), "ask": float(q["ask"]), "high": float(q["high"]), "low": float(q["low"]),
                    "change_pct": float(q["pctChange"]), "timestamp": q.get("create_date"),
                })
            except (httpx.HTTPError, KeyError, ValueError):
                resp = await client.get("https://api.frankfurter.app/latest", params={"from": base, "to": quote})
                resp.raise_for_status()
                data = resp.json()
                return _dump({
                    "source": "Frankfurter / European Central Bank reference rate", "pair": f"{base}/{quote}",
                    "rate": data["rates"][quote], "date": data["date"],
                })
    except Exception as exc:  # noqa: BLE001
        logger.warning("fx_failed", extra={"fields": {"error": str(exc)}})
        return _dump({"error": f"exchange-rate service unavailable: {type(exc).__name__}"})

