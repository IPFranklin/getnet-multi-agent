"""The real web tools are tested against mocked HTTP (respx), bypassing the autouse stubs."""

import importlib
import json

import httpx
import pytest
import respx

import app.tools.web_tools as wt


@pytest.fixture
def real_tools():
    return importlib.reload(wt)  # undo the monkeypatched stubs for this module object


@respx.mock
async def test_weather_forecast_parses_open_meteo(settings, real_tools):
    respx.get("https://geocoding-api.open-meteo.com/v1/search").mock(return_value=httpx.Response(200, json={
        "results": [{"name": "Porto Alegre", "admin1": "Rio Grande do Sul", "country": "Brasil", "latitude": -30.03, "longitude": -51.23}]}))
    respx.get("https://api.open-meteo.com/v1/forecast").mock(return_value=httpx.Response(200, json={"daily": {
        "time": ["2026-10-23", "2026-10-24"], "weather_code": [0, 61], "temperature_2m_max": [25, 21],
        "temperature_2m_min": [14, 15], "precipitation_probability_max": [5, 70], "precipitation_sum": [0, 8.2],
        "wind_speed_10m_max": [10, 22]}}))
    out = json.loads(await real_tools.get_weather_forecast("Porto Alegre", settings, day_offset=1))
    assert out["date"] == "2026-10-24" and out["condition"] == "light rain" and out["precipitation_probability_pct"] == 70


@respx.mock
async def test_exchange_rate_falls_back_to_frankfurter(settings, real_tools):
    respx.get("https://economia.awesomeapi.com.br/json/last/EUR-BRL").mock(return_value=httpx.Response(503))
    respx.get("https://api.frankfurter.app/latest").mock(return_value=httpx.Response(200, json={"date": "2026-10-23", "rates": {"BRL": 6.21}}))
    out = json.loads(await real_tools.get_exchange_rate("eur", settings))
    assert out["rate"] == 6.21 and "Frankfurter" in out["source"]


@respx.mock
async def test_tools_never_raise_on_network_errors(settings, real_tools):
    respx.get("https://geocoding-api.open-meteo.com/v1/search").mock(side_effect=httpx.ConnectError("down"))
    out = json.loads(await real_tools.get_weather_forecast("Porto Alegre", settings))
    assert "error" in out
