"""HTTP contract tests for the FastAPI layer."""


def test_chat_contract(client):
    resp = client.post("/chat", json={"message": "Can I sell through WhatsApp using the Payment Link?", "user_id": "cliente1988"})
    assert resp.status_code == 200
    body = resp.json()
    for field in ("response", "agent", "route", "sources", "tools_used", "trace_id", "latency_ms", "guardrails"):
        assert field in body
    assert body["agent"] == "knowledge" and body["sources"][0]["url"].startswith("http")


def test_versioned_route_alias(client):
    resp = client.post("/v1/chat", json={"message": "What's the euro exchange rate today?", "user_id": "cliente1988"})
    assert resp.status_code == 200 and resp.json()["agent"] == "knowledge"


def test_validation_errors(client):
    assert client.post("/chat", json={"message": "", "user_id": "x"}).status_code == 422
    assert client.post("/chat", json={"message": "hi"}).status_code == 422
    assert client.post("/chat", json={"message": "hi", "user_id": "bad id with spaces"}).status_code == 422


def test_health_ready_metrics(client):
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready").json()
    assert ready["status"] == "ready" and ready["chunks"] > 0
    client.post("/chat", json={"message": "My card machine is showing a transaction decline error.", "user_id": "cliente1988"})
    metrics = client.get("/metrics").text
    assert "chat_requests_total" in metrics and "agent_tool_calls_total" in metrics


def test_admin_endpoints(client):
    client.post("/chat", json={"message": "quero falar com um atendente", "user_id": "cliente1988"})
    handoffs = client.get("/admin/handoffs", params={"user_id": "cliente1988"}).json()
    assert handoffs and handoffs[-1]["status"] == "waiting_agent"
    assert client.post("/admin/reindex").json()["status"] == "reindexed"
