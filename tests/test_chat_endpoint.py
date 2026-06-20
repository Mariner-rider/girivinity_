from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_chat_returns_200(monkeypatch):
    from app.api.routes import chat

    monkeypatch.setattr(
        chat.QueryRouter,
        "route",
        lambda self, query: {
            "source": "knowledge_base",
            "context_string": "Context:\n[1] Test chunk",
            "confidence": 0.9,
            "urls": [],
        },
    )

    client = TestClient(app)
    response = client.post("/chat/message", json={"query": "test question"})

    assert response.status_code == 200
    payload = response.json()
    assert "answer" in payload
    assert "source" in payload
    assert "confidence" in payload


def test_empty_query_returns_400():
    client = TestClient(app)
    response = client.post("/chat/message", json={"query": "  "})
    assert response.status_code == 400


def test_no_results_still_returns_200(monkeypatch):
    from app.api.routes import chat

    monkeypatch.setattr(
        chat.QueryRouter,
        "route",
        lambda self, query: {
            "source": "none",
            "context_string": "",
            "confidence": 0.0,
            "urls": [],
        },
    )

    client = TestClient(app)
    response = client.post("/chat/message", json={"query": "unknown"})

    assert response.status_code == 200
    assert "could not find" in response.json()["answer"].lower()
