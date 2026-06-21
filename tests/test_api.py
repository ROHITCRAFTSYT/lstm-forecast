"""API integration tests via FastAPI TestClient (offline, no API key)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from lstm_forecast.api.main import create_app


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def _values_request(horizon=4, test_length=8, epochs=10, lags=8):
    rng = np.random.default_rng(0)
    series = (np.cumsum(rng.normal(0, 1, size=120)) + 100).tolist()
    return {
        "series": {"values": series},
        "horizon": horizon,
        "test_length": test_length,
        "lags": lags,
        "epochs": epochs,
    }


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "ai_enabled" in body and "device" in body


def test_forecast_endpoint(client):
    r = client.post("/forecast", json=_values_request())
    assert r.status_code == 200
    body = r.json()
    assert len(body["forecast"]) == 4
    assert "lstm" in body["metrics"]
    assert body["best_model"] is not None
    pt = body["forecast"][0]
    assert pt["lower"] <= pt["point"] <= pt["upper"]


def test_forecast_validation_error(client):
    bad = _values_request()
    bad["series"] = {}  # neither ticker nor values
    r = client.post("/forecast", json=bad)
    assert r.status_code == 422


def test_insights_endpoint_offline(client):
    r = client.post("/insights", json=_values_request())
    assert r.status_code == 200
    assert "offline mode" in r.json()["insights"].lower()


def test_chat_endpoint_offline(client):
    r = client.post("/chat", json={"question": "What intervals does the model use?"})
    assert r.status_code == 200
    body = r.json()
    assert body["ai_enabled"] is False
    assert isinstance(body["answer"], str)


def test_transfer_endpoint(client):
    rng = np.random.default_rng(1)
    src = (np.cumsum(rng.normal(0, 1, size=120)) + 100).tolist()
    tgt = (np.cumsum(rng.normal(0, 1, size=120)) + 120).tolist()
    r = client.post(
        "/transfer",
        json={
            "source": {"values": src},
            "target": {"values": tgt},
            "horizon": 4,
            "test_length": 8,
            "lags": 8,
            "epochs": 8,
        },
    )
    assert r.status_code == 200
    assert len(r.json()["forecast"]) == 4
