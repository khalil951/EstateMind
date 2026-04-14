from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from src.agents.langgraph_agent_api import (
    BatchListingRequest,
    create_agent_api,
    delete_listing_by_id,
    fetch_listing_by_id,
    fetch_recent_listings,
    search_listings,
)


class DummyGraph:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(state)
        return {
            "logs": ["dummy-start", f"source={state['source_url']}"] ,
            "final_listing": {
                "id": "listing-1",
                "created_at": "2026-04-11T12:00:00+00:00",
                "source_url": state["source_url"],
                "title": "Bright apartment",
                "property_type": "Appartement",
                "governorate": "Tunis",
                "city": "La Marsa",
                "neighborhood": "Sidi Abdelaziz",
                "surface_m2": 120.0,
                "bedrooms": 3,
                "bathrooms": 2,
                "condition": "Excellent",
                "predicted_price_tnd": 540000.0,
                "confidence_score": 0.82,
                "prediction_mode": "mock_regression",
            },
        }


def _seed_listing(db_path: Path, *, listing_id: str, title: str, city: str, price: float) -> None:
    from src.agents.langgraph_agent_api import _ensure_db_exists

    _ensure_db_exists(db_path)
    conn = __import__("sqlite3").connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO listings (
                id, created_at, source_url, title, property_type, governorate, city, neighborhood,
                surface_m2, bedrooms, bathrooms, condition, predicted_price_tnd, confidence, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                listing_id,
                "2026-04-11T12:00:00+00:00",
                f"https://example.com/{listing_id}",
                title,
                "Appartement",
                "Tunis",
                city,
                "Center",
                120.0,
                3,
                2,
                "Excellent",
                price,
                0.81,
                json.dumps({"id": listing_id, "title": title, "predicted_price_tnd": price}),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def test_batch_request_validation_rejects_empty_urls() -> None:
    try:
        BatchListingRequest(urls=[])
        raise AssertionError("expected validation error")
    except Exception as exc:
        assert "urls must not be empty" in str(exc)


def test_agent_api_endpoints_round_trip(tmp_path: Path) -> None:
    db_path = tmp_path / "listings.db"
    graph = DummyGraph()
    app = create_agent_api(graph, db_path=db_path)
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

        response = client.post("/ingest-and-value", json={"url": "https://example.com/listing-1"})
        assert response.status_code == 200
        body = response.json()
        assert body["final_listing"]["title"] == "Bright apartment"
        assert body["logs"][0] == "dummy-start"

        # Ingest endpoint should persist returned listing in SQLite.
        persisted = fetch_recent_listings(db_path, limit=10)
        assert len(persisted) >= 1

        response = client.post(
            "/ingest-and-value/batch",
            json={"urls": ["https://example.com/listing-1", "https://example.com/listing-2"]},
        )
        assert response.status_code == 200
        batch_body = response.json()
        assert batch_body["count"] == 2
        assert batch_body["error_count"] == 0


def test_listing_lookup_search_and_delete(tmp_path: Path) -> None:
    db_path = tmp_path / "listings.db"
    _seed_listing(db_path, listing_id="a1", title="Seaview apartment", city="La Marsa", price=650000.0)
    _seed_listing(db_path, listing_id="a2", title="Central apartment", city="Tunis", price=430000.0)

    recent = fetch_recent_listings(db_path, limit=5)
    assert len(recent) == 2

    one = fetch_listing_by_id(db_path, "a1")
    assert one is not None
    assert one["payload"]["title"] == "Seaview apartment"

    matches = search_listings(db_path, city="La Marsa", limit=10)
    assert len(matches) == 1
    assert matches[0]["id"] == "a1"

    deleted = delete_listing_by_id(db_path, "a1")
    assert deleted is True
    assert fetch_listing_by_id(db_path, "a1") is None


def test_scheduler_run_once_and_status(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "listings.db"
    graph = DummyGraph()

    import src.agents.langgraph_agent_api as api_module

    monkeypatch.setattr(
        api_module,
        "discover_listing_urls",
        lambda source_url, **kwargs: [f"{source_url}/listing-1"],
    )

    app = create_agent_api(graph, db_path=db_path)
    with TestClient(app) as client:
        response = client.post(
            "/scheduler/run-once",
            json={"target_per_source": 1, "timeout_s": 10, "sleep_s": 0.0},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["processed"] >= 1

        status = client.get("/scheduler/status")
        assert status.status_code == 200
        assert status.json()["last_run_at"] is not None

        start = client.post(
            "/scheduler/start",
            json={"interval_hours": 24.0, "target_per_source": 1, "timeout_s": 10, "sleep_s": 0.0},
        )
        assert start.status_code == 200
        assert start.json()["running"] is True

        stop = client.post("/scheduler/stop")
        assert stop.status_code == 200
        assert stop.json()["running"] is False
