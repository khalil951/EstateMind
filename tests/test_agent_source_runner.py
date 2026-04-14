from __future__ import annotations

import csv
from typing import Any
from pathlib import Path

from src.agents.agent_source_runner import (
    _extract_candidate_links,
    discover_listing_urls,
    export_listings_csv,
    process_sources,
    process_sources_with_listings,
)


class _FakeResponse:
    def __init__(self, text: str = "", payload: dict[str, Any] | None = None, status_code: int = 200) -> None:
        self.text = text
        self._payload = payload or {}
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeSession:
    def __init__(self, html_map: dict[str, str] | None = None) -> None:
        self.html_map = html_map or {}
        self.posts: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, timeout: int = 25) -> _FakeResponse:
        if url.endswith("/health"):
            return _FakeResponse(payload={"status": "ok"})
        return _FakeResponse(text=self.html_map.get(url, ""))

    def post(self, url: str, json: dict[str, Any], timeout: int = 30) -> _FakeResponse:
        self.posts.append((url, json))
        return _FakeResponse(payload={"final_listing": {"id": "x"}, "logs": []})


def test_extract_candidate_links_normalizes_relative_and_filters_non_http() -> None:
    html = """
    <html><body>
      <a href="/item/1">A</a>
      <a href="https://example.com/annonce/2">B</a>
      <a href="mailto:test@example.com">Mail</a>
      <a href="javascript:void(0)">JS</a>
      <a href="#fragment">F</a>
    </body></html>
    """
    links = _extract_candidate_links(html, base_url="https://example.com/base")
    assert "https://example.com/item/1" in links
    assert "https://example.com/annonce/2" in links
    assert all(not link.startswith("mailto:") for link in links)


def test_discover_listing_urls_collects_and_limits() -> None:
    src = "https://example.com/source"
    html_map = {
        src: """
            <a href='https://example.com/category/real-estate'>cat</a>
            <a href='https://example.com/annonce/1'>l1</a>
            <a href='https://example.com/annonce/2'>l2</a>
        """,
        "https://example.com/category/real-estate": """
            <a href='https://example.com/listing/3'>l3</a>
            <a href='https://example.com/listing/4'>l4</a>
        """,
    }
    session = _FakeSession(html_map=html_map)
    urls = discover_listing_urls(src, max_urls=3, session=session)
    assert len(urls) == 3
    assert all("example.com" in u for u in urls)


def test_process_sources_attempts_discovered_urls(monkeypatch) -> None:
    def _fake_discover(source_url: str, **kwargs: Any):
        return [f"{source_url}/listing-1", f"{source_url}/listing-2"]

    fake_session = _FakeSession()

    import src.agents.agent_source_runner as runner

    monkeypatch.setattr(runner, "discover_listing_urls", _fake_discover)
    monkeypatch.setattr(runner.requests, "Session", lambda: fake_session)

    report = process_sources(
        api_base_url="http://127.0.0.1:8010",
        sources=["https://s1.test", "https://s2.test"],
        target_per_source=20,
        timeout_s=10,
    )

    assert report["total_sources"] == 2
    assert report["total_attempted"] == 4
    assert report["total_succeeded"] == 4
    assert report["total_failed"] == 0
    assert len(fake_session.posts) == 4


def test_export_listings_csv_writes_rows(tmp_path: Path) -> None:
    output_path = tmp_path / "listings.csv"
    rows = [
        {
            "source": "https://example.com/source",
            "source_page_url": "https://example.com/source",
            "listing_url": "https://example.com/listing/1",
            "id": "a1",
            "created_at": "2026-04-11T12:00:00+00:00",
            "title": "Demo listing",
            "property_type": "Appartement",
            "governorate": "Tunis",
            "city": "La Marsa",
            "neighborhood": "Center",
            "surface_m2": 120,
            "bedrooms": 3,
            "bathrooms": 2,
            "condition": "Excellent",
            "predicted_price_tnd": 540000,
            "confidence_score": 0.82,
            "prediction_mode": "mock_regression",
            "explanation_short": "demo",
            "payload_json": "{}",
        }
    ]

    written = export_listings_csv(rows, output_path)
    assert written == output_path
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        parsed = list(reader)

    assert len(parsed) == 1
    assert parsed[0]["title"] == "Demo listing"
    assert parsed[0]["listing_url"] == "https://example.com/listing/1"


def test_process_sources_with_listings_collects_rows(monkeypatch) -> None:
    def _fake_discover(source_url: str, **kwargs: Any):
        return [f"{source_url}/listing-1"]

    fake_session = _FakeSession()

    import src.agents.agent_source_runner as runner

    monkeypatch.setattr(runner, "discover_listing_urls", _fake_discover)
    monkeypatch.setattr(runner.requests, "Session", lambda: fake_session)

    report, rows = process_sources_with_listings(
        api_base_url="http://127.0.0.1:8010",
        sources=["https://s1.test"],
        target_per_source=20,
        timeout_s=10,
    )

    assert report["listings_count"] == 1
    assert len(rows) == 1
    assert rows[0]["listing_url"] == "https://s1.test/listing-1"
