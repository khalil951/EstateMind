from __future__ import annotations

from typing import Any

from src.agent import listing_graph_factory as lgf


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


def test_default_graph_varies_price_by_listing(monkeypatch) -> None:
    def _fake_get(url: str, headers: dict[str, Any] | None = None, timeout: int = 25):
        if "listing-a" in url:
            return _FakeResponse(
                """
                <html><body>
                  <h1>Appartement A</h1>
                  <p class='description'>Renovated apartment with sea view and parking.</p>
                  <span class='location'>Tunis, La Marsa</span>
                  <span class='surface'>80 m2</span>
                  <span class='rooms'>3</span>
                  <span class='bathrooms'>1</span>
                  <span class='condition'>Excellent</span>
                  <span class='property_type'>Appartement</span>
                </body></html>
                """
            )
        return _FakeResponse(
            """
            <html><body>
              <h1>Maison B</h1>
              <p class='description'>Family house in need of renovation.</p>
              <span class='location'>Sfax, Sfax Ville</span>
              <span class='surface'>180 m2</span>
              <span class='rooms'>5</span>
              <span class='bathrooms'>2</span>
              <span class='condition'>Fair</span>
              <span class='property_type'>Maison</span>
            </body></html>
            """
        )

    monkeypatch.setattr(lgf.requests, "get", _fake_get)
    monkeypatch.setattr(lgf, "VALUATION_SERVICE", None)

    graph = lgf.build_default_listing_graph()
    first = graph.invoke({"source_url": "https://example.com/listing-a", "logs": []})
    second = graph.invoke({"source_url": "https://example.com/listing-b", "logs": []})

    first_price = float(first["final_listing"]["predicted_price_tnd"])
    second_price = float(second["final_listing"]["predicted_price_tnd"])

    assert first_price > 0
    assert second_price > 0
    assert first_price != second_price


def test_default_graph_returns_expected_fields(monkeypatch) -> None:
    monkeypatch.setattr(
        lgf.requests,
        "get",
        lambda url, headers=None, timeout=25: _FakeResponse(
            """
            <html><body>
              <h1>Appartement C</h1>
              <p class='description'>Premium property with garden and elevator.</p>
              <span class='location'>Ariana, Ariana Ville</span>
              <span class='surface'>110 m2</span>
              <span class='rooms'>4</span>
              <span class='bathrooms'>2</span>
              <span class='condition'>Good</span>
              <span class='property_type'>Appartement</span>
            </body></html>
            """
        ),
    )
    monkeypatch.setattr(lgf, "VALUATION_SERVICE", None)

    graph = lgf.build_default_listing_graph()
    result = graph.invoke({"source_url": "https://example.com/listing-c", "logs": []})
    listing = result["final_listing"]

    assert listing["property_type"] == "Appartement"
    assert listing["city"] == "Ariana Ville"
    assert listing["predicted_price_tnd"] > 0
    assert "predicted_price_range_tnd" in listing
