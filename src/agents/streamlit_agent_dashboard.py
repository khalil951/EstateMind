"""Streamlit dashboard for browsing agent-processed real estate listings."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

DB_PATH = Path("artifacts/langgraph_listings.db")
CSV_FALLBACK = Path("artifacts/reports/agent_source_runner_listings.csv")
TEMPLATE_IMAGE_URL = "https://images.unsplash.com/photo-1564013799919-ab600027ffc6?auto=format&fit=crop&w=1200&q=80"


def _load_from_sqlite(db_path: Path, limit: int = 500) -> List[Dict[str, Any]]:
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT id, created_at, source_url, title, property_type, governorate, city, neighborhood,
                   surface_m2, bedrooms, bathrooms, condition, predicted_price_tnd, confidence, payload_json
            FROM listings
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    finally:
        conn.close()

    listings: List[Dict[str, Any]] = []
    for row in rows:
        payload = row["payload_json"]
        parsed: Dict[str, Any]
        try:
            parsed = json.loads(payload)
        except Exception:
            parsed = {}
        listing = dict(row)
        for key, value in parsed.items():
            if key not in listing:
                listing[key] = value
        listings.append(listing)
    return listings


def _load_from_csv(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return []
    frame = pd.read_csv(csv_path)
    return frame.to_dict(orient="records")


def _load_listings() -> tuple[List[Dict[str, Any]], str]:
    sqlite_rows = _load_from_sqlite(DB_PATH)
    if sqlite_rows:
        return sqlite_rows, "sqlite"

    csv_rows = _load_from_csv(CSV_FALLBACK)
    if csv_rows:
        return csv_rows, "csv"

    return [], "none"


def _price_label(value: Any) -> str:
    try:
        return f"{float(value):,.0f} TND"
    except Exception:
        return "N/A"


def _render_card(item: Dict[str, Any]) -> None:
    title = str(item.get("title") or "Untitled Listing")
    city = str(item.get("city") or "Unknown City")
    governorate = str(item.get("governorate") or "Unknown Governorate")
    ptype = str(item.get("property_type") or "Unknown")
    surface = item.get("surface_m2")
    bedrooms = item.get("bedrooms")
    bathrooms = item.get("bathrooms")
    condition = str(item.get("condition") or "N/A")
    predicted_price = _price_label(item.get("predicted_price_tnd"))
    listing_url = str(item.get("listing_url") or item.get("source_url") or "")

    st.markdown(
        """
        <style>
            .listing-card {
                border: 1px solid #e8eaed;
                border-radius: 16px;
                padding: 12px;
                background: #ffffff;
                box-shadow: 0 2px 8px rgba(15,23,42,0.06);
                height: 100%;
            }
            .listing-title {
                font-size: 18px;
                font-weight: 700;
                color: #1f2937;
                margin-bottom: 6px;
            }
            .listing-meta {
                font-size: 13px;
                color: #4b5563;
                margin-bottom: 4px;
            }
            .listing-price {
                font-size: 20px;
                font-weight: 800;
                color: #0f766e;
                margin-top: 8px;
                margin-bottom: 6px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.image(TEMPLATE_IMAGE_URL, use_container_width=True)
    st.markdown(f"<div class='listing-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='listing-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='listing-meta'>{ptype} • {city}, {governorate}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='listing-meta'>Surface: {surface} m2 • Beds: {bedrooms} • Baths: {bathrooms}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='listing-meta'>Condition: {condition}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='listing-price'>{predicted_price}</div>", unsafe_allow_html=True)
    if listing_url:
        st.markdown(f"[Open Source]({listing_url})")
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Agent Listings Dashboard", page_icon="🏠", layout="wide")

    st.title("Agent Listings Dashboard")
    st.caption("Shows data from SQLite first, then falls back to CSV when scheduler data is unavailable.")

    listings, source = _load_listings()

    if source == "sqlite":
        st.success("Data source: SQLite")
    elif source == "csv":
        st.warning("Data source: CSV fallback")
    else:
        st.info("No listing data available yet.")
        return

    if not listings:
        st.info("No records found.")
        return

    frame = pd.DataFrame(listings)

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_city = st.selectbox("City", options=["All"] + sorted({str(v) for v in frame.get("city", []).dropna().tolist()}))
    with col2:
        selected_type = st.selectbox("Property Type", options=["All"] + sorted({str(v) for v in frame.get("property_type", []).dropna().tolist()}))
    with col3:
        max_rows = st.slider("Listings to display", min_value=6, max_value=120, value=24, step=6)

    filtered = frame.copy()
    if selected_city != "All":
        filtered = filtered[filtered["city"].astype(str) == selected_city]
    if selected_type != "All":
        filtered = filtered[filtered["property_type"].astype(str) == selected_type]

    st.write(f"Displaying {min(len(filtered), max_rows)} of {len(filtered)} listings")

    rows = filtered.head(max_rows).to_dict(orient="records")
    cols_per_row = 3
    for idx in range(0, len(rows), cols_per_row):
        cols = st.columns(cols_per_row)
        for card_col, item in zip(cols, rows[idx : idx + cols_per_row]):
            with card_col:
                _render_card(item)


if __name__ == "__main__":
    main()
