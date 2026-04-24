"""Reusable FastAPI module for the LangGraph listing agent."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, HttpUrl, field_validator

from src.agent.agent_source_runner import SAMPLE_SOURCES, discover_listing_urls


class ListingAgentState(TypedDict, total=False):
    """Minimal state expected by the graph entrypoint."""

    source_url: str
    logs: List[str]


class ListingRequest(BaseModel):
    """Request payload for listing ingestion and valuation."""

    url: HttpUrl


class BatchListingRequest(BaseModel):
    """Request payload for ingesting and valuing multiple listing URLs."""

    urls: List[HttpUrl]

    @field_validator("urls")
    @classmethod
    def validate_urls(cls, value: List[HttpUrl]) -> List[HttpUrl]:
        if not value:
            raise ValueError("urls must not be empty")
        if len(value) > 100:
            raise ValueError("urls must contain at most 100 items")
        return value


class DeleteListingResponse(BaseModel):
    """Response returned when deleting a persisted listing."""

    deleted: bool
    id: str


class SchedulerStartRequest(BaseModel):
    interval_hours: float = Field(default=24.0, gt=0)
    target_per_source: int = Field(default=20, ge=1, le=200)
    timeout_s: int = Field(default=30, ge=5, le=120)
    sleep_s: float = Field(default=0.0, ge=0.0, le=5.0)


class SchedulerRunOnceRequest(BaseModel):
    target_per_source: int = Field(default=20, ge=1, le=200)
    timeout_s: int = Field(default=30, ge=5, le=120)
    sleep_s: float = Field(default=0.0, ge=0.0, le=5.0)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_or_empty(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _translate_property_type_to_french(value: str) -> str:
    normalized = _clean_or_empty(value).lower()
    if not normalized:
        return ""

    mapping = {
        "apartment": "Appartement",
        "house": "Maison",
        "land": "Terrain",
        "apartments": "Appartements",
        "houses": "Maisons",
        "lands": "Terrains",
        "villa": "Villa",
        "duplex": "Duplex",
        "studio": "Studio",
        "commercial": "Commercial",
        "bureau": "Bureau",
        "office": "Bureau",
        "retail space": "Local commercial",
        "local": "Local",
        "appartement": "Appartement",
        "maison": "Maison",
        "terrain": "Terrain",
    }
    return mapping.get(normalized, _clean_or_empty(value))


def _build_display_title(listing: Dict[str, Any]) -> str:
    title = _clean_or_empty(listing.get("title"))
    if title:
        return title

    property_type = _translate_property_type_to_french(listing.get("property_type") or "")
    city = _clean_or_empty(listing.get("city"))
    governorate = _clean_or_empty(listing.get("governorate"))
    neighborhood = _clean_or_empty(listing.get("neighborhood"))

    location_bits = [bit for bit in [neighborhood, city, governorate] if bit]
    if property_type and location_bits:
        return f"{property_type} à {', '.join(location_bits)}"
    if property_type:
        return property_type
    if location_bits:
        return ", ".join(location_bits)
    return ""


def _normalize_stored_listing(final_listing: Dict[str, Any], source_url: str) -> Dict[str, Any]:
    listing = dict(final_listing)
    if not listing.get("id"):
        listing["id"] = str(uuid.uuid4())
    if not listing.get("created_at"):
        listing["created_at"] = _utc_now_iso()
    listing["title"] = _build_display_title(listing)
    listing.setdefault("property_type", "")
    listing.setdefault("governorate", "")
    listing.setdefault("city", "")
    listing.setdefault("neighborhood", "")
    listing.setdefault("surface_m2", 0.0)
    listing.setdefault("bedrooms", 0)
    listing.setdefault("bathrooms", 0)
    listing.setdefault("condition", "")
    listing.setdefault("predicted_price_tnd", 0.0)
    listing.setdefault("confidence_score", 0.0)
    listing["source_url"] = source_url
    return listing


def _ensure_db_exists(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS listings (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                source_url TEXT NOT NULL,
                title TEXT,
                property_type TEXT,
                governorate TEXT,
                city TEXT,
                neighborhood TEXT,
                surface_m2 REAL,
                bedrooms INTEGER,
                bathrooms INTEGER,
                condition TEXT,
                predicted_price_tnd REAL,
                confidence REAL,
                payload_json TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _extract_listing_summary(listing: Dict[str, Any], source_url: str) -> Dict[str, Any]:
    return {
        "id": str(listing.get("id") or ""),
        "created_at": str(listing.get("created_at") or _utc_now_iso()),
        "source_url": source_url,
        "title": str(listing.get("title") or ""),
        "property_type": str(listing.get("property_type") or ""),
        "governorate": str(listing.get("governorate") or ""),
        "city": str(listing.get("city") or ""),
        "predicted_price_tnd": float(listing.get("predicted_price_tnd") or 0.0),
        "confidence": float(listing.get("confidence_score") or 0.0),
    }


def store_listing(db_path: Path, final_listing: Dict[str, Any], source_url: str) -> Dict[str, Any]:
    """Persist one processed listing in SQLite and return normalized stored payload."""

    listing = _normalize_stored_listing(final_listing, source_url=source_url)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO listings (
                id, created_at, source_url, title, property_type, governorate, city, neighborhood,
                surface_m2, bedrooms, bathrooms, condition, predicted_price_tnd, confidence, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(listing.get("id") or ""),
                str(listing.get("created_at") or _utc_now_iso()),
                str(source_url),
                str(listing.get("title") or ""),
                str(listing.get("property_type") or ""),
                str(listing.get("governorate") or ""),
                str(listing.get("city") or ""),
                str(listing.get("neighborhood") or ""),
                float(listing.get("surface_m2") or 0.0),
                int(listing.get("bedrooms") or 0),
                int(listing.get("bathrooms") or 0),
                str(listing.get("condition") or ""),
                float(listing.get("predicted_price_tnd") or 0.0),
                float(listing.get("confidence_score") or 0.0),
                json.dumps(listing, ensure_ascii=False),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return listing


def _invoke_graph(listing_graph: Any, url: str) -> Dict[str, Any]:
    state: ListingAgentState = {"source_url": url, "logs": []}
    out = listing_graph.invoke(state)
    final_listing = out.get("final_listing")
    if not isinstance(final_listing, dict):
        raise RuntimeError("graph result does not contain final_listing")
    return {
        "final_listing": final_listing,
        "logs": out.get("logs", []),
    }


def run_source_cycle(
    *,
    listing_graph: Any,
    db_path: Path,
    sources: List[str],
    target_per_source: int = 20,
    timeout_s: int = 30,
    sleep_s: float = 0.0,
) -> Dict[str, Any]:
    """Run one full scrape/process/store cycle over all configured sources."""

    started = time.time()
    processed = 0
    failed = 0
    by_source: List[Dict[str, Any]] = []

    for source in sources:
        src_start = time.time()
        discovered = discover_listing_urls(source_url=source, max_urls=target_per_source, timeout_s=timeout_s)
        source_processed = 0
        source_failed = 0
        errors: List[Dict[str, Any]] = []

        for listing_url in discovered:
            try:
                out = _invoke_graph(listing_graph=listing_graph, url=listing_url)
                final_listing = out.get("final_listing") if isinstance(out, dict) else {}
                if not isinstance(final_listing, dict):
                    raise RuntimeError("invalid final_listing from graph")
                store_listing(db_path=db_path, final_listing=final_listing, source_url=listing_url)
                source_processed += 1
                processed += 1
            except Exception as exc:
                source_failed += 1
                failed += 1
                errors.append({"url": listing_url, "error": str(exc)})
            if sleep_s > 0:
                time.sleep(sleep_s)

        by_source.append(
            {
                "source": source,
                "discovered": len(discovered),
                "processed": source_processed,
                "failed": source_failed,
                "duration_s": round(time.time() - src_start, 3),
                "errors": errors,
            }
        )

    return {
        "started_at": _utc_now_iso(),
        "duration_s": round(time.time() - started, 3),
        "processed": processed,
        "failed": failed,
        "sources": by_source,
    }


class AgentScheduler:
    """Simple in-process scheduler that runs source cycles at a fixed interval."""

    def __init__(self, listing_graph: Any, db_path: Path, sources: List[str]) -> None:
        self.listing_graph = listing_graph
        self.db_path = db_path
        self.sources = list(sources)
        self.interval_seconds: float = 24 * 3600
        self.target_per_source: int = 20
        self.timeout_s: int = 30
        self.sleep_s: float = 0.0
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._running = False
        self._last_run_at: Optional[str] = None
        self._last_result: Optional[Dict[str, Any]] = None
        self._next_run_at: Optional[str] = None

    def _runner(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                self._next_run_at = datetime.fromtimestamp(
                    time.time() + self.interval_seconds,
                    tz=timezone.utc,
                ).isoformat()

            result = run_source_cycle(
                listing_graph=self.listing_graph,
                db_path=self.db_path,
                sources=self.sources,
                target_per_source=self.target_per_source,
                timeout_s=self.timeout_s,
                sleep_s=self.sleep_s,
            )

            with self._lock:
                self._last_run_at = _utc_now_iso()
                self._last_result = result

            if self._stop_event.wait(self.interval_seconds):
                break

        with self._lock:
            self._running = False
            self._next_run_at = None

    def start(self, *, interval_hours: float, target_per_source: int, timeout_s: int, sleep_s: float) -> Dict[str, Any]:
        already_running = False
        with self._lock:
            if self._running:
                already_running = True
            else:
                self.interval_seconds = float(interval_hours) * 3600.0
                self.target_per_source = int(target_per_source)
                self.timeout_s = int(timeout_s)
                self.sleep_s = float(sleep_s)
                self._stop_event.clear()
                self._running = True
                self._thread = threading.Thread(target=self._runner, daemon=True, name="agent-scheduler")
                self._thread.start()
                self._next_run_at = datetime.fromtimestamp(time.time() + self.interval_seconds, tz=timezone.utc).isoformat()

        if already_running:
            return self.status()
        return self.status()

    def stop(self) -> Dict[str, Any]:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=3)
        with self._lock:
            self._running = False
            self._next_run_at = None
        return self.status()

    def run_once(self, *, target_per_source: int, timeout_s: int, sleep_s: float) -> Dict[str, Any]:
        result = run_source_cycle(
            listing_graph=self.listing_graph,
            db_path=self.db_path,
            sources=self.sources,
            target_per_source=target_per_source,
            timeout_s=timeout_s,
            sleep_s=sleep_s,
        )
        with self._lock:
            self._last_run_at = _utc_now_iso()
            self._last_result = result
        return result

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "running": bool(self._running),
                "interval_hours": round(self.interval_seconds / 3600.0, 4),
                "target_per_source": self.target_per_source,
                "timeout_s": self.timeout_s,
                "sleep_s": self.sleep_s,
                "next_run_at": self._next_run_at,
                "last_run_at": self._last_run_at,
                "last_result": self._last_result,
            }


def fetch_recent_listings(db_path: Path, limit: int = 5) -> List[Dict[str, Any]]:
    """Read the latest persisted listings from SQLite."""

    if limit < 1:
        raise ValueError("limit must be >= 1")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT id, created_at, title, predicted_price_tnd, confidence
            FROM listings
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def fetch_listing_by_id(db_path: Path, listing_id: str) -> Optional[Dict[str, Any]]:
    """Read one persisted listing payload by identifier."""

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            SELECT id, created_at, source_url, payload_json
            FROM listings
            WHERE id = ?
            LIMIT 1
            """,
            (listing_id,),
        ).fetchone()
        if row is None:
            return None

        payload = row["payload_json"]
        parsed_payload: Dict[str, Any]
        try:
            import json

            parsed_payload = json.loads(payload)
        except Exception:
            parsed_payload = {"raw_payload_json": payload}

        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "source_url": row["source_url"],
            "payload": parsed_payload,
        }
    finally:
        conn.close()


def delete_listing_by_id(db_path: Path, listing_id: str) -> bool:
    """Delete one persisted listing by identifier."""

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute("DELETE FROM listings WHERE id = ?", (listing_id,))
        conn.commit()
        return (cursor.rowcount or 0) > 0
    finally:
        conn.close()


def search_listings(
    db_path: Path,
    *,
    city: Optional[str] = None,
    governorate: Optional[str] = None,
    property_type: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    limit: int = 25,
) -> List[Dict[str, Any]]:
    """Search persisted listings with optional filters."""

    if limit < 1:
        raise ValueError("limit must be >= 1")

    filters: List[str] = []
    values: List[Any] = []

    if city:
        filters.append("LOWER(city) = LOWER(?)")
        values.append(city)
    if governorate:
        filters.append("LOWER(governorate) = LOWER(?)")
        values.append(governorate)
    if property_type:
        filters.append("LOWER(property_type) = LOWER(?)")
        values.append(property_type)
    if min_price is not None:
        filters.append("predicted_price_tnd >= ?")
        values.append(float(min_price))
    if max_price is not None:
        filters.append("predicted_price_tnd <= ?")
        values.append(float(max_price))

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"""
                 SELECT id, created_at, source_url, title, property_type, governorate, city,
                     predicted_price_tnd, confidence
            FROM listings
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (*values, int(limit)),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def create_agent_api(listing_graph: Any | None = None, db_path: Path | None = None) -> FastAPI:
    """Create a FastAPI app bound to a LangGraph listing graph and SQLite path."""

    if listing_graph is None:
        from src.agent.listing_graph_factory import build_default_listing_graph

        listing_graph = build_default_listing_graph()

    if db_path is None:
        db_path = Path("artifacts/langgraph_listings.db")

    _ensure_db_exists(db_path)
    app = FastAPI(title="LangGraph Real Estate Agent API", version="1.0.0")
    scheduler = AgentScheduler(listing_graph=listing_graph, db_path=db_path, sources=SAMPLE_SOURCES)

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/ingest-and-value")
    def ingest_and_value(req: ListingRequest) -> Dict[str, Any]:
        try:
            out = _invoke_graph(listing_graph=listing_graph, url=str(req.url))
            final_listing = out.get("final_listing") if isinstance(out, dict) else {}
            if not isinstance(final_listing, dict):
                raise RuntimeError("missing final listing")
            out["final_listing"] = store_listing(db_path=db_path, final_listing=final_listing, source_url=str(req.url))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"graph execution failed: {exc}") from exc
        return out

    @app.post("/ingest-and-value/batch")
    def ingest_and_value_batch(req: BatchListingRequest) -> Dict[str, Any]:
        items: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        for raw_url in req.urls:
            url = str(raw_url)
            try:
                out = _invoke_graph(listing_graph=listing_graph, url=url)
                listing = out.get("final_listing") if isinstance(out, dict) else {}
                if isinstance(listing, dict):
                    listing = store_listing(db_path=db_path, final_listing=listing, source_url=url)
                items.append(
                    {
                        "url": url,
                        "summary": _extract_listing_summary(listing or {}, source_url=url),
                        "final_listing": listing,
                    }
                )
            except Exception as exc:
                errors.append({"url": url, "error": str(exc)})

        return {
            "count": len(items),
            "error_count": len(errors),
            "items": items,
            "errors": errors,
        }

    @app.get("/recent-listings")
    def recent_listings(limit: int = Query(default=5, ge=1, le=100)) -> Dict[str, Any]:
        try:
            rows = fetch_recent_listings(db_path=db_path, limit=limit)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"database read failed: {exc}") from exc
        return {"count": len(rows), "items": rows}

    @app.get("/listings/{listing_id}")
    def listing_by_id(listing_id: str) -> Dict[str, Any]:
        try:
            row = fetch_listing_by_id(db_path=db_path, listing_id=listing_id)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"database read failed: {exc}") from exc

        if row is None:
            raise HTTPException(status_code=404, detail="listing not found")
        return row

    @app.get("/listings")
    def listings(
        city: Optional[str] = Query(default=None),
        governorate: Optional[str] = Query(default=None),
        property_type: Optional[str] = Query(default=None),
        min_price: Optional[float] = Query(default=None, ge=0),
        max_price: Optional[float] = Query(default=None, ge=0),
        limit: int = Query(default=25, ge=1, le=200),
    ) -> Dict[str, Any]:
        if min_price is not None and max_price is not None and min_price > max_price:
            raise HTTPException(status_code=400, detail="min_price cannot exceed max_price")

        try:
            rows = search_listings(
                db_path=db_path,
                city=city,
                governorate=governorate,
                property_type=property_type,
                min_price=min_price,
                max_price=max_price,
                limit=limit,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"database search failed: {exc}") from exc

        return {
            "count": len(rows),
            "filters": {
                "city": city,
                "governorate": governorate,
                "property_type": property_type,
                "min_price": min_price,
                "max_price": max_price,
                "limit": limit,
            },
            "items": rows,
        }

    @app.post("/scheduler/start")
    def scheduler_start(req: SchedulerStartRequest) -> Dict[str, Any]:
        return scheduler.start(
            interval_hours=req.interval_hours,
            target_per_source=req.target_per_source,
            timeout_s=req.timeout_s,
            sleep_s=req.sleep_s,
        )

    @app.post("/scheduler/stop")
    def scheduler_stop() -> Dict[str, Any]:
        return scheduler.stop()

    @app.get("/scheduler/status")
    def scheduler_status() -> Dict[str, Any]:
        return scheduler.status()

    @app.post("/scheduler/run-once")
    def scheduler_run_once(req: SchedulerRunOnceRequest) -> Dict[str, Any]:
        return scheduler.run_once(
            target_per_source=req.target_per_source,
            timeout_s=req.timeout_s,
            sleep_s=req.sleep_s,
        )

    @app.delete("/listings/{listing_id}", response_model=DeleteListingResponse)
    def delete_listing(listing_id: str) -> DeleteListingResponse:
        try:
            deleted = delete_listing_by_id(db_path=db_path, listing_id=listing_id)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"database delete failed: {exc}") from exc

        if not deleted:
            raise HTTPException(status_code=404, detail="listing not found")

        return DeleteListingResponse(deleted=True, id=listing_id)

    @app.get("/services")
    def services() -> Dict[str, Any]:
        return {
            "name": "LangGraph Real Estate Agent API",
            "version": "1.0.0",
            "endpoints": [
                {"method": "GET", "path": "/health"},
                {"method": "POST", "path": "/ingest-and-value"},
                {"method": "POST", "path": "/ingest-and-value/batch"},
                {"method": "GET", "path": "/recent-listings"},
                {"method": "GET", "path": "/listings"},
                {"method": "GET", "path": "/listings/{listing_id}"},
                {"method": "DELETE", "path": "/listings/{listing_id}"},
                {"method": "POST", "path": "/scheduler/start"},
                {"method": "POST", "path": "/scheduler/stop"},
                {"method": "GET", "path": "/scheduler/status"},
                {"method": "POST", "path": "/scheduler/run-once"},
                {"method": "GET", "path": "/services"},
            ],
        }

    return app
