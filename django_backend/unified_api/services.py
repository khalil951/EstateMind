from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from src.agent.agent_source_runner import SAMPLE_SOURCES
from src.agent.langgraph_agent_api import AgentScheduler, _ensure_db_exists
from src.inference.valuation_service import ValuationService

_valuation_service: ValuationService | None = None
_listing_graph: Any | None = None
_scheduler: AgentScheduler | None = None
_lock = threading.Lock()


def get_db_path() -> Path:
    return Path("artifacts") / "langgraph_listings.db"


def get_valuation_service() -> ValuationService:
    global _valuation_service
    with _lock:
        if _valuation_service is None:
            _valuation_service = ValuationService()
    return _valuation_service


def get_listing_graph() -> Any:
    global _listing_graph
    with _lock:
        if _listing_graph is None:
            from src.agent.listing_graph_factory import build_default_listing_graph

            _listing_graph = build_default_listing_graph()
    return _listing_graph


def get_scheduler() -> AgentScheduler:
    global _scheduler
    with _lock:
        if _scheduler is None:
            db_path = get_db_path()
            _ensure_db_exists(db_path)
            _scheduler = AgentScheduler(
                listing_graph=get_listing_graph(),
                db_path=db_path,
                sources=SAMPLE_SOURCES,
            )
    return _scheduler
