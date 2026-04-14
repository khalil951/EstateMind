"""Command-line launcher for the LangGraph real-estate agent API."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from src.agents.langgraph_agent_api import create_agent_api


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LangGraph real-estate agent API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--db-path", default="artifacts/langgraph_listings.db")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    app = create_agent_api(db_path=Path(args.db_path))
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
