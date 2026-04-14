"""Run end-to-end agent functionality checks across multiple listing sources.

This script discovers up to N listing URLs per source and submits them to the
LangGraph agent API endpoint for ingestion and valuation.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

LOGGER = logging.getLogger("agent_source_runner")

SAMPLE_SOURCES: List[str] = [
    "https://www.tayara.tn/c/immobilier",
    "https://www.mubawab.tn/fr/listing-promotion:p:1",
    "http://www.tunisie-annonce.com/AnnoncesImmobilier.asp",
    "https://www.tecnocasa.tn/vendre/immeubles/nord-est-ne/grand-tunis.html",
    "https://bigdatis.tn/immobilier/vente/appartement/?o=date&q=eyJwZiI6W3sicCI6InRyYW5zYWN0aW9uVHlwZSIsInYiOlsic2FsZSJdfSx7InAiOiJwcm9wZXJ0eVR5cGUiLCJ2IjpbImZsYXQiXX1dfQ%3D%3D",
]

_LISTING_HINTS = (
    "annonce",
    "listing",
    "item",
    "immobilier",
    "appartement",
    "maison",
    "terrain",
    "vendre",
    "vente",
    "rent",
    "sale",
)


@dataclass
class ProcessingResult:
    source: str
    discovered: int
    attempted: int
    succeeded: int
    failed: int
    duration_s: float
    errors: List[Dict[str, Any]]


CSV_BASE_COLUMNS = [
    "source",
    "source_page_url",
    "listing_url",
    "id",
    "created_at",
    "title",
    "property_type",
    "governorate",
    "city",
    "neighborhood",
    "surface_m2",
    "bedrooms",
    "bathrooms",
    "condition",
    "predicted_price_tnd",
    "confidence_score",
    "prediction_mode",
    "explanation_short",
    "payload_json",
]


def _is_http_url(url: str) -> bool:
    """Return True when the given string is an absolute HTTP or HTTPS URL."""

    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _is_listingish(url: str) -> bool:
    """Heuristically determine whether a URL looks like a listing page."""

    lowered = url.lower()
    return any(token in lowered for token in _LISTING_HINTS)


def _extract_candidate_links(html: str, base_url: str) -> List[str]:
    """Extract absolute HTTP(S) links from HTML relative to a base URL."""

    soup = BeautifulSoup(html, "html.parser")
    links: List[str] = []
    for anchor in soup.find_all("a", href=True):
        href = str(anchor.get("href") or "").strip()
        if not href or href.startswith("#"):
            continue
        if href.startswith("javascript:") or href.startswith("mailto:"):
            continue
        absolute = urljoin(base_url, href)
        if not _is_http_url(absolute):
            continue
        links.append(absolute)
    return links


def discover_listing_urls(
    source_url: str,
    *,
    max_urls: int = 20,
    timeout_s: int = 25,
    session: Optional[requests.Session] = None,
) -> List[str]:
    """Discover up to ``max_urls`` listing-like URLs from a source page.

    The crawler performs one shallow expansion pass over the first page and,
    if needed, a second pass over a subset of discovered category links. This
    keeps the discovery step lightweight while still finding enough listing
    URLs for the downstream agent run.
    """

    sess = session or requests.Session()
    seen: set[str] = set()
    selected: List[str] = []

    def _collect(url: str) -> List[str]:
        response = sess.get(url, timeout=timeout_s)
        response.raise_for_status()
        return _extract_candidate_links(response.text, base_url=url)

    try:
        first_level = _collect(source_url)
    except Exception as exc:
        LOGGER.warning("Failed source fetch: %s (%s)", source_url, exc)
        return []

    for link in first_level:
        if link in seen:
            continue
        seen.add(link)
        if _is_listingish(link):
            selected.append(link)
        if len(selected) >= max_urls:
            return selected[:max_urls]

    # For pages with mostly category links, crawl a shallow second level.
    for parent in first_level[:30]:
        if len(selected) >= max_urls:
            break
        try:
            second_level = _collect(parent)
        except Exception:
            continue
        for link in second_level:
            if link in seen:
                continue
            seen.add(link)
            if _is_listingish(link):
                selected.append(link)
            if len(selected) >= max_urls:
                break

    return selected[:max_urls]


def _post_ingest(api_base_url: str, listing_url: str, session: requests.Session, timeout_s: int) -> Dict[str, Any]:
    """Submit one listing URL to the agent API and return the JSON response."""

    endpoint = f"{api_base_url.rstrip('/')}/ingest-and-value"
    response = session.post(endpoint, json={"url": listing_url}, timeout=timeout_s)
    response.raise_for_status()
    return response.json()


def _json_safe(value: Any) -> Any:
    """Convert nested values into JSON-safe text when needed."""

    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _build_csv_row(source: str, source_page_url: str, listing_url: str, response_body: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one API response into a CSV-friendly listing row."""

    final_listing = response_body.get("final_listing") if isinstance(response_body, dict) else {}
    if not isinstance(final_listing, dict):
        final_listing = {}

    row = {
        "source": source,
        "source_page_url": source_page_url,
        "listing_url": listing_url,
        "id": final_listing.get("id", ""),
        "created_at": final_listing.get("created_at", ""),
        "title": final_listing.get("title", ""),
        "property_type": final_listing.get("property_type", ""),
        "governorate": final_listing.get("governorate", ""),
        "city": final_listing.get("city", ""),
        "neighborhood": final_listing.get("neighborhood", ""),
        "surface_m2": final_listing.get("surface_m2", ""),
        "bedrooms": final_listing.get("bedrooms", ""),
        "bathrooms": final_listing.get("bathrooms", ""),
        "condition": final_listing.get("condition", ""),
        "predicted_price_tnd": final_listing.get("predicted_price_tnd", ""),
        "confidence_score": final_listing.get("confidence_score", ""),
        "prediction_mode": final_listing.get("prediction_mode", ""),
        "explanation_short": final_listing.get("explanation_short", ""),
        "payload_json": json.dumps(final_listing, ensure_ascii=False),
    }
    return {key: _json_safe(value) for key, value in row.items()}


def export_listings_csv(rows: List[Dict[str, Any]], output_path: Path) -> Path:
    """Write processed listing rows to CSV using stable base columns plus extras.

    The output always includes the core listing fields plus any extra keys found
    in the first row so the export remains easy to inspect and extend.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return output_path

    extra_columns = [column for column in rows[0].keys() if column not in CSV_BASE_COLUMNS]
    fieldnames = CSV_BASE_COLUMNS + extra_columns

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(row.get(key, "")) for key in fieldnames})

    return output_path


def process_sources(
    *,
    api_base_url: str,
    sources: Iterable[str],
    target_per_source: int = 20,
    timeout_s: int = 30,
    sleep_s: float = 0.0,
) -> Dict[str, Any]:
    """Discover, process, and summarize listings for each source.

    This helper is intended for smoke testing the agent API. It validates the
    API health endpoint, discovers listing URLs for each configured source, and
    submits each discovered listing to the ingestion endpoint.
    """

    session = requests.Session()
    started = time.time()

    health = session.get(f"{api_base_url.rstrip('/')}/health", timeout=timeout_s)
    health.raise_for_status()

    results: List[ProcessingResult] = []

    for source in sources:
        src_start = time.time()
        urls = discover_listing_urls(
            source,
            max_urls=target_per_source,
            timeout_s=timeout_s,
            session=session,
        )
        errors: List[Dict[str, Any]] = []
        succeeded = 0

        for url in urls:
            try:
                _post_ingest(api_base_url=api_base_url, listing_url=url, session=session, timeout_s=timeout_s)
                succeeded += 1
            except Exception as exc:
                errors.append({"url": url, "error": str(exc)})
            if sleep_s > 0:
                time.sleep(sleep_s)

        results.append(
            ProcessingResult(
                source=source,
                discovered=len(urls),
                attempted=len(urls),
                succeeded=succeeded,
                failed=len(urls) - succeeded,
                duration_s=round(time.time() - src_start, 3),
                errors=errors,
            )
        )

    total_attempted = sum(item.attempted for item in results)
    total_succeeded = sum(item.succeeded for item in results)
    total_failed = sum(item.failed for item in results)

    return {
        "api_base_url": api_base_url,
        "target_per_source": target_per_source,
        "total_sources": len(list(sources)) if not isinstance(sources, list) else len(sources),
        "total_attempted": total_attempted,
        "total_succeeded": total_succeeded,
        "total_failed": total_failed,
        "duration_s": round(time.time() - started, 3),
        "results": [
            {
                "source": r.source,
                "discovered": r.discovered,
                "attempted": r.attempted,
                "succeeded": r.succeeded,
                "failed": r.failed,
                "duration_s": r.duration_s,
                "errors": r.errors,
            }
            for r in results
        ],
    }


def process_sources_with_listings(
    *,
    api_base_url: str,
    sources: Iterable[str],
    target_per_source: int = 20,
    timeout_s: int = 30,
    sleep_s: float = 0.0,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Process sources and collect flattened listing rows for CSV export.

    This is the main end-to-end runner used by the CLI. It returns both the
    overall execution report and the row-level listing payloads that can be
    exported into a CSV file.
    """

    source_list = list(sources)
    session = requests.Session()
    started = time.time()

    health = session.get(f"{api_base_url.rstrip('/')}/health", timeout=timeout_s)
    health.raise_for_status()

    results: List[ProcessingResult] = []
    rows: List[Dict[str, Any]] = []

    for source in source_list:
        src_start = time.time()
        urls = discover_listing_urls(
            source,
            max_urls=target_per_source,
            timeout_s=timeout_s,
            session=session,
        )
        errors: List[Dict[str, Any]] = []
        succeeded = 0

        for url in urls:
            try:
                response_body = _post_ingest(api_base_url=api_base_url, listing_url=url, session=session, timeout_s=timeout_s)
                rows.append(_build_csv_row(source=source, source_page_url=source, listing_url=url, response_body=response_body))
                succeeded += 1
            except Exception as exc:
                errors.append({"url": url, "error": str(exc)})
            if sleep_s > 0:
                time.sleep(sleep_s)

        results.append(
            ProcessingResult(
                source=source,
                discovered=len(urls),
                attempted=len(urls),
                succeeded=succeeded,
                failed=len(urls) - succeeded,
                duration_s=round(time.time() - src_start, 3),
                errors=errors,
            )
        )

    total_attempted = sum(item.attempted for item in results)
    total_succeeded = sum(item.succeeded for item in results)
    total_failed = sum(item.failed for item in results)

    report = {
        "api_base_url": api_base_url,
        "target_per_source": target_per_source,
        "total_sources": len(source_list),
        "total_attempted": total_attempted,
        "total_succeeded": total_succeeded,
        "total_failed": total_failed,
        "duration_s": round(time.time() - started, 3),
        "results": [
            {
                "source": r.source,
                "discovered": r.discovered,
                "attempted": r.attempted,
                "succeeded": r.succeeded,
                "failed": r.failed,
                "duration_s": r.duration_s,
                "errors": r.errors,
            }
            for r in results
        ],
        "listings_count": len(rows),
    }

    return report, rows


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the source runner CLI."""

    parser = argparse.ArgumentParser(description="Run agent functionality checks on multiple listing sources")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--target-per-source", type=int, default=20)
    parser.add_argument("--timeout-s", type=int, default=30)
    parser.add_argument("--sleep-s", type=float, default=0.0)
    parser.add_argument("--output", default="artifacts/reports/agent_source_runner_report.json")
    parser.add_argument("--csv-output", default="artifacts/reports/agent_source_runner_listings.csv")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def main() -> int:
    """Execute the source runner CLI and write the JSON report and CSV export."""

    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")

    report, rows = process_sources_with_listings(
        api_base_url=args.api_base_url,
        sources=SAMPLE_SOURCES,
        target_per_source=args.target_per_source,
        timeout_s=args.timeout_s,
        sleep_s=args.sleep_s,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    csv_path = export_listings_csv(rows, Path(args.csv_output))

    LOGGER.info("Completed source run: attempted=%s succeeded=%s failed=%s", report["total_attempted"], report["total_succeeded"], report["total_failed"])
    LOGGER.info("Report written to %s", output_path)
    LOGGER.info("CSV written to %s", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
