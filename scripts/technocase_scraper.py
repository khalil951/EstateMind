import argparse
import json
import logging
import re
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.scraper import normalize_tunisian_data
except ModuleNotFoundError:
    from scraper import normalize_tunisian_data

START_URL = "https://www.tecnocasa.tn/vendre/immeubles/nord-est-ne/grand-tunis.html"
SOURCE = "Tecnocasa"
logger = logging.getLogger(__name__)


def _make_soup(html: str, parser: str = "auto") -> BeautifulSoup:
    if parser == "lxml":
        return BeautifulSoup(html, "lxml")
    if parser == "html.parser":
        return BeautifulSoup(html, "html.parser")
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")


def _clean_text(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    text = re.sub(r"\s+", " ", value).strip()
    return text or None


def _extract_surface(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"(\d{2,4})\s?(m²|m2|m)", text, re.IGNORECASE)
    return f"{m.group(1)} m2" if m else None


def _candidate_cards(soup: BeautifulSoup) -> List[Any]:
    selectors = [
        "article",
        "li[class*='result']",
        "div[class*='result']",
        "div[class*='card']",
    ]
    for selector in selectors:
        cards = soup.select(selector)
        if cards:
            return cards
    return []


def _extract_row(card: Any, page_url: str, page_number: int) -> Optional[Dict[str, Any]]:
    link = card.select_one("a[href]")
    if not link:
        return None

    href = link.get("href")
    listing_url = urljoin(page_url, href) if href else None

    title = _clean_text(link.get("title"))
    if not title:
        title = _clean_text(link.get_text(" ", strip=True))

    full_text = _clean_text(card.get_text(" ", strip=True))
    price_raw = None
    if full_text:
        price_match = re.search(r"(\d[\d\s\.]*)\s*(DT|TND|€)", full_text, re.IGNORECASE)
        if price_match:
            price_raw = f"{price_match.group(1)} {price_match.group(2)}"

    location = None
    if full_text:
        loc_match = re.search(r"(Tunis|Ariana|Ben Arous|Manouba|Nabeul|Sousse|Sfax)", full_text, re.IGNORECASE)
        if loc_match:
            location = loc_match.group(1)

    surface_hint = _extract_surface(full_text)

    if not any([listing_url, title, full_text]):
        return None

    normalized = normalize_tunisian_data(
        {
            "price_raw": price_raw,
            "location": location,
            "description": full_text,
            "surface": surface_hint,
            "property_type": "immeuble",
        }
    )

    return {
        "source": SOURCE,
        "title": title,
        "price": normalized.get("price"),
        "governorate": normalized.get("governorate"),
        "city": normalized.get("city"),
        "property_type": normalized.get("property_type"),
        "surface_area": normalized.get("surface_area"),
        "url": listing_url,
        "description": full_text,
        "page": page_number,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


def _is_same_site(url: str) -> bool:
    return "tecnocasa.tn" in urlparse(url).netloc


def _discover_next_pages(soup: BeautifulSoup, current_url: str) -> List[str]:
    candidates: List[str] = []
    for a in soup.select("a[href]"):
        href = a.get("href")
        if not href:
            continue
        absolute = urljoin(current_url, href)
        if not _is_same_site(absolute):
            continue

        text = _clean_text(a.get_text(" ", strip=True)) or ""
        if re.search(r"next|suiv|successiva|>", text, re.IGNORECASE):
            candidates.append(absolute)
            continue
        if re.search(r"page|pag=|/p\d+", absolute, re.IGNORECASE):
            candidates.append(absolute)

    deduped: List[str] = []
    seen: Set[str] = set()
    for url in candidates:
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    return deduped


def scrape_technocase(
    max_pages: int,
    headless: bool = True,
    parser: str = "auto",
) -> List[Dict[str, Any]]:
    if max_pages < 1:
        raise ValueError("max_pages must be >= 1")

    rows: List[Dict[str, Any]] = []
    seen_rows: Set[str] = set()
    seen_pages: Set[str] = set()
    queue: deque[str] = deque([START_URL])

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        context = browser.new_context()
        page = context.new_page()

        page_number = 0
        while queue and page_number < max_pages:
            url = queue.popleft()
            if url in seen_pages:
                continue
            seen_pages.add(url)
            page_number += 1

            logger.info("Loading page %s: %s", page_number, url)
            page.goto(url, wait_until="domcontentloaded", timeout=45000)
            try:
                page.wait_for_load_state("networkidle", timeout=15000)
            except PlaywrightTimeoutError:
                logger.info("Network idle timeout on %s", url)

            soup = _make_soup(page.content(), parser=parser)
            cards = _candidate_cards(soup)
            logger.info("Found %s candidate cards on page %s", len(cards), page_number)

            for card in cards:
                row = _extract_row(card, page_url=page.url, page_number=page_number)
                if not row:
                    continue
                key = f"{row.get('url') or ''}|{row.get('title') or ''}|{row.get('price') or ''}"
                if key in seen_rows:
                    continue
                seen_rows.add(key)
                rows.append(row)

            for next_url in _discover_next_pages(soup, page.url):
                if next_url not in seen_pages:
                    queue.append(next_url)

        context.close()
        browser.close()

    return rows


def save_output(rows: List[Dict[str, Any]], output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "technocase_listings.json"

    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(rows, fp, ensure_ascii=False, indent=2)

    return {"json": str(output_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape Tecnocasa Tunisia listings.")
    parser.add_argument("--max-pages", type=int, default=3)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--headed", action="store_true", help="Run browser in headed mode")
    parser.add_argument(
        "--parser",
        default="auto",
        choices=["auto", "lxml", "html.parser"],
        help="HTML parser engine to use.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    rows = scrape_technocase(
        max_pages=args.max_pages,
        headless=not args.headed,
        parser=args.parser,
    )
    paths = save_output(rows, Path(args.output_dir))

    logger.info("Scraped %s listings", len(rows))
    logger.info("Saved JSON: %s", paths["json"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
