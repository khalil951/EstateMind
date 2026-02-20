import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

import sys
# Ensure project root is on sys.path so `scraper` can be imported when running
# this script directly (python scripts/tayara_scraper.py).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scraper import normalize_tunisian_data

BASE_URL = "https://www.tayara.tn/listing/c/immobilier/appartements/?page={page}"
SOURCE = "Tayara"
logger = logging.getLogger(__name__)


def _first_text(container: Any, selectors: List[str]) -> Optional[str]:
    for sel in selectors:
        try:
            el = container.query_selector(sel)
            if el:
                text = (el.inner_text() or "").strip()
                if text:
                    return text
        except Exception:
            continue
    return None


def _extract_cards(page: Any) -> List[Any]:
    # Target the actual listing card shape from Tayara UI.
    selectors = [
        "div:has(h2.card-title):has(data[value])",
        "div.relative.rounded-md.group:has(h2.card-title)",
    ]
    for sel in selectors:
        cards = page.query_selector_all(sel)
        if cards:
            return cards
    return []


def _extract_url(card: Any, current_url: str) -> Optional[str]:
    link_selectors = [
        "a[href*='/item/']",
        "a[href*='/listing/']",
        "a[href*='/annonce/']",
        "a[href*='/ad/']",
        "a[href]",
    ]
    for sel in link_selectors:
        try:
            el = card.query_selector(sel)
            if not el:
                continue
            href = el.get_attribute("href")
            if not href:
                continue
            return urljoin(current_url, href)
        except Exception:
            continue

    try:
        href = card.evaluate(
            """
            (node) => {
                let el = node;
                for (let i = 0; i < 6 && el; i += 1) {
                    if (el.tagName === 'A' && el.href) return el.href;
                    const nested = el.querySelector('a[href]');
                    if (nested && nested.href) return nested.href;
                    if (el.dataset && (el.dataset.href || el.dataset.url)) {
                        return el.dataset.href || el.dataset.url;
                    }
                    el = el.parentElement;
                }
                return null;
            }
            """
        )
        return urljoin(current_url, href) if href else None
    except Exception:
        return None


def _extract_price_raw(card: Any) -> Optional[str]:
    try:
        data_el = card.query_selector("data[value]")
        if data_el:
            value_attr = data_el.get_attribute("value")
            if value_attr and value_attr.strip().isdigit():
                return f"{value_attr.strip()} DT"
            txt = (data_el.inner_text() or "").strip()
            if txt:
                return txt
    except Exception:
        pass

    return _first_text(
        card,
        ["[class*='price']", "[data-testid*='price']", "span:has-text('DT')", "span:has-text('TND')"],
    )


def _extract_meta(card: Any) -> Dict[str, Optional[str]]:
    category = None
    location = None

    try:
        spans = card.query_selector_all("span")
        texts = [((s.inner_text() or "").strip()) for s in spans]
        texts = [t for t in texts if t]

        for t in texts:
            if not category and "Appart" in t:
                category = t
            if not location and "," in t and "ago" in t.lower():
                location = t.split(",", 1)[0].strip()
            elif not location and t.lower().endswith("ago"):
                continue
            elif not location and t and len(t) <= 50 and "DT" not in t and "Appart" not in t:
                location = t
    except Exception:
        pass

    return {"category": category, "location": location}


def _row_key(title: Optional[str], listing_url: Optional[str], page_number: int, price: Optional[float]) -> str:
    return f"{listing_url or ''}|{title or ''}|{page_number}|{price or ''}"


def _scrape_page(page: Any, page_number: int, seen_keys: Set[str]) -> List[Dict[str, Any]]:
    url = BASE_URL.format(page=page_number)
    logger.info("Loading page %s: %s", page_number, url)
    page.goto(url, wait_until="domcontentloaded", timeout=45000)
    try:
        page.wait_for_load_state("networkidle", timeout=15000)
    except PlaywrightTimeoutError:
        logger.info("Network idle timeout on page %s; continuing", page_number)

    cards = _extract_cards(page)
    logger.info("Found %s listing cards on page %s", len(cards), page_number)
    rows: List[Dict[str, Any]] = []

    for card in cards:
        title = _first_text(card, ["h2.card-title", "h2", "h3", "h4", "[class*='title']", "a[title]"])
        price_raw = _extract_price_raw(card)
        meta = _extract_meta(card)
        location = meta.get("location")
        description = _first_text(card, ["[class*='description']", "p"])
        listing_url = _extract_url(card, page.url)

        normalized = normalize_tunisian_data(
            {
                "price_raw": price_raw,
                "location": location,
                "description": description,
                "property_type": "appartement",
            }
        )
        row = {
            "source": SOURCE,
            "title": title,
            "price": normalized.get("price"),
            "governorate": normalized.get("governorate"),
            "city": normalized.get("city"),
            "property_type": normalized.get("property_type"),
            "surface_area": normalized.get("surface_area"),
            "url": listing_url,
            "description": description,
            "category": meta.get("category"),
            "page": page_number,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
        }

        key = _row_key(title=row["title"], listing_url=row["url"], page_number=page_number, price=row["price"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        rows.append(row)

    return rows


def scrape_tayara(start_page: int, end_page: int, headless: bool = True) -> List[Dict[str, Any]]:
    if start_page < 1 or end_page < start_page:
        raise ValueError("Invalid page range")

    all_rows: List[Dict[str, Any]] = []
    seen_keys: Set[str] = set()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        context = browser.new_context()
        page = context.new_page()

        for page_number in range(start_page, end_page + 1):
            try:
                rows = _scrape_page(page, page_number, seen_keys)
                all_rows.extend(rows)
            except Exception as exc:
                logger.exception("Failed scraping page %s: %s", page_number, exc)

        context.close()
        browser.close()

    return all_rows


def save_output(rows: List[Dict[str, Any]], output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "tayara_listings.jsonl"
    json_path = output_dir / "tayara_listings.json"

    with jsonl_path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(rows, fp, ensure_ascii=False, indent=2)

    return {"jsonl": str(jsonl_path), "json": str(json_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape Tayara apartment listings by page.")
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int, default=3)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--headed", action="store_true", help="Run browser in headed mode")
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

    rows = scrape_tayara(
        start_page=args.start_page,
        end_page=args.end_page,
        headless=not args.headed,
    )
    paths = save_output(rows, Path(args.output_dir))
    logger.info("Scraped %s listings", len(rows))
    logger.info("Saved JSONL: %s", paths["jsonl"])
    logger.info("Saved JSON: %s", paths["json"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
