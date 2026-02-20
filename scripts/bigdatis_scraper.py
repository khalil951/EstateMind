import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.scraper import normalize_tunisian_data
except ModuleNotFoundError:
    from scraper import normalize_tunisian_data

BASE_URL = "https://bigdatis.tn/immobilier/vente/appartement/?o=date&q=eyJwZiI6W3sicCI6InRyYW5zYWN0aW9uVHlwZSIsInYiOlsic2FsZSJdfSx7InAiOiJwcm9wZXJ0eVR5cGUiLCJ2IjpbImZsYXQiXX1dfQ%3D%3D"
SOURCE = "Bigdatis"
CARD_SELECTOR = "app-unit-preview article a.card[href], app-unit-preview a.card[href], a.card[href*='/details/']"
LOAD_MORE_SELECTOR = "button:has-text('Voir plus de résultats')"
logger = logging.getLogger(__name__)


def _clean_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", value).strip()
    return text or None


def _first_text(card, selectors: List[str]) -> Optional[str]:
    for sel in selectors:
        try:
            el = card.query_selector(sel)
            if el:
                txt = _clean_text(el.inner_text())
                if txt:
                    return txt
        except Exception:
            continue
    return None


def _extract_labels(card) -> List[str]:
    labels: List[str] = []
    try:
        for el in card.query_selector_all(".additional-info .label"):
            txt = _clean_text(el.inner_text())
            if txt:
                labels.append(txt)
    except Exception:
        pass
    return labels


def _extract_row(card, page_url: str, logical_page: int) -> Optional[Dict[str, object]]:
    href = card.get_attribute("href")
    listing_url = urljoin(page_url, href) if href else None

    title = _first_text(card, [".info-panel .title", ".title"])
    meta_title = _first_text(card, [".meta-title"])
    price_raw = _first_text(card, [".price-wrapper .price", ".price", "[class*='price']"])
    location = _first_text(card, [".location-name", ".location-wrapper .location-name", "[class*='location']"])
    posted_at = _first_text(card, [".side-badge span[elapsedtime]", ".side-badge span"])
    labels = _extract_labels(card)

    if not title and not listing_url:
        return None

    surface_hint = None
    for label in labels:
        m = re.search(r"(\d[\d\s]*)\s?m²", label, re.IGNORECASE)
        if m:
            surface_hint = m.group(0)
            break

    normalized = normalize_tunisian_data(
        {
            "price_raw": price_raw,
            "location": location,
            "description": meta_title or surface_hint,
            "surface": surface_hint,
            "property_type": "appartement",
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
        "description": meta_title,
        "meta_title": meta_title,
        "labels": labels,
        "posted_at": posted_at,
        "page": logical_page,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


def _load_more_once(page, previous_count: int, timeout_ms: int = 25000) -> bool:
    btn = page.locator(LOAD_MORE_SELECTOR).first
    if btn.count() == 0:
        return False

    try:
        btn.scroll_into_view_if_needed(timeout=3000)
    except Exception:
        pass

    try:
        btn.click(timeout=5000)
    except Exception:
        return False

    try:
        page.wait_for_function(
            "(sel, prev) => document.querySelectorAll(sel).length > prev",
            arg=[CARD_SELECTOR, previous_count],
            timeout=timeout_ms,
        )
        return True
    except Exception:
        return False


def scrape_bigdatis(
    start_page: int,
    end_page: int,
    headless: bool = True,
    storage_state: Optional[str] = None,
    browser_channel: Optional[str] = None,
    user_data_dir: Optional[str] = None,
) -> List[Dict[str, object]]:
    if start_page < 1 or end_page < start_page:
        raise ValueError("Invalid page range")

    rows: List[Dict[str, object]] = []
    seen: Set[str] = set()

    with sync_playwright() as playwright:
        browser = None
        if user_data_dir:
            context = playwright.chromium.launch_persistent_context(
                user_data_dir=str(Path(user_data_dir)),
                headless=headless,
                channel=browser_channel,
            )
            if storage_state:
                logger.warning("--storage-state is ignored when --user-data-dir is used.")
        else:
            browser = playwright.chromium.launch(headless=headless, channel=browser_channel)
            context_kwargs = {}
            if storage_state:
                context_kwargs["storage_state"] = storage_state
            context = browser.new_context(**context_kwargs)
        page = context.new_page()

        logger.info("Loading listings page: %s", BASE_URL)
        page.goto(BASE_URL, wait_until="domcontentloaded", timeout=45000)

        try:
            page.wait_for_selector(CARD_SELECTOR, timeout=30000)
        except PlaywrightTimeoutError:
            logger.info("No listing cards found")
            context.close()
            if browser:
                browser.close()
            return rows

        current_count = page.locator(CARD_SELECTOR).count()
        logger.info("Initial listing cards loaded: %s", current_count)

        target_page = 1
        while target_page < start_page:
            loaded = _load_more_once(page, current_count)
            if not loaded:
                logger.warning("Could not load batch %s. Bigdatis requires login to load more results.", target_page + 1)
                context.close()
                if browser:
                    browser.close()
                return rows
            current_count = page.locator(CARD_SELECTOR).count()
            target_page += 1

        logical_page = start_page
        while logical_page <= end_page:
            cards = page.query_selector_all(CARD_SELECTOR)
            logger.info("Visible cards after batch %s: %s", logical_page, len(cards))

            for card in cards:
                row = _extract_row(card, page_url=page.url, logical_page=logical_page)
                if not row:
                    continue
                key = row.get("url") or f"{row.get('title') or ''}|{row.get('price') or ''}"
                key = str(key)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)

            if logical_page == end_page:
                break

            previous_count = len(cards)
            loaded = _load_more_once(page, previous_count)
            if not loaded:
                logger.warning(
                    "Stopped at batch %s. No additional cards loaded. "
                    "If you need all batches, run with a logged-in storage state via --storage-state.",
                    logical_page,
                )
                break

            logical_page += 1

        context.close()
        if browser:
            browser.close()

    return rows


def save_output(rows: List[Dict[str, object]], output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "bigdatis_listings.jsonl"
    json_path = output_dir / "bigdatis_listings.json"

    with jsonl_path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(rows, fp, ensure_ascii=False, indent=2)

    return {"jsonl": str(jsonl_path), "json": str(json_path)}


def save_storage_state_interactive(
    storage_state_path: Path,
    login_url: str,
    browser_channel: Optional[str] = None,
) -> None:
    storage_state_path.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False, channel=browser_channel)
        context = browser.new_context()
        page = context.new_page()
        logger.info("Opening login page: %s", login_url)
        page.goto(login_url, wait_until="domcontentloaded", timeout=45000)
        logger.info("Complete login in the opened browser, then press Enter here to save storage state.")
        input("Press Enter after login is complete...")
        context.storage_state(path=str(storage_state_path))
        context.close()
        browser.close()

    logger.info("Saved storage state: %s", storage_state_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape Bigdatis apartment sale listings.")
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int, default=3)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--headed", action="store_true", help="Run browser in headed mode")
    parser.add_argument("--storage-state", default=None, help="Optional Playwright storage state JSON (logged-in session)")
    parser.add_argument(
        "--save-storage-state",
        default=None,
        help="Interactive mode: open browser, login manually, and save Playwright storage state JSON to this path.",
    )
    parser.add_argument(
        "--login-url",
        default="https://bigdatis.tn/",
        help="URL to open when using --save-storage-state.",
    )
    parser.add_argument(
        "--browser-channel",
        default=None,
        help="Optional browser channel (example: chrome, msedge). Useful when a site blocks bundled Chromium login.",
    )
    parser.add_argument(
        "--user-data-dir",
        default=None,
        help="Optional persistent browser profile directory. If provided, scraper uses this logged-in profile directly.",
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

    if args.save_storage_state:
        save_storage_state_interactive(
            Path(args.save_storage_state),
            args.login_url,
            browser_channel=args.browser_channel,
        )
        return 0

    rows = scrape_bigdatis(
        start_page=args.start_page,
        end_page=args.end_page,
        headless=not args.headed,
        storage_state=args.storage_state,
        browser_channel=args.browser_channel,
        user_data_dir=args.user_data_dir,
    )
    paths = save_output(rows, Path(args.output_dir))

    logger.info("Scraped %s listings", len(rows))
    logger.info("Saved JSONL: %s", paths["jsonl"])
    logger.info("Saved JSON: %s", paths["json"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
