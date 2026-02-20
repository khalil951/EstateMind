import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.scraper import normalize_tunisian_data
except ModuleNotFoundError:
    from scraper import normalize_tunisian_data

BASE_URL = "https://www.mubawab.tn/fr/listing-promotion:p:{page}"
SOURCE = "Mubawab"
logger = logging.getLogger(__name__)


def _make_soup(html: str, parser: str = "auto") -> BeautifulSoup:
    if parser == "lxml":
        return BeautifulSoup(html, "lxml")
    if parser == "html.parser":
        return BeautifulSoup(html, "html.parser")
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        # Fallback when lxml is not installed in the environment.
        return BeautifulSoup(html, "html.parser")


def _clean_text(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    text = re.sub(r"\s+", " ", value).strip()
    return text or None


def _first_text(node: Any, selectors: List[str]) -> Optional[str]:
    for selector in selectors:
        el = node.select_one(selector)
        if el:
            text = _clean_text(el.get_text(" ", strip=True))
            if text:
                return text
    return None


def _extract_surface(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"(\d{2,4})\s?(m²|m2|m)", text, re.IGNORECASE)
    return f"{match.group(1)} m2" if match else None


def _row_key(row: Dict[str, Any]) -> str:
    return f"{row.get('url') or ''}|{row.get('title') or ''}|{row.get('price') or ''}|{row.get('page') or ''}"


def _extract_row(card: Any, page_url: str, page_number: int) -> Optional[Dict[str, Any]]:
    link = card.select_one("a[href]")
    href = link.get("href") if link else None
    listing_url = None
    if href:
        listing_url = urljoin(page_url, href)
    else:
        linkref = card.get("linkref")
        if linkref:
            listing_url = urljoin(page_url, linkref)
        else:
            promotion_id = card.get("promotion-id")
            if promotion_id:
                listing_url = f"https://www.mubawab.tn/fr/p/{promotion_id}"

    title = _first_text(card, ["h2.listingTit a", ".listingTit a", "h2", "h3", ".listing-title", "a[title]"])
    if not title and link:
        title = _clean_text(link.get("title")) or _clean_text(link.get_text(" ", strip=True))

    price_raw = _first_text(card, [".priceTag", ".price", "[class*='price']"])
    location = _first_text(card, [".listingH3", ".listingLocality", ".city", "[class*='local']"])
    description = _first_text(card, [".listingP.descLi", ".listingP", ".listingH4", ".listingH2", ".listingTxt", ".listing-description", "p"])

    card_text = _clean_text(card.get_text(" ", strip=True))
    surface_hint = _extract_surface(card_text)

    if not any([title, listing_url, price_raw, description]):
        return None

    normalized = normalize_tunisian_data(
        {
            "price_raw": price_raw,
            "location": location,
            "description": description,
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
        "description": description,
        "page": page_number,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


def _extract_cards(soup: BeautifulSoup) -> List[Any]:
    selectors = [
        "li.promotionListing.listingBox",
        "li.listingBox",
        "li.listingLi",
        "div.listingBox",
        "div[class*='listing']",
    ]
    for selector in selectors:
        cards = soup.select(selector)
        if cards:
            return cards
    return []


def scrape_mubaweb(
    start_page: int,
    end_page: int,
    timeout: int = 30,
    parser: str = "auto",
) -> List[Dict[str, Any]]:
    if start_page < 1 or end_page < start_page:
        raise ValueError("Invalid page range")

    rows: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
        }
    )

    for page_number in range(start_page, end_page + 1):
        url = BASE_URL.format(page=page_number)
        logger.info("Loading page %s: %s", page_number, url)
        response = session.get(url, timeout=timeout)
        response.raise_for_status()

        soup = _make_soup(response.text, parser=parser)
        cards = _extract_cards(soup)
        logger.info("Found %s cards on page %s", len(cards), page_number)
        if not cards:
            break

        page_added = 0
        for card in cards:
            row = _extract_row(card, page_url=url, page_number=page_number)
            if not row:
                continue
            key = _row_key(row)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
            page_added += 1

        if page_added == 0:
            logger.info("No new listings on page %s; stopping", page_number)
            break

    return rows


def save_output(rows: List[Dict[str, Any]], output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mubaweb_listings.json"

    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(rows, fp, ensure_ascii=False, indent=2)

    return {"json": str(output_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape Mubawab listing promotions.")
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int, default=3)
    parser.add_argument("--output-dir", default="data")
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

    rows = scrape_mubaweb(
        start_page=args.start_page,
        end_page=args.end_page,
        parser=args.parser,
    )
    paths = save_output(rows, Path(args.output_dir))

    logger.info("Scraped %s listings", len(rows))
    logger.info("Saved JSON: %s", paths["json"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
