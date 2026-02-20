import argparse
import logging
from pathlib import Path
from typing import List

from scripts.mubaweb_scraper import save_output as save_mubaweb_output
from scripts.mubaweb_scraper import scrape_mubaweb
from scripts.technocase_scraper import save_output as save_technocase_output
from scripts.technocase_scraper import scrape_technocase

try:
    from scripts.scraper import (
        MubawabScraper,
        RequestManager,
        TunisieAnnonceScraper,
        scraper_factory,
    )
except ModuleNotFoundError:
    from scripts.scraper import (
        MubawabScraper,
        RequestManager,
        TunisieAnnonceScraper,
        scraper_factory,
    )

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Tunisian real-estate scrapers and save outputs in data/."
    )
    parser.add_argument(
        "--source",
        default="all",
        choices=[
            "all",
            "tayara",
            "mubawab",
            "mubaweb",
            "tunisieannonce",
            "tecnocasa",
            "technocase",
        ],
        help="Scraper source to run.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where output files are written.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=3,
        help="Used by paginated scrapers (mubaweb/mubawab/tunisieannonce/technocase).",
    )
    parser.add_argument(
        "--max-listings",
        type=int,
        default=40,
        help="Used by remax scraper.",
    )
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--backoff-factor", type=float, default=1.0)
    parser.add_argument(
        "--proxy",
        action="append",
        default=None,
        help="Optional proxy URL; pass multiple times for rotation.",
    )
    parser.add_argument("--headed", action="store_true", help="Run browsers in headed mode")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def run_single_source(
    source: str,
    output_dir: Path,
    request_manager: RequestManager,
    max_pages: int,
    max_listings: int,
    headless: bool,
) -> int:
    logger.info("Running scraper: %s", source)

    if source == "mubaweb":
        rows = scrape_mubaweb(start_page=1, end_page=max_pages)
        paths = save_mubaweb_output(rows, output_dir)
        logger.info("Completed %s: %s listing(s), output=%s", source, len(rows), paths["json"])
        return len(rows)

    if source in ("technocase", "tecnocasa"):
        rows = scrape_technocase(max_pages=max_pages, headless=headless)
        paths = save_technocase_output(rows, output_dir)
        logger.info("Completed %s: %s listing(s), output=%s", source, len(rows), paths["json"])
        return len(rows)


    output_path = output_dir / f"{source}.jsonl"
    scraper = scraper_factory(source, request_manager=request_manager, output_path=str(output_path))

    if isinstance(scraper, (MubawabScraper, TunisieAnnonceScraper)):
        listings = scraper.scrape(max_pages=max_pages)
    else:
        listings = scraper.scrape()

    count = len(listings)
    logger.info("Completed %s: %s listing(s), output=%s", source, count, output_path)
    return count


def resolve_sources(source_arg: str) -> List[str]:
    if source_arg == "all":
        return ["tayara", "mubaweb", "technocase", "remax"]
    return [source_arg]


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    request_manager = RequestManager(
        proxies=args.proxy,
        max_retries=args.max_retries,
        backoff_factor=args.backoff_factor,
    )

    sources = resolve_sources(args.source)
    total = 0

    for source in sources:
        try:
            total += run_single_source(
                source=source,
                output_dir=output_dir,
                request_manager=request_manager,
                max_pages=args.max_pages,
                max_listings=args.max_listings,
                headless=not args.headed,
            )
        except Exception as exc:
            logger.exception("Scraper failed for source=%s: %s", source, exc)

    logger.info("Done. Total listings scraped: %s", total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
