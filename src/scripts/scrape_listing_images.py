import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests


URL_COLUMN_HINTS = {
    "listing_link",
    "listing_url",
    "url",
    "link",
    "path",
    "listing_path",
}
IMAGE_COLUMN_HINTS = {
    "listing_image",
    "image",
    "image_url",
    "photo",
    "photo_url",
}

CITY_COLUMN_HINTS = {"city", "governorate", "region", "listing_location"}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".avif", ".gif", ".bmp")
NOISE_IMAGE_KEYWORDS = ("logo", "icon", "sprite", "favicon", "placeholder")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; EstateMindImageScraper/1.0)"}


def discover_csv_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.rglob("*.csv"))


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\ufeff", "")).strip()


def _looks_like_url(value: str) -> bool:
    lower = value.lower()
    return lower.startswith("http://") or lower.startswith("https://")


def _find_url_columns(fieldnames: list[str]) -> list[str]:
    cols: list[str] = []
    for name in fieldnames:
        key = _clean_text(name).lower()
        if key in URL_COLUMN_HINTS or key.endswith("_url") or key.endswith("_link"):
            cols.append(name)
    return cols


def _find_image_columns(fieldnames: list[str]) -> list[str]:
    cols: list[str] = []
    for name in fieldnames:
        key = _clean_text(name).lower()
        if key in IMAGE_COLUMN_HINTS or key.endswith("_image") or key.endswith("_image_url"):
            cols.append(name)
    return cols


def _guess_city(row: dict[str, Any]) -> str:
    for col in CITY_COLUMN_HINTS:
        for key in row.keys():
            if _clean_text(key).lower() == col:
                value = _clean_text(row.get(key))
                if value:
                    return value
    return ""


def _read_csv_rows(csv_path: Path) -> list[dict[str, Any]]:
    for enc in ("utf-8", "latin-1"):
        try:
            with csv_path.open("r", encoding=enc, newline="") as fp:
                return list(csv.DictReader(fp))
        except UnicodeDecodeError:
            continue
    return []


def extract_listing_index(input_dir: Path) -> list[dict[str, Any]]:
    listing_by_url: dict[str, dict[str, Any]] = {}
    for csv_path in discover_csv_files(input_dir):
        rows = _read_csv_rows(csv_path)
        if not rows:
            continue
        url_cols = _find_url_columns(list(rows[0].keys()))
        image_cols = _find_image_columns(list(rows[0].keys()))
        if not url_cols:
            continue

        for row_idx, row in enumerate(rows, start=2):
            seed_images: list[str] = []
            for img_col in image_cols:
                raw_img = _clean_text(row.get(img_col))
                if _looks_like_url(raw_img) and _keep_image_url(raw_img):
                    seed_images.append(raw_img)

            for col in url_cols:
                raw = _clean_text(row.get(col))
                if not raw or not _looks_like_url(raw):
                    continue
                if raw not in listing_by_url:
                    listing_by_url[raw] = {
                        "listing_url": raw,
                        "source_csv": str(csv_path),
                        "source_row": row_idx,
                        "city": _guess_city(row),
                        "seed_images": seed_images[:],
                    }
                else:
                    existing = listing_by_url[raw].get("seed_images", [])
                    merged = list(dict.fromkeys([*existing, *seed_images]))
                    listing_by_url[raw]["seed_images"] = merged
    return sorted(listing_by_url.values(), key=lambda r: r["listing_url"])


def _extract_attribute_urls(html: str, attr_name: str) -> list[str]:
    pattern = re.compile(rf"{attr_name}\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
    return pattern.findall(html)


def _extract_srcset_urls(html: str) -> list[str]:
    candidates: list[str] = []
    pattern = re.compile(r"srcset\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
    for srcset in pattern.findall(html):
        parts = [p.strip() for p in srcset.split(",") if p.strip()]
        for part in parts:
            url = part.split(" ")[0].strip()
            if url:
                candidates.append(url)
    return candidates


def _keep_image_url(url: str) -> bool:
    lower = url.lower()
    if lower.startswith("data:"):
        return False
    if any(k in lower for k in NOISE_IMAGE_KEYWORDS):
        return False
    if lower.endswith(IMAGE_EXTENSIONS):
        return True
    if "/images/" in lower or "gallery" in lower:
        return True
    return False


def extract_image_urls_from_html(base_url: str, html: str) -> list[str]:
    raw_urls: list[str] = []
    for attr in ("src", "data-src", "data-original", "data-lazy", "content"):
        raw_urls.extend(_extract_attribute_urls(html, attr))
    raw_urls.extend(_extract_srcset_urls(html))

    seen: set[str] = set()
    final_urls: list[str] = []
    for raw in raw_urls:
        absolute = urljoin(base_url, raw.strip())
        if not absolute.startswith("http"):
            continue
        if not _keep_image_url(absolute):
            continue
        if absolute in seen:
            continue
        seen.add(absolute)
        final_urls.append(absolute)
    return final_urls


def scrape_listing_images(
    listing_url: str,
    timeout_s: int,
    max_retries: int,
) -> tuple[list[str], str]:
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(listing_url, headers=HEADERS, timeout=timeout_s)
        except requests.exceptions.RequestException as exc:
            if attempt == max_retries:
                return [], f"request_error: {exc}"
            continue

        if response.status_code != 200:
            if attempt == max_retries:
                return [], f"http_{response.status_code}"
            continue

        images = extract_image_urls_from_html(listing_url, response.text)
        return images, ""

    return [], "unknown_error"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract listing URLs from CSV files and scrape listing images.")
    parser.add_argument("--input-dir", default="data/csv", help="Directory containing CSV files.")
    parser.add_argument("--output-dir", default="data/images_dataset", help="Directory for output datasets.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of listings to scrape (0 = all).")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=2, help="Retry count per listing.")
    parser.add_argument("--sleep-ms", type=int, default=100, help="Delay between listing requests.")
    parser.add_argument("--start-offset", type=int, default=0, help="Skip first N listing URLs from index.")
    parser.add_argument("--append", action="store_true", help="Append to existing output files instead of overwriting.")
    parser.add_argument(
        "--url-contains",
        default="",
        help="Optional substring filter for listing URLs (example: tecnocasa.tn).",
    )
    parser.add_argument("--extract-only", action="store_true", help="Only build listing index without scraping pages.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    listing_index = extract_listing_index(input_dir)
    if args.url_contains:
        needle = args.url_contains.lower().strip()
        listing_index = [row for row in listing_index if needle in str(row["listing_url"]).lower()]
    if not listing_index:
        print(f"No listing URLs found in CSV files under: {input_dir}")
        return 1

    if args.start_offset > 0:
        listing_index = listing_index[args.start_offset :]

    if args.limit > 0:
        listing_index = listing_index[: args.limit]

    index_csv = output_dir / "listing_index.csv"
    index_rows = [
        {
            "listing_url": row["listing_url"],
            "source_csv": row["source_csv"],
            "source_row": row["source_row"],
            "city": row["city"],
            "seed_image_count": len(row.get("seed_images", [])),
        }
        for row in listing_index
    ]
    write_csv(index_csv, index_rows, ["listing_url", "source_csv", "source_row", "city", "seed_image_count"])
    print(f"Listing index saved: {index_csv} ({len(listing_index)} URLs)")

    if args.extract_only:
        return 0

    images_csv = output_dir / "listing_images.csv"
    images_jsonl = output_dir / "listing_images.jsonl"
    failures_csv = output_dir / "listing_image_failures.csv"

    if not args.append:
        write_csv(
            images_csv,
            [],
            ["listing_url", "image_url", "image_index", "source_csv", "source_row", "city"],
        )
        write_jsonl(images_jsonl, [])
        write_csv(
            failures_csv,
            [],
            ["listing_url", "source_csv", "source_row", "city", "error"],
        )

    image_count = 0
    failure_count = 0
    total = len(listing_index)
    for i, item in enumerate(listing_index, start=1):
        listing_url = item["listing_url"]
        images, error = scrape_listing_images(
            listing_url=listing_url,
            timeout_s=args.timeout,
            max_retries=args.max_retries,
        )
        seed_images = [u for u in item.get("seed_images", []) if isinstance(u, str)]
        merged_images = list(dict.fromkeys([*seed_images, *images]))
        one_failure: list[dict[str, Any]] = []
        if error:
            one_failure.append(
                {
                    "listing_url": listing_url,
                    "source_csv": item["source_csv"],
                    "source_row": item["source_row"],
                    "city": item["city"],
                    "error": error,
                }
            )
        one_images: list[dict[str, Any]] = []
        if merged_images:
            for idx, img in enumerate(merged_images, start=1):
                one_images.append(
                    {
                        "listing_url": listing_url,
                        "image_url": img,
                        "image_index": idx,
                        "source_csv": item["source_csv"],
                        "source_row": item["source_row"],
                        "city": item["city"],
                    }
                )
            one_failure = []

        append_csv(
            images_csv,
            one_images,
            ["listing_url", "image_url", "image_index", "source_csv", "source_row", "city"],
        )
        append_jsonl(images_jsonl, one_images)
        append_csv(
            failures_csv,
            one_failure,
            ["listing_url", "source_csv", "source_row", "city", "error"],
        )
        image_count += len(one_images)
        failure_count += len(one_failure)

        if i % 25 == 0 or i == total:
            print(f"Progress: {i}/{total} | images: {image_count} | failures: {failure_count}")
        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000)

    print(f"Images dataset saved: {images_csv} ({image_count} rows)")
    print(f"Images dataset JSONL saved: {images_jsonl}")
    print(f"Failures saved: {failures_csv} ({failure_count} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
