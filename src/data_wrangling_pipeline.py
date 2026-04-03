from __future__ import annotations

"""Normalize heterogeneous raw listing CSVs into a unified tabular dataset.

This module scans a directory tree for CSV files produced by different
scrapers/sources, maps inconsistent source columns into a shared schema, and
writes both a consolidated dataset and a lightweight data-quality report.

The wrangling stage is intentionally heuristic:
- text values are trimmed and normalized
- numeric values are parsed from mixed free-text fields
- transaction type, property type, and room counts are inferred when possible
- a stable ``record_id`` is generated per normalized row

The output is designed to be a clean intermediate dataset for downstream
enrichment and preprocessing steps.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

CANONICAL_COLUMNS = [
    "record_id",
    "source",
    "source_file",
    "listing_url",
    "title",
    "description",
    "transaction_type",
    "property_type",
    "price_tnd",
    "surface_m2",
    "price_per_m2",
    "rooms",
    "bedrooms",
    "bathrooms",
    "governorate",
    "city",
    "neighborhood",
    "location_raw",
    "posted_at",
    "scraped_at",
    "currency",
    "image_url",
    "image_count",
]


def discover_csv_files(input_dir: Path) -> List[Path]:
    """Return all CSV files under ``input_dir`` in deterministic order."""
    return sorted(input_dir.rglob("*.csv"))


def _clean_text(value: Any) -> str | None:
    """Normalize a scalar value into stripped text or ``None``.

    Placeholder-like values such as ``nan``, ``none``, and ``null`` are treated
    as missing.
    """
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value).replace("\ufeff", "")).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _to_float(value: Any) -> float | None:
    """Extract the first numeric value from a noisy scalar field as ``float``."""
    text = _clean_text(value)
    if not text:
        return None
    cleaned = (
        text.replace("DT", "")
        .replace("TND", "")
        .replace("m?", "")
        .replace("m2", "")
        .replace(" ", "")
        .replace(",", ".")
    )
    match = re.search(r"\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _to_int(value: Any) -> int | None:
    """Convert a noisy numeric field to an integer using rounded float parsing."""
    n = _to_float(value)
    if n is None:
        return None
    return int(round(n))


def _infer_transaction_type(*texts: Any) -> str | None:
    """Infer ``sale`` or ``rent`` from URLs/titles/descriptions."""
    blob = " ".join(_clean_text(t) or "" for t in texts).lower()
    if any(k in blob for k in ["/louer/", "? louer", "a louer", "location", "rent"]):
        return "rent"
    if any(k in blob for k in ["/vendre/", "? vendre", "a vendre", "vente", "sale"]):
        return "sale"
    return None


def _infer_property_type(*texts: Any) -> str | None:
    """Infer a coarse property type from multilingual free text."""
    blob = " ".join(_clean_text(t) or "" for t in texts).lower()
    if any(k in blob for k in ["appart", "apartment", "flat", "s+"]):
        return "Appartement"
    if any(k in blob for k in ["villa", "maison", "house", "duplex", "triplex"]):
        return "Maison"
    if any(k in blob for k in ["terrain", "land", "ferme"]):
        return "Terrain"
    return None

def _extract_bedrooms(*texts: Any) -> int | None:
    """Extract bedroom counts from English/French free text when available."""
    blob = " ".join(_clean_text(t) or "" for t in texts).lower()
    match = re.search(r"\b(\d+)\s*bedrooms?\b", blob)
    if match:
        return int(match.group(1))
    match = re.search(r"\b(\d+)\s*chambres?\b", blob)
    if match:
        return int(match.group(1))
    return None


def _extract_price(*texts: Any) -> float | None:
    """Parse a price from mixed-format free text.

    The parser handles common thousands/decimal separator combinations and
    ignores currency markers such as ``DT`` and ``TND``.
    """
    def _parse_numeric_token(token: str) -> float | None:
        s = token.strip().replace(" ", "")
        if not s:
            return None

        if "," in s and "." in s:
            last_comma = s.rfind(",")
            last_dot = s.rfind(".")
            dec_sep = "," if last_comma > last_dot else "."
            frac_len = len(s.split(dec_sep)[-1])
            if frac_len in (1, 2):
                thousands_sep = "." if dec_sep == "," else ","
                s = s.replace(thousands_sep, "").replace(dec_sep, ".")
            else:
                s = s.replace(",", "").replace(".", "")
        elif "," in s:
            frac_len = len(s.split(",")[-1])
            s = s.replace(",", ".") if frac_len in (1, 2) else s.replace(",", "")
        elif "." in s:
            frac_len = len(s.split(".")[-1])
            s = s if frac_len in (1, 2) else s.replace(".", "")

        try:
            return float(s)
        except ValueError:
            return None

    for value in texts:
        cleaned = _clean_text(value)
        if not cleaned:
            continue

        blob = cleaned.lower()
        blob = re.sub(r"\b(dt|tnd|dinar|dinars)\b", " ", blob, flags=re.IGNORECASE)
        blob = re.sub(r"\s+", " ", blob).strip()

        for token in re.findall(r"\d{1,3}(?:[.,\s]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?", blob):
            parsed = _parse_numeric_token(token)
            if parsed is None:
                continue
            if parsed == 0:
                return None
            return parsed

    return None



def _extract_rooms(*texts: Any) -> int | None:
    """Extract total room counts from common listing patterns such as ``S+2``."""
    blob = " ".join(_clean_text(t) or "" for t in texts).lower()
    splus = re.search(r"\bs\s*\+\s*(\d+)\b", blob)
    if splus:
        return int(splus.group(1))
    pieces = re.search(r"\b(\d+)\s*pi[e?]ces?\b", blob)
    if pieces:
        return int(pieces.group(1))
    return None


def _split_location(location: str | None) -> tuple[str | None, str | None, str | None]:
    """Split a raw location string into governorate, city, and neighborhood."""
    loc = _clean_text(location)
    if not loc:
        return None, None, None
    parts = [p.strip() for p in re.split(r",|-", loc) if p.strip()]
    city = parts[0] if parts else None
    neighborhood = parts[1] if len(parts) > 1 else None
    governorate = parts[-1] if len(parts) > 2 else None
    return governorate, city, neighborhood


def _build_record_id(source: str | None, url: str | None, title: str | None, source_file: str | None, row_idx: int) -> str:
    """Build a stable hash identifier for a normalized source row."""
    raw = f"{source or ''}|{url or ''}|{title or ''}|{source_file or ''}|{row_idx}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _normalize_row(row: Dict[str, Any], source_file: str, row_idx: int) -> Dict[str, Any]:
    """Map a raw source row into the canonical schema.

    This function centralizes column alias handling and heuristic inference so
    that downstream code only needs to reason about ``CANONICAL_COLUMNS``.
    """
    source = _clean_text(row.get("source"))
    if not source:
        source = Path(source_file).stem.split("_")[0].title()

    listing_url = _clean_text(row.get("listing_url") or row.get("listing_link") or row.get("url") or row.get("link"))
    title = _clean_text(row.get("title") or row.get("titles") or row.get("listing_property_type"))
    description = _clean_text(row.get("description") or row.get("descriptions") or row.get("meta_title"))

    location_raw = _clean_text(row.get("location_raw") or row.get("location") or row.get("listing_location") or row.get("governorate"))
    governorate = _clean_text(row.get("governorate") or row.get("region"))
    city = _clean_text(row.get("city"))
    neighborhood = _clean_text(row.get("neighborhood"))

    if not city or not governorate:
        g2, c2, n2 = _split_location(location_raw)
        governorate = governorate or g2
        city = city or c2
        neighborhood = neighborhood or n2

    property_type = _clean_text(row.get("property_type") or row.get("type"))
    property_type = _infer_property_type(property_type, title, description, row.get("category"), row.get("listing_property_type"))

    transaction_type = _clean_text(row.get("transaction_type") or row.get("transaction"))
    if transaction_type:
        transaction_type = "rent" if transaction_type.lower().startswith("rent") else "sale" if transaction_type.lower().startswith("sale") else _infer_transaction_type(transaction_type)
    transaction_type = transaction_type or _infer_transaction_type(listing_url, title, description, row.get("category"))

    price_tnd = _extract_price(row.get("price_tnd") or row.get("price") or row.get("listing_price") or row.get("cost"))
    surface_m2 = _to_float(row.get("surface_m2") or row.get("surface_area") or row.get("superficie") or row.get("size") or row.get("listing_size") or row.get("area"))

    rooms = _extract_rooms(row.get("rooms"), row.get("listing_rooms"), title, description)
    bedrooms = _to_int(row.get("bedrooms") or row.get("chambres") or row.get("room_count"))
    bathrooms = _to_int(row.get("bathrooms") or row.get("salles_de_bains") or row.get("bathroom_count"))

    image_url = _clean_text(row.get("listing_image") or row.get("image_url") or row.get("image"))
    image_count = 1 if image_url and image_url.startswith("http") else 0

    price_per_m2 = None
    if price_tnd and surface_m2 and surface_m2 > 0:
        price_per_m2 = round(price_tnd / surface_m2, 2)

    normalized = {
        "record_id": _build_record_id(source, listing_url, title, source_file, row_idx),
        "source": source,
        "source_file": source_file,
        "listing_url": listing_url,
        "title": title,
        "description": description,
        "transaction_type": transaction_type,
        "property_type": property_type,
        "price_tnd": price_tnd,
        "surface_m2": surface_m2,
        "price_per_m2": price_per_m2,
        "rooms": rooms,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "governorate": governorate,
        "city": city,
        "neighborhood": neighborhood,
        "location_raw": location_raw,
        "posted_at": _clean_text(row.get("posted_at") or row.get("date")),
        "scraped_at": _clean_text(row.get("scraped_at")),
        "currency": _clean_text(row.get("currency")) or "TND",
        "image_url": image_url if image_url and image_url.startswith("http") else None,
        "image_count": image_count,
    }
    return normalized


def build_unified_dataset(input_dir: Path) -> pd.DataFrame:
    """Read, normalize, and concatenate all CSV rows under ``input_dir``.

    The function tolerates UTF-8 and Latin-1 encoded files, skips empty inputs,
    and de-duplicates near-identical normalized listings at the end.
    """
    rows: List[Dict[str, Any]] = []
    for csv_path in discover_csv_files(input_dir):
        for enc in ("utf-8", "latin-1"):
            try:
                df = pd.read_csv(csv_path, encoding=enc)
                break
            except UnicodeDecodeError:
                df = None
                continue
        if df is None or df.empty:
            continue
        for idx, record in enumerate(df.to_dict(orient="records"), start=1):
            rows.append(_normalize_row(record, source_file=csv_path.name, row_idx=idx))  # type: ignore

    unified = pd.DataFrame(rows)
    if unified.empty:
        for col in CANONICAL_COLUMNS:
            unified[col] = []
        return unified

    for col in CANONICAL_COLUMNS:
        if col not in unified.columns:
            unified[col] = None
    unified = unified[CANONICAL_COLUMNS]

    unified = unified.drop_duplicates(subset=["source", "listing_url", "title", "price_tnd", "surface_m2"], keep="last")
    unified = unified.reset_index(drop=True)
    return unified


def quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute a compact wrangling quality report for the unified dataset."""
    if df.empty:
        return {
            "row_count": 0,
            "column_count": 0,
            "missing_price_ratio": 1.0,
            "missing_surface_ratio": 1.0,
            "missing_city_ratio": 1.0,
            "invalid_price_rows": 0,
            "invalid_surface_rows": 0,
            "url_present_ratio": 0.0,
            "duplicate_ratio": 0.0,
        }

    dup_ratio = float(df.duplicated(subset=["source", "listing_url", "title", "price_tnd", "surface_m2"]).mean())
    report = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "missing_price_ratio": float(df["price_tnd"].isna().mean()),
        "missing_surface_ratio": float(df["surface_m2"].isna().mean()),
        "missing_city_ratio": float(df["city"].isna().mean()),
        "invalid_price_rows": int(((df["price_tnd"].fillna(0) < 0)).sum()),
        "invalid_surface_rows": int(((df["surface_m2"].fillna(0) < 0)).sum()),
        "url_present_ratio": float(df["listing_url"].notna().mean()),
        "duplicate_ratio": dup_ratio,
    }
    return report


def run_pipeline(input_dir: Path, output_csv: Path, dq_json: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Execute wrangling end-to-end and persist both dataset and QA report."""
    df = build_unified_dataset(input_dir)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    report = quality_report(df)
    dq_json.parent.mkdir(parents=True, exist_ok=True)
    dq_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return df , report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the wrangling CLI."""
    parser = argparse.ArgumentParser(description="Wrangle and unify CSV datasets under data/csv")
    parser.add_argument("--input-dir", default="data/csv")
    parser.add_argument("--output-csv", default="data/csv/final_listings_wrangled.csv")
    parser.add_argument("--dq-json", default="data/csv/final_listings_dq_report.json")
    return parser.parse_args()


def main() -> int:
    """CLI entry point for the wrangling pipeline."""
    args = parse_args()
    report = run_pipeline(Path(args.input_dir), Path(args.output_csv), Path(args.dq_json))
    print("Wrangling completed")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
