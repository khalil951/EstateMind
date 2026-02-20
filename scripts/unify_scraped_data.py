from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

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
    "governorate",
    "city",
    "neighborhood",
    "location_raw",
    "raw_governorate",
    "raw_city",
    "posted_at",
    "scraped_at",
    "page",
    "category",
    "meta_title",
    "labels",
]

GOVERNORATES = {
    "tunis": "Tunis",
    "ariana": "Ariana",
    "ben arous": "Ben Arous",
    "manouba": "Manouba",
    "nabeul": "Nabeul",
    "zaghouan": "Zaghouan",
    "bizerte": "Bizerte",
    "beja": "Beja",
    "jendouba": "Jendouba",
    "kef": "Kef",
    "siliana": "Siliana",
    "sousse": "Sousse",
    "monastir": "Monastir",
    "mahdia": "Mahdia",
    "sfax": "Sfax",
    "kairouan": "Kairouan",
    "kasserine": "Kasserine",
    "sidi bouzid": "Sidi Bouzid",
    "gabes": "Gabes",
    "medenine": "Medenine",
    "tataouine": "Tataouine",
    "gafsa": "Gafsa",
    "tozeur": "Tozeur",
    "kebili": "Kebili",
}

NOISE_GOV_VALUES = {
    "publicite",
    "publicité",
    "a la une",
    "à la une",
}


def slug_to_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("-", " ")).strip().title()


def normalize_space(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = re.sub(r"\s+", " ", str(value)).strip()
    return cleaned or None


def normalize_governorate(value: Optional[str]) -> Optional[str]:
    value = normalize_space(value)
    if not value:
        return None

    lower = value.lower()
    if lower.isdigit() or lower in NOISE_GOV_VALUES or "boutique" in lower:
        return None

    key = (
        lower.replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("à", "a")
        .replace("ï", "i")
    )
    key = re.sub(r"[^a-z\s]", " ", key)
    key = re.sub(r"\s+", " ", key).strip()
    return GOVERNORATES.get(key, value)


def normalize_property_type(value: Optional[str]) -> Optional[str]:
    value = (value or "").lower()
    if any(token in value for token in ["appart", "apartment"]):
        return "Appartement"
    if any(token in value for token in ["maison", "villa", "house"]):
        return "Maison"
    if any(token in value for token in ["terrain", "land", "ferme"]):
        return "Terrain"
    return None


def infer_transaction_type(*texts: Optional[str]) -> Optional[str]:
    blob = " ".join(t for t in texts if t).lower()
    if any(k in blob for k in ["a louer", "à louer", "location", "rent", "/louer/"]):
        return "rent"
    if any(k in blob for k in ["a vendre", "à vendre", "vente", "sale", "/vendre/"]):
        return "sale"
    return None


def parse_rooms(*texts: Optional[str]) -> Optional[int]:
    blob = " ".join(t for t in texts if t).lower()
    match = re.search(r"\bs\s*\+\s*(\d)\b", blob)
    if match:
        return int(match.group(1))
    return None


def parse_tayara_url(url: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not url:
        return None, None
    path_parts = [p for p in urlparse(url).path.split("/") if p]
    # /item/{category}/{governorate}/{city}/...
    if len(path_parts) >= 5 and path_parts[0] == "item":
        return slug_to_text(path_parts[2]), slug_to_text(path_parts[3])
    return None, None


def parse_bigdatis_location(raw: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    raw = normalize_space(raw)
    if not raw:
        return None, None, None
    parts = [normalize_space(p) for p in raw.split(",") if normalize_space(p)]
    if len(parts) >= 3:
        return parts[-1], parts[-2], parts[0]
    if len(parts) == 2:
        return parts[-1], parts[0], None
    return None, parts[0], None


def parse_mubawab_location(raw: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    raw = normalize_space(raw)
    if not raw:
        return None, None
    if " à " in raw:
        left, right = raw.split(" à ", 1)
        return normalize_space(right), normalize_space(left)
    return None, raw


def compute_record_id(source: Optional[str], url: Optional[str], title: Optional[str], scraped_at: Optional[str]) -> str:
    base = f"{source or ''}|{url or ''}|{title or ''}|{scraped_at or ''}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def read_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows
    return json.loads(path.read_text(encoding="utf-8"))


def discover_input_files(input_dir: Path) -> List[Path]:
    json_files = sorted(input_dir.glob("*_listings.json"))
    jsonl_files = sorted(input_dir.glob("*_listings.jsonl"))

    by_stem = {p.stem: p for p in json_files}
    for p in jsonl_files:
        if p.stem not in by_stem:
            by_stem[p.stem] = p

    return sorted(by_stem.values(), key=lambda p: p.name)


def normalize_row(raw: Dict[str, Any], source_file: str) -> Dict[str, Any]:
    source = normalize_space(raw.get("source"))
    url = normalize_space(raw.get("url"))
    title = normalize_space(raw.get("title"))
    description = normalize_space(raw.get("description"))
    location_raw = normalize_space(raw.get("governorate"))

    governorate = normalize_governorate(raw.get("governorate"))
    city = normalize_space(raw.get("city"))
    neighborhood = None

    src_key = (source or "").lower()
    if src_key == "tayara":
        url_gov, url_city = parse_tayara_url(url)
        governorate = normalize_governorate(url_gov) or governorate
        city = city or normalize_space(url_city)
    elif src_key in {"bigdatis"}:
        loc_gov, loc_city, loc_neighborhood = parse_bigdatis_location(raw.get("governorate"))
        governorate = governorate or normalize_governorate(loc_gov)
        city = city or loc_city
        neighborhood = loc_neighborhood
    elif src_key in {"mubawab", "mubaweb"}:
        loc_city, loc_neighborhood = parse_mubawab_location(raw.get("governorate"))
        city = city or loc_city
        neighborhood = loc_neighborhood

    property_type = normalize_property_type(raw.get("property_type"))

    price = raw.get("price")
    surface = raw.get("surface_area")

    try:
        price = float(price) if price is not None else None
    except (TypeError, ValueError):
        price = None

    try:
        surface = float(surface) if surface is not None else None
    except (TypeError, ValueError):
        surface = None

    price_per_m2 = None
    if price and surface and surface > 0:
        price_per_m2 = round(price / surface, 2)

    labels = raw.get("labels")
    if isinstance(labels, list):
        labels = " | ".join(str(x) for x in labels)
    else:
        labels = normalize_space(labels)

    row = {
        "record_id": compute_record_id(source, url, title, raw.get("scraped_at")),
        "source": source,
        "source_file": source_file,
        "listing_url": url,
        "title": title,
        "description": description,
        "transaction_type": infer_transaction_type(url, title, description, raw.get("meta_title")),
        "property_type": property_type,
        "price_tnd": price,
        "surface_m2": surface,
        "price_per_m2": price_per_m2,
        "rooms": parse_rooms(title, description, raw.get("meta_title"), labels),
        "governorate": governorate,
        "city": city,
        "neighborhood": neighborhood,
        "location_raw": location_raw,
        "raw_governorate": normalize_space(raw.get("governorate")),
        "raw_city": normalize_space(raw.get("city")),
        "posted_at": normalize_space(raw.get("posted_at")),
        "scraped_at": normalize_space(raw.get("scraped_at")),
        "page": raw.get("page"),
        "category": normalize_space(raw.get("category")),
        "meta_title": normalize_space(raw.get("meta_title")),
        "labels": labels,
    }

    return row


def build_centralized_csv(input_dir: Path, output_csv: Path) -> pd.DataFrame:
    files = discover_input_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No *_listings.json or *_listings.jsonl files found in: {input_dir}")

    normalized_rows: List[Dict[str, Any]] = []
    for file_path in files:
        rows = read_rows(file_path)
        normalized_rows.extend(normalize_row(row, file_path.name) for row in rows)

    df = pd.DataFrame(normalized_rows)
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[CANONICAL_COLUMNS]

    # Remove exact duplicates by source+URL and keep latest scrape if available.
    df = df.sort_values(by=["scraped_at"], na_position="first")
    df = df.drop_duplicates(subset=["source", "listing_url"], keep="last")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unify scraped listings into one centralized CSV schema.")
    parser.add_argument("--input-dir", default="data", help="Directory containing *_listings.json/jsonl files.")
    parser.add_argument(
        "--output-csv",
        default="data/csv/centralized_listings.csv",
        help="Output path for unified CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)

    df = build_centralized_csv(input_dir=input_dir, output_csv=output_csv)
    print(f"Wrote {len(df)} rows to {output_csv}")
    print(f"Columns ({len(df.columns)}): {', '.join(df.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
