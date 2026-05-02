from __future__ import annotations

"""Preprocess the wrangled listings dataset into a model-ready table.

This module operates on the enriched/wrangled CSV and applies a sequence of
lightweight feature engineering and data-quality steps:
- infer missing structured fields from listing titles
- synthesize normalized descriptions
- fill transaction type with title/price heuristics
- optionally geocode cities into latitude/longitude
- remove implausible numeric values
- impute remaining numeric/categorical gaps

The output is a cleaner dataset intended for downstream ML, analytics, and
multimodal experiments.
"""

import argparse
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

try:
    from geopy.geocoders import Nominatim# type: ignore[import]
except (ImportError, ModuleNotFoundError):  
    Nominatim = None  # type: ignore[assignment]


@dataclass
class PreprocessConfig:
    """Runtime configuration for the preprocessing pipeline."""
    input_csv: str = "data/csv/final_listings_wrangled_enriched.csv"
    output_csv: str = "data/csv/final_listings_preprocessed.csv"
    report_json: str = "data/csv/final_listings_preprocessing_report.json"
    geocode_sleep_sec: float = 0.25
    geocode_max_cities: int = 250
    knn_neighbors: int = 5


class BasePreprocessor:
    """Encapsulate preprocessing heuristics and end-to-end execution.

    The class keeps geocoding state/caching local to one run and exposes a
    ``preprocess`` method for in-memory use plus ``run`` for file-based CLI use.
    """
    DROP_COLUMNS = [
        "source",
        "source_file",
        "listing_url",
        "neighborhood",
        "posted_at",
        "scraped_at",
        "currency",
        "image_url",
        "image_count",
    ]

    AMENITY_KEYWORDS = [
        "parking",
        "garage",
        "ascenseur",
        "elevator",
        "terrasse",
        "balcon",
        "jardin",
        "piscine",
        "clim",
        "chauffage",
        "meuble",
        "haut standing",
        "vue mer",
        "securite",
        "gardee",
        "metro",
        "proche commodites",
    ]

    def __init__(self, cfg: PreprocessConfig = PreprocessConfig()):
        """Initialize the preprocessor and optional geocoder resources."""
        self.cfg = cfg
        self._geo_cache: dict[str, tuple[float | None, float | None]] = {}
        self._geolocator = Nominatim(user_agent="estate_mind_preprocessor") if Nominatim is not None else None

    @staticmethod
    def _clean_text(value: Any) -> str:
        """Convert a scalar to normalized display text, preserving empty as ``''``."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        return re.sub(r"\s+", " ", str(value)).strip()

    @classmethod
    def _norm(cls, value: Any) -> str:
        """Create a lowercased ASCII-like text form for rule-based matching."""
        text = cls._clean_text(value).lower()
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        text = re.sub(r"[^a-z0-9+\s'-]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _to_numeric(value: Any) -> float | None:
        """Extract a numeric value from noisy text fields."""
        text = re.sub(r"\s+", "", str(value or ""))
        if not text:
            return None
        text = text.replace("DT", "").replace("TND", "").replace(",", ".")
        m = re.search(r"-?\d+(?:\.\d+)?", text)
        if not m:
            return None
        try:
            return float(m.group(0))
        except ValueError:
            return None

    def _extract_property_type(self, text: str) -> str | None:
        """Infer a coarse property type from title text."""
        blob = self._norm(text)
        if any(k in blob for k in ["appart", "apartment", "flat", "studio", "s+"]):
            return "Appartement"
        if any(k in blob for k in ["maison", "house", "villa", "duplex", "triplex"]):
            return "Maison"
        if any(k in blob for k in ["terrain", "land", "lot"]):
            return "Terrain"
        return None

    def _extract_transaction_type(self, text: str) -> str | None:
        """Infer whether a listing is for rent or sale from text cues."""
        blob = self._norm(text)
        if any(k in blob for k in ["a louer", "louer", "location", "rent"]):
            return "rent"
        if any(k in blob for k in ["a vendre", "vendre", "vente", "sale"]):
            return "sale"
        return None

    def _extract_rooms(self, text: str) -> int | None:
        """Infer room counts from common listing title conventions."""
        blob = self._norm(text)
        if re.search(r"\b(studio|garconniere)\b", blob):
            return 2
        m = re.search(r"\bs\s*\+\s*(\d+)\b", blob)
        if m:
            return int(m.group(1)) + 1
        m = re.search(r"\bs\s*(\d+)\b", blob)
        if m:
            return int(m.group(1)) + 1
        m = re.search(r"\b(\d+)\s*(?:pieces?|piece|p)\b", blob)
        if m:
            return int(m.group(1))
        return None

    def _extract_bedrooms(self, text: str) -> int | None:
        """Infer bedroom counts from listing title conventions."""
        blob = self._norm(text)
        if re.search(r"\b(studio|garconniere)\b", blob):
            return 1
        m = re.search(r"\b(\d+)\s*(?:bedrooms?|chambres?|chb)\b", blob)
        if m:
            return int(m.group(1))
        m = re.search(r"\bs\s*\+\s*(\d+)\b", blob)
        if m:
            return int(m.group(1))
        m = re.search(r"\bs\s*(\d+)\b", blob)
        if m:
            return int(m.group(1))
        return None

    def _extract_bathrooms(self, text: str, inferred_from_splus: bool) -> int | None:
        """Infer bathroom counts, with a conservative fallback for ``S+n`` titles."""
        blob = self._norm(text)
        m = re.search(r"\b(\d+)\s*(?:bathrooms?|sdb|salle(?:s)? de bain)\b", blob)
        if m:
            return int(m.group(1))
        if re.search(r"\b(studio|garconniere)\b", blob):
            return 1
        if inferred_from_splus:
            return 1
        return None

    def _build_description(self, row: pd.Series, title: str) -> str:
        """Generate a synthetic French description from structured listing fields."""
        parts: list[str] = []
        prop = self._clean_text(row.get("property_type")) or "Bien immobilier"
        trans = self._clean_text(row.get("transaction_type"))
        city = self._clean_text(row.get("city"))
        gov = self._clean_text(row.get("governorate"))
        rooms = row.get("rooms")
        bedrooms = row.get("bedrooms")
        bathrooms = row.get("bathrooms")
        surface = row.get("surface_m2")
        price = row.get("price_tnd")

        action = "à louer" if trans == "rent" else "à vendre" if trans == "sale" else "disponible"
        location = ", ".join([v for v in [city, gov] if v])

        parts.append(f"{prop} {action}" + (f" à {location}." if location else "."))

        details: list[str] = []
        if pd.notna(surface):
            details.append(f"Surface: {float(surface):.0f} m2")
        if pd.notna(rooms):
            details.append(f"Pièces: {int(float(rooms))}")
        if pd.notna(bedrooms):
            details.append(f"Chambres: {int(float(bedrooms))}")
        if pd.notna(bathrooms):
            details.append(f"SDB: {int(float(bathrooms))}")
        if pd.notna(price):
            details.append(f"Prix: {float(price):.0f} TND")
        if details:
            parts.append(" | ".join(details) + ".")

        blob = self._norm(title)
        amenities = [k for k in self.AMENITY_KEYWORDS if k in blob]
        if amenities:
            parts.append("Atouts: " + ", ".join(sorted(set(amenities))) + ".")

        parts.append("Informations déduites automatiquement à partir du titre de l'annonce.")
        return " ".join(parts)

    def _apply_title_extraction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Populate missing structured fields by parsing the listing title."""
        out = df.copy()
        numeric_defaults = ["rooms", "bedrooms", "bathrooms"]
        object_defaults = ["property_type", "transaction_type", "description"]
        for col in numeric_defaults:
            if col not in out.columns:
                out[col] = np.nan
        for col in object_defaults:
            if col not in out.columns:
                out[col] = pd.Series([None] * len(out), dtype="object")
        if "title" not in out.columns:
            out["title"] = ""

        for idx, row in out.iterrows():
            title = self._clean_text(row.get("title"))
            if not title:
                continue

            inferred_splus = bool(re.search(r"\bs\s*\+?\s*\d+\b", self._norm(title)))

            if pd.isna(row.get("rooms")):
                rooms = self._extract_rooms(title)
                if rooms is not None:
                    out.at[idx, "rooms"] = rooms

            if pd.isna(row.get("bedrooms")):
                bedrooms = self._extract_bedrooms(title)
                if bedrooms is not None:
                    out.at[idx, "bedrooms"] = bedrooms

            if pd.isna(row.get("bathrooms")):
                bathrooms = self._extract_bathrooms(title, inferred_from_splus=inferred_splus)
                if bathrooms is not None:
                    out.at[idx, "bathrooms"] = bathrooms

            if not self._clean_text(row.get("property_type")):
                ptype = self._extract_property_type(title)
                if ptype:
                    out.at[idx, "property_type"] = ptype

            if not self._clean_text(row.get("transaction_type")):
                ttype = self._extract_transaction_type(title)
                if ttype:
                    out.at[idx, "transaction_type"] = ttype

            out.at[idx, "description"] = self._build_description(out.loc[idx], title)# type: ignore

        return out

    def _fill_transaction_type_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """Backfill missing transaction types using title cues and price heuristics."""
        out = df.copy()
        if "transaction_type" not in out.columns:
            out["transaction_type"] = np.nan
        if "price_tnd" not in out.columns:
            return out

        prices = pd.to_numeric(out["price_tnd"], errors="coerce")
        q1 = prices.quantile(0.25)
        threshold = q1 * 0.7 if pd.notna(q1) else np.nan

        for idx, row in out.iterrows():
            existing = self._clean_text(row.get("transaction_type")).lower()
            if existing in {"sale", "rent"}:
                continue
            title = self._clean_text(row.get("title"))
            cue = self._extract_transaction_type(title)
            if cue:
                out.at[idx, "transaction_type"] = cue
                continue

            price = self._to_numeric(row.get("price_tnd"))
            if price is not None and pd.notna(threshold):
                out.at[idx, "transaction_type"] = "rent" if price < threshold else "sale"
            else:
                out.at[idx, "transaction_type"] = "sale"
        return out

    def _impute_missing_with_knn(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute remaining gaps using KNN for numeric fields and mode for categoricals."""
        out = df.copy()

        # Numeric KNN on relevant housing/geospatial fields only.
        num_cols = [
            c
            for c in ["price_tnd", "surface_m2", "rooms", "bedrooms", "bathrooms", "latitude", "longitude"]
            if c in out.columns
        ]
        if num_cols:
            usable_cols = [c for c in num_cols if out[c].notna().any()]
        else:
            usable_cols = []
        if usable_cols:
            imputer = KNNImputer(n_neighbors=self.cfg.knn_neighbors, weights="distance")
            out[usable_cols] = imputer.fit_transform(out[usable_cols])

        for col in ["rooms", "bedrooms", "bathrooms"]:
            if col in out.columns:
                out[col] = np.round(pd.to_numeric(out[col], errors="coerce")).astype(float)

        # Categorical fallback (except transaction_type handled separately)
        cat_cols = out.select_dtypes(include=["object"]).columns.tolist()
        for col in cat_cols:
            if col == "transaction_type":
                continue
            mode = out[col].mode(dropna=True)
            fill_val = mode.iat[0] if not mode.empty else "Unknown"
            out[col] = out[col].fillna(fill_val)

        return out

    def _geocode_city(self, city: str) -> tuple[float | None, float | None]:
        """Resolve a city name to latitude/longitude with caching."""
        key = city.strip().lower()
        if key in self._geo_cache:
            return self._geo_cache[key]
        if self._geolocator is None:
            self._geo_cache[key] = (None, None)
            return (None, None)

        query = f"{city}, Tunisia"
        try:
            loc = self._geolocator.geocode(query, timeout=10.0) # type: ignore
            if loc:
                coords = (float(loc.latitude), float(loc.longitude)) # type: ignore
                self._geo_cache[key] = coords
                time.sleep(self.cfg.geocode_sleep_sec)
                return coords
        except Exception:
            pass
        self._geo_cache[key] = (None, None)
        return (None, None)

    def _add_lat_lon_from_city(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill latitude/longitude from city names, capped by config limits."""
        out = df.copy()
        if "latitude" not in out.columns:
            out["latitude"] = np.nan
        if "longitude" not in out.columns:
            out["longitude"] = np.nan
        if "city" not in out.columns:
            return out

        unique_cities = [
            self._clean_text(c)
            for c in out["city"].dropna().astype(str).unique().tolist()
            if self._clean_text(c)
        ]
        unique_cities = unique_cities[: self.cfg.geocode_max_cities]
        city_coords: dict[str, tuple[float | None, float | None]] = {}
        for city in unique_cities:
            city_coords[city] = self._geocode_city(city)

        for idx, row in out.iterrows():
            city = self._clean_text(row.get("city"))
            if not city:
                continue
            lat, lon = city_coords.get(city, (None, None))
            if lat is not None and lon is not None:
                out.at[idx, "latitude"] = lat
                out.at[idx, "longitude"] = lon
        return out

    def _fix_quality_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove implausible numeric values and deduplicate records."""
        out = df.copy()
        for col in ["price_tnd", "surface_m2", "rooms", "bedrooms", "bathrooms"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        if "price_tnd" in out.columns:
            q99_price = out["price_tnd"].quantile(0.99)
            upper_price = float(q99_price * 3) if pd.notna(q99_price) else 1e8
            out.loc[(out["price_tnd"] < 50) | (out["price_tnd"] > upper_price), "price_tnd"] = np.nan
        if "surface_m2" in out.columns:
            out.loc[(out["surface_m2"] < 15) | (out["surface_m2"] > 3000), "surface_m2"] = np.nan
        if "rooms" in out.columns:
            out.loc[(out["rooms"] < 1) | (out["rooms"] > 15), "rooms"] = np.nan
        if "bedrooms" in out.columns:
            out.loc[(out["bedrooms"] < 0) | (out["bedrooms"] > 10), "bedrooms"] = np.nan
        if "bathrooms" in out.columns:
            out.loc[(out["bathrooms"] < 0) | (out["bathrooms"] > 10), "bathrooms"] = np.nan

        if "record_id" in out.columns:
            out = out.drop_duplicates(subset=["record_id"], keep="last")
        else:
            out = out.drop_duplicates(keep="last")
        out = out.reset_index(drop=True)
        return out

    def build_report(self, df: pd.DataFrame) -> dict[str, Any]:
        """Build a compact preprocessing quality report for key tracked fields."""
        report: dict[str, Any] = {
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
        }
        tracked = ["price_tnd", "surface_m2", "rooms", "bedrooms", "bathrooms", "transaction_type", "city", "latitude", "longitude", "description"]
        for col in tracked:
            if col in df.columns:
                report[f"missing_{col}_ratio"] = float(df[col].isna().mean())
        if "price_tnd" in df.columns:
            report["invalid_price_rows"] = int((pd.to_numeric(df["price_tnd"], errors="coerce").fillna(0) <= 0).sum())
        if "surface_m2" in df.columns:
            report["invalid_surface_rows"] = int((pd.to_numeric(df["surface_m2"], errors="coerce").fillna(0) <= 0).sum())
        if "record_id" in df.columns:
            report["duplicate_record_id_ratio"] = float(df.duplicated(subset=["record_id"]).mean())
        return report

    def preprocess(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Run preprocessing in memory and return both data and QA metrics."""
        out = df.copy()
        out = self._apply_title_extraction(out)
        out = self._fill_transaction_type_advanced(out)
        out = out.drop(columns=[c for c in self.DROP_COLUMNS if c in out.columns], errors="ignore")
        out = self._add_lat_lon_from_city(out)
        out = self._fix_quality_issues(out)
        out = self._impute_missing_with_knn(out)
        if "price_tnd" in out.columns and "surface_m2" in out.columns:
            out["price_per_m2"] = out["price_tnd"] / out["surface_m2"].replace({0: np.nan})
        if "title" in out.columns:
            out = out.drop(columns=["title"])
        report = self.build_report(out)
        return out, report

    def run(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Run preprocessing from configured input/output paths."""
        input_path = Path(self.cfg.input_csv)
        output_path = Path(self.cfg.output_csv)
        report_path = Path(self.cfg.report_json)

        df = pd.read_csv(input_path)
        processed, report = self.preprocess(df)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        processed.to_csv(output_path, index=False, encoding="utf-8")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return processed, report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the preprocessing CLI."""
    parser = argparse.ArgumentParser(description="Run data preprocessing for EstateMind listings dataset.")
    parser.add_argument("--input-csv", default=PreprocessConfig.input_csv)
    parser.add_argument("--output-csv", default=PreprocessConfig.output_csv)
    parser.add_argument("--report-json", default=PreprocessConfig.report_json)
    parser.add_argument("--geocode-max-cities", type=int, default=PreprocessConfig.geocode_max_cities)
    parser.add_argument("--geocode-sleep-sec", type=float, default=PreprocessConfig.geocode_sleep_sec)
    parser.add_argument("--knn-neighbors", type=int, default=PreprocessConfig.knn_neighbors)
    return parser.parse_args()


def main() -> int:
    """CLI entry point for the preprocessing pipeline."""
    args = parse_args()
    cfg = PreprocessConfig(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        report_json=args.report_json,
        geocode_max_cities=args.geocode_max_cities,
        geocode_sleep_sec=args.geocode_sleep_sec,
        knn_neighbors=args.knn_neighbors,
    )
    preprocessor = BasePreprocessor(cfg)
    _, report = preprocessor.run()
    print("Preprocessing completed")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
