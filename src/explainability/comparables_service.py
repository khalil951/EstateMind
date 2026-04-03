"""Comparable listing retrieval and market context estimation.

The comparable search is built on the preprocessed historical listings
dataset. It returns human-readable comparable entries for the report layer
and also derives a lightweight market context used by heuristic pricing and
confidence scoring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class ComparablesService:
    """Find structured comparables from the historical listings dataset."""

    def __init__(self, listings_path: str | Path | None = None) -> None:
        """Load the comparable listings source from the preferred CSV path."""

        root = Path(__file__).resolve().parents[2]
        default_candidates = [
            root / "data" / "csv" / "preprocessed" / "final_listings_preprocessed.csv",
            root / "data" / "csv" / "final_listings_preprocessed.csv",
        ]
        self.listings_path = Path(listings_path) if listings_path else next((p for p in default_candidates if p.exists()), default_candidates[0])
        self._df = self._load_df()

    def _load_df(self) -> pd.DataFrame:
        """Load and lightly normalize the listings dataset used for matching."""

        if not self.listings_path.exists():
            return pd.DataFrame()
        df = pd.read_csv(self.listings_path)
        for col in ("price_tnd", "surface_m2", "rooms", "bedrooms", "bathrooms", "price_per_m2"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ("property_type", "governorate", "city", "description", "transaction_type"):
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)
        return df

    @staticmethod
    def _normalize_transaction_type(value: Any) -> str:
        raw = str(value or "").strip().lower()
        if raw.startswith("rent"):
            return "rent"
        if raw.startswith("sale"):
            return "sale"
        return "unknown"

    @staticmethod
    def _detect_time_column(df: pd.DataFrame) -> str | None:
        for candidate in ("listing_date", "listed_at", "created_at", "scraped_at", "published_at", "date"):
            if candidate in df.columns:
                return candidate
        return None

    def _derive_trend(self, subset: pd.DataFrame) -> tuple[str, str | None]:
        time_col = self._detect_time_column(subset)
        if time_col is None:
            return "Unavailable", "Missing temporal listing columns in comparable dataset"

        trend_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(subset[time_col], errors="coerce"),
                "price_per_m2": pd.to_numeric(subset.get("price_per_m2"), errors="coerce"), # type: ignore
            }
        ).dropna()
        if len(trend_df) < 8:
            return "Unavailable", f"Insufficient dated samples for trend calculation ({len(trend_df)} found)"

        trend_df = trend_df.sort_values("timestamp")
        split_idx = max(int(len(trend_df) * 0.7), 1)
        earlier = float(trend_df.iloc[:split_idx]["price_per_m2"].median())
        recent_slice = trend_df.iloc[split_idx:]
        if recent_slice.empty:
            return "Unavailable", "Insufficient recent samples for trend comparison"
        recent = float(recent_slice["price_per_m2"].median())
        if earlier <= 0:
            return "Unavailable", "Invalid historical baseline for trend comparison"

        delta_pct = ((recent / earlier) - 1.0) * 100.0
        if delta_pct >= 3.0:
            return "Rising", f"Recent median price/m2 is {delta_pct:.1f}% above earlier period"
        if delta_pct <= -3.0:
            return "Softening", f"Recent median price/m2 is {abs(delta_pct):.1f}% below earlier period"
        return "Stable", f"Median price/m2 shift is {delta_pct:.1f}% across comparable periods"

    def _similarity(self, row: pd.Series, mapped: dict[str, Any]) -> int:
        """Score how similar one historical listing is to the request."""
        score = 100.0
        request_txn = self._normalize_transaction_type(mapped.get("transaction_type"))
        row_txn = self._normalize_transaction_type(row.get("transaction_type"))
        if request_txn != "unknown" and row_txn != "unknown" and row_txn != request_txn:
            score -= 8
        if str(row.get("property_type", "")).strip().lower() != str(mapped["property_type"]).strip().lower():
            score -= 30
        if str(row.get("governorate", "")).strip().lower() != str(mapped["governorate"]).strip().lower():
            score -= 15
        if str(row.get("city", "")).strip().lower() != str(mapped["city"]).strip().lower():
            score -= 10
        surface = float(row.get("surface_m2", 0.0) or 0.0)
        target_surface = float(mapped["surface_m2"])
        if surface > 0 and target_surface > 0:
            score -= min(abs(surface - target_surface) / max(target_surface, 1.0) * 25.0, 20.0)
        for field, weight in (("bedrooms", 6), ("bathrooms", 4)):
            value = row.get(field)
            if pd.notna(value):
                score -= min(abs(float(value) - float(mapped[field])) * weight, weight * 2)
        return max(0, min(99, int(round(score))))

    def _difference_summary(self, row: pd.Series, mapped: dict[str, Any]) -> str:
        """Describe the most salient structured differences for reporting."""
        parts: list[str] = []
        surface = float(row.get("surface_m2", 0.0) or 0.0)
        if surface > 0:
            delta = int(round(surface - float(mapped["surface_m2"])))
            if delta != 0:
                parts.append(f"{abs(delta)} m2 {'larger' if delta > 0 else 'smaller'}")
        for field, label in (("bedrooms", "bedroom"), ("bathrooms", "bathroom")):
            value = row.get(field)
            if pd.notna(value):
                delta = int(round(float(value) - float(mapped[field])))
                if delta != 0:
                    parts.append(f"{abs(delta)} {label}{'s' if abs(delta) > 1 else ''} {'more' if delta > 0 else 'less'}")
        return ", ".join(parts[:2]) if parts else "Closest structured match"

    def _market_context(self, mapped: dict[str, Any], predicted_per_m2: int) -> dict[str, Any]:
        """Summarize market-level context for the relevant city/type scope."""
        if self._df.empty or "price_per_m2" not in self._df.columns:
            avg_m2 = max(int(predicted_per_m2 * 0.93), 1)
            return {
                "avg_m2": avg_m2,
                "property_m2": int(predicted_per_m2),
                "delta_pct": round(((predicted_per_m2 / avg_m2) - 1) * 100),
                "trend": "Unavailable",
                "trend_reason": "Comparable dataset is unavailable for trend computation",
                "demand": "Limited market sample",
                "city": mapped["city"],
            }
        df = self._df
        city_mask = df["city"].astype(str).str.strip().str.lower() == str(mapped["city"]).strip().lower()
        type_mask = df["property_type"].astype(str).str.strip().str.lower() == str(mapped["property_type"]).strip().lower()
        subset = df[city_mask & type_mask]
        if subset.empty:
            subset = df[type_mask]
        if subset.empty:
            subset = df
        avg_m2 = int(round(float(subset["price_per_m2"].dropna().median()))) if subset["price_per_m2"].notna().any() else max(int(predicted_per_m2 * 0.93), 1)
        sample_size = int(len(subset))
        trend, trend_reason = self._derive_trend(subset)
        return {
            "avg_m2": avg_m2,
            "property_m2": int(predicted_per_m2),
            "delta_pct": round(((predicted_per_m2 / max(avg_m2, 1)) - 1) * 100),
            "trend": trend,
            "trend_reason": trend_reason,
            "demand": f"{sample_size} comparable listing(s) in scope",
            "city": mapped["city"],
        }

    def find(self, mapped: dict[str, Any], limit: int = 4) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Return the top comparable listings and a derived market summary."""

        if self._df.empty:
            fallback_market = self._market_context(mapped, predicted_per_m2=1450)
            return [], fallback_market

        df = self._df.copy()
        df["similarity"] = df.apply(lambda row: self._similarity(row, mapped), axis=1)
        df = df.sort_values(["similarity", "price_tnd"], ascending=[False, True]).head(limit)
        comparables: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            size = int(round(float(row.get("surface_m2", mapped["surface_m2"]) or mapped["surface_m2"])))
            price = int(round(float(row.get("price_tnd", 0.0) or 0.0)))
            comparables.append(
                {
                    "address": ", ".join(
                        [part for part in [str(row.get("city", "")).strip(), str(row.get("governorate", "")).strip()] if part]
                    )
                    or "Comparable listing",
                    "price": price,
                    "size": size,
                    "transaction_type": self._normalize_transaction_type(row.get("transaction_type")),
                    "similarity": int(row["similarity"]),
                    "difference": self._difference_summary(row, mapped),
                }
            )

        valid_ppm = [comp["price"] / max(comp["size"], 1) for comp in comparables if comp["price"] > 0 and comp["size"] > 0]
        predicted_per_m2 = int(round(sum(valid_ppm) / len(valid_ppm))) if valid_ppm else 1450
        market_context = self._market_context(mapped, predicted_per_m2=predicted_per_m2)
        return comparables, market_context
