"""Location sentiment lookup based on prepared review aggregates.

The service reads precomputed review metadata and exposes a stable,
deterministic sentiment summary keyed by city or neighborhood. This keeps the
online inference path lightweight and avoids live generative sentiment
analysis at request time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class LocationSentimentService:
    """Lookup neighborhood sentiment from the prepared review metadata."""

    def __init__(self, reviews_path: str | Path | None = None) -> None:
        """Load review aggregates from disk when they are available."""

        root = Path(__file__).resolve().parents[2]
        default_path = root / "data" / "nlp" / "real_estate_reviews_metadata.csv"
        self.reviews_path = Path(reviews_path) if reviews_path else default_path
        self._summary = self._load_summary()

    def _load_summary(self) -> dict[str, dict[str, Any]]:
        """Build a lowercase lookup table of aggregated sentiment by place."""

        if not self.reviews_path.exists():
            return {}
        df = pd.read_csv(self.reviews_path)
        if "city" not in df.columns or "sentiment" not in df.columns:
            return {}
        mapping = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
        df = df.copy()
        df["city"] = df["city"].fillna("").astype(str).str.strip().str.lower()
        df["sentiment_num"] = df["sentiment"].astype(str).str.lower().map(mapping).fillna(0.0)
        grouped = (
            df[df["city"].ne("")]
            .groupby("city")
            .agg(review_count=("sentiment_num", "size"), avg_sentiment=("sentiment_num", "mean"))
            .reset_index()
        )
        out: dict[str, dict[str, Any]] = {}
        for _, row in grouped.iterrows():
            avg = float(row["avg_sentiment"])
            if avg > 0.2:
                label = "positive"
            elif avg < -0.2:
                label = "negative"
            else:
                label = "neutral"
            out[str(row["city"])] = {
                "review_count": int(row["review_count"]),
                "avg_sentiment": avg,
                "label": label,
                "score_01": round((avg + 1.0) / 2.0, 3),
            }
        return out

    def analyze(self, city: str, neighborhood: str = "") -> dict[str, Any]:
        """Return sentiment metadata for a neighborhood or city fallback."""

        keys = [str(neighborhood).strip().lower(), str(city).strip().lower()]
        for key in keys:
            if key and key in self._summary:
                row = self._summary[key]
                return {
                    "sentiment": row["score_01"],
                    "sentiment_label": row["label"],
                    "review_count": row["review_count"],
                    "location_key": key,
                }
        return {
            "sentiment": 0.5,
            "sentiment_label": "neutral",
            "review_count": 0,
            "location_key": "",
        }
