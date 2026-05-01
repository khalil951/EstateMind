"""Token-level NLP explainability helpers for EstateMind."""

from __future__ import annotations

import re
from typing import Any


POSITIVE_WORDS = {
    "luxury": 0.95,
    "renovated": 0.9,
    "renove": 0.82,
    "bright": 0.65,
    "spacious": 0.75,
    "modern": 0.72,
    "premium": 0.8,
    "sea view": 1.0,
    "parking": 0.35,
    "garden": 0.45,
    "pool": 0.7,
    "piscine": 0.7,
    "clean": 0.4,
    "maintained": 0.6,
    "close": 0.45,
}

NEGATIVE_WORDS = {
    "small": -0.75,
    "unfinished": -0.9,
    "far": -0.6,
    "dark": -0.7,
    "noisy": -0.6,
    "old": -0.45,
    "damaged": -0.85,
    "basic": -0.35,
    "limited": -0.3,
}

QUALITY_WORDS = {
    "luxury": 1.0,
    "renovated": 0.85,
    "modern": 0.7,
    "bright": 0.55,
    "spacious": 0.8,
    "premium": 0.75,
    "clean": 0.4,
    "maintained": 0.6,
}


class NLPExplainabilityService:
    """Generate simple token contributions for sentiment, quality, and marketing signals."""

    @staticmethod
    def _tokenize(description: str) -> list[str]:
        return [token.lower() for token in re.findall(r"\b[\w']+\b", str(description or "")) if len(token) > 1]

    @staticmethod
    def _score_tokens(tokens: list[str], lexicon: dict[str, float], *, aspect: str) -> list[dict[str, Any]]:
        scores: list[dict[str, Any]] = []
        token_set = set(tokens)
        joined = " ".join(tokens)
        for term, score in lexicon.items():
            if " " in term:
                if term not in joined:
                    continue
                token = term
            else:
                if term not in token_set:
                    continue
                token = term
            scores.append(
                {
                    "token": token,
                    "contribution": round(score, 3),
                    "aspect": aspect,
                    "magnitude": round(abs(score), 3),
                    "polarity": "positive" if score > 0 else "negative",
                    "explanation": f"{token} is a {aspect.replace('_', ' ')} signal with weight {score:+.2f}.",
                }
            )
        return scores

    def analyze(
        self,
        description: str,
        description_analysis: dict[str, Any],
        description_sentiment: dict[str, Any],
        location_sentiment: dict[str, Any],
        comparables: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return token contributions and comparative location explainability."""

        tokens = self._tokenize(description)
        sentiment_tokens = self._score_tokens(tokens, {**POSITIVE_WORDS, **NEGATIVE_WORDS}, aspect="description_sentiment")
        quality_tokens = self._score_tokens(tokens, QUALITY_WORDS, aspect="description_quality")
        marketing_tokens = self._score_tokens(tokens, {**QUALITY_WORDS, **POSITIVE_WORDS}, aspect="marketing_quality")

        key_phrases = [str(item).strip().lower() for item in description_analysis.get("key_phrases", []) if str(item).strip()]
        for phrase in key_phrases[:5]:
            quality_tokens.append(
                {
                    "token": phrase,
                    "contribution": 0.35,
                    "aspect": "description_quality",
                    "magnitude": 0.35,
                    "polarity": "positive",
                    "explanation": f"{phrase} is a high-signal phrase in the listing description.",
                }
            )

        location_value = float(location_sentiment.get("sentiment", 0.5))
        location_label = str(location_sentiment.get("sentiment_label", "neutral"))
        location_percentile = round(max(0.0, min(100.0, location_value * 100.0)), 1)

        comparable_locations: list[str] = []
        for item in comparables or []:
            address = str(item.get("address", "")).strip()
            if not address:
                continue
            city = address.split(",")[0].strip()
            if city and city not in comparable_locations:
                comparable_locations.append(city)
            if len(comparable_locations) >= 3:
                break

        if location_label == "positive":
            benchmark_message = "This location scores above the neutral midpoint thanks to stronger local demand signals."
        elif location_label == "negative":
            benchmark_message = "This location scores below the neutral midpoint, suggesting weaker local demand or convenience."
        else:
            benchmark_message = "This location sits near the neutral benchmark, so neighborhood effects are limited."

        return {
            "description_sentiment_tokens": sorted(sentiment_tokens, key=lambda item: abs(float(item["contribution"])), reverse=True)[:12],
            "description_quality_tokens": sorted(quality_tokens, key=lambda item: abs(float(item["contribution"])), reverse=True)[:12],
            "marketing_quality_tokens": sorted(marketing_tokens, key=lambda item: abs(float(item["contribution"])), reverse=True)[:12],
            "location_comparison": {
                "location_sentiment": location_value,
                "location_sentiment_label": location_label,
                "location_percentile": location_percentile,
                "benchmark_message": benchmark_message,
                "similar_locations": comparable_locations,
                "description_sentiment": float(description_sentiment.get("description_sentiment", 0.5)),
            },
        }