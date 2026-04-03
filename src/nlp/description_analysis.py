"""Description-level NLP heuristics used during property valuation.

This module extracts lightweight signals from the free-text listing
description, such as key phrases, amenity mentions, and a coarse marketing
quality score. The outputs are intended as auxiliary evidence for pricing and
explainability rather than as a standalone valuation model.
"""

from __future__ import annotations

import re
from typing import Any


STOPWORDS = {
    "avec",
    "dans",
    "pour",
    "this",
    "that",
    "the",
    "and",
    "les",
    "des",
    "une",
    "appartement",
    "maison",
    "terrain",
    "property",
}

AMENITY_KEYWORDS = {
    "pool",
    "piscine",
    "garden",
    "jardin",
    "parking",
    "garage",
    "sea",
    "mer",
    "terrace",
    "terrasse",
    "elevator",
    "ascenseur",
    "renove",
    "renovated",
    "meuble",
}


class DescriptionAnalysisService:
    """Produce simple text-derived quality and feature signals."""

    def analyze(self, description: str) -> dict[str, Any]:
        """Analyze a property description and return explainability features."""

        text = str(description or "").strip()
        tokens = [t for t in re.findall(r"\b[\w']+\b", text.lower()) if len(t) > 2]
        keywords = [t for t in tokens if t not in STOPWORDS]
        unique_keywords: list[str] = []
        for token in keywords:
            if token not in unique_keywords:
                unique_keywords.append(token)
            if len(unique_keywords) >= 5:
                break

        amenity_hits = sum(1 for token in set(tokens) if token in AMENITY_KEYWORDS)
        richness = min(len(tokens) / 40.0, 1.0)
        score = min(0.35 + (0.35 * richness) + (0.06 * amenity_hits), 0.95) if text else 0.1
        if score >= 0.8:
            quality = "Professional"
            effectiveness = "High"
        elif score >= 0.6:
            quality = "Good"
            effectiveness = "Medium"
        else:
            quality = "Basic"
            effectiveness = "Low"

        return {
            "description_quality": f"{quality} ({round(score * 10, 1)}/10)",
            "description_score": round(score, 3),
            "marketing_effectiveness": effectiveness,
            "key_phrases": unique_keywords,
            "token_count": len(tokens),
        }
